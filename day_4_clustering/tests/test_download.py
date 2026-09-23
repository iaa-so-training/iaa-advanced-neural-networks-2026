"""Tests for the allStar downloader without any network access."""

from __future__ import annotations

import http.server
import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import click
import pytest

from cluster import download as download_module
from cluster.download import _already_downloaded, download_allstar, stream_to_file


# --------------------------------------------------------------------------- #
# A local HTTP server that speaks Range requests, so the resumable downloader
# can be tested for real without touching the network.
# --------------------------------------------------------------------------- #

_PAYLOAD = bytes(range(256)) * 4096          # 1 MB, byte-patterned


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = _PAYLOAD
    honour_range = True
    seen_range_headers: list[str | None] = []

    def do_GET(self) -> None:                # noqa: N802 (http.server API)
        type(self).seen_range_headers.append(self.headers.get("Range"))
        data = type(self).payload
        requested = self.headers.get("Range")
        if requested and type(self).honour_range:
            start = int(requested.split("=")[1].split("-")[0])
            body = data[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:   # noqa: A002 (http.server API)
        return


@pytest.fixture
def file_server() -> Any:
    """Yields ``(url, handler_class)`` backed by a throwaway local server."""
    handler = type("Handler", (_RangeHandler,), {"seen_range_headers": []})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}/data.bin", handler
    finally:
        server.shutdown()
        server.server_close()


def test_stream_to_file_resumes_from_a_partial_file(
    tmp_path: Path,
    file_server: Any,
) -> None:
    url, handler = file_server
    destination = tmp_path / "partial.bin"
    offset = 300_000
    destination.write_bytes(_PAYLOAD[:offset])

    stream_to_file(url, destination)

    assert destination.read_bytes() == _PAYLOAD
    assert handler.seen_range_headers == [f"bytes={offset}-"], "must ask for the rest, not restart"


def test_stream_to_file_restarts_when_server_ignores_range(
    tmp_path: Path,
    file_server: Any,
) -> None:
    url, handler = file_server
    handler.honour_range = False
    destination = tmp_path / "stale.bin"
    destination.write_bytes(b"garbage from an older attempt")

    stream_to_file(url, destination)

    assert destination.read_bytes() == _PAYLOAD, "a 200 answer must overwrite, not append"


def test_download_allstar_fails_loudly_on_a_truncated_file(
    tmp_path: Path,
    file_server: Any,
) -> None:
    url, _ = file_server
    destination = tmp_path / "catalogue.fits.gz"

    with pytest.raises(click.ClickException, match="incomplete"):
        download_allstar(destination, url=url)     # payload is 1 MB, not ALLSTAR_BYTES


def test_download_allstar_skips_a_complete_file_without_contacting_the_server(
    tmp_path: Path,
    file_server: Any,
    monkeypatch: Any,
) -> None:
    url, handler = file_server
    monkeypatch.setattr(download_module, "ALLSTAR_BYTES", len(_PAYLOAD))
    destination = tmp_path / "catalogue.fits.gz"
    destination.write_bytes(_PAYLOAD)

    download_allstar(destination, url=url)

    assert handler.seen_range_headers == [], "a complete local file must not hit the network"


def test_already_downloaded_requires_exact_byte_size(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(download_module, "ALLSTAR_BYTES", 5)
    good = tmp_path / "good.fits"
    good.write_bytes(b"12345")
    assert _already_downloaded(good) is True

    short = tmp_path / "short.fits"
    short.write_bytes(b"1234")
    assert _already_downloaded(short) is False

    assert _already_downloaded(tmp_path / "missing.fits") is False


def test_download_allstar_skips_when_cache_is_complete(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    destination = tmp_path / "data" / "allStar.fits"
    destination.parent.mkdir()
    destination.write_bytes(b"12345")
    monkeypatch.setattr(download_module, "ALLSTAR_BYTES", 5)

    def fail_if_called(cmd: list[str]) -> SimpleNamespace:
        raise AssertionError(f"wget should not run: {cmd}")

    monkeypatch.setattr(download_module.subprocess, "run", fail_if_called)

    result = download_allstar(destination)
    assert result == destination
    assert "already present" in capsys.readouterr().out


def test_download_allstar_success(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
    file_server: Any,
) -> None:
    url, _ = file_server
    monkeypatch.setattr(download_module, "ALLSTAR_BYTES", len(_PAYLOAD))
    destination = tmp_path / "allStar.fits"

    result = download_allstar(destination, url=url)

    assert result == destination
    assert _already_downloaded(destination) is True
    assert destination.read_bytes() == _PAYLOAD
    assert "Download complete" in capsys.readouterr().out


def test_download_allstar_propagates_network_errors(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A dead endpoint must raise, not silently leave a stub file behind."""
    import socket

    import requests

    with socket.socket() as probe:              # a port nobody is listening on
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
    monkeypatch.setattr(download_module, "ALLSTAR_BYTES", 5)

    with pytest.raises(requests.exceptions.RequestException):
        download_allstar(tmp_path / "allStar.fits", url=f"http://127.0.0.1:{dead_port}/x")


def test_download_allstar_keeps_the_partial_file_for_a_resume(
    tmp_path: Path,
    monkeypatch: Any,
    file_server: Any,
) -> None:
    url, _ = file_server
    monkeypatch.setattr(download_module, "ALLSTAR_BYTES", len(_PAYLOAD) + 10_000)  # never satisfied
    destination = tmp_path / "allStar.fits"

    with pytest.raises(click.ClickException, match="incomplete"):
        download_allstar(destination, url=url)

    assert destination.exists(), "the bytes already fetched must survive for the next run"
    assert destination.stat().st_size == len(_PAYLOAD)


def test_download_allstar_error_names_both_byte_counts(
    tmp_path: Path,
    monkeypatch: Any,
    file_server: Any,
) -> None:
    """The message has to be actionable: what came down vs what was expected."""
    url, _ = file_server
    monkeypatch.setattr(download_module, "ALLSTAR_BYTES", 5)
    destination = tmp_path / "allStar.fits"

    with pytest.raises(click.ClickException) as excinfo:
        download_allstar(destination, url=url)

    message = str(excinfo.value)
    assert "Download incomplete" in message
    assert str(len(_PAYLOAD)) in message      # what we got
    assert "5" in message                     # what we expected


def test_download_click_command(monkeypatch: Any) -> None:
    """The standalone click command delegates to download_allstar."""
    from click.testing import CliRunner

    seen: list[str] = []

    def fake_download(destination: str | Path, url: str = download_module.ALLSTAR_URL) -> Path:
        seen.append(str(destination))
        return Path(destination)

    monkeypatch.setattr(download_module, "download_allstar", fake_download)
    result = CliRunner().invoke(download_module.download, ["--destination", "x.fits"])
    assert result.exit_code == 0
    assert seen == ["x.fits"]


# --------------------------------------------------------------------------- #
# Asset bundle (embeddings + checkpoints) — no network, fake Hub
# --------------------------------------------------------------------------- #


def _fake_manifest(tmp_path: Path, files: dict[str, bytes]) -> dict:
    """Manifest whose sha256/bytes match real bytes written to *tmp_path*."""
    import hashlib

    entries = []
    for name, payload in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(payload)
        entries.append(
            {
                "path": name,
                "source": f"data/{name}",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "consumer": "test",
            }
        )
    return {"n_files": len(entries), "files": entries}


def _patch_hub(monkeypatch: Any, manifest: dict, remote: Path, name: str = "MANIFEST.json") -> list[str]:
    """Serve the manifest + files from a fake Hub directory.

    The manifest is written to disk and pointed at through ``ASSETS_MANIFEST``
    (that is how it ships in real life); ``calls`` therefore records only the
    payload fetches.
    """
    (remote / name).write_text(json.dumps(manifest))
    monkeypatch.setattr(download_module, "ASSETS_MANIFEST", str(remote / name))
    calls: list[str] = []

    def fake_hf(repo_id: str, filename: str, repo_type: str) -> Path:
        calls.append(filename)
        return remote / filename

    monkeypatch.setattr(download_module, "_hf_download", fake_hf)
    return calls


def test_target_path_maps_models_next_to_parquets() -> None:
    assert download_module.target_path("data", "embeddings/a.parquet") == Path(
        "data/embeddings/a.parquet"
    )
    assert download_module.target_path("data", "models/m.pt") == Path("data/embeddings/m.pt")
    assert download_module.target_path("data", "optional/s.tar") == Path("data/s.tar")


def test_select_entries_filters_optional_and_only() -> None:
    manifest = {
        "files": [
            {"path": "embeddings/a.parquet"},
            {"path": "optional/s.tar"},
        ]
    }
    default = download_module.select_entries(manifest)
    assert [e["path"] for e in default] == ["embeddings/a.parquet"]

    with_optional = download_module.select_entries(manifest, include_optional=True)
    assert len(with_optional) == 2

    only = download_module.select_entries(manifest, only=("embeddings/a.parquet",))
    assert [e["path"] for e in only] == ["embeddings/a.parquet"]

    with pytest.raises(click.ClickException, match="unknown bundle path"):
        download_module.select_entries(manifest, only=("embeddings/nope.parquet",))


def test_download_assets_fetches_missing_and_skips_valid(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    remote = tmp_path / "hub"
    remote.mkdir()
    manifest = _fake_manifest(
        tmp_path / "hub", {"embeddings/a.parquet": b"aaaa", "models/m.pt": b"mmmm"}
    )
    calls = _patch_hub(monkeypatch, manifest, remote)

    root = tmp_path / "data"
    # pre-place the parquet so the second run is a pure skip
    fetched = download_module.download_assets(root=root, repo_id="x/y")
    assert len(fetched) == 2
    assert (root / "embeddings/a.parquet").read_bytes() == b"aaaa"
    assert (root / "embeddings/m.pt").read_bytes() == b"mmmm"  # models land in embeddings/

    out = capsys.readouterr().out
    assert "embeddings/a.parquet" in calls and "models/m.pt" in calls
    assert "MANIFEST.json" not in calls, "the manifest ships with the code — no network"
    assert "bundle ready" in out

    calls.clear()
    again = download_module.download_assets(root=root, repo_id="x/y")
    assert len(again) == 2
    assert "already present" in capsys.readouterr().out
    assert calls == []  # nothing at all is re-fetched


def test_download_assets_rejects_corrupt_payload(
    tmp_path: Path, monkeypatch: Any
) -> None:
    remote = tmp_path / "hub"
    remote.mkdir()
    manifest = _fake_manifest(remote, {"embeddings/a.parquet": b"good"})
    # corrupt the served bytes after the manifest hash was computed
    (remote / "embeddings/a.parquet").write_bytes(b"bad!")
    _patch_hub(monkeypatch, manifest, remote)

    with pytest.raises(click.ClickException, match="failed verification"):
        download_module.download_assets(root=tmp_path / "data", repo_id="x/y")


def test_download_assets_check_only_reports_missing(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    remote = tmp_path / "hub"
    remote.mkdir()
    manifest = _fake_manifest(remote, {"embeddings/a.parquet": b"good"})
    _patch_hub(monkeypatch, manifest, remote)

    with pytest.raises(click.ClickException, match="bundle incomplete"):
        download_module.download_assets(root=tmp_path / "data", repo_id="x/y", check_only=True)
    out = capsys.readouterr().out
    assert "missing" in out


def test_download_assets_list_only(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    remote = tmp_path / "hub"
    remote.mkdir()
    manifest = _fake_manifest(remote, {"embeddings/a.parquet": b"good"})
    _patch_hub(monkeypatch, manifest, remote)

    fetched = download_module.download_assets(
        root=tmp_path / "data", repo_id="x/y", list_only=True
    )
    assert fetched == []
    out = capsys.readouterr().out
    assert "embeddings/a.parquet" in out and "TOTAL" in out


def test_cli_download_assets_delegates(monkeypatch: Any) -> None:
    """`cluster download --assets` calls the bundle downloader, not wget."""
    from click.testing import CliRunner

    from cluster import cli

    seen: dict[str, Any] = {}

    def fake_assets(**kwargs: Any) -> list[Path]:
        seen.update(kwargs)
        return []

    def fail_allstar(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("catalogue download must not run for --assets")

    monkeypatch.setattr(download_module, "download_assets", fake_assets)
    monkeypatch.setattr(download_module, "download_allstar", fail_allstar)

    result = CliRunner().invoke(cli.download, ["--assets", "--only", "embeddings/a.parquet"])
    assert result.exit_code == 0, result.output
    assert seen["only"] == ("embeddings/a.parquet",)
    assert seen["list_only"] is False and seen["check_only"] is False


# --------------------------------------------------------------------------- #
# Manifest resolution: it ships with the code, so --list/--check work offline
# from any working directory (in the container the cwd is the mounted volume).
# --------------------------------------------------------------------------- #


def test_shipped_manifest_is_found_from_any_working_directory(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from cluster.config import ASSETS_MANIFEST, _default_manifest

    manifest_path = Path(ASSETS_MANIFEST)
    assert manifest_path.is_file(), "the repo/image must ship hf/MANIFEST.json"
    assert manifest_path.name == "MANIFEST.json"

    # resolving is cwd-independent
    monkeypatch.chdir(tmp_path)
    assert Path(_default_manifest()).is_file()
    manifest = download_module.load_manifest()
    assert manifest["files"], "the shipped manifest lists bundle entries"
    assert len(manifest["files"]) == manifest["n_files"]


def test_load_manifest_uses_the_local_copy_without_touching_the_hub(
    tmp_path: Path, monkeypatch: Any
) -> None:
    local = tmp_path / "MANIFEST.json"
    local.write_text(json.dumps({"n_files": 1, "files": [{"path": "embeddings/a.parquet"}]}))

    def boom(*args: Any, **kwargs: Any) -> Path:
        raise AssertionError("a manifest on disk must not be fetched from the Hub")

    monkeypatch.setattr(download_module, "_hf_download", boom)
    manifest = download_module.load_manifest(manifest_name=str(local))
    assert manifest["files"][0]["path"] == "embeddings/a.parquet"


def test_load_manifest_hub_failure_is_actionable(
    tmp_path: Path, monkeypatch: Any
) -> None:
    def dead_hub(*args: Any, **kwargs: Any) -> Path:
        raise RuntimeError("404 Client Error: Repository Not Found")

    monkeypatch.setattr(download_module, "_hf_download", dead_hub)

    with pytest.raises(click.ClickException, match="not found on disk"):
        download_module.load_manifest(manifest_name=str(tmp_path / "absent.json"))


def test_hub_hint_names_the_repo_and_the_override() -> None:
    error = download_module._hub_hint("owner/name", "dataset", RuntimeError("boom"))
    text = str(error)
    assert "datasets/owner/name" in text
    assert "CLUSTER_HF_REPO" in text
    assert "resumes" in text, "students should know a rerun continues the download"
