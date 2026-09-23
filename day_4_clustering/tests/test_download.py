"""Tests for the allStar downloader without any network access."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import click
import pytest

from cluster import download as download_module
from cluster.download import _already_downloaded, download_allstar


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
) -> None:
    monkeypatch.setattr(download_module, "ALLSTAR_BYTES", 5)
    destination = tmp_path / "allStar.fits"

    def fake_run(cmd: list[str]) -> SimpleNamespace:
        # cmd[-2] is the destination passed after ``-O``.
        Path(cmd[-2]).write_bytes(b"12345")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(download_module.subprocess, "run", fake_run)

    result = download_allstar(destination)
    assert result == destination
    assert _already_downloaded(destination) is True
    assert "Download complete" in capsys.readouterr().out


def test_download_allstar_raises_on_nonzero_wget(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    def fake_run(cmd: list[str]) -> SimpleNamespace:
        return SimpleNamespace(returncode=8)

    monkeypatch.setattr(download_module, "ALLSTAR_BYTES", 5)
    monkeypatch.setattr(download_module.subprocess, "run", fake_run)
    with pytest.raises(click.ClickException, match="Download failed"):
        download_allstar(tmp_path / "allStar.fits")


def test_download_allstar_raises_when_size_does_not_match(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(download_module, "ALLSTAR_BYTES", 5)
    destination = tmp_path / "allStar.fits"

    def fake_run(cmd: list[str]) -> SimpleNamespace:
        Path(cmd[-2]).write_bytes(b"123")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(download_module.subprocess, "run", fake_run)

    with pytest.raises(click.ClickException, match="Download incomplete"):
        download_allstar(destination)


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
    """Serve the manifest + files from a fake Hub directory."""
    (remote / name).write_text(json.dumps(manifest))
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
    assert "MANIFEST.json" in calls and "embeddings/a.parquet" in calls
    assert "bundle ready" in out

    calls.clear()
    again = download_module.download_assets(root=root, repo_id="x/y")
    assert len(again) == 2
    assert "already present" in capsys.readouterr().out
    assert set(calls) == {"MANIFEST.json"}  # no re-download of payloads


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
