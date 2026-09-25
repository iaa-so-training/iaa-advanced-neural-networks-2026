"""Download the SDSS-V DR19 Astra ASPCAP catalog (the DR17 allStar successor).

One file replaces the old ``allStar-dr17-synspec_rev1.fits``:

* ``astraAllStarASPCAP-0.6.0.fits.gz`` (1.17 GB) — stellar params + [X/H]
  abundances, Gaia DR3 astrometry/photometry, quality flags.

Uses ``requests`` with HTTP range requests, so interrupted transfers resume and
the download also works where no ``wget``/``curl`` binary exists — notably
inside the workshop container (``python:*-slim`` ships neither).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import click

from .config import (
    ALLSTAR_BYTES,
    ALLSTAR_URL,
    ASSETS_MANIFEST,
    ASTRA_ASPCAP_PATH,
    HF_REPO_ID,
    HF_REPO_TYPE,
)

DOWNLOAD_CHUNK = 8 << 20      # 8 MB per write
PROGRESS_STEPS = 10           # ~10 progress lines per download


def _already_downloaded(path: Path) -> bool:
    return path.exists() and path.stat().st_size == ALLSTAR_BYTES


def stream_to_file(url: str, destination: Path, expected_bytes: int | None = None) -> Path:
    """Resumable download: HTTP range request, append, verify the final size.

    A server that ignores ``Range`` (answering 200 instead of 206) makes us
    start over rather than corrupt the file.
    """
    import requests  # imported here so `import cluster` stays cheap

    destination.parent.mkdir(parents=True, exist_ok=True)
    offset = destination.stat().st_size if destination.exists() else 0
    mode = "ab" if offset else "wb"

    with requests.get(
        url,
        headers={"Range": f"bytes={offset}-"} if offset else {},
        stream=True,
        timeout=60,
    ) as response:
        if response.status_code == 416:          # nothing left to fetch
            return destination
        if offset and response.status_code != 206:
            click.echo("   (server ignored the resume request — starting over)")
            offset, mode = 0, "wb"
        response.raise_for_status()

        declared = int(response.headers.get("Content-Length") or 0)
        total = expected_bytes or (offset + declared) or None
        written = offset
        next_mark = 0

        with destination.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                if total and written >= next_mark:
                    click.echo(
                        f"   {written / 1e6:7.0f} / {total / 1e6:.0f} MB"
                        f"  ({100 * written / total:3.0f}%)"
                    )
                    next_mark = written + max(total // PROGRESS_STEPS, DOWNLOAD_CHUNK)

    return destination


def download_allstar(destination: str | Path, url: str = ALLSTAR_URL) -> Path:
    """Download the Astra ASPCAP catalog (the DR17 allStar successor)."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if _already_downloaded(destination):
        click.echo(f"✓ {destination} already present ({ALLSTAR_BYTES} bytes). Skipping.")
        _write_catalogue_sidecar(destination)
        return destination

    click.echo(f"🌐 Downloading {url} → {destination}")
    click.echo("   (resumable; 1.17 GB)")
    stream_to_file(url, destination, expected_bytes=ALLSTAR_BYTES)

    if not _already_downloaded(destination):
        size = destination.stat().st_size if destination.exists() else 0
        raise click.ClickException(
            f"Download incomplete: {size} bytes (expected {ALLSTAR_BYTES}). "
            "Rerun to resume; delete the file to start over."
        )
    click.echo("✓ Download complete.")
    _write_catalogue_sidecar(destination)
    return destination


def _write_catalogue_sidecar(catalogue: Path) -> Path:
    """Record the catalogue's sha256 beside it, and check it when it exists.

    A size check catches a truncated transfer but not a corrupted one. The
    sidecar makes the bytes checkable later (``cluster doctor --deep``); when one
    is already present its hash is compared now, and a mismatch is reported
    loudly rather than overwritten — it means the file is not the one this
    machine recorded.
    """
    sidecar = catalogue.with_name(catalogue.name + ".sha256")
    digest = _sha256(catalogue)
    if sidecar.is_file():
        recorded = sidecar.read_text().split()[0].strip()
        if recorded != digest:
            click.echo(
                f"⚠ {sidecar.name} records {recorded[:16]}… but the file hashes to "
                f"{digest[:16]}… — these are not the bytes this machine verified. "
                "Delete the file and rerun to download it again."
            )
        else:
            click.echo(f"✓ catalogue sha256 matches {sidecar.name} ({digest[:16]}…)")
        return sidecar
    sidecar.write_text(f"{digest}  {catalogue.name}\n")
    click.echo(f"✓ catalogue sha256 {digest} recorded in {sidecar.name}")
    return sidecar


@click.command()
@click.option(
    "--destination",
    default=ASTRA_ASPCAP_PATH,
    show_default=True,
    help="Where to save the catalog.",
)
def download(destination: str) -> None:
    """Fetch the SDSS-V DR19 Astra ASPCAP catalog."""
    download_allstar(destination)


# --------------------------------------------------------------------------- #
# Asset bundle (embeddings + checkpoints) from Hugging Face
# --------------------------------------------------------------------------- #

CHUNK = 1 << 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _hf_download(repo_id: str, filename: str, repo_type: str) -> Path:
    """Fetch one file from the Hub (imported lazily — network is optional)."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type))


def _hub_hint(repo_id: str, repo_type: str, exc: Exception) -> click.ClickException:
    """Turn any Hub failure into something a student can act on."""
    return click.ClickException(
        f"could not reach the asset bundle at "
        f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}\n"
        f"  reason: {exc}\n"
        "  • check your connection — the download resumes, just re-run it\n"
        "  • the bundle must be published there by the workshop organisers\n"
        "  • point elsewhere with CLUSTER_HF_REPO=<owner/name> (no retraining needed)"
    )


def load_manifest(
    repo_id: str | None = None,
    repo_type: str | None = None,
    manifest_name: str | None = None,
) -> dict:
    """Read ``MANIFEST.json``.

    The manifest ships with the code (``hf/MANIFEST.json``, and inside the
    container image), so this is normally an offline file read — ``--list``
    works before any download. Only a manifest that is not on disk is fetched
    from the Hub, which is what lets someone verify a bundle they found
    elsewhere.
    """
    repo_id = repo_id or HF_REPO_ID
    repo_type = repo_type or HF_REPO_TYPE
    manifest_name = manifest_name or ASSETS_MANIFEST
    local = Path(manifest_name)
    if local.is_file():
        return json.loads(local.read_text())
    try:
        fetched = _hf_download(repo_id, manifest_name, repo_type)
    except Exception as exc:  # RepositoryNotFoundError, HfHubHTTPError, offline, …
        raise click.ClickException(
            f"asset manifest '{manifest_name}' not found on disk and not fetchable "
            f"from the Hub ({exc})"
        ) from exc
    return json.loads(fetched.read_text())


def target_path(root: str | Path, entry_path: str) -> Path:
    """Local destination for a bundle entry.

    ``embeddings/x.parquet`` -> ``<root>/embeddings/x.parquet``
    ``models/x.pt``          -> ``<root>/embeddings/x.pt`` (the code expects the
                                checkpoints next to the parquets)
    ``optional/x.tar``       -> ``<root>/x.tar``
    """
    root = Path(root)
    parts = Path(entry_path).parts
    if not parts:
        raise ValueError("empty manifest entry path")
    kind, rest = parts[0], Path(*parts[1:])
    if kind == "models":
        return root / "embeddings" / rest
    if kind == "optional":
        return root / rest
    return root / kind / rest


def entry_is_valid(path: Path, entry: dict) -> bool:
    """Size + sha256 check against the manifest."""
    return (
        path.exists()
        and path.stat().st_size == entry["bytes"]
        and _sha256(path) == entry["sha256"]
    )


def select_entries(
    manifest: dict,
    include_optional: bool = False,
    only: tuple[str, ...] = (),
) -> list[dict]:
    entries = list(manifest.get("files", []))
    if not include_optional:
        entries = [e for e in entries if not e["path"].startswith("optional/")]
    if only:
        wanted = set(only)
        entries = [e for e in entries if e["path"] in wanted]
        missing = wanted - {e["path"] for e in entries}
        if missing:
            raise click.ClickException(
                "unknown bundle path(s): " + ", ".join(sorted(missing))
            )
    return entries


def format_manifest(entries: list[dict]) -> str:
    lines = [f"{'path':<52} {'size':>10}  consumer"]
    for e in entries:
        lines.append(f"{e['path']:<52} {e['bytes'] / 1e6:>8.1f} MB  {e['consumer']}")
    total = sum(e["bytes"] for e in entries)
    lines.append(f"{'TOTAL':<52} {total / 1e6:>8.1f} MB  ({len(entries)} files)")
    return "\n".join(lines)


def download_assets(
    root: str | Path = "data",
    repo_id: str = HF_REPO_ID,
    repo_type: str = HF_REPO_TYPE,
    include_optional: bool = False,
    only: tuple[str, ...] = (),
    list_only: bool = False,
    check_only: bool = False,
) -> list[Path]:
    """Fetch (or verify) the embeddings + checkpoint bundle."""
    manifest = load_manifest(repo_id, repo_type)
    entries = select_entries(manifest, include_optional=include_optional, only=only)
    if not entries:
        raise click.ClickException("nothing selected from the bundle")

    if list_only:
        click.echo(format_manifest(entries))
        return []

    fetched: list[Path] = []
    problems: list[str] = []
    for entry in entries:
        dest = target_path(root, entry["path"])
        if entry_is_valid(dest, entry):
            click.echo(f"✓ {dest} already present ({entry['bytes'] / 1e6:.1f} MB)")
            fetched.append(dest)
            continue
        if check_only:
            state = "size/hash mismatch" if dest.exists() else "missing"
            problems.append(f"{dest} ({state})")
            click.echo(f"✗ {dest} — {state}")
            continue

        click.echo(f"🌐 {entry['path']} → {dest} ({entry['bytes'] / 1e6:.1f} MB)")
        try:
            cached = _hf_download(repo_id, entry["path"], repo_type)
        except Exception as exc:
            raise _hub_hint(repo_id, repo_type, exc) from exc
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.resolve() != cached.resolve():
            dest.write_bytes(cached.read_bytes())
        if not entry_is_valid(dest, entry):
            raise click.ClickException(
                f"{dest} failed verification against the manifest "
                f"(expected {entry['bytes']} bytes / sha256 {entry['sha256'][:12]}…)"
            )
        fetched.append(dest)

    if check_only:
        click.echo(
            f"\n{len(fetched)} file(s) verified, {len(problems)} problem(s)"
        )
        if problems:
            raise click.ClickException(
                "bundle incomplete — rerun `cluster download --assets` "
                "or inspect: " + "; ".join(problems)
            )
    else:
        click.echo(f"\n✓ bundle ready: {len(fetched)} file(s) under {root}/")
    return fetched
