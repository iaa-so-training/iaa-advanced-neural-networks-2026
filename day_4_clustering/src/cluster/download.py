"""Download the SDSS-V DR19 Astra ASPCAP catalog (the DR17 allStar successor).

One file replaces the old ``allStar-dr17-synspec_rev1.fits``:

* ``astraAllStarASPCAP-0.6.0.fits.gz`` (1.17 GB) — stellar params + [X/H]
  abundances, Gaia DR3 astrometry/photometry, quality flags.

Uses ``wget -c`` so interrupted transfers resume.
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


def _already_downloaded(path: Path) -> bool:
    return path.exists() and path.stat().st_size == ALLSTAR_BYTES


def download_allstar(destination: str | Path, url: str = ALLSTAR_URL) -> Path:
    """Download the Astra ASPCAP catalog (the DR17 allStar successor)."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if _already_downloaded(destination):
        click.echo(f"✓ {destination} already present ({ALLSTAR_BYTES} bytes). Skipping.")
        return destination

    click.echo(f"🌐 Downloading {url} → {destination}")
    click.echo("   (resumable; 1.17 GB)")
    cmd = ["wget", "-c", "--no-check-certificate", "-O", str(destination), url]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise click.ClickException("Download failed — rerun to resume.")
    if not _already_downloaded(destination):
        raise click.ClickException(
            f"Download incomplete: {destination.stat().st_size} bytes "
            f"(expected {ALLSTAR_BYTES}). Delete the file and rerun."
        )
    click.echo("✓ Download complete.")
    return destination


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


def load_manifest(
    repo_id: str = HF_REPO_ID,
    repo_type: str = HF_REPO_TYPE,
    manifest_name: str = ASSETS_MANIFEST,
) -> dict:
    """Read ``MANIFEST.json`` from the published bundle."""
    return json.loads(_hf_download(repo_id, manifest_name, repo_type).read_text())


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
        cached = _hf_download(repo_id, entry["path"], repo_type)
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
