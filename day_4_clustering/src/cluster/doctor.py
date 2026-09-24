"""The environment fingerprint a quoted number belongs to.

Scores from this pipeline move by roughly ±0.02 across machines, and that is a
property of the stack rather than a bug to stamp out: sklearn's Barnes-Hut
t-SNE (OpenMP), numba (UMAP, EVoC, HDBSCAN) and BLAS each reduce floating-point
sums in an environment-dependent order, so identical code and identical data can
land on slightly different floats on a different CPU, core count or thread
setting. The workable response is to *record* the environment instead of
promising a constant — ``cluster doctor`` prints the record, and
``cluster doctor --json`` is the format ``docs/reference_runs/*.json`` uses.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from . import config
from .seeding import thread_report

#: Distribution names as installed, which is what importlib.metadata wants.
_DISTRIBUTIONS = (
    "numpy", "pandas", "scikit-learn", "umap-learn", "hdbscan", "evoc",
    "numba", "marimo", "torch", "opentsne",
)


def versions() -> dict[str, str]:
    """Installed versions of everything that touches the numbers."""
    out: dict[str, str] = {}
    for dist in _DISTRIBUTIONS:
        try:
            out[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[dist] = "absent"
    return out


def git_sha() -> str:
    """The commit the code came from: build-time stamp, else the checkout."""
    stamped = os.environ.get("DAY4_GIT_SHA", "").strip()
    if stamped:
        return stamped
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def image_ref() -> str:
    """The container this ran in, when it ran in one (stamped at build time)."""
    ref = os.environ.get("DAY4_IMAGE", "").strip()
    return ref or "native python (no container)"


def data_report(*, deep: bool = False) -> dict[str, Any]:
    """Catalogue identity, plus bundle coverage.

    With ``deep=True`` the catalogue *and* every bundle file are re-hashed
    against their recorded sha256 (about a minute for 2 GB); otherwise the
    catalogue is hashed only when its size is wrong, and the bundle is only
    counted. The catalogue's ``.sha256`` sidecar is written by ``cluster
    download --all``.
    """
    from .download import _sha256, entry_is_valid, load_manifest, target_path

    report: dict[str, Any] = {}

    catalogue = Path(config.ASTRA_ASPCAP_PATH)
    if catalogue.is_file():
        size = catalogue.stat().st_size
        sidecar = catalogue.with_name(catalogue.name + ".sha256")
        recorded = sidecar.read_text().split()[0].strip() if sidecar.is_file() else None
        entry: dict[str, Any] = {
            "path": str(catalogue),
            "bytes": size,
            "expected_bytes": config.ASTRA_ASPCAP_BYTES,
            "bytes_ok": size == config.ASTRA_ASPCAP_BYTES,
            "sidecar_sha256": recorded,
        }
        if deep or not entry["bytes_ok"]:
            entry["sha256"] = _sha256(catalogue)
            if recorded:
                entry["sha256_ok"] = entry["sha256"] == recorded
        report["catalogue"] = entry
    else:
        report["catalogue"] = {
            "path": str(catalogue),
            "status": "not downloaded — run `cluster download --all`",
        }

    try:
        manifest = load_manifest()
    except Exception as exc:
        report["bundle"] = {"manifest": "unavailable", "detail": str(exc)}
        return report

    entries = [
        e for e in manifest.get("files", []) if not e["path"].startswith("optional/")
    ]
    present, missing = 0, []
    for entry in entries:
        dest = target_path("data", entry["path"])
        if dest.is_file() and dest.stat().st_size == entry["bytes"]:
            present += 1
        else:
            missing.append(entry["path"])
    bundle: dict[str, Any] = {
        "manifest": manifest.get("dataset", "MANIFEST.json"),
        "manifest_files": len(entries),
        "present": present,
        "missing": missing,
    }
    if deep:
        verified = sum(
            1 for entry in entries
            if entry_is_valid(target_path("data", entry["path"]), entry)
        )
        bundle["verified_sha256"] = f"{verified}/{len(entries)}"
    report["bundle"] = bundle
    return report


def fingerprint(*, deep: bool = False) -> dict[str, Any]:
    """Everything a quoted number depends on, as a JSON-ready dict."""
    return {
        "seed": config.Settings().random_state,
        "python": platform.python_version(),
        "platform": f"{platform.system().lower()}/{platform.machine()}",
        "threads": thread_report(),
        "versions": versions(),
        "git_sha": git_sha(),
        "image": image_ref(),
        "data": data_report(deep=deep),
    }


def format_fingerprint(fp: dict[str, Any]) -> str:
    """The human-readable block ``cluster doctor`` prints."""
    threads = fp["threads"]
    absent = [k for k, v in fp["versions"].items() if v == "absent"]
    present = " ".join(f"{k}={v}" for k, v in fp["versions"].items() if v != "absent")
    lines = [
        f"seed={fp['seed']}  threads: OMP_NUM_THREADS={threads['OMP_NUM_THREADS']}"
        f"  NUMBA_NUM_THREADS={threads['NUMBA_NUM_THREADS']}"
        f"  (cpus={threads['cpu_count']}, numba effective={threads['numba_effective']},"
        f" torch={threads['torch_threads']})",
        f"versions: python={fp['python']}  {present}",
    ]
    if absent:
        lines.append(f"absent: {', '.join(absent)}")
    lines.append(f"platform={fp['platform']}  git={fp['git_sha']}")
    lines.append(f"image={fp['image']}")

    cat = fp["data"].get("catalogue", {})
    if "bytes" in cat:
        digest = cat.get("sha256") or cat.get("sidecar_sha256") or "not hashed (--deep)"
        flag = "ok" if cat.get("bytes_ok") else "SIZE MISMATCH"
        lines.append(f"catalogue: {cat['path']} ({cat['bytes']} B, {flag})")
        lines.append(f"  sha256={digest}")
        if cat.get("sha256_ok") is False:
            lines.append(
                "  ⚠ catalogue bytes differ from its .sha256 sidecar — "
                "re-run `cluster download --all`"
            )
    else:
        lines.append(f"catalogue: {cat.get('status', 'unknown')}")

    bundle = fp["data"].get("bundle", {})
    if "present" in bundle:
        extra = (
            f", sha256 verified {bundle['verified_sha256']}"
            if "verified_sha256" in bundle else ""
        )
        lines.append(f"bundle: {bundle['present']}/{bundle['manifest_files']} files present{extra}")
        if bundle["missing"]:
            shown = ", ".join(bundle["missing"][:4])
            more = " …" if len(bundle["missing"]) > 4 else ""
            lines.append(f"  missing: {shown}{more} — run `cluster download --assets`")

    lines += [
        "",
        "Scores move by ~±0.02 across machines. docs/reproducibility.md records the "
        "reference recipe, the tolerance and the values behind each quoted number.",
    ]
    return "\n".join(lines)
