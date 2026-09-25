#!/usr/bin/env python
"""Publish the asset bundle (``hf/MANIFEST.json``) to a Hugging Face dataset repo.

    hf auth login                      # once, as the account that owns the repo
    .venv/bin/python hf/publish.py --repo-id <user-or-org>/iaa-chemical-tagging-2026
    .venv/bin/python hf/publish.py --dry-run        # show what would be uploaded

Uploads the dataset card (``hf/README.md``), the manifest, and every file listed
in the manifest, mapping local paths -> ``path`` in the repo. Idempotent: a file
whose sha256 on the Hub already matches the manifest is skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "hf" / "MANIFEST.json"
CARD = REPO_ROOT / "hf" / "README.md"


def _sha256_local(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default=None, help="e.g. RafaelDias/iaa-chemical-tagging-2026")
    ap.add_argument("--private", action="store_true", help="Create the repo private (default public).")
    ap.add_argument("--dry-run", action="store_true", help="Print the upload plan and exit.")
    ap.add_argument("--card-only", action="store_true", help="Upload only the card + manifest.")
    args = ap.parse_args()

    if not MANIFEST.exists():
        print("hf/MANIFEST.json missing — run `python hf/make_manifest.py` first.", file=sys.stderr)
        return 2

    manifest = json.loads(MANIFEST.read_text())
    files = manifest["files"]

    uploads: list[tuple[Path, str]] = []
    missing: list[str] = []
    for entry in files:
        local = REPO_ROOT / entry["source"]
        if not local.exists():
            missing.append(entry["source"])
            continue
        if _sha256_local(local) != entry["sha256"]:
            print(f"!! {local} changed since the manifest was built — rerun make_manifest.py", file=sys.stderr)
            return 2
        uploads.append((local, entry["path"]))

    total = sum(p.stat().st_size for p, _ in uploads)
    print(f"bundle: {len(uploads)} files, {total / 1e6:.0f} MB")
    for src, dest in uploads:
        print(f"  {src.relative_to(REPO_ROOT)}  ->  {dest}")
    if missing:
        print("\nNOT STAGED (excluded from this run):")
        for name in missing:
            print(f"  {name}")

    if args.dry_run:
        print("\n(dry run — nothing uploaded)")
        return 0

    if not args.repo_id:
        print("\n--repo-id is required to upload (or use --dry-run).", file=sys.stderr)
        return 2

    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()["name"]
    print(f"\nlogged in as {who}; creating/using dataset repo {args.repo_id}")

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    api.upload_file(
        path_or_fileobj=str(MANIFEST),
        path_in_repo="MANIFEST.json",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="manifest: regenerate",
    )
    if CARD.exists():
        api.upload_file(
            path_or_fileobj=str(CARD),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="card: publish",
        )
    if args.card_only:
        print("card + manifest uploaded")
        return 0

    for src, dest in uploads:
        print(f"uploading {dest} ({src.stat().st_size / 1e6:.1f} MB)")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=dest,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"assets: add {dest}",
        )

    print(f"\ndone -> https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
