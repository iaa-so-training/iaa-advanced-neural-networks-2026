# AGENTS.md — hf

Publishing the data bundle to Hugging Face. The workshop's embeddings,
baselines and checkpoints are too large for git (~1.29 GB across 32 files), so
they live in a HF dataset repo that `cluster download --assets` fetches.

| File | Purpose |
|---|---|
| `MANIFEST.json` | **The single source of truth** for what the bundle contains |
| `make_manifest.py` | Build the manifest: walk the curated list, hash every file, record parquet shapes |
| `publish.py` | Upload the card, the manifest, and every file the manifest lists |
| `README.md` | The dataset card (YAML frontmatter + prose) shown on the Hub |

## Publishing

```bash
hf auth login                                   # once, as the account owning the repo
uv run python hf/make_manifest.py               # rebuild the manifest from local files
uv run python hf/publish.py --dry-run           # show what would be uploaded
uv run python hf/publish.py --repo-id <user-or-org>/iaa-chemical-tagging-2026
```

Default target is `RafaelDias/iaa-chemical-tagging-2026`, overridable with
`CLUSTER_HF_REPO`.

## What is in the bundle, and what is deliberately not

32 files: `embeddings/` (latents, PCA baselines, star lists), `models/`
(checkpoints), and one `optional/` entry.

`optional/mwmstar.tar` (~240 MB, 736 raw DR19 spectra) is **excluded unless
asked for** — `cluster download --assets` skips anything under `optional/`, and
`--with-optional` includes it. It also arrives as a tar that nothing unpacks
automatically (`tar -xf data/mwmstar.tar -C data/`).

**Derived results are not published here.** `results/casamiquela_comparison.csv`
and `data/galah_apogee_*.parquet` are computed by scripts, not downloaded —
they are cheap to regenerate from the catalogue and would go stale against the
code that produces them. An exercise needing one must say *build it*, not
*download it*; getting that wrong strands the student.

## The manifest is the contract

`make_manifest.py` hashes each file; `cluster download --assets` **verifies
against those hashes**. So the order is fixed:

1. change the local artifacts,
2. re-run `make_manifest.py`,
3. publish.

Publishing a file without regenerating the manifest gives every student a hash
mismatch on download. Regenerating the manifest without publishing gives them a
404. Neither fails on your machine, where the files are already present — so
**verify with `--dry-run`, then verify the download path**, not just the upload.

## Conventions

- **Never commit the artifacts themselves.** `data/` and `results/` are
  gitignored; this directory holds only the manifest and the tooling.
- **No credentials in the repo.** Authentication is `hf auth login` (or
  `HF_TOKEN` in the environment). A token must never reach a committed file, a
  notebook cell, or a log line.
- The card in `README.md` is public-facing: it states the license and
  provenance of the data, so keep it accurate about what the bundle contains
  and where it came from.
- `total_bytes` and `n_files` in the manifest are generated, not hand-edited —
  they are how a partial upload is detected.
