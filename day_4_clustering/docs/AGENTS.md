# AGENTS.md — docs

Result write-ups and the recorded runs that back them. This is where a number
goes once it has actually been measured.

| File | Covers |
|---|---|
| `baseline_results.md` | The published-baseline reproduction |
| `experiment_results.md` | Benchmark runs across configurations |
| `spectral_benchmark_results.md` | Spectral-embedding feature sets |
| `dr19_rerun_results.md` | The uniform DR19 re-run |
| `region_sweep_results.md` + `region_sweep.csv` | Region-mode sweep across clusters |
| `reproducibility.md` | What reproduces, what does not, and why |
| `data_releases.md` | DR17 vs DR19, and why mixing them is a batch effect |
| `cluster_assignment.md` | How members are assigned |
| `narrative.md` | The teaching narrative behind the session |
| `student_activities.md` | Session activities |
| `docker.md` | Container usage |
| `spectral_*_plan.md`, `spectral_gpu_runbook.md` | Plans and the GPU runbook |
| `reference_runs/*.json` | **The recorded runs the documents quote** |

## Reference runs are the evidence

A document here may quote a number only if a run produced it. `reference_runs/`
holds those runs as JSON, written by `../scripts/reference_run.py`:

```json
{
  "name": "fast",
  "command": "uv run python scripts/reference_run.py --fast",
  "measured_utc": "2026-09-24T14:58:44+00:00",
  "environment": { ... },
  "population": { ... },
  "per_cluster_recall": { ... }
}
```

The `environment` block is the point. Results in this project depend on
configuration that defaults *away* from the full run — `CLUSTER_FAST` is `True`
by default — so a bare number without its fingerprint is not reproducible and
not quotable. `uv run cluster doctor` prints the same fingerprint for a run
you are doing now.

```bash
uv run python scripts/reference_run.py --fast     # record a new reading
```

The existing files include deliberate pairs (`fast_…`, `fast_…_container`,
`fast_…_container_1thread`) that exist to show how much the environment alone
moves a result. Keep that habit: when a number changes, the first question is
whether the environment changed.

## Conventions

- **Never edit a number here by hand.** Re-run, re-record, and cite the new
  reference run. A hand-adjusted figure is indistinguishable from a fabricated
  one, in a project whose subject is that published numbers are hard to trust.
- **Report discrepancies rather than reconciling them.** Several documents state
  that a published result does not reproduce — including the §9.3 row-order
  range and the §2.3 NaN claim. Those findings are the material; tuning until
  they agree would destroy the lesson.
- Reference-run JSON is append-only in spirit: add a new dated file rather than
  overwriting an old reading, so the history of what was measured survives.
- Some files mention marimo in recorded run metadata. That is historical — the
  project ships Jupyter decks now — and rewriting the record of a past run to
  match present tooling would be falsifying it.
