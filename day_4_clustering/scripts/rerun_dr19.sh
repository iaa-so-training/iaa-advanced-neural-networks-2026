#!/usr/bin/env bash
# Full 25-cluster re-run on DR19/DR4/DR3 data (abundance pipeline).
# FAST=0 -> all clean stars (358k at DR19 quality cuts), not the 25k cap.
set -euo pipefail
cd "$(dirname "$0")/.."
export CLUSTER_FAST=0

echo "=== [1/3] paper baseline (abundances only, cluster-only) ==="
uv run cluster baseline 2>&1 | tee results/dr19_baseline_chem.txt

echo "=== [2/3] paper baseline (+ kinematics) ==="
uv run cluster baseline --kinematics 2>&1 | tee results/dr19_baseline_kin.txt

echo "=== [3/3] field retrieval (all clusters, all-sky) ==="
uv run cluster run 2>&1 | tee results/dr19_field_retrieval.txt

echo "=== DONE ==="
