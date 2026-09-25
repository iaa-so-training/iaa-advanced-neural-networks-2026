# iaa-advanced-neural-networks-2026
SO-IAA school on Advanced Neural Networks 

# Day 4 — Clustering and chemical tagging

Hands-on session: benchmark **t-SNE / UMAP / EVoC** on the 16-D APOGEE abundance
space and decide which stars in a field belong to a star cluster, scored against
kinematic ground truth.

Everything runs in Docker — section B of the School Software Installation Guide,
so it is already on your laptop. `uv` and all the Python live inside the image:

```bash
git clone https://github.com/iaa-so-training/iaa-advanced-neural-networks-2026.git
cd iaa-advanced-neural-networks-2026/day_4_clustering
mkdir -p data results notebooks
docker pull ghcr.io/iaa-so-training/day4-clustering:latest

export IMG=ghcr.io/iaa-so-training/day4-clustering:latest
export DAY4="-v $PWD/data:/app/data -v $PWD/results:/app/results -v $PWD/notebooks:/app/notebooks"

docker run --rm -it $DAY4 $IMG uv run cluster download --all   # catalogue + embeddings, ~2.2 GB, once
docker run --rm -it $DAY4 $IMG uv run cluster run --fast       # ~2 min smoke test
```

Windows (PowerShell): use `$Day4 = @("-v","$($PWD.Path)/data:/app/data", …)` and
splat it with `@Day4` — see
[`day_4_clustering/docs/docker.md`](day_4_clustering/docs/docker.md).
Prefer native Python? `uv sync && uv run cluster …` — identical flags and
commands.

- Instructions and full walk-through: [`day_4_clustering/README.md`](day_4_clustering/README.md)
- Student activities: [`day_4_clustering/docs/student_activities.md`](day_4_clustering/docs/student_activities.md)
- Container details (image, mounts, performance): [`day_4_clustering/docs/docker.md`](day_4_clustering/docs/docker.md)
