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

# Day 1 — Downloading the workshop materials

You will need:

- A computer with at least 5 GB of free disk space.
- A Google account, to run the notebooks in Google Colab.
- Git and Git Large File Storage (Git LFS).

You do not need a GitHub account to download the public workshop repository.

The LoTSS dataset is a large archive stored with Git LFS. A normal Git clone alone may download only a small pointer file, so the Git LFS steps below are essential.

## 1. Install Git and Git LFS

### macOS

If you use Homebrew, open Terminal and run:

```bash
brew install git git-lfs
```

Otherwise, install Git from [git-scm.com/downloads](https://git-scm.com/downloads/), then install Git LFS from [git-lfs.com](https://git-lfs.com/).

### Linux

For Ubuntu or Debian, open a terminal and run:

```bash
sudo apt update
sudo apt install git git-lfs
```

For Fedora:

```bash
sudo dnf install git git-lfs
```

For Arch Linux:

```bash
sudo pacman -S git git-lfs
```

### Windows

1. Install [Git for Windows](https://git-scm.com/download/win).
2. Install [Git LFS for Windows](https://git-lfs.com/).
3. Open **Git Bash** from the Windows Start menu for the commands below.

## 2. Enable Git LFS

Open Terminal (macOS/Linux) or Git Bash (Windows), then run:

```bash
git lfs install
```

This only needs to be done once on each computer.

## 3. Download the Day 1 materials

Choose a suitable location on your computer, then run:

```bash
git clone https://github.com/iaa-so-training/iaa-advanced-neural-networks-2026.git
cd iaa-advanced-neural-networks-2026
git lfs pull
```

The final command downloads the large LoTSS archive. It may take several minutes, depending on your connection.

The Day 1 files are in:

```text
day1_radio_surveys/
```

You should find:

```text
01_lotss_fits_scaling_augmentation.ipynb
02_lotss_transfer_learning.ipynb
Granada_School_LoTSS_raw.tar.gz
```

## 4. Check that the dataset downloaded properly

In Terminal or Git Bash, run:

```bash
ls -lh day1_radio_surveys/Granada_School_LoTSS_raw.tar.gz
```

The archive should be approximately **1.2 GB**. If it is only a few hundred bytes or a few kilobytes, Git LFS has not downloaded the data; rerun:

```bash
git lfs pull
```

## 5. Set up Google Drive and Colab

1. In Google Drive, create a folder named exactly:

   ```text
   Granada School
   ```

   directly inside **My Drive**.

2. Upload this file, without unpacking it:

   ```text
   Granada_School_LoTSS_raw.tar.gz
   ```

   to `My Drive/Granada School/`.

3. Upload or open the two `.ipynb` notebooks in Google Colab.

4. Start with:

   ```text
   01_lotss_fits_scaling_augmentation.ipynb
   ```

   Then continue with:

   ```text
   02_lotss_transfer_learning.ipynb
   ```

The notebooks will mount Google Drive and unpack the archive automatically. No external data download is needed during the practical.

## Updating the materials later

If the workshop repository is updated, return to its folder and run:

```bash
git pull
git lfs pull
```

If anything goes wrong, bring the exact error message to the session.
