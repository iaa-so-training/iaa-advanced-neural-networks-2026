# Advanced Neural Networks for Astronomy 2026

Welcome to the hands-on repository for the **SO–IAA School on Advanced Neural
Networks**. Across five teaching days, we will move from radio-galaxy images to
cosmic-web graphs, gamma-ray events, stellar chemistry, and black-hole imaging.

> **Student guide:** the friendly, session-by-session website will be available at
> <https://iaa-so-training.github.io/iaa-advanced-neural-networks-2026/>.

## Sessions at a glance

| Day | Theme | Tutor | What you will build or investigate | Start here |
|---|---|---|---|---|
| Monday (Day 1) | Radio survey pipelines | **Dr. Andrea DeMarco**<br>Institute of Space Sciences and Astronomy (ISSA), University of Malta, Malta | A defensible LoTSS image pipeline and a transfer-learning experiment | [`day1_radio_surveys/README.md`](day1_radio_surveys/README.md) |
| Tuesday (Day 2) | GNNs for cosmology | **Dr. Farida Farsian**<br>Italian National Institute for Astrophysics (INAF), Osservatorio Astrofisico di Catania (OACT), Italy | Node classification and graph-level Ω_m regression on halo catalogues | [`day2_GNN_LSS/README.md`](day2_GNN_LSS/README.md) |
| Wednesday (Day 3) | Gamma-ray astronomy | **Dr. Tjark Miener**<br>University of Geneva, Switzerland / IAA-CSIC, Granada, Spain | CTLearn training and inference for Cherenkov-telescope events | [`day3_ctlearn_gammaray/README.rst`](day3_ctlearn_gammaray/README.rst) |
| Thursday (Day 4) | Clustering and chemical tagging | **Dr. Rafael Garcia-Dias**<br>King’s College London, United Kingdom | t-SNE, UMAP, and EVoC benchmarks on stellar abundances | [`day_4_clustering/README.md`](day_4_clustering/README.md) |
| Friday (Day 5) | Imaging interferometry | **Dr. Joel Sánchez Bermúdez**<br>Instituto de Astronomía (IA-UNAM), Mexico | A deep-image-prior reconstruction of M87* | [`day5_imaging_interferometry/README.md`](day5_imaging_interferometry/README.md) |

## Before you arrive: software setup

Complete the **[Software Installation Guide](docs/resources/software-installation-guide.pdf)**
on the computer you will bring to the school. It covers Google Colab, Docker,
Git, Git LFS, Miniforge, CTLearn, and ViTables, with checks to confirm that each
tool is working before the first session.

## Download the repository

Day 1 includes a large archive managed with Git LFS, so enable LFS before
downloading the workshop materials:

```bash
git lfs install
git clone https://github.com/iaa-so-training/iaa-advanced-neural-networks-2026.git
cd iaa-advanced-neural-networks-2026
git lfs pull
```

If you already cloned the repository, update both the code and large files with:

```bash
git pull
git lfs pull
```

## A good way to work

1. Open the README for your session before class.
2. Complete large downloads early whenever the guide recommends it.
3. Start with the smallest working example.
4. Change one experimental choice at a time and record what happened.
5. Keep results and error messages—both are useful evidence during the session.

Most notebooks are designed for Google Colab. Day 3 uses a Conda environment,
and Day 4 uses a tested Docker image by default. Each session README gives the
exact setup and a lighter-weight route where one is available.
