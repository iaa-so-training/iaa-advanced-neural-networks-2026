# Advanced Neural Networks for Astronomy 2026

Welcome to the hands-on repository for the **SO–IAA School on Advanced Neural
Networks**. Across five teaching days, we will move from radio-galaxy images to
cosmic-web graphs, gamma-ray events, stellar chemistry, and black-hole imaging.

> **Conference website:** 
> <https://www.granadacongresos.com/ai-ml>.


> **Student guide:** the friendly, session-by-session website will be available at
> <https://iaa-so-training.github.io/iaa-advanced-neural-networks-2026/>.

## Sessions at a glance

| Day | Theme | What you will build or investigate | Start here |
|---|---|---|---|
| 1 | Radio survey pipelines | A defensible LoTSS image pipeline and a transfer-learning experiment | [`day1_radio_surveys/README.md`](day1_radio_surveys/README.md) |
| 2 | GNNs for cosmology | Node classification and graph-level Ω_m regression on halo catalogues | [`day2_GNN_LSS/README.md`](day2_GNN_LSS/README.md) |
| 3 | Gamma-ray astronomy | CTLearn training and inference for Cherenkov-telescope events | [`day3_ctlearn_gammaray/README.rst`](day3_ctlearn_gammaray/README.rst) |
| 4 | Clustering and chemical tagging | t-SNE, UMAP, and EVoC benchmarks on stellar abundances | [`day_4_clustering/README.md`](day_4_clustering/README.md) |
| 5 | Imaging interferometry | A deep-image-prior reconstruction of M87* | [`day5_imaging_interferometry/README.md`](day5_imaging_interferometry/README.md) |

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


--- 

![](docs/Combo_2025.png)

The organisers acknowledge financial support from the State Agency for Research of the Spanish MCIU through the "Center of Excellence Severo Ochoa" award for the Instituto de Astrofísica de Andalucía (grant CEX2021-001131-S 10.13039/501100011033)​​

​
