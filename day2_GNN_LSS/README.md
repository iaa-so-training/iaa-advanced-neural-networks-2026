# GNNs for Cosmology — Hands-on Sessions

Hands-on material for the Graph Neural Networks module of the Granada advanced ML/DL school for
astrophysics and cosmology.

## Notebooks

| Notebook | Task | Level |
|---|---|---|
| `Handson1_Node_Classification.ipynb` | **Node classification** — label each halo as living in a *cluster* (dense) or *void* (sparse) environment | node-level |
| `Handson2_Omega_m_Regression.ipynb` | **Graph-level regression** — infer the cosmological parameter **Ω_m** for a whole simulation box | graph-level |

Each box is a **100 Mpc/h** periodic volume; halos become graph nodes and are linked by spatial
proximity. Session 1 works within one box; Session 2 treats each box as one graph and is a
hands-on version of simulation-based (likelihood-free) inference.

## Running them

Designed for **Google Colab** — open a notebook, run the first cell to install dependencies, and
follow along. A **Google account** is all you need; the notebooks mount your Google Drive and
download the data into it (once).

## Data

Public **CAMELS-SAM** Rockstar halo catalogs:
<https://users.flatironinstitute.org/~camels/Rockstar/CAMELS-SAM/>

- **Session 1** downloads one box (`LH_35`, z = 0, ~200 MB).
- **Session 2** samples ~100 boxes spanning Ω_m at a higher-redshift snapshot (`out_16`, z ≈ 5.2,
  ~50 MB/box → ~5 GB total).

Hands-on 2 also includes a built-in **synthetic data generator**
(`USE_REAL_DATA = False`) so its graph-regression pipeline can run without the
5 GB multi-box download. Hands-on 1 uses the real `LH_35` catalogue (~200 MB),
so download that file before class if possible.
