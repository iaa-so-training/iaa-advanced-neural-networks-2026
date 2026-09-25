# Day 5 — Imaging M87* with a Deep Image Prior

This Colab tutorial reconstructs a **64 × 64 image of M87\*** from sparse Event
Horizon Telescope observables. The network is optimized for this one observation;
it is not trained on a library of black-hole images.

## What you will learn

By the end of the session, you will be able to:

- inspect interferometric sampling in the `(u,v)` plane;
- connect image pixels to complex visibilities through the Fourier measurement equation;
- explain the roles of visibility amplitudes, closure phases, the neural parameterization,
  and explicit regularizers;
- run a Deep Image Prior reconstruction on a CUDA GPU;
- evaluate the recovered image in **data space**, not only by appearance; and
- test how resolution, priors, loss choices, random seeds, and bootstrap weights affect the result.

## Before you start

You need a Google account and a Colab runtime with a CUDA GPU. Keep these three
course files together and do not rename them:

```text
EHT_DIP_Colab_Tutorial.ipynb
eht_dip.py
SR1_M87_2017_095_lo_hops_netcal_StokesI.npz
```

The notebook installs Astropy. Colab already supplies NumPy, Matplotlib, and a
CUDA-enabled PyTorch runtime.

## Quick start

1. [Open the tutorial in Google Colab](https://colab.research.google.com/github/iaa-so-training/iaa-advanced-neural-networks-2026/blob/main/day5_imaging_interferometry/EHT_DIP_Colab_Tutorial.ipynb).
2. Choose **Runtime → Change runtime type → T4 GPU** (or another CUDA GPU).
3. Run the notebook from the top and upload `eht_dip.py` plus the `.npz` data file
   when prompted.
4. For the first classroom pass, set:

   ```python
   QUICK_RUN = True
   RUN_BOOTSTRAP = False
   ```

5. Judge the reconstruction using the visibility-amplitude and closure-phase
   comparisons, not only the image.
6. Download `eht_dip_results.zip` before the Colab runtime closes.

The quick run demonstrates the workflow. A scientific result needs longer
optimization, several random seeds, hyperparameter tests, and enough bootstrap
realizations to show convergence.

## Session materials

- [`EHT_DIP_Colab_Tutorial.ipynb`](EHT_DIP_Colab_Tutorial.ipynb) — guided tutorial and exercises
- [`eht_dip.py`](eht_dip.py) — model, measurement operators, and optimizer
- [`SR1_M87_2017_095_lo_hops_netcal_StokesI.npz`](SR1_M87_2017_095_lo_hops_netcal_StokesI.npz) — example EHT observables
- [`IAA_school_EHTDIP.pdf`](IAA_school_EHTDIP.pdf) — companion session slides

## Common snags

- **CUDA is unavailable:** switch the Colab runtime to a GPU, then rerun from the first cell.
- **A file is missing:** upload both the `.py` and `.npz` files with their original names.
- **The quick image looks unfinished:** that is expected; compare it with a longer run before drawing conclusions.
- **Optimization becomes unstable:** return to the supplied settings and change one parameter at a time.
- **Your results disappeared:** Colab storage is temporary, so download the results archive before leaving.
