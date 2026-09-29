"""Chapter 14, exercise 1 — single-pixel versus block masking.

    Train the same autoencoder twice, once with single-pixel random masking
    and once with contiguous 200-pixel blocks, and compare the reconstruction
    loss on a held-out set. Then compare the two latents on the cluster-only
    task. Which model wins on reconstruction, and which on chemistry? Explain
    the disagreement.

NOT COMPUTABLE HERE, and the module says so rather than inventing numbers.
Training the masked autoencoder needs (a) a GPU, (b) the 40 939 spectra the
model was trained on, and (c) the training code, which lives in the lightsurf
repository on the GPU machine, not in this one. This repository consumes the
*exported* latents (``data/embeddings/masked_latent*.parquet``) precisely so
that clustering stays GPU-free — a design choice of ``cluster/spectral.py``.

What the module does instead is measure the premise the chapter's answer rests
on, on real data: how well linear interpolation from the visible neighbours
reconstructs a single masked pixel versus a contiguous 200-pixel block, using
the 736 DR19 ``mwmStar`` spectra that are on disk. That is the mechanism §14.3
describes ("copying from the left and right solves the task"), measured rather
than asserted, and it is the reason the two models must disagree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import DataNotAvailable, project_root

#: APOGEE spectra are 8 575 pixels; §14.2 masks 50% of them in 200-pixel blocks.
N_PIXELS = 8_575
BLOCK = 200
#: Spectra read for the interpolation probe (each file is ~0.3 MB, one extension
#: of 8 575 doubles); 40 is enough for a stable median and stays interactive.
DEFAULT_N_SPECTRA = 40
#: Single pixels probed per spectrum.
N_SINGLE = 200
#: Blocks probed per spectrum.
N_BLOCKS = 20


def mwmstar_files(limit: int = DEFAULT_N_SPECTRA) -> list[Path]:
    """The DR19 ``mwmStar`` spectra staged in the repository, sorted."""
    directory = project_root() / "data" / "mwmstar"
    files: list[Path] = sorted(directory.glob("*.fits")) if directory.is_dir() else []
    if not files:
        raise DataNotAvailable(
            f"no DR19 mwmStar spectra under {directory}\n\n"
            "They ship as the optional archive in the asset bundle (736 "
            "spectra, ~240 MB), which is not fetched by default:\n\n"
            "    uv run cluster download --assets --with-optional\n"
            "    tar -xf data/mwmstar.tar -C data/\n",
        )
    return files[:limit]


def _spectrum(path: Path) -> np.ndarray | None:
    """The combined-visit 8 575-pixel flux array from one ``mwmStar`` file."""
    from astropy.io import fits

    try:
        with fits.open(path) as hdus:
            for i in range(1, len(hdus)):
                hdu: Any = hdus[i]
                if hdu.data is None:
                    continue
                if hdu.header.get("NAXIS", 0) < 2 or hdu.header.get("NAXIS1", 0) < N_PIXELS:
                    continue
                if hdu.header.get("NAXIS2", 0) < 1:
                    continue
                flux = np.asarray(hdu.data["flux"][0], dtype=float)
                if flux.size >= N_PIXELS and np.isfinite(flux).all():
                    return flux
    except (KeyError, OSError, ValueError):
        return None
    return None


def interpolation_probe(
    n_spectra: int = DEFAULT_N_SPECTRA, seed: int = 0,
) -> dict[str, object]:
    """Reconstruction error of a masked pixel / block by linear interpolation.

    For each spectrum: ``N_SINGLE`` pixels are masked one at a time and filled
    with the mean of their two immediate neighbours; ``N_BLOCKS`` contiguous
    200-pixel blocks are masked and filled by a straight line between the two
    pixels bounding the gap. Errors are absolute flux differences, reported in
    units of the per-spectrum median absolute deviation so that a hot star and
    a cool one are comparable.
    """
    rng = np.random.default_rng(seed)
    single: list[np.ndarray] = []
    block: list[np.ndarray] = []
    scales: list[float] = []

    for path in mwmstar_files(n_spectra):
        flux = _spectrum(path)
        if flux is None or flux.size < BLOCK + 2:
            continue
        scales.append(float(np.median(np.abs(flux - np.median(flux)))))
        idx = rng.integers(2, flux.size - 2, N_SINGLE)
        single.append(np.abs(0.5 * (flux[idx - 1] + flux[idx + 1]) - flux[idx]))
        for start in rng.integers(2, flux.size - BLOCK - 2, N_BLOCKS):
            hole = flux[start:start + BLOCK]
            line = np.linspace(flux[start - 1], flux[start + BLOCK], BLOCK + 2)[1:-1]
            block.append(np.abs(line - hole))

    if not single or not block:
        raise DataNotAvailable("mwmStar spectra were present but unreadable")

    single_err = np.concatenate(single)
    block_err = np.concatenate(block)
    scale = float(np.median(scales))

    return {
        "n_spectra": int(len(scales)),
        "flux_scale": round(scale, 4),
        "single_median_err": round(float(np.median(single_err)), 4),
        "block_median_err": round(float(np.median(block_err)), 4),
        "single_fraction_of_scale": round(float(np.median(single_err) / scale), 4),
        "block_fraction_of_scale": round(float(np.median(block_err) / scale), 4),
        "ratio_block_over_single": round(
            float(np.median(block_err) / np.median(single_err)), 2,
        ),
        "single_err": single_err,
        "block_err": block_err,
    }


def solve(n_spectra: int = DEFAULT_N_SPECTRA) -> dict[str, object]:
    """The interpolation probe; the training experiment is out of reach here."""
    probe = interpolation_probe(n_spectra=n_spectra)
    return {
        "training_experiment_ran": False,
        "reason": (
            "no GPU and no torch in this environment; the 40 939 training "
            "spectra and the lightsurf training loop are not in this "
            "repository (only the exported latents are)"
        ),
        "what_was_measured_instead": probe,
        "comparison_table": pd.DataFrame([
            {"masking": "single pixels (random)",
             "linear_interpolation_error_median": probe["single_median_err"],
             "as_fraction_of_flux_scale": probe["single_fraction_of_scale"]},
            {"masking": f"contiguous {BLOCK}-pixel blocks",
             "linear_interpolation_error_median": probe["block_median_err"],
             "as_fraction_of_flux_scale": probe["block_fraction_of_scale"]},
        ]),
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """Distributions of the interpolation error for the two masking schemes."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    probe = result["what_was_measured_instead"]
    assert isinstance(probe, dict)
    scale = float(probe["flux_scale"])

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    edges = np.linspace(0, 6, 80)
    ax.hist(np.clip(probe["single_err"] / scale, 0, 6), bins=list(edges), density=True,
            color="#4c72b0", alpha=0.85, label="single masked pixel")
    ax.hist(np.clip(probe["block_err"] / scale, 0, 6), bins=list(edges), density=True,
            color="#dd8452", alpha=0.65, label=f"{BLOCK}-pixel masked block")
    ax.set_yscale("log")
    ax.set_xlabel("|interpolation residual| / per-spectrum flux MAD")
    ax.set_title("A single pixel is photocopyable; a 200-pixel hole is not")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "can this be run here? no — and this is what it would take": (
        "The exercise cannot be computed in this repository. Training the "
        f"masked autoencoder {cite('He:22')} requires a GPU, the 40 939 "
        "spectra the model was "
        "trained on, and the training loop, which lives in the lightsurf "
        "repository on the GPU machine. There is no torch in this "
        "environment, no CUDA device, and the repository deliberately ships "
        "only the *exported* latents (masked_latent*.parquet) so the "
        "clustering benchmark stays GPU-free — see the module docstring of "
        "cluster/spectral.py. Two full trainings plus two encodings plus two "
        "7-seed benchmark runs is an overnight job on the machine that has "
        "the data, and no number for it can be produced honestly from here. "
        "What is in this repository is the 736 staged DR19 mwmStar spectra, "
        "and that is enough to measure the mechanism the answer turns on."
    ),
    "the mechanism, measured": (
        "Filling a single masked pixel with the mean of its two neighbours "
        "leaves a median residual of 2.88 flux units on real DR19 spectra, "
        "which is 7.0% of the per-spectrum flux MAD — the pixel's own "
        "signal is largely reproduced by two visible pixels a few Angstrom "
        "away, because the continuum is smooth at that scale. Filling a "
        "contiguous 200-pixel hole by a straight line between its two "
        "bounding pixels leaves a median residual of 28.9 units, 70.6% of "
        "the flux MAD: the line crosses real line regions, so it cannot "
        "reproduce the depth of the lines the abundances are measured from — "
        f"the lines APOGEE's pipeline reads {cite('Majewski:17', bare=True)}. "
        "The ratio is a factor of 10.05 in the median. That is the entire "
        "argument of §14.3, measured on the survey's own spectra: a model "
        "trained with single-pixel masking can lower its loss by "
        "interpolating, and a model trained with block masking cannot."
    ),
    "which model wins, and why (the expected result)": (
        "Single-pixel masking wins decisively on reconstruction loss, and "
        "block masking wins on chemistry. The reasoning is the measurement "
        "above plus what the loss rewards: the reconstruction loss is "
        "computed only on the hidden pixels (§14.2, step 4), so a masking "
        "scheme whose hidden pixels are reconstructible from context gives "
        "the encoder a cheap solution. Interpolation is that solution, and it "
        "is a solution that requires learning almost nothing global — which "
        "is why the latent of the single-pixel model is expected to look "
        "respectable and carry less cluster information. The block-masked "
        "model is forced to use the spectrum's global shape — which line "
        "regions are present and how deep — before it can fill a 200-pixel "
        "gap, and that is where element ratios live, so its latent should "
        "score higher on the cluster-only task. That is the design argument "
        f"{cite('He:22', parenthetical=False)} make for images, and the "
        "reason this workbook masks in blocks. Note carefully which half of "
        "this is measured here and which half is a prediction: the "
        "interpolation gap is measured; the two-model comparison is not, and "
        "is stated as the expected outcome with its reason."
    ),
    "the training-scale caveat": (
        "The probe uses 40 spectra out of 736 staged files, at one random "
        "seed, and reports medians over 8 000 single pixels and 800 blocks. "
        "That is enough to establish a factor-of-ten difference in "
        "reconstructibility; it is not enough to calibrate a reconstruction "
        "loss, and it says nothing about the third regime — the 50% masking "
        "fraction, the five strided convolutional blocks, and the 256-"
        "dimensional bottleneck all shape what either model can learn. The "
        "probe isolates one design decision (contiguity) and holds everything "
        "else of the architecture out of scope by design."
    ),
    "what the disagreement means": (
        "The two models are not competing on the same task, which is the "
        "point §14.3 makes and the reason the chapter's design choice is not "
        "arbitrary. A reconstruction loss is a proxy for 'how much of the "
        "spectrum did you capture' and it can be minimised by a model that "
        "captures the wrong thing; this is the same failure mode as the "
        "provenance confound of §14.5 — a latent that scores well on the "
        "metric it was trained on while encoding something other than the "
        "physics. The correct report of such a result is the pair of numbers "
        "and the mechanism, never the reconstruction loss alone as evidence "
        "that one latent is better."
    ),
    "references": reference_list("He:22", "Majewski:17"),
}
