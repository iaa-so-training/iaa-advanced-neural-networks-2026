"""Teaching figures for the masked spectral autoencoder.

The deck walks K-means through four hand-drawn steps; this gives the masked AE
the same treatment, but with *real* data instead of a toy: a real DR19 mwmStar
spectrum, the real trained checkpoint, and the model's actual reconstruction of
the blocks it could not see.

Figures written (into the deck's asset folder):

1. ``mae_arch.png``          — the architecture, drawn to scale from the real
                               layer shapes (8575 → 5 conv blocks → 256-d z →
                               transposed-conv decoder → 8575).
2. ``mae_step1_mask.png``    — a real spectrum with contiguous blocks hidden.
3. ``mae_step2_encode.png``  — the conv stack squeezing length while growing
                               channels, annotated with the true tensor shapes.
4. ``mae_step3_recon.png``   — the model's ACTUAL reconstruction inside the
                               masked windows, with the MSE it is scored on.
5. ``mae_step4_latent.png``  — what the 256-d latent does downstream: real
                               member embeddings, PCA-2D, coloured by cluster.
6. ``mae_why_blocks.png``    — why contiguous blocks and not random pixels:
                               linear interpolation nails random holes and fails
                               on a block, which is the whole design argument.

Run (repo root) — needs torch, install the extra once:
    uv sync --extra torch
    uv run python scripts/make_masked_ae_figures.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Rectangle

# Palette mirrors the deck's astro-theme (RevealDeck.vue .astro-theme block), so
# the figures and the slides around them use one colour language.
INDIGO = "#3730a3"   # --accent          night sky / the model
BLUE = "#0369a1"     # --accent-cyan     hot star / visible pixels
AMBER = "#b45309"    # --accent-orange   K giant / the latent
ROSE = "#be123c"     # --accent-pink     H-alpha / the hidden blocks
GREEN = "#15803d"    # --accent-green    truth / reconstruction
INK = "#1c2333"
MUTED = "#55607a"
LINE = "#d5dae6"
SURFACE = "#f2f4f9"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": INK,
    "axes.edgecolor": LINE,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})

# Where the PNGs land. Override with DAY4_DECK_DIR=... (e.g. a slides folder).
DECK = os.path.expanduser(os.environ.get("DAY4_DECK_DIR", "results/deck"))
CKPT = "data/embeddings/masked_ae_rerun.pt"
SPEC_DIR = "data/mwmstar"

# The APOGEE spectrum is three detectors with gaps; this is the blue-green
# region where the deck's example lines live.
N_PIX = 8575


def _save(fig: Figure, name: str) -> None:
    path = os.path.join(DECK, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


def load_one_spectrum() -> np.ndarray:
    """One real DR19 mwmStar spectrum, standardised exactly as training did."""
    from astropy.io import fits

    files = sorted(f for f in os.listdir(SPEC_DIR) if f.endswith(".fits"))
    for fname in files:
        try:
            with fits.open(os.path.join(SPEC_DIR, fname)) as hdul:
                for hdu_idx in (3, 4):  # APO first, LCO fallback
                    if hdu_idx >= len(hdul):
                        continue
                    data = hdul[hdu_idx].data
                    if data is None or len(data) == 0:
                        continue
                    flux = np.asarray(data["flux"][0], dtype=np.float64)
                    if flux.size != N_PIX or not np.isfinite(flux).any():
                        continue
                    flux = np.nan_to_num(flux, nan=0.0)
                    if np.count_nonzero(flux) < N_PIX * 0.5:
                        continue
                    sd = flux.std()
                    if sd <= 0:
                        continue
                    print(f"  spectrum: {fname} (HDU {hdu_idx})", flush=True)
                    return (flux - flux.mean()) / sd
        except Exception:
            continue
    raise SystemExit("no usable spectrum found in " + SPEC_DIR)


def fig_architecture() -> None:
    """The model, drawn to scale from the real layer shapes."""
    import torch

    from cluster.models.masked_spectral_ae import MaskedSpectralAE

    model = MaskedSpectralAE(N_PIX, 256)
    n_params = sum(p.numel() for p in model.parameters())

    # Real per-layer shapes, taken from the model rather than retyped.
    with torch.no_grad():
        x = torch.zeros(1, 1, N_PIX)
        lengths: list[int] = []
        channels: list[int] = []
        h = x
        for layer in model.encoder:
            h = layer(h)
            if isinstance(layer, torch.nn.Conv1d):
                lengths.append(h.shape[2])
                channels.append(h.shape[1])

    fig, ax = plt.subplots(figsize=(13.4, 5.6))
    ax.set_xlim(-2, 102)
    ax.set_ylim(-1.6, 13.0)
    ax.axis("off")

    # Shape labels alternate between two rows so neighbouring (narrow) blocks
    # can never overprint each other, and stay horizontal for legibility.
    shape_rows = [0.30, 1.35]

    def block(
        x0: float, w: float, h: float, color: str, alpha: float,
        label: str, sub: str | None = None, row: int = 0,
    ) -> None:
        ax.add_patch(Rectangle((x0, 5 - h / 2), w, h, facecolor=color, alpha=alpha,
                               edgecolor=color, linewidth=1.4, zorder=3))
        ax.text(x0 + w / 2, 5, label, ha="center", va="center", fontsize=8.4,
                fontweight="bold", color="white", zorder=4, rotation=90)
        if sub:
            y = 5 - h / 2 - shape_rows[row % 2]
            ax.plot([x0 + w / 2, x0 + w / 2], [5 - h / 2 - 0.05, y + 0.18],
                    color=LINE, linewidth=0.7, zorder=2)
            ax.text(x0 + w / 2, y, sub, ha="center", va="top",
                    fontsize=7.0, color=MUTED, zorder=4)

    def band(x0: float, x1: float, y: float, text: str, color: str) -> None:
        """A stage header drawn as a bracket over exactly the blocks it covers,
        so no two headers can collide however the blocks are sized."""
        ax.plot([x0, x0, x1, x1], [y - 0.28, y, y, y - 0.28],
                color=color, linewidth=1.3, zorder=4)
        ax.text((x0 + x1) / 2, y + 0.25, text, ha="center", va="bottom",
                fontsize=8.8, color=color, fontweight="bold", zorder=4)

    # Input
    in_x, in_w = 0.5, 3.4
    block(in_x, in_w, 7.6, BLUE, 0.85, "spectrum", f"1 x {N_PIX}")
    band(in_x, in_x + in_w, 11.2, "masked input", BLUE)

    # Encoder — height shrinks with sequence length, width grows with channels.
    enc_x0 = 6.4
    x0 = enc_x0
    max_len = lengths[0]
    for i, (L, C) in enumerate(zip(lengths, channels, strict=True)):
        h = 1.2 + 6.4 * (L / max_len)
        w = 2.0 + 2.2 * (i / max(1, len(lengths) - 1))
        block(x0, w, h, INDIGO, 0.55 + 0.09 * i, f"conv {C}", f"{C}x{L}", row=i)
        x0 += w + 1.3
    band(enc_x0, x0 - 1.3, 9.3, "encoder — Conv1d stride 2 + BN + PReLU", INDIGO)

    # Latent
    lat_x, lat_w = x0 + 1.6, 3.2
    block(lat_x, lat_w, 2.3, AMBER, 0.92, "z (256)")
    ax.text(lat_x + lat_w / 2, 5 - 2.3 / 2 - 0.30, "global pool -> fc",
            ha="center", va="top", fontsize=7.0, color=MUTED)
    band(lat_x, lat_x + lat_w, 11.2, "the embedding", AMBER)
    ax.annotate("", xy=(lat_x + lat_w / 2, 6.35), xytext=(lat_x + lat_w / 2, 10.85),
                arrowprops=dict(arrowstyle="->", color=AMBER, linewidth=1.2))
    ax.text(lat_x + lat_w / 2, 2.35, "this is what\nwe cluster", ha="center", va="top",
            fontsize=8.4, color=AMBER, style="italic", fontweight="bold")

    # Decoder — mirror image.
    dec_x0 = lat_x + lat_w + 2.4
    x0 = dec_x0
    for i, (L, C) in enumerate(zip(reversed(lengths), reversed(channels), strict=True)):
        h = 1.2 + 6.4 * (L / max_len)
        w = 4.2 - 2.2 * (i / max(1, len(lengths) - 1))
        block(x0, w, h, GREEN, 0.30 + 0.07 * i, f"deconv {C}", f"{C}x{L}", row=i + 1)
        x0 += w + 1.3
    band(dec_x0, x0 - 1.3, 9.3, "decoder — ConvTranspose1d", GREEN)

    # Output
    out_x, out_w = x0 + 0.6, 3.4
    block(out_x, out_w, 7.6, ROSE, 0.80, "recon", f"1 x {N_PIX}")
    band(out_x, out_x + out_w, 11.2, "predicted", ROSE)

    # The loss arrow — only over the hidden pixels, which is the whole point.
    ax.add_patch(FancyArrowPatch((in_x + in_w / 2, -0.75), (out_x + out_w / 2, -0.75),
                                 arrowstyle="<->", color=ROSE, linewidth=1.5,
                                 mutation_scale=13, zorder=5))
    ax.text((in_x + in_w / 2 + out_x + out_w / 2) / 2, -1.05,
            "loss = MSE on the HIDDEN pixels only  —  no abundance label anywhere",
            ha="center", va="top", fontsize=9.6, color=ROSE, fontweight="bold")

    ax.set_title(f"Masked spectral autoencoder — {n_params / 1e6:.1f}M parameters, "
                 f"8575 pixels in, 256 numbers out",
                 fontsize=12.5, pad=16, color=INK)
    _save(fig, "mae_arch.png")


def _block_mask(length: int, ratio: float, block: int, seed: int) -> np.ndarray:
    """Same construction as the model's make_block_mask, fixed seed for teaching."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(length, dtype=bool)
    n_blocks = max(1, int(length * ratio / block))
    for s in rng.integers(0, max(1, length - block), n_blocks):
        mask[s:s + block] = True
    return mask


def _clean_window(flux: np.ndarray, width: int = 1600) -> tuple[int, int]:
    """Pick a plotting window that avoids APOGEE's inter-detector gaps.

    Those gaps are zero-flux in the raw file, which standardisation turns into a
    long flat line at ~-3 sigma. Plotted, it reads as a modelling artefact rather
    than as "no detector here", so the teaching figures step around it.
    """
    zero = np.isclose(flux, flux.min(), atol=1e-6) | (np.abs(flux) < 1e-9)
    best, best_score = 0, -1.0
    for start in range(0, flux.size - width, 50):
        score = 1.0 - zero[start:start + width].mean()
        if score > best_score:
            best, best_score = start, score
        if best_score >= 1.0:
            break
    return best, best + width


def fig_steps(flux: np.ndarray) -> float:
    """Steps 1-3, each a real spectrum and (for step 3) a real reconstruction."""
    import torch

    from cluster.models.masked_spectral_ae import MaskedSpectralAE

    mask = _block_mask(N_PIX, 0.5, 200, seed=7)
    # A gap-free window with a couple of masked blocks, so the figures are
    # readable rather than 8575 pixels of hairline.
    lo, hi = _clean_window(flux)
    xs = np.arange(lo, hi)

    # ---- step 1: mask ----
    fig, ax = plt.subplots(figsize=(11.2, 3.5))
    ax.plot(xs, flux[lo:hi], color=BLUE, linewidth=0.9, label="visible flux")
    seen = False
    for start in range(lo, hi):
        if mask[start] and not mask[start - 1]:
            end = start
            while end < N_PIX and mask[end]:
                end += 1
            ax.axvspan(start, min(end, hi), color=ROSE, alpha=0.16, zorder=0,
                       label=None if seen else "hidden block (200 px)")
            seen = True
    ax.set_xlabel("pixel (wavelength bin)")
    ax.set_ylabel("standardised flux")
    ax.set_title("Step 1 — hide contiguous wavelength blocks (~50% of the spectrum)",
                 fontsize=12, pad=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.text(0.01, 0.97, "the model never sees the shaded pixels",
            transform=ax.transAxes, fontsize=9, color=ROSE, va="top", style="italic")
    _save(fig, "mae_step1_mask.png")

    # ---- step 2: encode ----
    model = MaskedSpectralAE(N_PIX, 256)
    with torch.no_grad():
        h = torch.zeros(1, 1, N_PIX)
        shapes = [(1, N_PIX)]
        for layer in model.encoder:
            h = layer(h)
            if isinstance(layer, torch.nn.Conv1d):
                shapes.append((h.shape[1], h.shape[2]))

    fig, ax = plt.subplots(figsize=(11.2, 3.5))
    lens = [L for _, L in shapes]
    chans = [c for c, _ in shapes]
    xpos = np.arange(len(shapes))
    ax.bar(xpos, lens, color=[BLUE] + [INDIGO] * (len(shapes) - 1), alpha=0.85, width=0.62)
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels(["input"] + [f"conv {i + 1}" for i in range(len(shapes) - 1)])
    ax.set_ylabel("sequence length (log)")
    for x, L, c in zip(xpos, lens, chans, strict=True):
        ax.text(float(x), L * 1.18, f"{c} ch\n{L} long", ha="center", fontsize=8.2, color=INK)
    ax.set_ylim(top=max(lens) * 4)
    ax.set_title("Step 2 — each stride-2 conv halves the length and widens the channels",
                 fontsize=12, pad=10)
    ax.text(0.99, 0.94, "then global-average pool + one linear layer -> z (256-d)",
            transform=ax.transAxes, fontsize=9.2, color=AMBER, ha="right",
            va="top", fontweight="bold")
    _save(fig, "mae_step2_encode.png")

    # ---- step 3: the model's real reconstruction ----
    model = MaskedSpectralAE(N_PIX, 256)
    state = torch.load(CKPT, map_location="cpu")
    state = state.get("model", state)
    model.load_state_dict(state)
    model.eval()

    x = torch.tensor(flux, dtype=torch.float32).view(1, 1, -1)
    m = torch.tensor(mask).view(1, 1, -1)
    with torch.no_grad():
        z = model.encode(x * (~m))
        recon = model.dec_fc(z).view(1, -1, model._enc_len)
        recon = model.decoder(recon)[:, :, :N_PIX]
    rec = recon.numpy().ravel()
    mse_hidden = float(((rec[mask] - flux[mask]) ** 2).mean())

    fig, ax = plt.subplots(figsize=(11.2, 3.5))
    ax.plot(xs, flux[lo:hi], color=BLUE, linewidth=0.9, label="true flux", zorder=2)
    seen = False
    for start in range(lo, hi):
        if mask[start] and not mask[start - 1]:
            end = start
            while end < N_PIX and mask[end]:
                end += 1
            end = min(end, hi)
            ax.axvspan(start, end, color=ROSE, alpha=0.12, zorder=0)
            seg = np.arange(start, end)
            ax.plot(seg, rec[start:end], color=GREEN, linewidth=1.6, zorder=3,
                    label=None if seen else "model's prediction (never saw these)")
            seen = True
    ax.set_xlabel("pixel (wavelength bin)")
    ax.set_ylabel("standardised flux")
    ax.set_title("Step 3 — reconstruct the hidden blocks from the 256-d latent alone",
                 fontsize=12, pad=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.text(0.01, 0.97, f"MSE on hidden pixels = {mse_hidden:.3f}   (this is the entire loss)",
            transform=ax.transAxes, fontsize=9.4, color=GREEN, va="top", fontweight="bold")
    _save(fig, "mae_step3_recon.png")
    return mse_hidden


def fig_why_blocks(flux: np.ndarray) -> tuple[float, float]:
    """Why blocks, not random pixels — the design argument, made empirical."""
    rng = np.random.default_rng(3)
    wlo, whi = _clean_window(flux, width=500)
    lo, hi = wlo, whi
    seg = flux[lo:hi].copy()
    n = seg.size

    # (a) random pixels: trivially interpolated from immediate neighbours.
    rand_mask = np.zeros(n, dtype=bool)
    rand_mask[rng.choice(n, int(n * 0.5), replace=False)] = True
    idx = np.arange(n)
    interp_rand = np.asarray(np.interp(idx, idx[~rand_mask], seg[~rand_mask]))
    err_rand = float(((interp_rand[rand_mask] - seg[rand_mask]) ** 2).mean())

    # (b) one contiguous block: interpolation has nothing local to lean on.
    blk_mask = np.zeros(n, dtype=bool)
    blk_mask[150:350] = True
    interp_blk = np.asarray(np.interp(idx, idx[~blk_mask], seg[~blk_mask]))
    err_blk = float(((interp_blk[blk_mask] - seg[blk_mask]) ** 2).mean())

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.8), sharey=True)
    for ax, mk, interp, err, title in [
        (axes[0], rand_mask, interp_rand, err_rand, "random pixels — too easy"),
        (axes[1], blk_mask, interp_blk, err_blk, "one contiguous block — hard"),
    ]:
        ax.plot(idx, seg, color=BLUE, linewidth=0.9, label="true flux", zorder=2)
        ax.plot(idx[mk], interp[mk], ".", color=ROSE, markersize=2.6,
                label="linear interpolation", zorder=3)
        ax.set_title(f"{title}\nnaive-interpolation MSE = {err:.3f}", fontsize=11, pad=8)
        ax.set_xlabel("pixel")
    axes[0].set_ylabel("standardised flux")
    axes[0].legend(frameon=False, fontsize=8.6, loc="lower right")
    fig.suptitle("Why contiguous blocks: a hole you can interpolate teaches the model nothing",
                 fontsize=12.5, y=1.12)
    ratio = err_blk / err_rand if err_rand else float("nan")
    fig.text(0.5, -0.09,
             f"Interpolating a block is {ratio:.0f}x worse — so filling it "
             f"requires the line physics, not the neighbouring pixels.",
             ha="center", fontsize=9.6, color=INK, style="italic")
    _save(fig, "mae_why_blocks.png")
    return err_rand, err_blk


def fig_latent_downstream() -> None:
    """What the latent buys downstream — real members, real clusters.

    Uses the project's own ``spectral_prepared`` entry point rather than a
    hand-rolled merge, so the population here is exactly the population the
    benchmark scores.
    """
    from sklearn.decomposition import PCA

    sys.path.insert(0, "src")
    from cluster.config import Settings
    from cluster.spectral import spectral_prepared

    sys.path.insert(0, "scripts")
    from make_results_figures import load_prepared

    settings = Settings()
    settings.require_aspcap_flag_clean = False
    prepared = load_prepared()
    spec = spectral_prepared(prepared, "data/embeddings/masked_latent.parquet", settings)

    df = spec.df.drop_duplicates(subset=["APOGEE_ID"], keep="first")
    show = ["M 3", "M 67", "NGC 188", "Berkeley 66", "IC 166"]
    sub = df[df["cluster"].astype(str).isin(show)]
    if sub.empty:
        print("  (skipping latent figure: no overlap with the five clusters)", flush=True)
        return

    X = sub[list(spec.elements)].to_numpy(float)
    labels = sub["cluster"].astype(str).to_numpy()
    Y = PCA(n_components=2).fit_transform(X)

    fig, ax = plt.subplots(figsize=(7.6, 5.1))
    colors = {"M 3": INDIGO, "M 67": BLUE, "NGC 188": AMBER,
              "Berkeley 66": ROSE, "IC 166": GREEN}
    for name in show:
        m = labels == name
        if m.sum():
            ax.scatter(Y[m, 0], Y[m, 1], s=44, c=colors[name],
                       label=f"{name} (n={int(m.sum())})",
                       edgecolors="white", linewidths=0.7, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.set_title("Step 4 — the 256-d latent, projected to 2-D:\n"
                 "clusters the model was never told about",
                 fontsize=12, pad=10)
    _save(fig, "mae_step4_latent.png")
    print(f"  latent figure: {len(sub)} stars across {len(set(labels))} clusters", flush=True)


def main() -> None:
    os.makedirs(DECK, exist_ok=True)
    print("masked-AE teaching figures", flush=True)
    flux = load_one_spectrum()
    fig_architecture()
    mse = fig_steps(flux)
    errs = fig_why_blocks(flux)
    fig_latent_downstream()
    print(f"\n  numbers for the slides: hidden-pixel MSE = {mse:.3f}; "
          f"interp random = {errs[0]:.3f}, interp block = {errs[1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
