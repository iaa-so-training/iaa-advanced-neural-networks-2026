"""Re-embed the backfilled members from DR19 mwmStar (one uniform product).

Supersedes the DR17 aspcapStar backfill. The masked AE was pretrained on
per-star-standardised raw apStar flux (8575 px). mwmStar's APOGEE/APO HDU is
the same product family (raw combined spectrum, 8575 px, v_astra 0.6.0), so
the same preprocessing applies and the latents are comparable to the DR19 arm.

Usage:
    # 1) verify the checkpoint + preprocessing reproduce the published latents
    .venv/bin/python scripts/embed_dr19_rerun.py --verify

    # 2) download the 738 mwmStar files (if not present) and embed them
    .venv/bin/python scripts/embed_dr19_rerun.py \
        --model data/embeddings/model_dr19.pt \
        --out data/embeddings/masked_latent_dr19_rerun.parquet
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits

MODEL_PATH = "data/embeddings/masked_ae_rerun.pt"  # MaskedSpectralAE checkpoint (not model_dr19.pt, which is the supervised net)
URL_LIST = "data/dr19_rerun_members.csv"
MWM_DIR = Path("data/mwmstar")
OUT_PATH = "data/embeddings/masked_latent_dr19_rerun.parquet"
N_BINS = 8575
BATCH = 256


def load_model():
    import torch  # the `torch` extra: uv sync --extra torch
    from cluster.models.masked_spectral_ae import MaskedSpectralAE

    model = MaskedSpectralAE(n_features=N_BINS, latent_dim=256)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, torch


def standardise(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X, nan=0.0)
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True) + 1e-8
    return (X - m) / s


def embed(model: Any, torch: Any, X: np.ndarray) -> np.ndarray:
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).unsqueeze(1)
    Z = []
    with torch.no_grad():
        for i in range(0, len(Xt), BATCH):
            Z.append(model.embed(Xt[i:i + BATCH]).cpu().numpy())
    return np.concatenate(Z, axis=0)


def verify(flux_csv: str) -> None:
    """Reproduce a few published latents from the DR19 flux CSV.

    ``flux_csv`` is the spectrum->abundance table of the training project, which
    is not redistributed here: it is far too large for the bundle and the
    published latents are the product students need. Point ``--flux-csv`` at a
    copy if you have one.
    """
    model, torch = load_model()
    ref = pd.read_parquet("data/embeddings/masked_latent.parquet")
    ref["APOGEE_ID"] = ref["APOGEE_ID"].astype(str)
    test = ref.iloc[:3]
    ids = test["APOGEE_ID"].tolist()

    csv = Path(flux_csv)
    if not csv.exists():
        raise SystemExit(
            f"flux CSV not found: {csv}\n"
            "pass --flux-csv <path> (the file belongs to the training project "
            "and is not part of the workshop bundle)",
        )
    rows = []
    for chunk in pd.read_csv(csv, chunksize=200_000):
        f = chunk["FILE"].astype(str)
        f = f.str.removeprefix("apStar-1.3-apo25m-").str.replace(r"-\d{5}$", "", regex=True)
        hit = f.isin(ids)
        if hit.any():
            rows.append(chunk[hit.to_numpy()])
            if len(pd.concat(rows)) >= len(ids):
                break
    got = pd.concat(rows).drop_duplicates(subset="FILE", keep="first")
    got["APOGEE_ID"] = got["FILE"].astype(str).str.removeprefix("apStar-1.3-apo25m-")
    got = got.set_index("APOGEE_ID").loc[ids]
    cols = [c for c in got.columns if c != "FILE"]
    X = standardise(got[cols].to_numpy(dtype=np.float32))
    Z = embed(model, torch, X)
    refZ = test[[f"z{i}" for i in range(256)]].to_numpy()
    delta = np.abs(Z - refZ).max()
    rel = np.abs(Z - refZ).mean() / (np.abs(refZ).mean() + 1e-9)
    print(f"checkpoint reproduction on {len(ids)} DR19 stars: "
          f"max |Δ| = {delta:.2e}, mean rel = {rel:.1%}")
    print("=> PASS, preprocessing + checkpoint confirmed" if delta < 1e-4
          else "=> MISMATCH — do not re-embed until this is understood")


def download(urls: pd.DataFrame) -> None:
    MWM_DIR.mkdir(parents=True, exist_ok=True)

    def fetch(row: pd.Series) -> str:
        out = MWM_DIR / f"{row['APOGEE_ID']}.fits"
        if out.exists() and out.stat().st_size > 50_000:
            return "have"
        subprocess.run(
            ["curl", "-s", "--max-time", "180", "--retry", "3",
             "-o", str(out), row["url"]],
            check=False,
        )
        ok = out.exists() and out.stat().st_size > 50_000
        if not ok:
            out.unlink(missing_ok=True)
        return "ok" if ok else "FAIL"

    with ThreadPoolExecutor(12) as ex:
        results = list(ex.map(fetch, [r for _, r in urls.iterrows()]))
    print(f"downloads: {results.count('ok')} new, "
          f"{results.count('have')} cached, {results.count('FAIL')} failed")


def read_mwmstar(path: Path) -> np.ndarray | None:
    """Combined APOGEE spectrum from a mwmStar file (raw, 8575 px).

    Prefers the APO HDU; falls back to LCO for the southern stars.
    """
    try:
        with fits.open(path, memmap=False) as h:
            candidates: list[np.ndarray] = []
            for hdu in h:
                name = (hdu.name or "").upper()
                if not name.startswith("APOGEE") or hdu.data is None:
                    continue
                if "flux" not in hdu.columns.names or hdu.data.shape[0] == 0:
                    continue
                flux = np.asarray(hdu.data["flux"], dtype=np.float64)
                if flux.ndim == 2:
                    flux = flux[0]
                if np.nanmedian(flux) > 0 and flux.size == N_BINS:
                    if name.endswith("APO"):
                        return np.asarray(flux, dtype=np.float32)
                    candidates.append(np.asarray(flux, dtype=np.float32))
            if candidates:
                return candidates[0]
    except Exception as exc:
        print(f"  !! {path.name}: {exc}", flush=True)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--download-only", action="store_true",
                        help="fetch the mwmStar files, do not embed")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--flux-csv", default="data/raw_data/flux_abundances.csv",
        help="Spectrum->abundance CSV used by --verify (training-project data "
             "product, not shipped here).",
    )
    args = parser.parse_args()

    urls = pd.read_csv(URL_LIST, dtype=str)
    if args.download_only or not (args.verify or args.skip_download):
        download(urls)
    if args.download_only:
        return
    if args.verify:
        verify(args.flux_csv)
        return

    model, torch = load_model()
    print(f"re-embedding {len(urls)} members from DR19 mwmStar", flush=True)

    ids: list[str] = []
    spectra: list[np.ndarray] = []
    t0 = time.time()
    for _, row in urls.iterrows():
        path = MWM_DIR / f"{row['APOGEE_ID']}.fits"
        if not path.exists():
            print(f"  !! missing {path.name}", flush=True)
            continue
        flux = read_mwmstar(path)
        if flux is None or flux.size != N_BINS:
            print(f"  !! bad flux {path.name}: {None if flux is None else flux.size}", flush=True)
            continue
        ids.append(row["APOGEE_ID"])
        spectra.append(flux)
    print(f"read {len(ids)} spectra in {time.time() - t0:.0f}s", flush=True)

    X = standardise(np.vstack(spectra))
    Z = embed(model, torch, X)

    frame = pd.DataFrame({"APOGEE_ID": ids})
    for d in range(Z.shape[1]):
        frame[f"z{d}"] = Z[:, d].astype("float32")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    print(f"{len(frame)} x {Z.shape[1]} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
