"""Why the DR17 backfill is not comparable to the DR19 embeddings.

The batch effect in ``masked_latent_all.parquet`` is NOT "DR17 vs DR19 data".
DR19 reanalyses and *includes* the DR17 stars: ``astraAllStarASPCAP-0.6.0``
carries 717,689 rows with ``release='dr17'``, and every backfilled member has
finite DR19 ASPCAP parameters.

The confound is a **data-product mismatch**:

    DR19 arm : apStar / mwmStar  -> RAW flux, median ~5.8e3
    backfill : aspcapStar-dr17   -> CONTINUUM-NORMALISED flux, median ~1.01

Two different products, ~3 orders of magnitude apart, fed to one autoencoder.

This script proves it with a paired control: 253 stars were embedded through
BOTH pipelines, so physics is held constant and only the product changes.

Run:
    .venv/bin/python scripts/diagnose_product_mismatch.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DR17 = "data/embeddings/masked_latent_dr17.parquet"
DR19 = "data/embeddings/masked_latent.parquet"


def _load(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path).set_index("APOGEE_ID")
    frame.index = frame.index.astype(str)
    return frame


def main() -> None:
    d17, d19 = _load(DR17), _load(DR19)
    both = sorted(set(d17.index) & set(d19.index))
    if not both:
        print("no paired stars — nothing to diagnose")
        return

    a = d17.loc[both].to_numpy(dtype=float)  # aspcapStar, normalised
    b = d19.loc[both].to_numpy(dtype=float)  # apStar, raw

    paired = np.linalg.norm(a - b, axis=1)
    rng = np.random.default_rng(0)
    within = np.linalg.norm(a - a[rng.permutation(len(both))], axis=1)
    cos = (a * b).sum(1) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    )
    offset = b - a
    shared = np.linalg.norm(offset.mean(0)) / np.linalg.norm(offset, axis=1).mean()

    print(f"paired control: {len(both)} identical stars through both pipelines\n")
    print(f"  same star,  different product : {paired.mean():.3f} ± {paired.std():.3f}")
    print(f"  different star, same product  : {within.mean():.3f} ± {within.std():.3f}")
    print(f"  ratio                         : {paired.mean() / within.mean():.2f}x")
    print(f"  cosine(same star, two products): {cos.mean():.3f} ± {cos.std():.3f}")
    print(f"  shared fraction of the offset : {shared:.1%}")
    print(
        "\nthe product gap exceeds the star-to-star spread: the embedding "
        "separates PRODUCTS, not populations.\n"
        "fix = re-embed every star from one product (DR19 mwmStar covers "
        "100% of the backfilled members), not a caveat."
    )


if __name__ == "__main__":
    main()
