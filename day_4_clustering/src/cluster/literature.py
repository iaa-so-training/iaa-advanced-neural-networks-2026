"""Literature parameters for the target clusters, for fit comparison.

Open clusters come from VizieR ``B/ocl/clusters`` (Dias et al. 2002-2015);
globulars from ``VII/202`` (Harris 1996, 2010 edition — arXiv:0904.2907).
Globular absolute ages are not in Harris, so they come from Dotter et al.
(2010) / Marin-Franch et al. (2009).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .clusters import Cluster

Z_SUN = 0.0152

# Harris does not tabulate absolute ages; literature values (Gyr).
_GLOBULAR_AGE_GYR: dict[str, float] = {
    "M 5": 10.5, "M 3": 11.0, "M 13": 11.5, "M 15": 12.5,
    "M 71": 10.0, "M 107": 11.5, "M 92": 12.5,
}

# Dias does not resolve these two by our names.
_FALLBACK: dict[str, dict[str, float]] = {
    "Pleiades": {"age_Gyr": 0.10, "dist_pc": 136.0, "dm": 5.67, "feh": 0.03},
    "Berkeley 66": {"age_Gyr": 1.0, "dist_pc": 5346.0, "dm": 13.64, "feh": -0.1},
}


def _dm(dist_pc: float) -> float:
    return float(5.0 * np.log10(dist_pc / 10.0))


def _feh(z: float) -> float:
    return float(np.log10(z / Z_SUN)) if z > 0 else float("nan")


def fetch_literature(cluster: Cluster, cache_dir: str | Path = "data/literature") -> dict[str, float]:
    """Return ``{age_Gyr, dist_pc, dm, feh}`` for one cluster (VizieR, cached)."""
    if cluster.name in _FALLBACK:
        return dict(_FALLBACK[cluster.name])

    cache_dir = Path(cache_dir)
    cache = cache_dir / f"lit_{cluster.name.replace(' ', '_')}.csv"
    if cache.exists():
        row = pd.read_csv(cache).iloc[0]
        return {
            "age_Gyr": float(row["age_Gyr"]), "dist_pc": float(row["dist_pc"]),
            "dm": float(row["dm"]), "feh": float(row["feh"]),
        }

    from astroquery.vizier import Vizier

    Vizier.ROW_LIMIT = 3
    if cluster.kind == "open":
        result = Vizier.query_object(cluster.name, catalog=["B/ocl/clusters"])  # pyrefly: ignore[missing-attribute]
        table = result[0]
        age_log = float(table["Age"][0])
        dist_pc = float(table["Dist"][0])
        feh = float("nan")
        for col in table.colnames:
            if "Fe" in col and table[col][0] is not None:
                feh = float(table[col][0])
                break
    else:
        result = Vizier.query_object(cluster.name, catalog=["VII/202"])  # pyrefly: ignore[missing-attribute]
        table = result[0]
        feh = float(table["[Fe/H]"][0])
        dist_pc = float(table["Rsun"][0]) * 1000.0  # kpc -> pc
        age_log = float("nan")
        if cluster.name in _GLOBULAR_AGE_GYR:
            age_log = np.log10(_GLOBULAR_AGE_GYR[cluster.name] * 1e9)

    out = {
        "age_Gyr": 10.0 ** (age_log - 9.0) if np.isfinite(age_log) else float("nan"),
        "dist_pc": dist_pc,
        "dm": _dm(dist_pc),
        "feh": feh,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([out]).to_csv(cache, index=False)
    return out


def literature_table(clusters: list[Cluster]) -> pd.DataFrame:
    rows = []
    for c in clusters:
        try:
            lit = fetch_literature(c)
            rows.append({
                "cluster": c.name, "kind": c.kind,
                "lit_age_Gyr": lit["age_Gyr"], "lit_dm": lit["dm"],
                "lit_feh": lit["feh"], "lit_dist_pc": lit["dist_pc"],
            })
        except Exception as exc:
            rows.append({
                "cluster": c.name, "kind": c.kind,
                "lit_age_Gyr": float("nan"), "lit_dm": float("nan"),
                "lit_feh": float("nan"), "lit_dist_pc": float("nan"),
            })
    return pd.DataFrame(rows)


def compare_fit(fit_age_gyr: float, fit_dm: float, fit_met_z: float, lit: dict[str, float]) -> dict[str, float]:
    """Residuals of a fit vs literature (absolute differences)."""
    fit_feh = _feh(fit_met_z)
    return {
        "age_Gyr": fit_age_gyr,
        "lit_age_Gyr": lit["age_Gyr"],
        "age_resid": abs(fit_age_gyr - lit["age_Gyr"]) if np.isfinite(lit["age_Gyr"]) else float("nan"),
        "dm": fit_dm,
        "lit_dm": lit["dm"],
        "dm_resid": abs(fit_dm - lit["dm"]),
        "feh": fit_feh,
        "lit_feh": lit["feh"],
        "feh_resid": abs(fit_feh - lit["feh"]) if np.isfinite(lit["feh"]) else float("nan"),
    }
