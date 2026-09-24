# Data-release migration (DR19 / GALAH DR4 / Gaia DR3)

Why: the workshop originally pinned APOGEE DR17 (SDSS-IV final), GALAH DR3,
Gaia EDR3. These are superseded. This document records the target releases
and the exact column mapping used to migrate.

## Target releases

| Survey | Was | Now | Notes |
|--------|-----|-----|-------|
| APOGEE | DR17 `allStar-dr17-synspec_rev1.fits` (3.7 GB, ASPCAP) | SDSS-V DR19 `astraAllStarASPCAP-0.6.0.fits.gz` (1.17 GB) | ASPCAP → **Astra** |
| GALAH | DR3 (VizieR `J/MNRAS/506/150`) | DR4 `galah_dr4_allstar_240705.fits` (Data Central) | lowercase columns, built-in Gaia DR3 crossmatch |
| Gaia | EDR3 (`GAIAEDR3_*` embedded in allStar) | DR3 | embedded in both new catalogs |

## APOGEE DR19 — one file (`astraAllStarASPCAP`)

SDSS-V replaced the ASPCAP pipeline with the **Astra** framework. The single
DR17 allStar (ASPCAP synspec) is now `astraAllStarASPCAP-0.6.0.fits.gz`
(1.17 GB, 1 095 480 MWM targets — APOGEE + BOSS). It carries stellar params,
`[X/H]` abundances, Gaia DR3 astrometry/photometry, and quality flags.

Two schema changes matter:

1. Abundances are **`[X/H]`** (lowercase `_h`), not `[X/Fe]` → convert
   `X_FE = x_h − fe_h`.
2. Flags renamed: `ASPCAPFLAG`→`flag_bad`, `STARFLAG`→`spectrum_flags`.

### Verifying the download

`cluster download --all` records the catalogue's sha256 in a sidecar beside it
(`data/astraAllStarASPCAP-0.6.0.fits.gz.sha256`) and prints it; a later run
compares the file against it. As fetched on 2026-09-24, at 1 171 102 556 bytes:

```
5324bf39baeede7553b0a3c8a50bea1760f47eb67a95a8a84e6db303aa2edb04
```

`cluster doctor --deep` re-hashes both the catalogue and the asset bundle and
checks them against these records — a size check alone catches a truncated
transfer but not a corrupted one.

### Column mapping (Astra ASPCAP → internal schema)

| Internal | Source | Notes |
|----------|--------|-------|
| `APOGEE_ID` | `sdss4_apogee_id` | DR17 cross-ref id |
| `RA`, `DEC` | `ra`, `dec` | |
| `GLON`, `GLAT` | computed from `ra`/`dec` | Astra ships no galactic coords |
| `SNR` | `snr` | |
| `TEFF`, `LOGG` | `teff`, `logg` | |
| `FE_H` | `fe_h` | |
| `C_FE` … `NI_FE` | `c_h` … `ni_h` − `fe_h` | [X/H]→[X/Fe] |
| `{e}_ERR` | sqrt(`e_{x_h}`² + `e_fe_h`²) | quadrature |
| `{e}_FLAG` | `{x_h}_flags` | |
| `GAIAEDR3_PARALLAX` | `plx` | Gaia DR3 (name kept for churn-free internal schema) |
| `GAIAEDR3_PMRA` | `pmra` | |
| `GAIAEDR3_PMDEC` | `pmde` | |
| `VHELIO_AVG` | `v_rad` | |
| `GAIAEDR3_PHOT_*_MEAN_MAG` | `g_mag`/`bp_mag`/`rp_mag` | |
| `J`, `H`, `K` | `j_mag`, `h_mag`, `k_mag` | |
| `ASPCAPFLAG` | `flag_bad` | 0 = analysis not flagged bad |
| `STARFLAG` | `spectrum_flags` | 0 = clean |

Element name map (Astra `_h` → internal `_FE`):

```
c_h→C_FE  n_h→N_FE   o_h→O_FE   na_h→NA_FE  mg_h→MG_FE  al_h→AL_FE
si_h→SI_FE  s_h→S_FE  k_h→K_FE  ca_h→CA_FE  ti_h→TI_FE  v_h→V_FE
cr_h→CR_FE  mn_h→MN_FE  ni_h→NI_FE  fe_h→FE_H
```

Note: the internal `GAIAEDR3_*` / `VHELIO_AVG` names are kept so the rest of
the pipeline is release-agnostic; they now carry Gaia **DR3** values (the
"EDR3" in the name is historical).

## GALAH DR4

`galah_dr4_allstar_240705.fits` (723 MB) from Data Central. One entry per star,
917 588 stars, up to 30 abundances.

| DR3 (old) | DR4 (new) |
|-----------|-----------|
| VizieR `J/MNRAS/506/150` | Data Central FITS |
| `RA_ICRS`, `DE_ICRS` | `ra`, `dec` (via Gaia DR3 crossmatch) |
| `Teff`, `logg`, `[Fe/H]` | `teff`, `logg`, `fe_h` |
| `[X/Fe]` | `x_fe` |
| no Gaia astrometry | `gaiadr3_source_id` + VAC (`parallax`, `pmra`, `pmdec`, `phot_g_mean_mag`) |

DR4 ships a Gaia DR3 crossmatch VAC (`galah_dr4_vac_wise_tmass_gaiadr3_240705.fits`),
so the manual `astroquery.gaia` cone crossmatch in
`scripts/build_galah_apogee.py` is no longer needed.
