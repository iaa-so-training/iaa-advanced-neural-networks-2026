"""Tests for kinematic membership labelling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cluster.clusters import Cluster
from cluster.membership import (
    _pleiades_members,
    angular_separation,
    combined_members,
    kinematic_members,
    label_clusters,
)


def test_angular_separation_zero_ninety_and_small_angle() -> None:
    ra = np.array([0.0, 90.0, 10.0])
    dec = np.array([0.0, 0.0, 0.0])
    sep = angular_separation(ra, dec, 0.0, 0.0)
    assert np.allclose(sep, [0.0, 90.0, 10.0])


def test_angular_separation_clips_floating_point_errors() -> None:
    sep = angular_separation(np.array([0.0]), np.array([0.0]), 0.0, 0.0)
    assert np.allclose(sep, 0.0)


def test_pleiades_members_exact_box(allstar_frame: pd.DataFrame) -> None:
    mask = _pleiades_members(allstar_frame)
    assert mask.sum() == 2
    assert bool(mask[0]) is True
    assert bool(mask[1]) is True
    assert not mask[2:].any()


def test_kinematic_members_pleiades_special_case(allstar_frame: pd.DataFrame) -> None:
    from cluster.clusters import CLUSTER_BY_NAME

    members = kinematic_members(
        allstar_frame,
        CLUSTER_BY_NAME["Pleiades"],
        seed_position_radius_deg=0.6,
        seed_parallax_frac=0.30,
        seed_pm_tol=5.0,
        seed_rv_tol=25.0,
        n_refine_passes=3,
        refine_sigma=2.5,
    )
    assert isinstance(members, pd.Series)
    assert members.name == "member_Pleiades"
    assert members.sum() == 2


def _tight_frame() -> pd.DataFrame:
    n = 8
    df = pd.DataFrame(
        {
            "RA": [100.0, 100.1, 100.2, 130.0, 150.0, 80.0, 120.0, 70.0],
            "DEC": [20.0, 20.0, 20.1, 40.0, -30.0, 50.0, 0.0, -40.0],
            "GAIAEDR3_PARALLAX": [1.00, 1.05, 0.95, 2.0, 0.5, 1.2, 1.3, 0.8],
            "GAIAEDR3_PMRA": [10.0, 10.1, 9.9, 5.0, -5.0, 2.0, 3.0, -2.0],
            "GAIAEDR3_PMDEC": [-5.0, -5.1, -4.9, 0.0, 5.0, 2.0, -1.0, 1.0],
            "VHELIO_AVG": [20.0, 20.2, 19.8, 0.0, 50.0, 30.0, -10.0, -5.0],
        }
    )
    return df


def _tight_cluster() -> Cluster:
    return Cluster(
        name="Tight",
        kind="open",
        ra_deg=100.0,
        dec_deg=20.0,
        dist_pc=1000.0,
        pmra=10.0,
        pmdec=-5.0,
        rv=20.0,
        n_ref=None,
    )


def test_kinematic_members_generic_refines_seed_box() -> None:
    df = _tight_frame()
    members = kinematic_members(
        df,
        _tight_cluster(),
        seed_position_radius_deg=1.0,
        seed_parallax_frac=0.30,
        seed_pm_tol=5.0,
        seed_rv_tol=25.0,
        n_refine_passes=3,
        refine_sigma=2.5,
    )
    assert members.iloc[0:3].all()
    assert not members.iloc[3:].any()


def test_kinematic_members_generic_fewer_than_two_stars_returns_seed() -> None:
    df = _tight_frame().iloc[:1].copy()
    members = kinematic_members(
        df,
        _tight_cluster(),
        seed_position_radius_deg=1.0,
        seed_parallax_frac=0.30,
        seed_pm_tol=5.0,
        seed_rv_tol=25.0,
        n_refine_passes=3,
        refine_sigma=2.5,
    )
    assert members.sum() == 1


def test_label_clusters_assigns_names_and_member_flag() -> None:
    df = _tight_frame()
    labelled = label_clusters(
        df,
        [_tight_cluster()],
        seed_position_radius_deg=1.0,
        seed_parallax_frac=0.30,
        seed_pm_tol=5.0,
        seed_rv_tol=25.0,
        n_refine_passes=3,
        refine_sigma=2.5,
    )
    assert set(labelled["cluster"]) == {"Tight", "field"}
    assert labelled.loc[0:2, "cluster"].eq("Tight").all()
    assert labelled.loc[3:, "cluster"].eq("field").all()
    assert labelled["is_member"].iloc[0:3].all()
    assert not labelled["is_member"].iloc[3:].any()


def test_label_clusters_does_not_mutate_input() -> None:
    df = _tight_frame()
    original = df.copy()
    label_clusters(
        df,
        [_tight_cluster()],
        seed_position_radius_deg=1.0,
        seed_parallax_frac=0.30,
        seed_pm_tol=5.0,
        seed_rv_tol=25.0,
        n_refine_passes=3,
        refine_sigma=2.5,
    )
    pd.testing.assert_frame_equal(df, original)


def test_combined_members_equals_kinematic_without_expansion() -> None:
    df = _tight_frame()
    X = np.zeros((len(df), 2), dtype=float)
    X[4:, 0] = 10.0  # field stars chemically far from the core
    comb = combined_members(
        df, _tight_cluster(), X,
        seed_position_radius_deg=1.0, seed_parallax_frac=0.30,
        seed_pm_tol=5.0, seed_rv_tol=25.0,
        n_refine_passes=3, refine_sigma=2.5,
        combined_chem_sigma=3.0, combined_kin_sigma=3.0,
    )
    assert comb.iloc[0:3].all()
    assert not comb.iloc[3:].any()


def test_combined_members_adds_chemically_close_loosely_kinematic_star() -> None:
    df = _tight_frame()
    X = np.zeros((len(df), 2), dtype=float)
    X[3] = [0.0, 0.0]  # star 3 chemically identical to the core
    comb = combined_members(
        df, _tight_cluster(), X,
        seed_position_radius_deg=1.0, seed_parallax_frac=0.30,
        seed_pm_tol=5.0, seed_rv_tol=25.0,
        n_refine_passes=3, refine_sigma=2.5,
        combined_chem_sigma=10.0, combined_kin_sigma=10.0,
    )
    assert comb.iloc[0:3].all()  # core preserved
    assert comb.iloc[3]          # added: chemically close + loosely kinematic
    assert not comb.iloc[4:].any()


def test_combined_members_under_two_kinematic_returns_seed() -> None:
    df = _tight_frame().iloc[:1].copy()
    X = np.zeros((1, 2), dtype=float)
    comb = combined_members(
        df, _tight_cluster(), X,
        seed_position_radius_deg=1.0, seed_parallax_frac=0.30,
        seed_pm_tol=5.0, seed_rv_tol=25.0,
        n_refine_passes=3, refine_sigma=2.5,
        combined_chem_sigma=3.0, combined_kin_sigma=3.0,
    )
    assert comb.sum() == 1


def test_label_clusters_combined_method() -> None:
    df = _tight_frame()
    X = np.zeros((len(df), 2), dtype=float)
    labelled = label_clusters(
        df, [_tight_cluster()],
        seed_position_radius_deg=1.0, seed_parallax_frac=0.30,
        seed_pm_tol=5.0, seed_rv_tol=25.0,
        n_refine_passes=3, refine_sigma=2.5,
        membership_method="combined", X=X,
        combined_chem_sigma=3.0, combined_kin_sigma=3.0,
    )
    assert set(labelled["cluster"]) == {"Tight", "field"}
    assert labelled["is_member"].iloc[0:3].all()
