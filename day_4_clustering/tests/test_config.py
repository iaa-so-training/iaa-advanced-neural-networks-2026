"""Tests for :mod:`cluster.config`."""

from __future__ import annotations

from typing import Any

from cluster import config


def test_parse_bool_variants() -> None:
    for raw in ("1", "true", "TRUE", "yes", "on"):
        assert config._parse(raw) is True
    for raw in ("0", "false", "FALSE", "no", "off"):
        assert config._parse(raw) is False


def test_parse_none_variants() -> None:
    for raw in ("none", "null", "~"):
        assert config._parse(raw) is None


def test_parse_numeric_and_text() -> None:
    assert config._parse("42") == 42
    assert config._parse("  42  ") == 42
    assert config._parse("-3") == -3
    assert config._parse("3.5") == 3.5
    assert config._parse("hello") == "hello"


def test_parse_lists() -> None:
    assert config._parse("[a, b,c]") == ["a", "b", "c"]
    assert config._parse("a,b") == ["a", "b"]
    assert config._parse("[]") == []
    assert config._parse("single") == "single"


def test_env_uses_prefix_and_falls_back(monkeypatch: Any) -> None:
    monkeypatch.setenv("CLUSTER_TEST_KEY", "yes")
    assert config.env("TEST_KEY", False) is True
    monkeypatch.delenv("CLUSTER_TEST_KEY")
    assert config.env("TEST_KEY", 99) == 99


def test_env_parses_int_float_bool_list(monkeypatch: Any) -> None:
    monkeypatch.setenv("CLUSTER_INT_VALUE", "17")
    monkeypatch.setenv("CLUSTER_FLOAT_VALUE", "1.25")
    monkeypatch.setenv("CLUSTER_BOOL_VALUE", "off")
    monkeypatch.setenv("CLUSTER_LIST_VALUE", "[x, y]")
    assert config.env("INT_VALUE", 0) == 17
    assert config.env("FLOAT_VALUE", 0.0) == 1.25
    assert config.env("BOOL_VALUE", True) is False
    assert config.env("LIST_VALUE", []) == ["x", "y"]


def test_settings_default_builds() -> None:
    settings = config.Settings.default()
    assert isinstance(settings, config.Settings)
    assert settings.fast == config.FAST
    assert settings.snr_min == config.SNR_MIN
    assert settings.elements == config.ELEMENTS
    assert settings.tsne["perplexity"] == config.TSNE["perplexity"]
    assert settings.cluster_names == config.CLUSTER_NAMES


def test_settings_accepts_explicit_values() -> None:
    settings = config.Settings(fast=False, snr_min=12.5, standardize=False)
    assert settings.fast is False
    assert settings.snr_min == 12.5
    assert settings.standardize is False


def test_resolve_cluster_names_all_variants() -> None:
    available = ["Pleiades", "M 67", "NGC 6819"]
    for raw in ("all", ["all"]):
        settings = config.Settings(cluster_names=raw)
        assert settings.resolve_cluster_names(available) == available


def test_resolve_cluster_names_specific_and_unknown() -> None:
    available = ["Pleiades", "M 67", "NGC 6819"]
    settings = config.Settings(cluster_names=["Pleiades", "missing"])
    assert settings.resolve_cluster_names(available) == ["Pleiades"]
    settings_single = config.Settings(cluster_names="Pleiades")
    assert settings_single.resolve_cluster_names(available) == ["Pleiades"]
