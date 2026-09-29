"""Tests for the deterministic consensus math (StatsTool)."""
import numpy as np
import pandas as pd
import pytest

from src.tools import StatsTool


def test_consensus_fraction_tight_is_high():
    s = pd.Series([100, 101, 99, 102, 98, 100])
    assert StatsTool.consensus_fraction(s) == 1.0


def test_consensus_fraction_spread_is_low():
    s = pd.Series([10, 1000, 20, 2000, 30, 3000])
    assert StatsTool.consensus_fraction(s) < 0.6


def test_consensus_fraction_too_few_is_nan():
    assert np.isnan(StatsTool.consensus_fraction(pd.Series([100.0])))


def test_consensus_fraction_drops_nonpositive():
    s = pd.Series([0, -5, 100, 101, 99])
    val = StatsTool.consensus_fraction(s)
    assert 0.0 <= val <= 1.0


def test_band_basic():
    b = StatsTool.band(pd.Series([100, 200, 300, 400]))
    assert b["n"] == 4
    assert b["q25"] < b["median"] < b["q75"]


def test_band_empty():
    b = StatsTool.band(pd.Series([], dtype=float))
    assert b["n"] == 0
    assert np.isnan(b["median"])


def test_cost_tier_partition_two_tiers(synthetic_claims):
    low = StatsTool.cost_tier_partition(synthetic_claims, "gsar_tot_cost_gross", "low", 2)
    high = StatsTool.cost_tier_partition(synthetic_claims, "gsar_tot_cost_gross", "high", 2)
    assert low["gsar_tot_cost_gross"].max() < high["gsar_tot_cost_gross"].min()


def test_cost_tier_partition_three_tiers():
    df = pd.DataFrame({"gsar_tot_cost_gross": list(range(1, 31))})
    lo = StatsTool.cost_tier_partition(df, "gsar_tot_cost_gross", "low", 3)
    mid = StatsTool.cost_tier_partition(df, "gsar_tot_cost_gross", "mid", 3)
    hi = StatsTool.cost_tier_partition(df, "gsar_tot_cost_gross", "high", 3)
    assert lo["gsar_tot_cost_gross"].max() <= mid["gsar_tot_cost_gross"].min()
    assert mid["gsar_tot_cost_gross"].max() <= hi["gsar_tot_cost_gross"].min()


def test_cost_tier_partition_single_repair_returns_all(synthetic_claims):
    out = StatsTool.cost_tier_partition(synthetic_claims, "gsar_tot_cost_gross", "all", 1)
    assert len(out) == len(synthetic_claims)


def test_confidence_monotonic_in_support():
    assert StatsTool.confidence(30, 0.9) > StatsTool.confidence(5, 0.9)


def test_confidence_handles_nan_consensus():
    assert StatsTool.confidence(10, float("nan")) == 0.0


@pytest.mark.parametrize("n,cons", [(1, 1.0), (1000, 0.5), (10, 0.0)])
def test_confidence_in_unit_range(n, cons):
    assert 0.0 <= StatsTool.confidence(n, cons) <= 1.0
