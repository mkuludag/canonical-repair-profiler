"""Tests for RegionIndexTool + RegionDelegatorAgent."""
import pandas as pd
import pytest

from src.tools import RegionIndexTool
from src.agents.region_delegator_agent import RegionDelegatorAgent


@pytest.fixture
def region_csv(tmp_path):
    df = pd.DataFrame({
        "state": ["CA", "TX", "MI"],
        "claims": [1000, 1000, 1000],
        "med_cost": [120.0, 100.0, 80.0],
        "med_labor_cost": [60.0, 50.0, 40.0],
    })
    p = tmp_path / "region.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_index_relative_to_national(region_csv):
    t = RegionIndexTool(region_csv)
    assert t.available()
    assert abs(t.national_cost - 100.0) < 1e-6
    assert t.index_for("CA") == 1.2
    assert t.index_for("MI") == 0.8


def test_unknown_state_neutral(region_csv):
    assert RegionIndexTool(region_csv).index_for("ZZ") == 1.0
    assert RegionIndexTool(region_csv).index_for("") == 1.0


def test_missing_file_degrades_gracefully(tmp_path):
    t = RegionIndexTool(str(tmp_path / "nope.csv"))
    assert not t.available()
    assert t.index_for("CA") == 1.0
    assert t.spread()["ratio"] == 1.0


def test_states_sorted_by_index(region_csv):
    assert RegionIndexTool(region_csv).states()[0] == "CA"  # priciest first


def test_spread(region_csv):
    sp = RegionIndexTool(region_csv).spread()
    assert sp["priciest"] == "CA" and sp["cheapest"] == "MI"
    assert sp["ratio"] == 1.5


def test_delegator_adjusts_solution(region_csv):
    agent = RegionDelegatorAgent(RegionIndexTool(region_csv))
    sol = {"cost_med": 1000.0, "cost_q25": 800.0, "cost_q75": 1200.0}
    out = agent.adjust_solution(sol, "CA")
    assert out["region_cost_index"] == 1.2
    assert out["cost_med_regional"] == 1200.0
    assert out["cost_q75_regional"] == 1440.0
    assert "above" in out["region_note"]


def test_delegator_apply_to_context(region_csv, base_ctx):
    base_ctx.solutions = [{"cost_med": 500.0}]
    agent = RegionDelegatorAgent(RegionIndexTool(region_csv))
    ctx = agent.apply(base_ctx, "MI")
    assert ctx.region["state"] == "MI"
    assert ctx.solutions[0]["cost_med_regional"] == 400.0
