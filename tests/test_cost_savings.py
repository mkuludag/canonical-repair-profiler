"""Tests for CostSavingsAgent (deterministic avoidable-overspend math)."""
import pandas as pd

from src.agents.cost_savings_agent import CostSavingsAgent


def test_compute_basic():
    costs = pd.Series([100, 100, 200, 300])  # canonical 100 -> excess 0+0+100+200 = 300
    m = CostSavingsAgent.compute(costs, 100.0)
    assert m["avoidable_overspend"] == 300.0
    assert m["n_above"] == 2
    assert m["savings_per_claim"] == 75.0
    assert m["pct_claims_above"] == 50.0


def test_compute_no_overspend_when_canonical_is_max():
    costs = pd.Series([100, 200, 300])
    m = CostSavingsAgent.compute(costs, 300.0)
    assert m["avoidable_overspend"] == 0.0
    assert m["n_above"] == 0


def test_compute_handles_empty():
    m = CostSavingsAgent.compute(pd.Series([], dtype=float), 100.0)
    assert m["n_claims"] == 0 and m["avoidable_overspend"] == 0.0


def test_compute_ignores_bad_canonical():
    m = CostSavingsAgent.compute(pd.Series([100, 200]), float("nan"))
    assert m["avoidable_overspend"] == 0.0


def test_analyze_writes_solution_fields(base_ctx):
    base_ctx.solutions = [{"cost_med": 2000.0}]  # synthetic_claims span ~1.9k-9.3k
    ctx = CostSavingsAgent().analyze(base_ctx)
    s = ctx.solutions[0]
    assert s["savings_avoidable_overspend"] > 0
    assert s["savings_n_claims"] == len(base_ctx.claims)
    assert ctx.savings["avoidable_overspend"] == s["savings_avoidable_overspend"]
