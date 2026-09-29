"""
CostSavingsAgent - quantifies the dollars saved by converging a repair group to its canonical cost.

Business framing: within a single Repair Signature, claims are scattered around the consensus
(canonical) cost. Claims ABOVE the canonical cost are avoidable overspend - if every dealer executed
the Canonical Repair at the consensus cost, that excess would not be paid. This agent measures that
gap, giving a grounded, data-derived savings number per group (distinct from the model-uplift
business case in the README).

    avoidable_overspend = sum over claims of max(0, claim_cost - canonical_cost)

It is deterministic (no LLM). HAND-OFF: reads ctx.claims + ctx.solutions, writes savings fields onto
each solution and records ctx-level totals on the primary solution.
"""
import numpy as np
import pandas as pd

from .context import RepairContext
from .. import config as C


class CostSavingsAgent:
    def __init__(self, stats_tool=None):
        self.stats = stats_tool  # not required; kept for symmetry with other agents

    @staticmethod
    def compute(costs: pd.Series, canonical_cost: float) -> dict:
        """Savings metrics for a set of claim costs converging to `canonical_cost`."""
        v = pd.to_numeric(costs, errors="coerce").dropna()
        v = v[(v > 0) & (v < C.COST_CAP)]
        if len(v) == 0 or not canonical_cost or canonical_cost != canonical_cost:
            return {"n_claims": int(len(v)), "n_above": 0, "avoidable_overspend": 0.0,
                    "savings_per_claim": 0.0, "pct_claims_above": 0.0, "total_spend": float(v.sum())}
        excess = (v - canonical_cost).clip(lower=0)
        n_above = int((v > canonical_cost).sum())
        return {
            "n_claims": int(len(v)),
            "n_above": n_above,
            "avoidable_overspend": round(float(excess.sum()), 2),
            "savings_per_claim": round(float(excess.sum() / len(v)), 2),
            "pct_claims_above": round(100.0 * n_above / len(v), 1),
            "total_spend": round(float(v.sum()), 2),
        }

    def analyze(self, ctx: RepairContext) -> RepairContext:
        if ctx.claims is None or not ctx.solutions:
            return ctx
        costs = ctx.claims[C.COST_COL]
        for sol in ctx.solutions:
            metrics = self.compute(costs, sol.get("cost_med"))
            sol["savings_avoidable_overspend"] = metrics["avoidable_overspend"]
            sol["savings_per_claim"] = metrics["savings_per_claim"]
            sol["savings_pct_claims_above"] = metrics["pct_claims_above"]
            sol["savings_n_claims"] = metrics["n_claims"]
        ctx.savings = self.compute(costs, ctx.solutions[0].get("cost_med")) if ctx.solutions else {}
        return ctx
