"""
ConsensusAgent — computes the deterministic, auditable NUMERIC half of a Canonical Repair.

Uses StatsTool to derive cost/labor median+IQR bands and the agreement (consensus) fractions for the
group. The LLM never invents numbers — it only writes text — so the dollar/labor ground truths are
reproducible from the data alone.

HAND-OFF: reads ctx.claims, writes ctx.consensus, passes ctx to the RepairAnalystAgent.
"""
from .context import RepairContext


class ConsensusAgent:
    def __init__(self, stats_tool):
        self.stats = stats_tool

    def enrich(self, ctx: RepairContext) -> RepairContext:
        claims = ctx.claims
        cost = claims["gsar_tot_cost_gross"]
        labor = claims["gsar_labor_hrs"]
        cost_band = self.stats.band(cost)
        labor_band = self.stats.band(labor)
        cost_cons = self.stats.consensus_fraction(cost)
        labor_cons = self.stats.consensus_fraction(labor)
        ctx.consensus = {
            "n_claims": len(claims),
            "cost_med": cost_band["median"], "cost_q25": cost_band["q25"], "cost_q75": cost_band["q75"],
            "labor_med": labor_band["median"], "labor_q25": labor_band["q25"], "labor_q75": labor_band["q75"],
            "cost_consensus": None if cost_cons != cost_cons else round(cost_cons, 3),
            "labor_consensus": None if labor_cons != labor_cons else round(labor_cons, 3),
        }
        return ctx
