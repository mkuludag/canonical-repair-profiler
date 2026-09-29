"""
RegionDelegatorAgent - localizes a national Canonical Repair to a dealer's state.

The base pipeline produces ONE national consensus cost per Canonical Repair. But the same repair
costs materially more in some states than others (our region study measured ~1.7x spread). At
inference time a dealer/agent works in a specific state, so this agent "delegates" the national
solution to the right region by applying the deterministic RegionIndexTool multiplier.

It is a thin reasoning layer over a TOOL (no LLM), so the localized number is fully reproducible.

HAND-OFF: reads ctx.solutions (from SolutionAssemblyAgent), writes region-localized fields onto each
solution and records ctx.region = {state, cost_index, labor_index}.
"""
from .context import RepairContext


class RegionDelegatorAgent:
    def __init__(self, region_tool):
        self.region = region_tool

    def adjust_solution(self, solution: dict, state: str) -> dict:
        """Return a copy of a solution with region-localized cost fields added."""
        idx = self.region.index_for(state)
        lab = self.region.labor_index_for(state)
        out = dict(solution)
        out["region_state"] = (state or "").upper()
        out["region_cost_index"] = round(idx, 3)
        out["region_labor_index"] = round(lab, 3)
        for src, dst in (("cost_med", "cost_med_regional"),
                         ("cost_q25", "cost_q25_regional"),
                         ("cost_q75", "cost_q75_regional")):
            base = solution.get(src)
            out[dst] = round(float(base) * idx, 2) if isinstance(base, (int, float)) and base == base else None
        delta = (idx - 1.0) * 100.0
        direction = "above" if delta >= 0 else "below"
        out["region_note"] = (
            f"{out['region_state']} runs {abs(delta):.0f}% {direction} the national average "
            f"for this repair (index {idx:.2f}x)." if state else "National average (no state selected).")
        return out

    def apply(self, ctx: RepairContext, state: str) -> RepairContext:
        """Localize every assembled solution to the given state (context hand-off)."""
        ctx.region = {"state": (state or "").upper(),
                      "cost_index": self.region.index_for(state),
                      "labor_index": self.region.labor_index_for(state)}
        ctx.solutions = [self.adjust_solution(s, state) for s in ctx.solutions]
        return ctx
