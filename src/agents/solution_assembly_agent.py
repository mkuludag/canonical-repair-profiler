"""
SolutionAssemblyAgent — fuses the LLM text with the deterministic numbers into final solutions.

For each repair the analyst identified, it partitions the group's claims into the matching cost tier
(StatsTool) and recomputes per-tier cost/labor bands + that tier's own consensus, then assembles the
Canonical Repair record: unified diagnosis + suggested correction (LLM) and labor/cost + confidence
(data). A deterministic SEPARATION GUARD collapses a "split" whose tiers are not genuinely bimodal,
so the agent never reports two repairs that are really one repair with normal price variation.

HAND-OFF: reads ctx.claims + ctx.consensus + ctx.llm_decision, writes ctx.solutions (terminal agent).
"""
import numpy as np

from .context import RepairContext
from ..config import COST_COL as COST, SPLIT_SEPARATION_MIN


class SolutionAssemblyAgent:
    def __init__(self, stats_tool):
        self.stats = stats_tool

    def _row(self, ctx: RepairContext, rep: dict, tier: str, n_repairs: int) -> dict:
        band_df = self.stats.cost_tier_partition(ctx.claims, COST, tier, n_repairs)
        has = band_df is not None and len(band_df)
        cost_band = self.stats.band(band_df[COST]) if has else {"n": 0, "median": None, "q25": None, "q75": None}
        labor_band = self.stats.band(band_df["gsar_labor_hrs"]) if has else {"median": None}
        cost_c = self.stats.consensus_fraction(band_df[COST]) if has else float("nan")
        labor_c = self.stats.consensus_fraction(band_df["gsar_labor_hrs"]) if has else float("nan")
        cons_vals = [x for x in (cost_c, labor_c) if x == x]
        consensus = round(float(np.mean(cons_vals)), 3) if cons_vals else None
        support = cost_band["n"]
        theme = (ctx.archetype_theme.split(",")[0].strip().title() if ctx.archetype_theme else "")
        name = rep.get("name") or (f"{theme} Repair" if theme else "Canonical Repair")
        return {
            "signature_id": ctx.signature_id, "vehicle_line": ctx.vehicle_line, "causal_part": ctx.causal_part,
            "n_repairs_detected": n_repairs, "repair_name": name, "cost_tier": tier,
            "tier_n_claims": support, "split_rationale": ctx.llm_decision.get("split_rationale", ""),
            "unified_diagnosis": rep.get("unified_diagnosis", ""),
            "suggested_correction": rep.get("suggested_correction", ""),
            "typical_parts": rep.get("typical_parts", ""),
            "labor_med_hrs": labor_band["median"], "cost_med": cost_band["median"],
            "cost_q25": cost_band["q25"], "cost_q75": cost_band["q75"],
            "consensus": consensus, "support": support,
            "confidence": self.stats.confidence(support, consensus), "usable": bool(cost_band["median"]),
        }

    def assemble(self, ctx: RepairContext) -> RepairContext:
        repairs = (ctx.llm_decision.get("repairs") or [{}])[:2]   # cap at 2 (minor vs full replacement)
        n_repairs = len(repairs)
        # keep each repair dict paired with the row it produced so the guard can map tier -> source
        pairs = [(rep, self._row(ctx, rep, rep.get("cost_tier", "all"), n_repairs)) for rep in repairs]
        rows = [row for _, row in pairs]

        # separation guard: collapse a split that isn't genuinely bimodal
        if n_repairs == 2 and all(r["cost_med"] for r in rows):
            (_lo_rep, lo), (hi_rep, hi) = sorted(pairs, key=lambda pr: pr[1]["cost_med"])
            if hi["cost_med"] / lo["cost_med"] < SPLIT_SEPARATION_MIN:
                merged = self._row(ctx, hi_rep, "all", 1)   # full-group stats, higher-tier diagnosis
                merged["split_rationale"] = (
                    f"LLM proposed 2 tiers but they were not separated "
                    f"(${hi['cost_med']:.0f}/${lo['cost_med']:.0f} < {SPLIT_SEPARATION_MIN}x) -> one repair.")
                rows = [merged]
        ctx.solutions = rows
        return ctx
