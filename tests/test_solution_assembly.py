"""Tests for SolutionAssemblyAgent - including the deterministic separation guard."""
from src.tools import StatsTool
from src.agents.solution_assembly_agent import SolutionAssemblyAgent, SPLIT_SEPARATION_MIN


def _assemble(ctx, llm_decision):
    ctx.llm_decision = llm_decision
    return SolutionAssemblyAgent(StatsTool()).assemble(ctx).solutions


def test_honors_genuine_split(base_ctx):
    """Bimodal ~2k vs ~9k (>1.5x) -> two repairs kept."""
    sols = _assemble(base_ctx, {
        "n_repairs": 2, "split_rationale": "bimodal",
        "repairs": [
            {"name": "Reseal", "cost_tier": "low", "unified_diagnosis": "minor leak",
             "suggested_correction": "reseal"},
            {"name": "Replace", "cost_tier": "high", "unified_diagnosis": "major failure",
             "suggested_correction": "replace"},
        ]})
    assert len(sols) == 2
    costs = sorted(s["cost_med"] for s in sols)
    assert costs[1] / costs[0] >= SPLIT_SEPARATION_MIN


def test_collapses_non_separated_split(tight_claims, base_ctx):
    """Tight single-mode distribution -> a proposed 2-way split collapses to ONE repair."""
    base_ctx.claims = tight_claims
    sols = _assemble(base_ctx, {
        "n_repairs": 2, "split_rationale": "guess",
        "repairs": [
            {"name": "Minor", "cost_tier": "low", "unified_diagnosis": "a", "suggested_correction": "a"},
            {"name": "Major", "cost_tier": "high", "unified_diagnosis": "b", "suggested_correction": "b"},
        ]})
    assert len(sols) == 1
    assert "not separated" in sols[0]["split_rationale"]


def test_single_repair_passthrough(tight_claims, base_ctx):
    base_ctx.claims = tight_claims
    sols = _assemble(base_ctx, {"n_repairs": 1, "split_rationale": "",
                                "repairs": [{"name": "Long Block", "cost_tier": "all",
                                             "unified_diagnosis": "x", "suggested_correction": "y"}]})
    assert len(sols) == 1
    assert sols[0]["cost_med"] is not None
    assert sols[0]["support"] == len(tight_claims)


def test_name_fallback_uses_theme(tight_claims, base_ctx):
    base_ctx.claims = tight_claims
    sols = _assemble(base_ctx, {"n_repairs": 1, "split_rationale": "",
                                "repairs": [{"name": "", "cost_tier": "all"}]})
    assert sols[0]["repair_name"]  # never empty
    assert "Oil Leak" in sols[0]["repair_name"] or "Repair" in sols[0]["repair_name"]


def test_caps_at_two_repairs(base_ctx):
    repairs = [{"name": f"r{i}", "cost_tier": "all"} for i in range(4)]
    sols = _assemble(base_ctx, {"n_repairs": 4, "split_rationale": "", "repairs": repairs})
    assert len(sols) <= 2
