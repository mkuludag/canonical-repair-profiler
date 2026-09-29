"""Tests for ConsensusAgent (deterministic numeric enrichment)."""
from src.tools import StatsTool
from src.agents.consensus_agent import ConsensusAgent


def test_enrich_populates_bands(base_ctx):
    ctx = ConsensusAgent(StatsTool()).enrich(base_ctx)
    c = ctx.consensus
    assert c["n_claims"] == len(base_ctx.claims)
    assert c["cost_q25"] <= c["cost_med"] <= c["cost_q75"]
    assert c["labor_med"] is not None


def test_enrich_consensus_in_range(tight_claims, base_ctx):
    base_ctx.claims = tight_claims
    ctx = ConsensusAgent(StatsTool()).enrich(base_ctx)
    assert 0.0 <= ctx.consensus["cost_consensus"] <= 1.0
    # a tight group should agree strongly on cost
    assert ctx.consensus["cost_consensus"] >= 0.9
