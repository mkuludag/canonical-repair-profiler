"""Tests for RepairAnalystAgent: prompt construction + graceful LLM-failure fallback."""
from src.agents.repair_analyst_agent import RepairAnalystAgent
from src.agents.consensus_agent import ConsensusAgent
from src.tools import StatsTool


def test_build_prompt_includes_context_and_samples(base_ctx, fake_llm):
    base_ctx = ConsensusAgent(StatsTool()).enrich(base_ctx)
    agent = RepairAnalystAgent(fake_llm())
    prompt = agent._build_prompt(base_ctx)
    assert "vehicle_line=TEST F-150" in prompt
    assert "causal_part=6019" in prompt
    assert "$" in prompt  # cost-tagged comment samples present


def test_analyze_sets_decision(base_ctx, fake_llm):
    base_ctx = ConsensusAgent(StatsTool()).enrich(base_ctx)
    llm = fake_llm({"_ok": True, "n_repairs": 1, "split_rationale": "",
                    "repairs": [{"name": "X", "cost_tier": "all"}]})
    ctx = RepairAnalystAgent(llm).analyze(base_ctx)
    assert ctx.llm_decision["n_repairs"] == 1
    assert "_ok" not in ctx.llm_decision  # stripped


def test_analyze_falls_back_on_llm_error(base_ctx, fake_llm):
    base_ctx = ConsensusAgent(StatsTool()).enrich(base_ctx)
    llm = fake_llm({"_ok": False, "error": "503 UNAVAILABLE"})
    ctx = RepairAnalystAgent(llm).analyze(base_ctx)
    assert ctx.llm_decision["n_repairs"] == 1  # graceful single-repair shell
    assert ctx.llm_decision["error"] == "503 UNAVAILABLE"


def test_analyze_falls_back_on_malformed_payload(base_ctx, fake_llm):
    """Valid JSON that violates the prompt's schema must not poison downstream agents."""
    base_ctx = ConsensusAgent(StatsTool()).enrich(base_ctx)
    llm = fake_llm({"_ok": True, "n_repairs": 2, "repairs": "not-a-list"})
    ctx = RepairAnalystAgent(llm).analyze(base_ctx)
    assert ctx.llm_decision["n_repairs"] == 1  # schema-validated -> single-repair shell
    assert "schema" in ctx.llm_decision["error"]
