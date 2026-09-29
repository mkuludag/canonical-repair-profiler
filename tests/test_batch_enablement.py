"""Tests for the batch-run enablement: GroupingAgent.preload, Orchestrator.run_one_ctx,
the enriched run_batch cache record, and env-overridable artifact paths.

All offline (sample data + fake LLM), like the rest of the suite.
"""
import importlib
import json

import pandas as pd

from src.tools import DataStore
from src.agents.grouping_agent import GroupingAgent
from src.agents.orchestrator import Orchestrator


# --------------------------------------------------------------------- preload

def test_preload_equivalent_to_cold_path(sample_env):
    """A preloaded context must be indistinguishable from one built by the per-signature path."""
    cold = GroupingAgent(DataStore()).load_signature(32954)

    warm_agent = GroupingAgent(DataStore())
    n = warm_agent.preload([32954, 40109])
    assert n == 2
    warm = warm_agent.load_signature(32954)

    assert warm.signature_id == cold.signature_id
    assert warm.vehicle_line == cold.vehicle_line
    assert warm.causal_part == cold.causal_part
    assert warm.archetype == cold.archetype
    assert len(warm.claims) == len(cold.claims)
    pd.testing.assert_series_equal(
        warm.claims["gsar_tot_cost_gross"].reset_index(drop=True),
        cold.claims["gsar_tot_cost_gross"].reset_index(drop=True))


def test_preload_applies_cost_cap(sample_env):
    agent = GroupingAgent(DataStore())
    agent.preload([32954])
    valid = agent.load_signature(32954).claims["gsar_tot_cost_gross"].dropna()
    assert (valid > 0).all() and (valid < 100000).all()


def test_preload_missing_signature_still_raises(sample_env):
    agent = GroupingAgent(DataStore())
    agent.preload([32954])
    try:
        agent.load_signature(-12345)
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_live_path_untouched_without_preload(sample_env):
    agent = GroupingAgent(DataStore())
    assert agent._claims_index is None
    ctx = agent.load_signature(32954)
    assert len(ctx.claims) > 0


# --------------------------------------------------------------------- run_one_ctx

FULL_PAYLOAD = {
    "_ok": True, "n_repairs": 1, "split_rationale": "",
    "repairs": [{"name": "Long Block Engine Replacement", "cost_tier": "all",
                 "unified_diagnosis": "internal engine failure",
                 "suggested_correction": "replace long block"}],
}


def test_run_one_ctx_returns_full_context(sample_env, fake_llm):
    orch = Orchestrator()
    orch.analyst.llm = fake_llm(FULL_PAYLOAD)
    ctx = orch.run_one_ctx(32954)
    assert ctx.solutions and ctx.solutions[0]["critic_verdict"] == "PASS"
    assert ctx.critique["n_pass"] == 1 and ctx.critique["revised"] is False
    assert ctx.llm_decision.get("n_repairs") == 1
    # run_one still returns just the solutions (back-compat)
    assert orch.run_one(32954) == ctx.solutions


def test_run_batch_caches_critique(sample_env, fake_llm, tmp_path):
    orch = Orchestrator()
    orch.analyst.llm = fake_llm(FULL_PAYLOAD)
    orch.datastore = DataStore(root=str(tmp_path))
    sols = orch.run_batch([32954], cache_rel="cache.jsonl")
    assert len(sols) == 1
    rec = json.loads((tmp_path / "cache.jsonl").read_text().strip())
    assert rec["_ok"] is True and rec["signature_id"] == 32954
    assert rec["critique"] == {"n_pass": 1, "n_flag": 0, "n_fail": 0, "revised": False}
    assert rec["n_repairs"] == 1 and rec["llm_error"] is None
    # the fake payload carries no _attempts (real ones do, via VertexGeminiTool.generate_json);
    # the cache must record it verbatim rather than KeyError
    assert rec["llm_attempts"] == FULL_PAYLOAD.get("_attempts")


def test_run_batch_resumes_from_cache(sample_env, fake_llm, tmp_path):
    orch = Orchestrator()
    orch.analyst.llm = fake_llm(FULL_PAYLOAD)
    orch.datastore = DataStore(root=str(tmp_path))
    orch.run_batch([32954], cache_rel="cache.jsonl")
    calls_after_first = len(orch.analyst.llm.calls)
    orch.run_batch([32954], cache_rel="cache.jsonl")  # already cached -> no new LLM call
    assert len(orch.analyst.llm.calls) == calls_after_first


# --------------------------------------------------------------------- env-overridable paths

def test_artifact_paths_env_overridable(monkeypatch):
    import src.config as config
    monkeypatch.setenv("CRP_GOLDEN_SOLUTIONS", "grouping/out_v2/golden_solutions.csv")
    monkeypatch.setenv("CRP_GROUP_SAVINGS", "grouping/out_v2/group_savings.csv")
    importlib.reload(config)
    try:
        assert config.GOLDEN_SOLUTIONS == "grouping/out_v2/golden_solutions.csv"
        assert config.GROUP_SAVINGS == "grouping/out_v2/group_savings.csv"
    finally:
        monkeypatch.delenv("CRP_GOLDEN_SOLUTIONS")
        monkeypatch.delenv("CRP_GROUP_SAVINGS")
        importlib.reload(config)
        assert config.GOLDEN_SOLUTIONS == "grouping/out/golden_solutions.csv"
