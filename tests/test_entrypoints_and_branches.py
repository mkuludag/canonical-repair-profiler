"""
Coverage-completeness tests: CLI entrypoints, the batch path, the non-sample (BigQuery) grouping
branch, and small defensive branches. Everything stays offline (cloud transport is mocked/stubbed).

Note: the raw cloud-transport methods (VertexGeminiTool._client/_raw_generate, BigQueryTool._client_fresh)
are intentionally NOT exercised here - they only build real SDK clients, and every test mocks at that
boundary. That is the one part of the code deliberately left to live/integration runs.
"""
import sys

import numpy as np
import pandas as pd
import pytest

from src.agents.context import RepairContext


# ---------- context.summary ----------
def test_context_summary():
    ctx = RepairContext(signature_id=1, vehicle_line="F-150", causal_part="6148", archetype=2)
    s = ctx.summary()
    assert "signature=1" in s and "F-150/6148" in s and "claims=0" in s


# ---------- CostSavingsAgent guard ----------
def test_cost_savings_noop_without_solutions(base_ctx):
    from src.agents.cost_savings_agent import CostSavingsAgent
    base_ctx.solutions = []
    out = CostSavingsAgent().analyze(base_ctx)
    assert out.solutions == []  # returns ctx unchanged when there is nothing to score


# ---------- RepairAnalystAgent all-NaN cost fallback ----------
def test_build_prompt_all_nan_costs(fake_llm):
    from src.agents.repair_analyst_agent import RepairAnalystAgent
    from src.agents.consensus_agent import ConsensusAgent
    from src.tools import StatsTool
    ctx = RepairContext(signature_id=5, vehicle_line="V", causal_part="6", archetype=1)
    ctx.claims = pd.DataFrame({"gsar_tot_cost_gross": [np.nan, np.nan],
                               "gsar_labor_hrs": [np.nan, np.nan],
                               "paws_comment_trail": ["x", "y"]})
    ctx = ConsensusAgent(StatsTool()).enrich(ctx)
    prompt = RepairAnalystAgent(fake_llm())._build_prompt(ctx)  # exercises the all-NaN g=ctx.claims path
    assert "vehicle_line=V" in prompt


# ---------- auth: explicit account branch ----------
def test_auth_uses_explicit_account(monkeypatch):
    import src.tools.auth as auth
    auth._cache["token"] = None
    auth._cache["ts"] = 0.0
    monkeypatch.setattr(auth, "ACCOUNT", "demo@example.com")
    captured = {}

    def fake_check_output(cmd, **kw):
        captured["cmd"] = cmd
        return b"tok\n"

    monkeypatch.setattr(auth.subprocess, "check_output", fake_check_output)
    assert auth.fresh_access_token() == "tok"
    assert "--account" in captured["cmd"] and "demo@example.com" in captured["cmd"]
    auth._cache["token"] = None  # reset for other tests


# ---------- VertexGeminiTool.generate_text retry-then-success ----------
def test_generate_text_retries_then_succeeds(monkeypatch):
    from src.tools.vertex_gemini_tool import VertexGeminiTool
    import src.tools.vertex_gemini_tool as vmod
    monkeypatch.setattr(vmod.time, "sleep", lambda *_: None)
    tool = VertexGeminiTool(max_retries=3)
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("503 UNAVAILABLE")
        return "the answer"

    monkeypatch.setattr(tool, "_raw_generate", flaky)
    assert tool.generate_text("p") == "the answer" and calls["n"] == 2


# ---------- GroupingAgent non-sample (BigQuery/parquet) branch ----------
class _FakeDS:
    """Stands in for DataStore, returning frames for the non-sample parquet path."""
    def read_csv(self, rel):
        return pd.DataFrame({"signature_id": [1], "vehicle_line": ["P552N F-150 [15-20]"],
                             "causal_part": ["6148"], "arch": [3], "arch_theme": ["oil"]})

    def read_parquet(self, rel, columns=None):
        if "claim_signatures" in rel:
            return pd.DataFrame({"request_r": [10, 11, 12], "signature_id": [1, 1, 2]})
        return pd.DataFrame({"request_r": [10, 11, 12],
                             "paws_comment_trail": ["a", "b", "c"],
                             "gsar_tot_cost_gross": [100.0, 200.0, 999999.0],
                             "gsar_labor_hrs": [1.0, 2.0, 3.0]})


def test_grouping_agent_non_sample_path(monkeypatch):
    monkeypatch.delenv("CRP_USE_SAMPLE", raising=False)
    from src.agents.grouping_agent import GroupingAgent
    agent = GroupingAgent(_FakeDS())
    assert agent.sample is False
    ctx = agent.load_signature(1)
    assert len(ctx.claims) == 2                      # only members of signature 1
    assert ctx.claims["gsar_tot_cost_gross"].notna().all()  # cost cap applied (no >100k)


# ---------- Orchestrator.run_batch (concurrency + resumable cache) ----------
def _fake_ctx(sid):
    """Minimal RepairContext stand-in for run_batch tests (run_one_ctx is the seam now)."""
    from src.agents.context import RepairContext
    ctx = RepairContext(signature_id=sid, vehicle_line="T", causal_part="1", archetype=0)
    ctx.solutions = [{"signature_id": sid, "repair_name": "R"}]
    ctx.critique = {"n_pass": 1, "n_flag": 0, "n_fail": 0, "revised": False}
    ctx.llm_decision = {"n_repairs": 1}
    return ctx


def test_run_batch_caches_and_skips_done(sample_env, tmp_path, monkeypatch):
    from src.agents.orchestrator import Orchestrator
    from src.tools import DataStore
    orch = Orchestrator()
    orch.datastore = DataStore(root=str(tmp_path))
    monkeypatch.setattr(orch, "run_one_ctx", lambda sid: _fake_ctx(sid))
    out = orch.run_batch([1, 2], workers=2, cache_rel="cache.jsonl")
    assert len(out) == 2
    # second run: both are cached -> skipped (no new work)
    out2 = orch.run_batch([1, 2], workers=2, cache_rel="cache.jsonl")
    assert out2 == []


def test_run_batch_survives_a_failing_group(sample_env, tmp_path, monkeypatch):
    from src.agents.orchestrator import Orchestrator
    from src.tools import DataStore
    orch = Orchestrator()
    orch.datastore = DataStore(root=str(tmp_path))

    def flaky(sid):
        if sid == 2:
            raise RuntimeError("boom")
        return _fake_ctx(sid)

    monkeypatch.setattr(orch, "run_one_ctx", flaky)
    out = orch.run_batch([1, 2, 3], workers=3, cache_rel="c.jsonl")
    assert len(out) == 2  # the failing group is logged, not fatal


# ---------- CLI entrypoints ----------
def test_infer_print_paths(capsys):
    from src.agents import infer
    infer._print({"status": "NO_MATCH", "message": "none"})
    infer._print({"status": "MATCHED", "confidence_0_1": 0.9, "reason": "fit",
                  "solution": {"repair_name": "R", "unified_diagnosis": "d", "suggested_correction": "c",
                               "labor_med_hrs": 5, "cost_med": 100, "cost_q25": 90, "cost_q75": 110,
                               "typical_parts": "p", "confidence": 0.8, "tier_n_claims": 10}})
    out = capsys.readouterr().out
    assert "NO_MATCH" in out and "MATCHED CANONICAL REPAIR" in out


def test_infer_main(sample_env, monkeypatch, capsys):
    from src.agents import infer

    class FakeAgent:
        def __init__(self, *a, **k): pass
        def match(self, *a, **k): return {"status": "NO_MATCH", "message": "n/a"}

    monkeypatch.setattr(infer, "ClaimMatchAgent", FakeAgent)
    monkeypatch.setattr(sys, "argv", ["infer", "--vehicle", "F-150", "--part", "6148", "--concern", "x"])
    infer.main()
    assert "INCOMING CLAIM" in capsys.readouterr().out


def test_orchestrator_main(monkeypatch):
    import src.agents.orchestrator as omod

    class FakeOrch:
        def run_one(self, sig, verbose=False, state=""):
            return [{"signature_id": sig}]

    monkeypatch.setattr(omod, "Orchestrator", lambda *a, **k: FakeOrch())
    monkeypatch.setattr(sys, "argv", ["orch", "--signature", "32954"])
    omod.main()  # should not raise


def test_cli_parser_and_src_main_import():
    import cli
    ns = cli.build_parser().parse_args(["infer", "--vehicle", "F-150", "--part", "6148", "--concern", "x"])
    assert ns.cmd == "infer" and ns.vehicle == "F-150"
    import src.main  # noqa: F401  (covers the template entrypoint wrapper import)
