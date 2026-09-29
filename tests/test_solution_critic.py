"""Tests for SolutionCriticAgent: verdict gates + the bounded revision-request logic."""
from src.agents.context import RepairContext
from src.agents.solution_critic_agent import SolutionCriticAgent


def _ctx(solutions, llm_decision=None):
    ctx = RepairContext(signature_id=1, vehicle_line="TEST", causal_part="0000", archetype=0)
    ctx.solutions = solutions
    ctx.llm_decision = llm_decision or {"n_repairs": 1}
    return ctx


def _good_solution(**over):
    s = {"repair_name": "Good Repair", "unified_diagnosis": "clear diagnosis",
         "suggested_correction": "clear correction", "cost_med": 100.0,
         "cost_q25": 80.0, "cost_q75": 120.0, "support": 50, "consensus": 0.8, "usable": True}
    s.update(over)
    return s


def test_pass_verdict_on_complete_solution():
    ctx = SolutionCriticAgent().review(_ctx([_good_solution()]))
    assert ctx.solutions[0]["critic_verdict"] == "PASS"
    assert ctx.solutions[0]["usable"] is True
    assert ctx.critique["n_pass"] == 1 and ctx.critique["n_fail"] == 0


def test_fail_on_missing_cost_consensus():
    ctx = SolutionCriticAgent().review(_ctx([_good_solution(cost_med=None)]))
    assert ctx.solutions[0]["critic_verdict"] == "FAIL"
    assert ctx.solutions[0]["usable"] is False


def test_fail_on_cost_band_out_of_order():
    ctx = SolutionCriticAgent().review(_ctx([_good_solution(cost_med=200.0)]))  # med > q75
    assert ctx.solutions[0]["critic_verdict"] == "FAIL"
    assert any("band" in r for r in ctx.critique["verdicts"][0]["reasons"])


def test_flag_on_low_support_and_low_consensus():
    ctx = SolutionCriticAgent(min_support=5, min_consensus=0.3).review(
        _ctx([_good_solution(support=2, consensus=0.1)]))
    assert ctx.solutions[0]["critic_verdict"] == "FLAG"
    assert ctx.solutions[0]["usable"] is True  # flagged, still emitted
    assert len(ctx.critique["verdicts"][0]["reasons"]) == 2


def test_flag_on_missing_text():
    ctx = SolutionCriticAgent().review(_ctx([_good_solution(unified_diagnosis="")]))
    assert ctx.solutions[0]["critic_verdict"] == "FLAG"


def test_revision_requested_only_for_text_defects():
    critic = SolutionCriticAgent()
    ctx = critic.review(_ctx([_good_solution(unified_diagnosis="")]))
    assert "unified_diagnosis" in critic.revision_request(ctx) or "Good Repair" in critic.revision_request(ctx)
    # numeric-only defects are NOT revisable (the LLM does not own the numbers)
    ctx2 = critic.review(_ctx([_good_solution(support=2)]))
    assert critic.revision_request(ctx2) == ""


def test_no_revision_after_llm_error_or_second_round():
    critic = SolutionCriticAgent()
    # upstream LLM already failed -> a revision would just fail again
    ctx = critic.review(_ctx([_good_solution(unified_diagnosis="")],
                             llm_decision={"n_repairs": 1, "error": "503"}))
    assert critic.revision_request(ctx) == ""
    # already revised once -> bounded loop stops
    ctx2 = critic.review(_ctx([_good_solution(unified_diagnosis="")]), revised=True)
    assert critic.revision_request(ctx2) == ""
