"""End-to-end test of the agent hand-off chain (offline sample + injected fake LLM).

Exercises GroupingAgent -> ConsensusAgent -> RepairAnalystAgent -> SolutionAssemblyAgent
-> SolutionCriticAgent (incl. the bounded revision loop) over the shared RepairContext,
with no BigQuery and a stubbed Gemini.
"""
from src.agents.orchestrator import Orchestrator


class SeqLLM:
    """FakeLLM that returns a DIFFERENT payload per call - drives the critic revision loop."""
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def generate_json(self, prompt, temperature=0.2):
        self.calls.append(prompt)
        return dict(self.payloads[min(len(self.calls), len(self.payloads)) - 1])


def test_run_one_single_repair(sample_env, fake_llm):
    orch = Orchestrator()
    orch.analyst.llm = fake_llm({
        "_ok": True, "n_repairs": 1, "split_rationale": "",
        "repairs": [{"name": "Long Block Engine Replacement", "cost_tier": "all",
                     "unified_diagnosis": "internal engine failure",
                     "suggested_correction": "replace long block",
                     "typical_parts": "long block assembly"}],
    })
    sols = orch.run_one(32954)
    assert len(sols) == 1
    s = sols[0]
    assert s["signature_id"] == 32954
    assert s["cost_med"] and s["cost_med"] > 0
    assert s["support"] > 0
    assert s["unified_diagnosis"]


def test_run_one_split_kept_when_separated(sample_env, fake_llm):
    orch = Orchestrator()
    orch.analyst.llm = fake_llm({
        "_ok": True, "n_repairs": 2, "split_rationale": "bimodal reseal vs replace",
        "repairs": [
            {"name": "Upper Oil Pan Reseal", "cost_tier": "low",
             "unified_diagnosis": "minor seepage", "suggested_correction": "reseal"},
            {"name": "Upper Oil Pan Replacement", "cost_tier": "high",
             "unified_diagnosis": "pan failure", "suggested_correction": "replace pan"},
        ],
    })
    sols = orch.run_one(40109)  # Super Duty oil pan - a real split signature
    assert 1 <= len(sols) <= 2
    for s in sols:
        assert s["signature_id"] == 40109


def test_critic_passes_complete_solution(sample_env, fake_llm):
    orch = Orchestrator()
    orch.analyst.llm = fake_llm({
        "_ok": True, "n_repairs": 1, "split_rationale": "",
        "repairs": [{"name": "Long Block Engine Replacement", "cost_tier": "all",
                     "unified_diagnosis": "internal engine failure",
                     "suggested_correction": "replace long block"}],
    })
    sols = orch.run_one(32954)
    assert sols[0]["critic_verdict"] == "PASS"
    assert len(orch.analyst.llm.calls) == 1  # no revision needed


def test_critic_revision_loop_fixes_missing_text(sample_env):
    """Round 1: LLM omits the diagnosis -> critic sends the critique back -> round 2 fixes it."""
    incomplete = {"_ok": True, "n_repairs": 1, "split_rationale": "",
                  "repairs": [{"name": "Long Block Engine Replacement", "cost_tier": "all"}]}
    complete = {"_ok": True, "n_repairs": 1, "split_rationale": "",
                "repairs": [{"name": "Long Block Engine Replacement", "cost_tier": "all",
                             "unified_diagnosis": "internal engine failure",
                             "suggested_correction": "replace long block"}]}
    orch = Orchestrator()
    orch.analyst.llm = SeqLLM([incomplete, complete])
    sols = orch.run_one(32954)
    assert len(orch.analyst.llm.calls) == 2                      # exactly ONE revision round
    assert "REVISION REQUEST" in orch.analyst.llm.calls[1]       # critique reached the analyst
    assert sols[0]["unified_diagnosis"]                          # revision fixed the text
    assert sols[0]["critic_verdict"] == "PASS"


def test_critic_bounded_when_revision_does_not_help(sample_env, fake_llm):
    """The LLM never returns text: the loop must stop after one round and FLAG, not retry forever."""
    incomplete = {"_ok": True, "n_repairs": 1, "split_rationale": "",
                  "repairs": [{"name": "Stubborn Repair", "cost_tier": "all"}]}
    orch = Orchestrator()
    orch.analyst.llm = fake_llm(incomplete)
    sols = orch.run_one(32954)
    assert len(orch.analyst.llm.calls) == 2          # initial + one bounded revision, then stop
    assert sols[0]["critic_verdict"] == "FLAG"       # emitted, but marked for analyst review
    assert sols[0]["usable"] is True


def test_critic_skips_revision_when_llm_already_failed(sample_env, fake_llm):
    orch = Orchestrator()
    orch.analyst.llm = fake_llm({"_ok": False, "error": "503 UNAVAILABLE"})
    sols = orch.run_one(32954)
    assert len(orch.analyst.llm.calls) == 1          # a revision would just fail again
    assert sols[0]["critic_verdict"] == "FLAG"       # fallback shell has no text -> flagged


def test_run_one_with_region_and_savings(sample_env, fake_llm):
    orch = Orchestrator()
    orch.analyst.llm = fake_llm({
        "_ok": True, "n_repairs": 1, "split_rationale": "",
        "repairs": [{"name": "Long Block Engine Replacement", "cost_tier": "all",
                     "unified_diagnosis": "x", "suggested_correction": "y"}],
    })
    sols = orch.run_one(32954, state="CA")
    s = sols[0]
    # cost-savings agent ran
    assert "savings_avoidable_overspend" in s
    # region delegator localized to CA (index > 1)
    assert s["region_state"] == "CA"
    assert s["cost_med_regional"] >= s["cost_med"]
