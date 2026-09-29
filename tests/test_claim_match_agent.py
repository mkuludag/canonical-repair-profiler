"""Tests for ClaimMatchAgent inference routing against the bundled sample library."""
from src.tools import DataStore
from src.agents.infer import ClaimMatchAgent


def _agent(sample_env, llm):
    return ClaimMatchAgent(DataStore(), llm)


def test_single_candidate_matches_without_llm(sample_env, fake_llm):
    llm = fake_llm()  # should NOT be called when only one candidate exists
    res = _agent(sample_env, llm).match("F-150", "6148", "engine knock, oil consumption")
    assert res["status"] == "MATCHED"
    assert res["match_index"] == 0
    assert len(llm.calls) == 0
    assert res["solution"]["repair_name"]


def test_many_candidates_uses_llm_choice(sample_env, fake_llm):
    llm = fake_llm({"_ok": True, "match_index": 1, "confidence_0_1": 0.88, "reason": "best fit"})
    res = _agent(sample_env, llm).match("F-150", "6256", "cam phaser rattle, then full engine failure")
    assert res["status"] == "MATCHED"
    assert res["match_index"] == 1
    assert len(llm.calls) == 1


def test_no_candidates_returns_no_match(sample_env, fake_llm):
    res = _agent(sample_env, fake_llm()).match("DeLorean", "0000", "flux capacitor leak")
    assert res["status"] == "NO_MATCH"


def test_llm_negative_index_returns_no_match(sample_env, fake_llm):
    llm = fake_llm({"_ok": True, "match_index": -1, "reason": "none fit"})
    res = _agent(sample_env, llm).match("F-150", "6256", "totally unrelated symptom")
    assert res["status"] == "NO_MATCH"


def test_llm_out_of_range_index_returns_no_match(sample_env, fake_llm):
    llm = fake_llm({"_ok": True, "match_index": 99})
    res = _agent(sample_env, llm).match("F-150", "6256", "x")
    assert res["status"] == "NO_MATCH"


def test_llm_failure_surfaces_as_llm_error(sample_env, fake_llm):
    # when the LLM call fails (e.g. expired auth), report it - do not disguise it as NO_MATCH
    llm = fake_llm({"_ok": False, "error": "Reauthentication failed. Run `gcloud auth login`"})
    res = _agent(sample_env, llm).match("F-150", "6256", "engine knock")
    assert res["status"] == "LLM_ERROR"
    assert "gcloud auth login" in res["reason"]


def test_vehicle_line_with_regex_specials_does_not_crash(sample_env, fake_llm):
    # the full vehicle line "P552N F-150 [15-20]" must be matched literally, not as a regex
    # ("[15-20]" is an invalid character range and previously raised re.error)
    res = _agent(sample_env, fake_llm()).match("P552N F-150 [15-20]", "6148", "engine knock")
    assert res["status"] == "MATCHED"
    assert res["solution"]["repair_name"]
