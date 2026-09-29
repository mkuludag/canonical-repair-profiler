"""Tests for the eval-side LLMGateway gateway client: JSON extraction, retries, auth, path suffixing.

Mirrors test_vertex_gemini_tool.py: the transport seam `_raw_generate` is monkeypatched, so no
network, no gateway token, and no opencode install is needed. The point of these tests is that the
second-model runs behave EXACTLY like the shipped Gemini path everywhere the analyses read - the
`_ok`/`_attempts` contract and the jittered parse-retry - because every downstream number assumes it.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))

from llm_gateway_tool import (  # noqa: E402
    LLMGatewayTool, TokenError, extract_json, make_client, slug, suffixed, unwrap_tool_call,
)
import llm_gateway_tool as fmod  # noqa: E402


@pytest.fixture(autouse=True)
def _no_sleep_no_token(monkeypatch):
    monkeypatch.setattr(fmod.time, "sleep", lambda *_: None)
    monkeypatch.setattr(LLMGatewayTool, "_token", lambda self, force=False: "test-token")


# --------------------------------------------------------------------- extract_json

def test_extract_json_strict():
    assert extract_json('{"n_repairs": 2}') == {"n_repairs": 2}


def test_extract_json_strips_code_fence():
    """What a model without response_format support actually returns."""
    assert extract_json('```json\n{"n_repairs": 1}\n```') == {"n_repairs": 1}


def test_extract_json_ignores_surrounding_prose():
    text = 'Here is the result:\n{"n_repairs": 2, "note": "ok"}\nHope that helps.'
    assert extract_json(text)["n_repairs"] == 2


def test_extract_json_rejects_non_object():
    """A bare list must take the jittered-retry path, not be accepted as a decision."""
    for bad in ("[1, 2, 3]", "", "no json here", '"a string"'):
        with pytest.raises(json.JSONDecodeError):
            extract_json(bad)


# --------------------------------------------------------------------- tool-call envelope

def test_unwrap_tool_call_parameters():
    """claude-sonnet-5 through this gateway returns this shape on a minority of responses."""
    env = {"name": "json_tool_call", "parameters": {"n_repairs": 1, "repairs": [{}]}}
    payload, wrapped = unwrap_tool_call(env)
    assert wrapped is True and payload["n_repairs"] == 1


def test_unwrap_tool_call_arguments_as_json_string():
    env = {"name": "f", "arguments": '{"n_repairs": 2, "repairs": [{}, {}]}'}
    payload, wrapped = unwrap_tool_call(env)
    assert wrapped is True and payload["n_repairs"] == 2


def test_unwrap_handles_every_observed_payload_key():
    """The gateway used parameters/arguments/params for the same prompt across calls."""
    for key in ("parameters", "arguments", "params", "args", "input"):
        payload, wrapped = unwrap_tool_call({"name": "json_tool_call", key: {"n_repairs": 1}})
        assert wrapped is True and payload == {"n_repairs": 1}, key


def test_unwrap_handles_an_unseen_wrapper_key():
    """`json`, `$PARAMETERS` and a literal `{}` were all observed as wrapper keys in one sweep."""
    for key in ("json", "$PARAMETERS", "{}", "totally_new_key"):
        payload, wrapped = unwrap_tool_call({key: {"n_repairs": 1, "repairs": [{}]}},
                                            expect_keys=DECISION)
        assert wrapped is True and payload["n_repairs"] == 1, key


def test_generic_unwrap_requires_the_nested_object_to_look_like_the_contract():
    obj = {"unknown": {"something": "else"}}
    assert unwrap_tool_call(obj, expect_keys=DECISION) == (obj, False)


def test_generic_unwrap_never_fires_when_the_top_level_is_the_payload():
    """A real decision that happens to nest a dict must not be replaced by that dict."""
    real = {"n_repairs": 1, "repairs": [{}], "meta": {"split_rationale": "x"}}
    assert unwrap_tool_call(real, expect_keys=DECISION) == (real, False)


def test_unwrap_leaves_a_real_payload_alone():
    real = {"n_repairs": 1, "repairs": [{}], "split_rationale": "x"}
    assert unwrap_tool_call(real) == (real, False)


def test_unwrap_refuses_when_the_envelope_carries_its_own_keys():
    """Narrow by design: a malformed decision must never be rescued into looking valid."""
    mixed = {"name": "x", "parameters": {"n_repairs": 1}, "n_repairs": 9}
    assert unwrap_tool_call(mixed) == (mixed, False)


def test_generate_json_flags_unwrapped_responses(monkeypatch):
    tool = LLMGatewayTool("claude-sonnet-5")
    monkeypatch.setattr(tool, "_raw_generate", lambda *a, **k:
                        '{"name": "json_tool_call", "parameters": {"n_repairs": 1}}')
    out = tool.generate_json("p", temperature=0.0)
    assert out["_ok"] is True and out["n_repairs"] == 1 and out["_envelope"] is True


def test_generate_json_does_not_flag_clean_responses(monkeypatch):
    tool = LLMGatewayTool("claude-sonnet-5")
    monkeypatch.setattr(tool, "_raw_generate", lambda *a, **k: '{"n_repairs": 1}')
    assert "_envelope" not in tool.generate_json("p", temperature=0.0)


# --------------------------------------------------------------------- generate_json

def test_generate_json_success(monkeypatch):
    tool = LLMGatewayTool("claude-sonnet-5")
    monkeypatch.setattr(tool, "_raw_generate", lambda *a, **k: '{"n_repairs": 2, "ok": true}')
    out = tool.generate_json("p", temperature=0.0)
    assert out["_ok"] is True and out["n_repairs"] == 2
    assert out["_attempts"] == 1  # first-try success -> sampled at the requested temperature


def test_generate_json_retry_floors_temperature(monkeypatch):
    """A deterministic formatting glitch must not simply repeat at the same temperature."""
    seen = []

    def fake(prompt, temperature, json_mode, force=False):
        seen.append(temperature)
        return "not json" if len(seen) == 1 else '{"n_repairs": 1}'

    tool = LLMGatewayTool("deepseekv4-flash", max_retries=3)
    monkeypatch.setattr(tool, "_raw_generate", fake)
    out = tool.generate_json("p", temperature=0.0)
    assert out["_ok"] is True and out["_attempts"] == 2
    assert seen == [0.0, LLMGatewayTool.JSON_RETRY_MIN_TEMPERATURE]


def test_generate_json_permanent_failure_never_raises(monkeypatch):
    tool = LLMGatewayTool("claude-sonnet-5", max_retries=2)

    def boom(*a, **k):
        raise RuntimeError("400 malformed request")

    monkeypatch.setattr(tool, "_raw_generate", boom)
    out = tool.generate_json("p")
    assert out["_ok"] is False and "400" in out["error"]


def test_generate_json_retries_retryable_errors(monkeypatch):
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("429 rate limited")
        return '{"n_repairs": 1}'

    tool = LLMGatewayTool("claude-sonnet-5", max_retries=3)
    monkeypatch.setattr(tool, "_raw_generate", flaky)
    assert tool.generate_json("p")["_ok"] is True
    assert len(calls) == 3


# --------------------------------------------------------------------- transport debris

DECISION = ("n_repairs", "repairs", "split_rationale", "est_group_cost_med")


def test_debris_is_rerequested_not_returned(monkeypatch):
    """An empty object or a stray tool-call fragment never carried the model's answer."""
    seen = []

    def fake(prompt, temperature, json_mode, force=False):
        seen.append(temperature)
        return ['{}', '{"name": "n_repairs", "value": 1}',
                '{"n_repairs": 1, "repairs": [{}]}'][len(seen) - 1]

    tool = LLMGatewayTool("claude-sonnet-5", max_retries=3, expect_keys=DECISION)
    monkeypatch.setattr(tool, "_raw_generate", fake)
    out = tool.generate_json("p", temperature=0.0)
    assert out["_ok"] is True and out["n_repairs"] == 1
    assert out["_debris_retries"] == 2 and out["_attempts"] == 3


def test_a_real_but_wrong_decision_is_never_rerolled(monkeypatch):
    """Re-rolling the model's own schema failures would inflate the quantity under measurement."""
    calls = []

    def fake(*a, **k):
        calls.append(1)
        return '{"n_repairs": 1, "repairs": "not a list"}'

    tool = LLMGatewayTool("claude-sonnet-5", max_retries=3, expect_keys=DECISION)
    monkeypatch.setattr(tool, "_raw_generate", fake)
    out = tool.generate_json("p", temperature=0.0)
    assert out["_ok"] is True and out["repairs"] == "not a list"
    assert len(calls) == 1 and "_debris_retries" not in out


def test_debris_has_its_own_budget_beyond_max_retries(monkeypatch):
    """A dropped payload is a transport event; spending the model-error budget on it would abandon
    signatures the model never failed."""
    calls = []

    def fake(*a, **k):
        calls.append(1)
        return "{}" if len(calls) <= 5 else '{"n_repairs": 1, "repairs": [{}]}'

    tool = LLMGatewayTool("claude-sonnet-5", max_retries=3, debris_extra=5, expect_keys=DECISION)
    monkeypatch.setattr(tool, "_raw_generate", fake)
    out = tool.generate_json("p", temperature=0.0)
    assert out["_ok"] is True and out["_debris_retries"] == 5 and len(calls) == 6


def test_debris_budget_is_finite(monkeypatch):
    calls = []
    tool = LLMGatewayTool("claude-sonnet-5", max_retries=3, debris_extra=2, expect_keys=DECISION)
    monkeypatch.setattr(tool, "_raw_generate", lambda *a, **k: calls.append(1) or "{}")
    out = tool.generate_json("p")
    assert out["_ok"] is False and len(calls) == 3


def test_hard_errors_still_stop_at_max_retries(monkeypatch):
    """The debris budget must not turn a genuinely failing call into nine transport attempts."""
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise RuntimeError("503 upstream unavailable")

    tool = LLMGatewayTool("claude-sonnet-5", max_retries=3, debris_extra=5, expect_keys=DECISION)
    monkeypatch.setattr(tool, "_raw_generate", boom)
    assert tool.generate_json("p")["_ok"] is False
    assert len(calls) == 3


# --------------------------------------------------------------------- throttling

def test_throttling_has_its_own_budget_and_adapts_the_rate(monkeypatch):
    """gpt-5.4 answers 429 'limited to 100 per 60 seconds'; a 429 says nothing about the answer."""
    calls = []

    def fake(*a, **k):
        calls.append(1)
        if len(calls) <= 2:
            raise RuntimeError('429 {"error":{"code":"rate_limit_exceeded"}}')
        return '{"n_repairs": 1, "repairs": [{}]}'

    tool = LLMGatewayTool("gpt-5.4-2026-03-05", max_retries=3, expect_keys=DECISION)
    before = tool.limiter.per_min
    monkeypatch.setattr(tool, "_raw_generate", fake)
    out = tool.generate_json("p", temperature=0.0)
    assert out["_ok"] is True and out["_throttled"] == 2
    assert out["_attempts"] == 1          # no CONTENT retry happened
    assert tool.limiter.per_min < before  # the real ceiling is lower than assumed


def test_a_throttled_retry_does_not_move_the_temperature(monkeypatch):
    """Flooring the temperature after a 429 would silently change the sampling protocol."""
    seen = []

    def fake(prompt, temperature, json_mode, force=False):
        seen.append(temperature)
        if len(seen) == 1:
            raise RuntimeError("429 rate_limit_exceeded")
        return '{"n_repairs": 1, "repairs": [{}]}'

    tool = LLMGatewayTool("gpt-5.4-2026-03-05", expect_keys=DECISION)
    monkeypatch.setattr(tool, "_raw_generate", fake)
    tool.generate_json("p", temperature=0.0)
    assert seen == [0.0, 0.0]


def test_rate_limiter_admits_up_to_the_ceiling_without_waiting():
    lim = fmod._RateLimiter(5)
    for _ in range(5):
        lim.acquire()
    assert len(lim._times) == 5
    lim.penalize()
    assert lim.per_min == 20  # floored, never zero (which would mean "unlimited")


def test_no_expect_keys_means_no_debris_filtering(monkeypatch):
    tool = LLMGatewayTool("claude-sonnet-5")
    monkeypatch.setattr(tool, "_raw_generate", lambda *a, **k: "{}")
    assert tool.generate_json("p")["_ok"] is True


def test_failure_payload_carries_the_raw_response_for_diagnosis(monkeypatch):
    tool = LLMGatewayTool("claude-sonnet-5", max_retries=2, expect_keys=DECISION)
    monkeypatch.setattr(tool, "_raw_generate", lambda *a, **k: '{"name": "x", "value": 1}')
    out = tool.generate_json("p")
    assert out["_ok"] is False and '"value": 1' in out["_raw"]


# --------------------------------------------------------------------- json-mode support table

def test_response_format_omitted_for_unsupported_model():
    """deepseekv4-flash 400s on response_format; it must run prompt-enforced instead."""
    assert LLMGatewayTool("deepseekv4-flash").json_mode is False
    assert LLMGatewayTool("claude-sonnet-5").json_mode is True
    assert LLMGatewayTool("deepseekv4-flash", json_mode=True).json_mode is True  # explicit override


def test_request_body_carries_response_format_only_when_supported(monkeypatch):
    bodies = []

    class FakeResp:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": '{"n_repairs": 1}'}}]}

    class FakeSession:
        trust_env = True

        def post(self, url, json=None, timeout=None, headers=None):
            bodies.append(json)
            return FakeResp()

    for model, expect in (("claude-sonnet-5", True), ("deepseekv4-flash", False)):
        tool = LLMGatewayTool(model)
        monkeypatch.setattr(tool, "_session", lambda: FakeSession())
        tool.generate_json("p", temperature=0.0)
        assert ("response_format" in bodies[-1]) is expect
        assert bodies[-1]["temperature"] == 0.0
        assert bodies[-1]["model"] == model


# --------------------------------------------------------------------- auth

def test_assert_token_health_exits_with_the_fix(monkeypatch):
    tool = LLMGatewayTool("claude-sonnet-5")

    def expired(force=False):
        raise TokenError("opencode's LLMGateway token expired 3m ago")

    monkeypatch.setattr(tool, "_token", expired)
    with pytest.raises(SystemExit) as e:
        tool.assert_token_health()
    assert fmod.RELOGIN_HINT in str(e.value)


def test_mint_prefers_client_credentials(monkeypatch):
    posted = {}

    class FakeResp:
        status_code = 200

        @staticmethod
        def json():
            return {"access_token": "cc-token", "expires_in": 3600}

    monkeypatch.setenv("CRP_LLM_CLIENT_ID", "cid")
    monkeypatch.setenv("CRP_LLM_CLIENT_SECRET", "sec")
    monkeypatch.setattr(fmod.requests, "post",
                        lambda url, timeout=None, data=None: posted.update(data or {}) or FakeResp())
    tok, exp = LLMGatewayTool("claude-sonnet-5")._mint()
    assert tok == "cc-token" and exp > 0
    assert posted["grant_type"] == "client_credentials"


def test_mint_falls_back_to_opencode_token(monkeypatch, tmp_path):
    monkeypatch.delenv("CRP_LLM_CLIENT_ID", raising=False)
    monkeypatch.delenv("CRP_LLM_CLIENT_SECRET", raising=False)
    monkeypatch.setattr(fmod, "_read_secrets_env", lambda k: "")
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"llm_gateway": {"access": "oc-token",
                                            "expires": (fmod.time.time() + 3600) * 1000}}))
    monkeypatch.setattr(fmod, "OPENCODE_AUTH", auth)
    tok, _exp = LLMGatewayTool("claude-sonnet-5")._mint()
    assert tok == "oc-token"


def test_mint_reports_expiry_rather_than_returning_a_dead_token(monkeypatch, tmp_path):
    monkeypatch.delenv("CRP_LLM_CLIENT_ID", raising=False)
    monkeypatch.delenv("CRP_LLM_CLIENT_SECRET", raising=False)
    monkeypatch.setattr(fmod, "_read_secrets_env", lambda k: "")
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"llm_gateway": {"access": "stale",
                                            "expires": (fmod.time.time() - 600) * 1000}}))
    monkeypatch.setattr(fmod, "OPENCODE_AUTH", auth)
    with pytest.raises(TokenError, match="expired"):
        LLMGatewayTool("claude-sonnet-5")._mint()


# --------------------------------------------------------------------- artifact naming

def test_suffixed_leaves_shipped_names_untouched():
    """The shipped Gemini artifacts must be unreachable from a second-model run."""
    assert suffixed("b3_single_call.json", "") == "b3_single_call.json"
    assert suffixed("eval/out/ablation/single_call.jsonl", "") == \
        "eval/out/ablation/single_call.jsonl"


def test_suffixed_inserts_the_tag_before_the_extension():
    assert suffixed("b3_single_call.json", "claude-sonnet-5") == \
        "b3_single_call_claude-sonnet-5.json"
    assert suffixed("eval/out/ablation/llmnum_s8.jsonl", "gpt-5.4-2026-03-05") == \
        "eval/out/ablation/llmnum_s8_gpt-5.4-2026-03-05.jsonl"


def test_slug_is_filename_safe():
    assert slug("gpt-5.4-2026-03-05") == "gpt-5.4-2026-03-05"
    assert slug("vendor/model:v1") == "vendor-model-v1"


def test_make_client_defaults_to_the_shipped_vertex_path():
    client, probe, tag = make_client("")
    assert tag == "" and type(client).__name__ == "VertexGeminiTool"
    assert callable(probe)


def test_make_client_returns_gateway_client_and_its_own_probe():
    client, probe, tag = make_client("claude-sonnet-5")
    assert isinstance(client, LLMGatewayTool) and tag == "claude-sonnet-5"
    assert probe == client.assert_token_health
