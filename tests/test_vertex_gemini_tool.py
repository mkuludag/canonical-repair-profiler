"""Tests for VertexGeminiTool resilience: retries, JSON parsing, graceful failure.

The transport method `_raw_generate` is monkeypatched, so no google-genai SDK or network is needed.
"""
import pytest

from src.tools.vertex_gemini_tool import VertexGeminiTool
import src.tools.vertex_gemini_tool as vmod


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(vmod.time, "sleep", lambda *_: None)


def test_generate_json_success(monkeypatch):
    tool = VertexGeminiTool()
    monkeypatch.setattr(tool, "_raw_generate", lambda *a, **k: '{"n_repairs": 2, "ok": true}')
    out = tool.generate_json("p")
    assert out["_ok"] is True and out["n_repairs"] == 2
    assert out["_attempts"] == 1  # first-try success -> sampled at the requested temperature


def test_generate_json_retries_then_succeeds(monkeypatch):
    tool = VertexGeminiTool(max_retries=3)
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return '{"ok": true}'

    monkeypatch.setattr(tool, "_raw_generate", flaky)
    out = tool.generate_json("p")
    assert out["_ok"] is True and calls["n"] == 2


def test_generate_json_bad_json_retries_with_temperature_jitter(monkeypatch):
    """A parse failure at temperature 0 reproduces identically (observed on gemini-2.5-flash:
    dropped closing brace), so retries must sample at >= JSON_RETRY_MIN_TEMPERATURE."""
    tool = VertexGeminiTool()
    calls = {"n": 0, "temps": []}

    def flaky_json(prompt, temperature, json_mode, force=False):
        calls["n"] += 1
        calls["temps"].append(temperature)
        return "this is not json" if calls["n"] == 1 else '{"ok": true}'

    monkeypatch.setattr(tool, "_raw_generate", flaky_json)
    out = tool.generate_json("p", temperature=0.0)
    assert out["_ok"] is True and calls["n"] == 2
    assert calls["temps"][0] == 0.0
    assert calls["temps"][1] >= tool.JSON_RETRY_MIN_TEMPERATURE
    assert out["_attempts"] == 2  # marks the payload as coming from the jittered retry


def test_generate_json_bad_json_permanent_failure_reports(monkeypatch):
    tool = VertexGeminiTool(max_retries=3)
    calls = {"n": 0}

    def bad(*a, **k):
        calls["n"] += 1
        return "this is not json"

    monkeypatch.setattr(tool, "_raw_generate", bad)
    out = tool.generate_json("p")
    assert out["_ok"] is False and "JSON parse" in out["error"] and calls["n"] == 3


def test_generate_json_non_retryable_breaks_fast(monkeypatch):
    tool = VertexGeminiTool(max_retries=3)
    calls = {"n": 0}

    def fatal(*a, **k):
        calls["n"] += 1
        raise RuntimeError("400 INVALID_ARGUMENT")

    monkeypatch.setattr(tool, "_raw_generate", fatal)
    out = tool.generate_json("p")
    assert out["_ok"] is False and calls["n"] == 1  # 400 is not retryable


def test_generate_json_exhausts_retries(monkeypatch):
    tool = VertexGeminiTool(max_retries=3)
    monkeypatch.setattr(tool, "_raw_generate",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("503 UNAVAILABLE")))
    out = tool.generate_json("p")
    assert out["_ok"] is False and "503" in out["error"]


def test_generate_text_failure_is_graceful(monkeypatch):
    tool = VertexGeminiTool(max_retries=1)
    monkeypatch.setattr(tool, "_raw_generate",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert tool.generate_text("p").startswith("[LLM_ERROR]")


def test_is_retryable():
    assert VertexGeminiTool._is_retryable("429 RESOURCE_EXHAUSTED")
    assert VertexGeminiTool._is_retryable("UNAUTHENTICATED 401")
    assert not VertexGeminiTool._is_retryable("400 INVALID_ARGUMENT")
