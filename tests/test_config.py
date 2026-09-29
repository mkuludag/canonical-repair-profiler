"""Tests for config defaults + environment overrides + sample toggle."""
import importlib


def test_defaults_present():
    import src.config as c
    assert c.GEMINI_MODEL
    assert c.COST_COL == "gsar_tot_cost_gross"
    assert c.SPLIT_SEPARATION_MIN >= 1.0


def test_env_override(monkeypatch):
    monkeypatch.setenv("CRP_GEMINI_MODEL", "gemini-test-model")
    import src.config as c
    importlib.reload(c)
    assert c.GEMINI_MODEL == "gemini-test-model"
    monkeypatch.delenv("CRP_GEMINI_MODEL")
    importlib.reload(c)


def test_use_sample_toggle(monkeypatch):
    import src.config as c
    monkeypatch.delenv("CRP_USE_SAMPLE", raising=False)
    assert c.use_sample() is False
    monkeypatch.setenv("CRP_USE_SAMPLE", "1")
    assert c.use_sample() is True
