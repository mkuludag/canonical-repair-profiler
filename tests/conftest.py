"""
Shared pytest fixtures.

All tests run fully OFFLINE: every external call (Vertex Gemini, BigQuery, gcloud) is replaced by a
stub/mock, and data comes from the bundled sample (data/sample/). No network or credentials needed.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.agents.context import RepairContext  # noqa: E402


@pytest.fixture(scope="session")
def repo_root():
    return ROOT


@pytest.fixture
def sample_env(monkeypatch):
    """Route config path helpers to the bundled offline sample."""
    monkeypatch.setenv("CRP_USE_SAMPLE", "1")
    yield


@pytest.fixture
def synthetic_claims():
    """A claims frame with a clearly BIMODAL cost distribution (reseal ~2k vs replace ~9k)."""
    low = {"gsar_tot_cost_gross": [1900, 2000, 2100, 2200, 2300, 2000, 2150, 2050],
           "gsar_labor_hrs": [3.0, 3.2, 2.8, 3.1, 3.0, 2.9, 3.3, 3.0]}
    high = {"gsar_tot_cost_gross": [8800, 9000, 9200, 9100, 9300, 8900, 9050, 9150],
            "gsar_labor_hrs": [11.0, 11.5, 10.8, 11.2, 11.1, 10.9, 11.3, 11.0]}
    rows = []
    for d in (low, high):
        for c, h in zip(d["gsar_tot_cost_gross"], d["gsar_labor_hrs"]):
            rows.append({"gsar_tot_cost_gross": float(c), "gsar_labor_hrs": float(h),
                         "paws_comment_trail": "oil leak front cover / replace assembly"})
    return pd.DataFrame(rows)


@pytest.fixture
def tight_claims():
    """A claims frame with a TIGHT, single-mode cost distribution (one true repair)."""
    return pd.DataFrame({
        "gsar_tot_cost_gross": [12000, 12200, 12400, 12500, 12600, 12300, 12450, 12550],
        "gsar_labor_hrs": [16.0, 16.5, 16.2, 16.8, 16.1, 16.3, 16.4, 16.6],
        "paws_comment_trail": ["long block replacement"] * 8,
    })


@pytest.fixture
def base_ctx(synthetic_claims):
    ctx = RepairContext(signature_id=999, vehicle_line="TEST F-150", causal_part="6019",
                        archetype=1, archetype_theme="oil leak, front cover")
    ctx.claims = synthetic_claims
    return ctx


class FakeLLM:
    """Drop-in stand-in for VertexGeminiTool used in agent tests."""
    def __init__(self, json_payload=None, text_payload="ok"):
        self.json_payload = json_payload or {"_ok": True}
        self.text_payload = text_payload
        self.calls = []

    def generate_json(self, prompt, temperature=0.2):
        self.calls.append(prompt)
        return dict(self.json_payload)

    def generate_text(self, prompt, temperature=0.2):
        self.calls.append(prompt)
        return self.text_payload


@pytest.fixture
def fake_llm():
    return FakeLLM
