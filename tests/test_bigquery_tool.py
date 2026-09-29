"""Tests for BigQueryTool: Decimal->float coercion + retry on transient errors (client mocked)."""
from decimal import Decimal

import pandas as pd
import pytest

from src.tools.bigquery_tool import BigQueryTool
import src.tools.bigquery_tool as bqmod


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(bqmod.time, "sleep", lambda *_: None)


def test_coerce_decimal_to_float():
    df = pd.DataFrame({"cost": [Decimal("1.50"), Decimal("2.25")], "label": ["a", "b"]})
    out = BigQueryTool._coerce(df)
    assert out["cost"].dtype == float
    assert out["label"].dtype == object


class _FakeJob:
    def __init__(self, df, seen=None):
        self._df = df
        self._seen = seen

    def result(self, timeout=None):
        if self._seen is not None:
            self._seen["timeout"] = timeout
        return self

    def to_dataframe(self, **_):
        return self._df


class _FakeClient:
    def __init__(self, df, seen=None):
        self._df = df
        self._seen = seen

    def query(self, _sql):
        return _FakeJob(self._df, self._seen)


def test_query_returns_coerced_df(monkeypatch):
    df = pd.DataFrame({"cost": [Decimal("3.0")]})
    tool = BigQueryTool()
    monkeypatch.setattr(tool, "_client_fresh", lambda force=False: _FakeClient(df))
    out = tool.query("SELECT 1")
    assert out["cost"].dtype == float and out["cost"].iloc[0] == 3.0


def test_query_enforces_timeout(monkeypatch):
    """Every query carries the 60s timeout budget so a stuck job cannot hang the pipeline."""
    seen = {}
    tool = BigQueryTool()
    monkeypatch.setattr(tool, "_client_fresh",
                        lambda force=False: _FakeClient(pd.DataFrame({"x": [1]}), seen))
    tool.query("SELECT 1")
    assert seen["timeout"] == bqmod.REQUEST_TIMEOUT_MS / 1000


def test_query_retries_then_succeeds(monkeypatch):
    df = pd.DataFrame({"x": [1]})
    tool = BigQueryTool(max_retries=3)
    state = {"n": 0}

    def factory(force=False):
        state["n"] += 1
        if state["n"] < 2:
            class Boom:
                def query(self, _):
                    raise RuntimeError("503 backend error")
            return Boom()
        return _FakeClient(df)

    monkeypatch.setattr(tool, "_client_fresh", factory)
    out = tool.query("SELECT 1")
    assert len(out) == 1 and state["n"] == 2


def test_query_raises_after_retries(monkeypatch):
    tool = BigQueryTool(max_retries=2)

    class Boom:
        def query(self, _):
            raise RuntimeError("503 backend error")

    monkeypatch.setattr(tool, "_client_fresh", lambda force=False: Boom())
    with pytest.raises(RuntimeError, match="BigQuery query failed"):
        tool.query("SELECT 1")
