"""Tests for GroupingAgent in offline sample mode (reads data/sample/, no parquet/BigQuery)."""
import pytest

from src.tools import DataStore
from src.agents.grouping_agent import GroupingAgent


def test_load_signature_from_sample(sample_env):
    agent = GroupingAgent(DataStore())
    assert agent.sample is True
    ctx = agent.load_signature(32954)  # F-150 Long Block
    assert ctx.signature_id == 32954
    assert "F-150" in ctx.vehicle_line
    assert len(ctx.claims) > 0
    assert "gsar_tot_cost_gross" in ctx.claims.columns


def test_cost_cap_applied(sample_env):
    ctx = GroupingAgent(DataStore()).load_signature(32954)
    valid = ctx.claims["gsar_tot_cost_gross"].dropna()
    assert (valid < 100000).all()
    assert (valid > 0).all()


def test_unknown_signature_raises(sample_env):
    with pytest.raises(KeyError):
        GroupingAgent(DataStore()).load_signature(-12345)
