"""
StatsTool — deterministic consensus math used by the agents.

Pure, side-effect-free statistics so the numeric half of every Canonical Repair is reproducible
and auditable (the LLM only writes text; the numbers come from here). Functions:
  - consensus_fraction: share of values within +/-tol of the median (our "agreement" metric),
  - iqr / median bands for cost & labor,
  - cost_tier_partition: split a group's claims into N cost tiers (to realize an LLM split),
  - confidence: combine support (n) and consensus into a single 0-1 trust weight.
"""
import numpy as np
import pandas as pd

from ..config import CONSENSUS_TOL


class StatsTool:
    @staticmethod
    def consensus_fraction(values: pd.Series, tol: float = CONSENSUS_TOL) -> float:
        v = pd.to_numeric(values, errors="coerce").dropna()
        v = v[v > 0]
        if len(v) < 2:
            return float("nan")
        med = v.median()
        if med <= 0:
            return float("nan")
        return float((v.sub(med).abs() <= tol * med).mean())

    @staticmethod
    def band(values: pd.Series) -> dict:
        v = pd.to_numeric(values, errors="coerce").dropna()
        v = v[v > 0]
        if len(v) == 0:
            return {"n": 0, "median": np.nan, "q25": np.nan, "q75": np.nan}
        return {"n": int(len(v)), "median": round(float(v.median()), 2),
                "q25": round(float(v.quantile(0.25)), 2), "q75": round(float(v.quantile(0.75)), 2)}

    @staticmethod
    def cost_tier_partition(df: pd.DataFrame, cost_col: str, cost_tier: str, n_repairs: int) -> pd.DataFrame:
        """Slice a signature's claims to the cost band for a given repair tier (low/mid/high/all)."""
        if df is None or len(df) == 0 or n_repairs <= 1 or cost_tier in ("all", "", None):
            return df
        c = pd.to_numeric(df[cost_col], errors="coerce")
        if n_repairs == 2:
            q = c.quantile(0.5)
            return df[c < q] if cost_tier == "low" else df[c >= q]
        q1, q2 = c.quantile(1 / 3), c.quantile(2 / 3)
        if cost_tier == "low":
            return df[c < q1]
        if cost_tier == "high":
            return df[c >= q2]
        return df[(c >= q1) & (c < q2)]

    @staticmethod
    def confidence(n: int, consensus: float) -> float:
        """Blend agreement (consensus) with a gentle sample-size weight.
        support_w reaches 1.0 by ~30 claims (n=30 -> 1.0, n=10 -> ~0.70, n=6 -> ~0.57),
        so a tight small tier is mildly — not harshly — discounted."""
        if consensus is None or (isinstance(consensus, float) and np.isnan(consensus)):
            consensus = 0.0
        support_w = min(1.0, np.log10(max(n, 1) + 1) / np.log10(31))
        return round(float(consensus) * support_w, 3)
