"""
RegionIndexTool - deterministic per-state cost/labor index for warranty repairs.

Our data-playground study found warranty cost varies ~1.7x across US states (labor rates, taxes,
parts logistics). This tool turns that study (grouping/out/region_overall.csv, ~10.6M GSAR claims
aggregated by state) into a reproducible multiplier so a single national Canonical Repair cost can be
localized to the dealer's state - WITHOUT any per-state LLM calls (fully deterministic + auditable).

    cost_index[state]  = state_median_cost      / national_reference_cost
    labor_index[state] = state_median_labor_cost / national_reference_labor

national_reference = claims-weighted mean of the per-state medians (a stable national anchor).

This is a TOOL (no LLM): the RegionDelegatorAgent selects and calls it. It is defensive - a missing
file or unknown state degrades gracefully to a neutral 1.0 index rather than raising.
"""
import pandas as pd

from .. import config as C


class RegionIndexTool:
    def __init__(self, region_csv: str = C.REGION_OVERALL):
        self.region_csv = region_csv
        self._cost_index = {}
        self._labor_index = {}
        self.national_cost = None
        self.national_labor = None
        self._rows = {}
        self._load()

    def _load(self) -> None:
        try:
            df = pd.read_csv(self.region_csv)
        except (FileNotFoundError, OSError):
            return  # neutral indices; index_for() returns 1.0
        for col in ("med_cost", "mean_cost", "claims", "med_labor_cost", "med_labor_hrs"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["state", "med_cost", "claims"])
        w = df["claims"].sum()
        self.national_cost = float((df["med_cost"] * df["claims"]).sum() / w) if w else float(df["med_cost"].median())
        if "med_labor_cost" in df.columns:
            self.national_labor = float((df["med_labor_cost"] * df["claims"]).sum() / w) if w else None
        for _, r in df.iterrows():
            st = str(r["state"]).upper()
            self._cost_index[st] = round(float(r["med_cost"]) / self.national_cost, 4) if self.national_cost else 1.0
            if self.national_labor and pd.notna(r.get("med_labor_cost")):
                self._labor_index[st] = round(float(r["med_labor_cost"]) / self.national_labor, 4)
            self._rows[st] = {"claims": int(r["claims"]), "med_cost": float(r["med_cost"])}

    # ---- public API ----
    def available(self) -> bool:
        return bool(self._cost_index)

    def states(self) -> list:
        """States sorted most-expensive -> cheapest by cost index."""
        return sorted(self._cost_index, key=lambda s: self._cost_index[s], reverse=True)

    def index_for(self, state: str) -> float:
        """Cost multiplier for a state (1.0 = national average; unknown state -> 1.0)."""
        if not state:
            return 1.0
        return self._cost_index.get(str(state).upper(), 1.0)

    def labor_index_for(self, state: str) -> float:
        return self._labor_index.get(str(state).upper(), 1.0) if state else 1.0

    def spread(self) -> dict:
        """Min/max index across states (the size of the regional effect)."""
        if not self._cost_index:
            return {"min": 1.0, "max": 1.0, "ratio": 1.0}
        lo, hi = min(self._cost_index.values()), max(self._cost_index.values())
        return {"min": round(lo, 3), "max": round(hi, 3), "ratio": round(hi / lo, 2) if lo else 1.0,
                "cheapest": min(self._cost_index, key=self._cost_index.get),
                "priciest": max(self._cost_index, key=self._cost_index.get)}
