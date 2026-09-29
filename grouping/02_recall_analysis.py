#!/usr/bin/env python3
"""
Recall treatment analysis (local Parquet).
NOTE on recall *types*: repair_profile carries only gsar_camp_recall_flag (Y/N). Specific
campaign/FSA codes are NOT joined at claim grain here -- they live in fsa_ss_camp_vw /
fsa_ss_grid_vw (AWS) and sowsr23/24/25_cprcf_fsa_* (OWS). This script analyzes the Y/N flag:
  (a) recall vs non-recall overall cost/labor treatment
  (b) within recall claims, how the SAME repair (causal part) is treated differently across
      vehicle lines  -> "were different vehicles in recalls treated differently?"
Outputs grouping/out/recall_*.csv
"""
import os
import numpy as np
import pandas as pd

PARQUET = os.environ.get("REPAIR_PARQUET", "data/repair_profile.parquet")
OUT = "grouping/out"; os.makedirs(OUT, exist_ok=True)

NEEDED = ["gsar_camp_recall_flag", "gsar_veh_line_desc", "rep_sub_cat_label", "paws_causal_part",
          "det_approved_amt", "gsar_tot_cost_gross", "gsar_labor_hrs", "gsar_material_cost"]
MONEY = ["det_approved_amt", "gsar_tot_cost_gross", "gsar_labor_hrs", "gsar_material_cost"]


def main():
    df = pd.read_parquet(PARQUET, columns=NEEDED)
    for c in MONEY:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["recall"] = df["gsar_camp_recall_flag"].map({"Y": "recall", "N": "non_recall"}).fillna("unknown")

    # (a) overall recall vs non-recall treatment (only rows with a GSAR match carry the flag)
    overall = df[df["recall"].isin(["recall", "non_recall"])].groupby("recall").agg(
        claims=("recall", "size"),
        approved_med=("det_approved_amt", "median"),
        approved_mean=("det_approved_amt", "mean"),
        gsar_cost_med=("gsar_tot_cost_gross", "median"),
        gsar_cost_mean=("gsar_tot_cost_gross", "mean"),
        labor_hrs_med=("gsar_labor_hrs", "median"),
        material_med=("gsar_material_cost", "median"),
    ).round(2)
    overall.to_csv(f"{OUT}/recall_overall.csv")
    print("=== (a) Recall vs non-recall treatment ===")
    print(overall.to_string(), "\n")

    # (b) within RECALL claims: same causal part, treated differently across vehicle lines.
    rec = df[df["recall"] == "recall"].dropna(subset=["paws_causal_part", "gsar_veh_line_desc"])
    # part-level treatment by vehicle line
    by = rec.groupby(["paws_causal_part", "gsar_veh_line_desc"]).agg(
        n=("det_approved_amt", "size"),
        approved_med=("det_approved_amt", "median"),
        gsar_cost_med=("gsar_tot_cost_gross", "median"),
        labor_med=("gsar_labor_hrs", "median"),
    )
    # recall claims are sparse in this prior-approval table; loosen thresholds accordingly
    by = by[by["n"] >= 3]
    part_spread = by.groupby("paws_causal_part").agg(
        vehicle_lines=("approved_med", "size"),
        claims=("n", "sum"),
        cost_min=("gsar_cost_med", "min"),
        cost_max=("gsar_cost_med", "max"),
        labor_min=("labor_med", "min"),
        labor_max=("labor_med", "max"),
    )
    part_spread = part_spread[part_spread["vehicle_lines"] >= 2].copy()
    part_spread["cost_ratio_max_min"] = (part_spread["cost_max"] / part_spread["cost_min"]).round(2)
    part_spread["labor_ratio_max_min"] = (part_spread["labor_max"] / part_spread["labor_min"]).round(2)
    part_spread = part_spread.sort_values("cost_ratio_max_min", ascending=False)
    part_spread.round(2).to_csv(f"{OUT}/recall_part_treatment_by_vehline.csv")

    print("=== (b) Recall causal parts treated MOST differently across vehicle lines ===")
    print(f"(causal parts on >=3 vehicle lines, >=10 claims each: {len(part_spread)} parts)\n")
    print(part_spread.head(12)[["vehicle_lines", "claims", "cost_min", "cost_max",
                                "cost_ratio_max_min", "labor_min", "labor_max",
                                "labor_ratio_max_min"]].to_string())
    print(f"\nWrote {OUT}/recall_overall.csv  and  {OUT}/recall_part_treatment_by_vehline.csv")


if __name__ == "__main__":
    main()
