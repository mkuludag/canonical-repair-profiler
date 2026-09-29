#!/usr/bin/env python3
"""
Grouping-playground technique sweep: how groupable is repair_profile, and how much
WITHIN-GROUP variability / contradiction exists for cost ($), labor (hrs), and the
repair solution (parts).  Memory-aware: reads only the needed columns, uses vectorized
groupby.agg (no .apply).

Outputs:
  grouping/out/grouping_summary.csv          one row per technique
  grouping/out/<Tn>_largest_groups.csv       biggest groups + their variability
Prints a readable summary table to stdout.
"""
import os
import numpy as np
import pandas as pd

PARQUET = os.environ.get("REPAIR_PARQUET", "data/repair_profile.parquet")
OUT = "grouping/out"
os.makedirs(OUT, exist_ok=True)

COST   = "det_approved_amt"       # 99.9% filled  -> primary $ metric
GCOST  = "gsar_tot_cost_gross"    # 68% filled    -> GSAR gross $
GHRS   = "gsar_labor_hrs"         # 68% filled    -> labor hrs
SOLN   = "part_numbers"           # 15.9% filled  -> the parts "solution"

NEEDED = ["rep_cat_label", "rep_sub_cat_label", "paws_causal_part", "gsar_veh_line_desc",
          "model_year", "gsar_cust_concern_label", "gsar_condition_label", "gsar_wcc_label",
          COST, GCOST, GHRS, SOLN]

TECHNIQUES = [
    (["rep_cat_label"],                                          "T1_cat"),
    (["rep_cat_label", "rep_sub_cat_label"],                     "T2_cat_subcat"),
    (["paws_causal_part"],                                       "T3_causal_part"),
    (["rep_sub_cat_label", "gsar_veh_line_desc"],                "T4_subcat_vehline"),
    (["paws_causal_part", "gsar_veh_line_desc"],                 "T5_causal_vehline"),
    (["model_year", "gsar_veh_line_desc", "rep_sub_cat_label"],  "T6_year_line_subcat"),
    (["gsar_cust_concern_label", "gsar_condition_label"],        "T7_ccc_concern_cond"),
    (["gsar_wcc_label"],                                         "T8_wcc"),
]


def analyze(df, keys, tag):
    sub = df.dropna(subset=keys)
    grp = sub.groupby(keys, observed=True, sort=False)

    # vectorized per-group aggregates
    agg = grp.agg(
        n=(COST, "size"),
        cost_mean=(COST, "mean"), cost_std=(COST, "std"),
        gcost_mean=(GCOST, "mean"), gcost_std=(GCOST, "std"),
        ghrs_mean=(GHRS, "mean"), ghrs_std=(GHRS, "std"),
        soln_nunique=(SOLN, "nunique"), soln_count=(SOLN, "count"),
    )
    # coefficient of variation (std/mean); divide by NaN (not 0) to avoid Decimal/zero errors
    agg["cost_cv"]  = agg["cost_std"]  / agg["cost_mean"].where(agg["cost_mean"]  > 0)
    agg["gcost_cv"] = agg["gcost_std"] / agg["gcost_mean"].where(agg["gcost_mean"] > 0)
    agg["ghrs_cv"]  = agg["ghrs_std"]  / agg["ghrs_mean"].where(agg["ghrs_mean"]  > 0)
    # solution divergence: distinct part-sets per claim that HAS parts (1.0 = every claim a different set)
    agg["soln_divergence"] = agg["soln_nunique"] / agg["soln_count"].where(agg["soln_count"] >= 2)

    sizes = agg["n"]
    multi = agg[agg["n"] >= 2]
    summary = {
        "technique": tag, "keys": "+".join(keys),
        "n_groups": int(len(agg)),
        "n_multi_groups": int(len(multi)),
        "claims": int(sizes.sum()),
        "avg_per_group": round(float(sizes.mean()), 1),
        "median_per_group": int(sizes.median()),
        "p90_per_group": int(sizes.quantile(0.90)),
        "max_group": int(sizes.max()),
        "pct_claims_in_multi": round(100 * multi["n"].sum() / sizes.sum(), 1),
        # typical within-group spread (median of per-group CV across multi-claim groups)
        "med_cost_cv": round(float(multi["cost_cv"].median()), 3),
        "med_gsar_cost_cv": round(float(multi["gcost_cv"].median()), 3),
        "med_labor_cv": round(float(multi["ghrs_cv"].median()), 3),
        "med_soln_divergence": round(float(multi["soln_divergence"].median()), 3),
    }
    top = agg.sort_values("n", ascending=False).head(40).reset_index()
    top.to_csv(f"{OUT}/{tag}_largest_groups.csv", index=False)
    return summary


def main():
    print(f"Loading {len(NEEDED)} columns from {PARQUET} ...")
    df = pd.read_parquet(PARQUET, columns=NEEDED)
    # BigQuery NUMERIC -> Decimal objects; cast money/hrs to float for arithmetic
    for c in (COST, GCOST, GHRS):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"  {len(df):,} rows, {df.memory_usage(deep=True).sum()/1e6:.0f} MB\n")

    rows = [analyze(df, keys, tag) for keys, tag in TECHNIQUES if all(k in df.columns for k in keys)]
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/grouping_summary.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(out[["technique", "n_groups", "n_multi_groups", "avg_per_group", "median_per_group",
               "max_group", "pct_claims_in_multi", "med_cost_cv", "med_gsar_cost_cv",
               "med_labor_cv", "med_soln_divergence"]].to_string(index=False))
    print(f"\nWrote {OUT}/grouping_summary.csv  (+ per-technique *_largest_groups.csv)")
    print("\nReading guide:")
    print("  med_cost_cv / med_gsar_cost_cv: typical $ spread within a group (0=identical, >0.5=high variance)")
    print("  med_labor_cv: typical labor-hours spread within a group")
    print("  med_soln_divergence: 0->everyone used same parts; 1->every claim a different part-set")


if __name__ == "__main__":
    main()
