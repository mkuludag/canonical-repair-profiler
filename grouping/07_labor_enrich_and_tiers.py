#!/usr/bin/env python3
"""
Enhancement A (labor enrichment) + Enhancement B (cost-tier sub-split).

A) The PAWS labor-op view (sxpaa104) is empty; real labor-op CODES live in OWS (sowsc09w,
   226M rows) but need an OWS->VIN bridge. The directly-joinable, DENSE labor source is
   apply.repair_cost_labor_view -- keyed by request_id (== request_r), 1.05M requests,
   structured total_labor_hours / labor_rate / total_labor_cost. We extract it (1 row/request)
   and compare its coverage + consensus to the GSAR labor we'd been using.

B) Cost-tier sub-split: a single causal-part code can hide two repairs (reseal $800 vs replace
   $8k). For "fuzzy" G3 groups (low cost consensus), split each into 2 cost tiers (median split)
   and measure how much consensus improves -> turns 1 fuzzy ground truth into 2 clean ones.

Outputs: grouping/out/labor_enrich_summary.csv, cost_tier_split_summary.csv,
         data/labor_apply.parquet (cached extract)
"""
import os, subprocess
from decimal import Decimal
import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.oauth2.credentials import Credentials

PARQUET = os.environ.get("REPAIR_PARQUET", "data/repair_profile.parquet")
OUT = "grouping/out"; os.makedirs(OUT, exist_ok=True)
os.makedirs("data", exist_ok=True)
LABOR_CACHE = "data/labor_apply.parquet"
BILLING = "REDACTED-BILLING-PROJECT"
TOL = 0.25


def extract_labor():
    if os.path.exists(LABOR_CACHE):
        print(f"using cached {LABOR_CACHE}")
        return pd.read_parquet(LABOR_CACHE)
    tok = subprocess.check_output(
        ["gcloud", "auth", "print-access-token"],
        env={**os.environ, "CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT": ""}).decode().strip()
    c = bigquery.Client(project=BILLING, credentials=Credentials(tok))
    sql = """
      SELECT request_id AS request_r,
             SUM(total_labor_hours)        AS apply_labor_hours,
             SUM(total_labor_cost)         AS apply_labor_cost,
             AVG(NULLIF(labor_rate,0))     AS apply_labor_rate,
             SUM(scheduled_labor_hours)    AS apply_sched_hours
      FROM `REDACTED-SOURCE-PROJECT.dataset_paws_apply.repair_cost_labor_view`
      WHERE request_id IS NOT NULL
      GROUP BY request_id
    """
    print("extracting apply labor (1 row/request) ...")
    df = c.query(sql).result().to_dataframe(create_bqstorage_client=False)
    for col in df.columns:
        if df[col].dtype == object and df[col].map(lambda x: isinstance(x, Decimal)).any():
            df[col] = df[col].astype(float)
    df["request_r"] = df["request_r"].astype("int64")
    df.to_parquet(LABOR_CACHE, index=False)
    print(f"  cached {len(df):,} requests -> {LABOR_CACHE}")
    return df


def consensus(df, keys, col, tol=TOL):
    med = df.groupby(keys, observed=True)[col].transform("median")
    valid = df[col].notna() & (med > 0)
    within = (df[col].sub(med).abs() <= tol * med) & valid
    t = pd.DataFrame({"_v": valid.astype(int), "_w": within.astype(int)})
    for k in keys:
        t[k] = df[k].values
    a = t.groupby(keys, observed=True)[["_v", "_w"]].sum()
    return a["_w"] / a["_v"].where(a["_v"] > 0)


def main():
    labor = extract_labor()

    cols = ["gsar_veh_line_desc", "paws_causal_part", "det_approved_amt", "gsar_labor_hrs", "paws_comment_trail"]
    df = pd.read_parquet(PARQUET, columns=cols)
    # need request_r to join labor -> read separately and align by row order index
    rr = pd.read_parquet(PARQUET, columns=["request_r"])
    df["request_r"] = pd.to_numeric(rr["request_r"], errors="coerce").astype("Int64")
    df["det_approved_amt"] = pd.to_numeric(df["det_approved_amt"], errors="coerce")
    df.loc[(df["det_approved_amt"] <= 0) | (df["det_approved_amt"] >= 100000), "det_approved_amt"] = np.nan
    df["gsar_labor_hrs"] = pd.to_numeric(df["gsar_labor_hrs"], errors="coerce")

    df = df.merge(labor[["request_r", "apply_labor_hours"]], on="request_r", how="left")
    df.loc[(df["apply_labor_hours"] <= 0) | (df["apply_labor_hours"] >= 200), "apply_labor_hours"] = np.nan

    # ---- Enhancement A: labor coverage + consensus comparison ----
    keys = ["gsar_veh_line_desc", "paws_causal_part"]
    cov = pd.DataFrame({
        "source": ["gsar_labor_hrs", "apply_labor_hours"],
        "fill_pct": [round(df["gsar_labor_hrs"].notna().mean()*100, 1),
                     round(df["apply_labor_hours"].notna().mean()*100, 1)],
    })
    sub = df.dropna(subset=keys)
    lc_gsar = consensus(sub, keys, "gsar_labor_hrs")
    lc_apply = consensus(sub, keys, "apply_labor_hours")
    n = sub.groupby(keys, observed=True).size()
    big = n[n >= 5].index
    cov["median_labor_consensus_grp_ge5"] = [round(float(lc_gsar.reindex(big).median()), 3),
                                             round(float(lc_apply.reindex(big).median()), 3)]
    cov.to_csv(f"{OUT}/labor_enrich_summary.csv", index=False)
    print("\n=== Enhancement A: labor source comparison ===")
    print(cov.to_string(index=False))

    # ---- Enhancement B: cost-tier sub-split on fuzzy groups ----
    g = sub.groupby(keys, observed=True)
    n = g.size().rename("n")
    base_cons = consensus(sub, keys, "det_approved_amt").rename("cost_consensus")
    grp = pd.concat([n, base_cons], axis=1)
    fuzzy = grp[(grp["n"] >= 20) & (grp["cost_consensus"] < 0.5)]
    print(f"\n=== Enhancement B: cost-tier sub-split ===")
    print(f"fuzzy groups (n>=20, cost_consensus<0.5): {len(fuzzy):,}")

    # median split into 2 tiers, recompute consensus within (group, tier)
    med = sub.groupby(keys, observed=True)["det_approved_amt"].transform("median")
    sub2 = sub.assign(_tier=(sub["det_approved_amt"] >= med).astype("Int8"))
    keys2 = keys + ["_tier"]
    tier_cons = consensus(sub2, keys2, "det_approved_amt")
    # weighted mean tier consensus per original group
    tn = sub2.groupby(keys2, observed=True).size().rename("n")
    tc = pd.concat([tn, tier_cons.rename("c")], axis=1).reset_index()
    tc["wc"] = tc["n"] * tc["c"]
    grp_tier = tc.groupby(keys, observed=True).apply(
        lambda x: pd.Series({"tiered_consensus": x["wc"].sum() / x["n"].sum()})).reset_index()
    comp = grp.reset_index().merge(grp_tier, on=keys, how="left")
    fuzzy_comp = comp[(comp["n"] >= 20) & (comp["cost_consensus"] < 0.5)]
    lift = (fuzzy_comp["tiered_consensus"] - fuzzy_comp["cost_consensus"])
    became_clean = (fuzzy_comp["tiered_consensus"] >= 0.6).sum()
    summ = pd.DataFrame([{
        "fuzzy_groups": len(fuzzy_comp),
        "median_consensus_before": round(float(fuzzy_comp["cost_consensus"].median()), 3),
        "median_consensus_after_tiering": round(float(fuzzy_comp["tiered_consensus"].median()), 3),
        "median_lift": round(float(lift.median()), 3),
        "groups_became_clean(>=0.6)": int(became_clean),
        "pct_became_clean": round(100*became_clean/max(len(fuzzy_comp),1), 1),
    }])
    summ.to_csv(f"{OUT}/cost_tier_split_summary.csv", index=False)
    print(summ.to_string(index=False))
    print("\n(median split: each fuzzy group -> low/high cost tier; consensus recomputed within tiers)")


if __name__ == "__main__":
    main()
