#!/usr/bin/env python3
"""
RECALL VERTICAL DEMO (GSAR broad recall population, ~243k recall claims).

Recalls are campaign-defined, so they SHOULD be the most uniform repairs -> the cleanest Canonical
Repairs, and any dealer/state divergence is a clear process discrepancy. We:
  (1) emit recall Canonical Repairs: per (vehicle_line, causal_part, condition) consensus cost/labor
      + how many dealers/states executed it and how widely dealer medians spread;
  (2) test the hypothesis "recalls are more uniform than non-recall" (within-signature cost CV).

Outputs:
  grouping/out/recall_canonical.csv
  grouping/out/recall_uniformity.csv
  grouping/out/fig_recall_uniformity.png
"""
import os, subprocess
from decimal import Decimal
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from google.cloud import bigquery
from google.oauth2.credentials import Credentials

BILLING = "REDACTED-BILLING-PROJECT"
SRC = "REDACTED-SOURCE-PROJECT.dataset_fdp.claims_master_view"
OUT = "grouping/out"; os.makedirs(OUT, exist_ok=True)
GUARD = ("rpr_cntry_cd='USA' AND tot_cost_gross BETWEEN 50 AND 100000 "
         "AND prt_num_causl_base_cd NOT IN ('MAINT','*','') AND veh_line_desc IS NOT NULL "
         "AND cond_cd IS NOT NULL AND rpr_dt BETWEEN DATE '2022-01-01' AND CURRENT_DATE()")


def client():
    tok = subprocess.check_output(
        ["gcloud", "auth", "print-access-token"],
        env={**os.environ, "CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT": ""}).decode().strip()
    return bigquery.Client(project=BILLING, credentials=Credentials(tok))


Q_RECALL_CANON = f"""
WITH r AS (
  SELECT veh_line_desc, prt_num_causl_base_cd part, cond_cd, rpr_dlr_cd, rpr_st_prov_cd,
         tot_cost_gross, lbr_hrs
  FROM `{SRC}` WHERE {GUARD} AND UPPER(camp_recall_flag)='Y'
),
dealer AS (
  SELECT veh_line_desc, part, cond_cd, rpr_dlr_cd,
         APPROX_QUANTILES(tot_cost_gross,2)[OFFSET(1)] dlr_med, COUNT(*) dn
  FROM r GROUP BY 1,2,3,4 HAVING dn >= 5
)
SELECT r.veh_line_desc, r.part, r.cond_cd,
       COUNT(*) n,
       COUNT(DISTINCT r.rpr_dlr_cd) dealers,
       COUNT(DISTINCT r.rpr_st_prov_cd) states,
       ROUND(APPROX_QUANTILES(r.tot_cost_gross,4)[OFFSET(2)],2) cost_med,
       ROUND(APPROX_QUANTILES(r.tot_cost_gross,4)[OFFSET(1)],2) cost_q25,
       ROUND(APPROX_QUANTILES(r.tot_cost_gross,4)[OFFSET(3)],2) cost_q75,
       ROUND(APPROX_QUANTILES(r.lbr_hrs,2)[OFFSET(1)],2) labor_med,
       ROUND(MAX(d.dlr_med)/NULLIF(MIN(d.dlr_med),0),2) dealer_spread_ratio
FROM r LEFT JOIN dealer d
  ON r.veh_line_desc=d.veh_line_desc AND r.part=d.part AND r.cond_cd=d.cond_cd
GROUP BY 1,2,3 HAVING n >= 30
ORDER BY n DESC LIMIT 300
"""

# uniformity: within-signature cost CV, recall vs non-recall
Q_UNIFORMITY = f"""
WITH base AS (
  SELECT CASE WHEN UPPER(camp_recall_flag)='Y' THEN 'recall' ELSE 'non_recall' END flag,
         veh_line_desc, prt_num_causl_base_cd part, cond_cd, tot_cost_gross
  FROM `{SRC}` WHERE {GUARD}
),
sig AS (
  SELECT flag, veh_line_desc, part, cond_cd, COUNT(*) n,
         SAFE_DIVIDE(STDDEV(tot_cost_gross), AVG(tot_cost_gross)) cv
  FROM base GROUP BY 1,2,3,4 HAVING n >= 30
)
SELECT flag, COUNT(*) signatures,
       ROUND(APPROX_QUANTILES(cv,2)[OFFSET(1)],3) median_within_sig_cost_cv
FROM sig WHERE cv IS NOT NULL GROUP BY flag
"""


def run(c, sql, name):
    print(f"running {name} ...")
    df = c.query(sql).result().to_dataframe(create_bqstorage_client=False)
    for col in df.columns:
        if df[col].dtype == object and df[col].map(lambda x: isinstance(x, Decimal)).any():
            df[col] = df[col].astype(float)
    df.to_csv(f"{OUT}/{name}.csv", index=False)
    return df


def main():
    c = client()
    canon = run(c, Q_RECALL_CANON, "recall_canonical")
    print(f"  recall Canonical Repairs (n>=30): {len(canon):,}")
    if len(canon):
        print("\n  Top recall repairs (by volume):")
        print(canon.head(10)[["veh_line_desc", "part", "cond_cd", "n", "dealers", "states",
                              "cost_med", "labor_med", "dealer_spread_ratio"]].to_string(index=False))
        valid = canon[canon["dealer_spread_ratio"].notna()]
        if len(valid):
            print(f"\n  median within-recall dealer cost spread: {valid['dealer_spread_ratio'].median():.2f}x")

    uni = run(c, Q_UNIFORMITY, "recall_uniformity")
    print("\n=== Are recalls more uniform than non-recall? (within-signature cost CV) ===")
    print(uni.to_string(index=False))
    if len(uni) == 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(uni["flag"], uni["median_within_sig_cost_cv"], color=["#2a9d8f", "#e76f51"])
        ax.set_ylabel("median within-signature cost CV (lower = more uniform)")
        ax.set_title("Recall vs non-recall: repair-cost uniformity")
        for i, v in enumerate(uni["median_within_sig_cost_cv"]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
        fig.tight_layout(); fig.savefig(f"{OUT}/fig_recall_uniformity.png", dpi=120); plt.close(fig)
        print(f"\n  wrote {OUT}/fig_recall_uniformity.png")


if __name__ == "__main__":
    main()
