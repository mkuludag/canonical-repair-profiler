#!/usr/bin/env python3
"""
Idea #6: within a STATE, for the SAME repair signature (vehicle line + causal part + condition),
how much do DEALERS diverge from the consensus?  These are the actionable discrepancies -- dealers
to nudge toward the group ground truth. Runs on GSAR clm_master in BigQuery (raw dealer + state).

Outputs:
  grouping/out/dealer_spread_by_state_signature.csv   per (state,signature): dealer-median spread
  grouping/out/dealer_outliers_examples.csv           worst individual dealer outliers
  grouping/out/fig_dealer_spread.png                  distribution of within-state dealer spread
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
GUARD = ("rpr_cntry_cd='USA' AND rpr_st_prov_cd IS NOT NULL AND rpr_dlr_cd IS NOT NULL "
         "AND tot_cost_gross BETWEEN 50 AND 100000 AND prt_num_causl_base_cd NOT IN ('MAINT','*','') "
         "AND rpr_dt BETWEEN DATE '2022-01-01' AND CURRENT_DATE()")


def client():
    tok = subprocess.check_output(
        ["gcloud", "auth", "print-access-token"],
        env={**os.environ, "CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT": ""}).decode().strip()
    return bigquery.Client(project=BILLING, credentials=Credentials(tok))


# within-state dealer spread for the same repair signature
Q_SPREAD = f"""
WITH dealer_cell AS (
  SELECT rpr_st_prov_cd state, veh_line_desc, prt_num_causl_base_cd part, cond_cd, rpr_dlr_cd dealer,
         COUNT(*) n, APPROX_QUANTILES(tot_cost_gross,2)[OFFSET(1)] dealer_med
  FROM `{SRC}` WHERE {GUARD}
  GROUP BY 1,2,3,4,5 HAVING n >= 10
)
SELECT state, veh_line_desc, part, cond_cd,
       COUNT(*) dealers, SUM(n) claims,
       ROUND(APPROX_QUANTILES(dealer_med,2)[OFFSET(1)],2) consensus_med,
       ROUND(MIN(dealer_med),2) lo, ROUND(MAX(dealer_med),2) hi,
       ROUND(MAX(dealer_med)/NULLIF(MIN(dealer_med),0),2) dealer_spread_ratio
FROM dealer_cell
GROUP BY 1,2,3,4 HAVING dealers >= 5
ORDER BY dealer_spread_ratio DESC
"""

# individual dealer outliers vs their state+signature consensus (for the busiest cells)
Q_OUTLIERS = f"""
WITH dealer_cell AS (
  SELECT rpr_st_prov_cd state, veh_line_desc, prt_num_causl_base_cd part, cond_cd, rpr_dlr_cd dealer,
         COUNT(*) n, APPROX_QUANTILES(tot_cost_gross,2)[OFFSET(1)] dealer_med
  FROM `{SRC}` WHERE {GUARD}
  GROUP BY 1,2,3,4,5 HAVING n >= 15
),
cell AS (
  SELECT state, veh_line_desc, part, cond_cd,
         APPROX_QUANTILES(dealer_med,2)[OFFSET(1)] consensus_med, COUNT(*) dealers, SUM(n) claims
  FROM dealer_cell GROUP BY 1,2,3,4 HAVING dealers >= 8
)
SELECT d.state, d.veh_line_desc, d.part, d.cond_cd, d.dealer, d.n,
       ROUND(d.dealer_med,2) dealer_med, ROUND(c.consensus_med,2) consensus_med,
       ROUND(d.dealer_med/NULLIF(c.consensus_med,0),2) vs_consensus, c.claims cell_claims
FROM dealer_cell d JOIN cell c USING (state, veh_line_desc, part, cond_cd)
WHERE d.dealer_med/NULLIF(c.consensus_med,0) >= 1.8
ORDER BY c.claims DESC, vs_consensus DESC
LIMIT 300
"""


def run(c, sql, name):
    print(f"running {name} ...")
    df = c.query(sql).result().to_dataframe(create_bqstorage_client=False)
    # BigQuery NUMERIC -> Decimal objects; cast to float for local math/plots
    for col in df.columns:
        if df[col].dtype == object and df[col].map(lambda x: isinstance(x, Decimal)).any():
            df[col] = df[col].astype(float)
    df.to_csv(f"{OUT}/{name}.csv", index=False)
    return df


def main():
    c = client()
    spread = run(c, Q_SPREAD, "dealer_spread_by_state_signature")
    print(f"  state x signature cells (>=5 dealers, >=10 claims/dealer): {len(spread):,}")
    if len(spread):
        med = spread["dealer_spread_ratio"].median()
        p90 = spread["dealer_spread_ratio"].quantile(0.9)
        print(f"  within-state dealer cost spread (max/min dealer median): median {med:.2f}x, p90 {p90:.2f}x")
        print("\n  Top within-state dealer disparities (same repair, same state):")
        print(spread.head(10).to_string(index=False))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(spread["dealer_spread_ratio"].clip(upper=6), bins=40, color="#d1495b")
        ax.axvline(med, ls="--", c="k", label=f"median {med:.2f}x")
        ax.set_xlabel("within-state dealer cost spread (max/min dealer median, same repair)")
        ax.set_ylabel("# state x repair cells"); ax.legend()
        ax.set_title("Same repair, same state: how much do dealers diverge?")
        fig.tight_layout(); fig.savefig(f"{OUT}/fig_dealer_spread.png", dpi=120); plt.close(fig)
        print(f"\n  wrote {OUT}/fig_dealer_spread.png")

    outl = run(c, Q_OUTLIERS, "dealer_outliers_examples")
    print(f"\n  dealer outliers (>=1.8x consensus, busy cells): {len(outl):,}")
    if len(outl):
        print(outl.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
