#!/usr/bin/env python3
"""
Broad-population REGION + RECALL analysis on GSAR clm_master (millions of warranty claims),
which carries rpr_region_cd / rpr_st_prov_cd + camp_recall_flag + classification + cost.
Runs in BigQuery (one scan) via the user's gcloud token (ADC is blocked by Ford CAA),
pulls compact aggregates locally, writes CSVs. Memory-light: only aggregates come back.

Outputs:
  grouping/out/region_cost_by_signature.csv  region cost spread for the SAME repair signature
  grouping/out/region_overall.csv            per-region overall cost/labor
  grouping/out/recall_vs_nonrecall_gsar.csv  recall treatment across the broad population
"""
import os, subprocess
from google.cloud import bigquery
from google.oauth2.credentials import Credentials

BILLING = "REDACTED-BILLING-PROJECT"
SRC = "REDACTED-SOURCE-PROJECT.dataset_fdp.claims_master_view"
OUT = "grouping/out"; os.makedirs(OUT, exist_ok=True)
WIN = "rpr_dt BETWEEN DATE '2022-01-01' AND CURRENT_DATE()"
# US-only, plausible-cost guard, real causal parts (drop maintenance / unknown)
US = ("rpr_cntry_cd='USA' AND rpr_st_prov_cd IS NOT NULL "
      "AND tot_cost_gross BETWEEN 50 AND 100000 "
      "AND prt_num_causl_base_cd NOT IN ('MAINT','*','')")


def client():
    tok = subprocess.check_output(
        ["gcloud", "auth", "print-access-token"],
        env={**os.environ, "CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT": ""}).decode().strip()
    return bigquery.Client(project=BILLING, credentials=Credentials(tok))


# (1) US-STATE cost variation for the SAME repair signature (veh line + causal part + condition).
#     Signature must appear in >=5 states with >=30 claims each, so comparisons are fair.
Q_REGION_SIG = f"""
WITH per_state AS (
  SELECT veh_line_desc, prt_num_causl_base_cd, cond_cd, rpr_st_prov_cd,
         COUNT(*) n,
         APPROX_QUANTILES(tot_cost_gross, 2)[OFFSET(1)] med_cost,
         APPROX_QUANTILES(lbr_hrs, 2)[OFFSET(1)] med_hrs
  FROM `{SRC}`
  WHERE {WIN} AND {US} AND veh_line_desc IS NOT NULL AND cond_cd IS NOT NULL
  GROUP BY 1,2,3,4
  HAVING n >= 30
)
SELECT veh_line_desc, prt_num_causl_base_cd, cond_cd,
       COUNT(*) states, SUM(n) claims,
       ROUND(MIN(med_cost),2) cost_min, ROUND(MAX(med_cost),2) cost_max,
       ROUND(MAX(med_cost)/NULLIF(MIN(med_cost),0),2) cost_ratio,
       ROUND(MIN(med_hrs),2) hrs_min, ROUND(MAX(med_hrs),2) hrs_max
FROM per_state
GROUP BY 1,2,3
HAVING states >= 5
ORDER BY cost_ratio DESC
LIMIT 200
"""

# (2) Overall per-US-state cost/labor (do states differ systematically?).
Q_REGION_OVERALL = f"""
SELECT rpr_st_prov_cd state,
       COUNT(*) claims,
       ROUND(APPROX_QUANTILES(tot_cost_gross,2)[OFFSET(1)],2) med_cost,
       ROUND(AVG(tot_cost_gross),2) mean_cost,
       ROUND(APPROX_QUANTILES(lbr_hrs,2)[OFFSET(1)],2) med_labor_hrs,
       ROUND(APPROX_QUANTILES(lbr_cost,2)[OFFSET(1)],2) med_labor_cost,
       ROUND(APPROX_QUANTILES(mtrl_cost,2)[OFFSET(1)],2) med_material
FROM `{SRC}`
WHERE {WIN} AND {US}
GROUP BY 1 HAVING claims >= 1000 ORDER BY med_cost DESC
"""

# (3) Recall vs non-recall treatment across the broad GSAR population.
Q_RECALL = f"""
SELECT
  CASE WHEN UPPER(camp_recall_flag)='Y' THEN 'recall' ELSE 'non_recall' END recall,
  COUNT(*) claims,
  ROUND(APPROX_QUANTILES(tot_cost_gross,2)[OFFSET(1)],2) med_cost,
  ROUND(AVG(tot_cost_gross),2) mean_cost,
  ROUND(APPROX_QUANTILES(lbr_hrs,2)[OFFSET(1)],2) med_labor_hrs,
  ROUND(APPROX_QUANTILES(mtrl_cost,2)[OFFSET(1)],2) med_material
FROM `{SRC}`
WHERE {WIN} AND tot_cost_gross > 0
GROUP BY 1
"""


def run(c, sql, name):
    print(f"running {name} ...")
    df = c.query(sql).result().to_dataframe(create_bqstorage_client=False)
    df.to_csv(f"{OUT}/{name}.csv", index=False)
    print(df.to_string(index=False)[:3000], "\n")
    return df


def main():
    c = client()
    ov = run(c, Q_REGION_OVERALL, "region_overall")
    if len(ov):
        print(f"  US states: {len(ov)} | med_cost range ${ov.med_cost.min():.0f}-${ov.med_cost.max():.0f} "
              f"({ov.med_cost.max()/max(ov.med_cost.min(),1):.2f}x)\n")
    run(c, Q_RECALL, "recall_vs_nonrecall_gsar")
    sig = run(c, Q_REGION_SIG, "region_cost_by_signature")
    print(f"Top US-state cost-disparity signatures written: {len(sig)} rows")


if __name__ == "__main__":
    main()
