"""
Phase 12 - Group cost-savings: the grounded, data-derived $ value of converging each repair group
to its Canonical Repair consensus cost.

For every SINGLE-repair golden signature (where one consensus cost is meaningful), we measure the
avoidable overspend = sum over member claims of max(0, claim_cost - canonical_cost). Summed across
the library this is a conservative, bottom-up savings number that complements the model-uplift
business case in the README.

Outputs:
  grouping/out/group_savings.csv     - per-signature savings, ranked
  grouping/out/group_savings_headline.txt
  grouping/out/fig_group_savings.png + figures/fig_group_savings.png

Run:  python grouping/12_group_savings.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C  # noqa: E402
from src.agents.cost_savings_agent import CostSavingsAgent  # noqa: E402

OUT_CSV = C.GROUP_SAVINGS
OUT_HEADLINE = "grouping/out/group_savings_headline.txt"
OUT_FIG = "grouping/out/fig_group_savings.png"
OUT_FIG2 = "figures/fig_group_savings.png"


def main():
    gs = pd.read_csv(C.GOLDEN_SOLUTIONS)
    single = gs[gs["n_repairs_detected"] == 1].dropna(subset=["cost_med"]).copy()
    canon = single.set_index("signature_id")["cost_med"].to_dict()
    meta = single.set_index("signature_id")[["vehicle_line", "causal_part", "repair_name", "support"]]

    sig = pd.read_parquet(C.CLAIM_SIGNATURES, columns=["request_r", "signature_id"])
    sig["request_r"] = pd.to_numeric(sig["request_r"], errors="coerce").astype("Int64")
    sig = sig[sig["signature_id"].isin(canon)]

    rp = pd.read_parquet(C.REPAIR_PROFILE, columns=["request_r", C.COST_COL])
    rp["request_r"] = pd.to_numeric(rp["request_r"], errors="coerce").astype("Int64")
    rp[C.COST_COL] = pd.to_numeric(rp[C.COST_COL], errors="coerce")

    claims = sig.merge(rp, on="request_r", how="inner")
    claims = claims[(claims[C.COST_COL] > 0) & (claims[C.COST_COL] < C.COST_CAP)]

    rows = []
    for sid, grp in claims.groupby("signature_id"):
        m = CostSavingsAgent.compute(grp[C.COST_COL], canon[sid])
        info = meta.loc[sid]
        rows.append({
            "signature_id": sid, "vehicle_line": info["vehicle_line"],
            "causal_part": info["causal_part"], "repair_name": info["repair_name"],
            "canonical_cost": round(canon[sid], 2), **m})

    out = pd.DataFrame(rows).sort_values("avoidable_overspend", ascending=False)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    total = out["avoidable_overspend"].sum()
    total_spend = out["total_spend"].sum()
    n_claims = int(out["n_claims"].sum())
    headline = (
        f"Group cost-savings (bottom-up, {len(out)} single-repair canonical repairs):\n"
        f"  claims analyzed          : {n_claims:,}\n"
        f"  total historical spend   : ${total_spend/1e6:,.1f}M\n"
        f"  avoidable overspend      : ${total/1e6:,.1f}M  "
        f"({100*total/total_spend:.1f}% of spend above consensus)\n"
        f"  avg savings per claim    : ${total/max(n_claims,1):,.0f}\n"
        f"  top group ({out.iloc[0]['vehicle_line']} / {out.iloc[0]['repair_name']}): "
        f"${out.iloc[0]['avoidable_overspend']/1e6:,.2f}M\n")
    with open(OUT_HEADLINE, "w") as fh:
        fh.write(headline)
    print(headline)

    top = out.head(12).iloc[::-1]
    labels = [f"{r.vehicle_line.split(']')[0].split('[')[0].strip()[:18]} - {str(r.repair_name)[:24]}"
              for r in top.itertuples()]
    plt.figure(figsize=(10, 6))
    plt.barh(labels, top["avoidable_overspend"] / 1e6)
    plt.xlabel("Avoidable overspend ($M, claims above canonical consensus cost)")
    plt.title("Top repair groups by data-derived savings opportunity")
    plt.tight_layout()
    for p in (OUT_FIG, OUT_FIG2):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        plt.savefig(p, dpi=120)
    print(f"Wrote {OUT_CSV}, {OUT_HEADLINE}, {OUT_FIG}")


if __name__ == "__main__":
    main()
