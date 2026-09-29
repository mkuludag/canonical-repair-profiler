#!/usr/bin/env python3
"""H4 / A7 — overspend baseline sensitivity curve at p50 / p60 / p75.

Area-chair item H4: "Removes the strongest IAAI objection and retires
'addressable ceiling' as an overclaim." Ruled in, never run, untracked.

The shipped figure sums, per single-repair canonical group,
    avoidable_overspend = SUM_claims max(0, claim_cost - consensus_median)
Because the baseline is the median, ~half of each group's claims sit above it by
construction. The question this answers: how fast does the total decay as the
baseline moves up the group's own cost distribution?

BASIS SAFETY — this reads paws_discovery/repair_profile.parquet, which is the REAL
basis, not the rescaled public basis that eval/out/ and the paper use. This script
therefore prints ONLY shares and ratios, which are basis-independent, plus dollar
figures derived from the already-published public-basis total. No real-basis dollar
value is printed or written. Nothing is written to eval/out/.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, f"{REPO}/src")
import config as C  # noqa: E402

PUBLISHED_TOTAL_OVERSPEND = 96_823_942.82   # public basis, eval/out/a5_savings_summary.json
PUBLISHED_TOTAL_SPEND = 482_009_419.89      # public basis
LEVELS = [0.50, 0.55, 0.60, 0.65, 0.75]


def main():
    gs = pd.read_csv(f"{REPO}/grouping/out_v2_public/group_savings.csv")
    gs["signature_id"] = pd.to_numeric(gs["signature_id"], errors="coerce").astype("Int64")
    wanted = set(gs["signature_id"].dropna().astype(int))
    print(f"groups in group_savings.csv: {len(gs)}   claims claimed: {int(gs['n_claims'].sum()):,}")

    sig = pd.read_parquet(f"{REPO}/data/claim_signatures.parquet",
                          columns=["request_r", "signature_id"])
    sig["request_r"] = pd.to_numeric(sig["request_r"], errors="coerce").astype("Int64")
    sig = sig[sig["signature_id"].isin(wanted)]

    rp = pd.read_parquet(f"{REPO}/paws_discovery/repair_profile.parquet",
                         columns=["request_r", C.COST_COL])
    rp["request_r"] = pd.to_numeric(rp["request_r"], errors="coerce").astype("Int64")

    claims = sig.merge(rp, on="request_r", how="inner")
    claims[C.COST_COL] = pd.to_numeric(claims[C.COST_COL], errors="coerce")
    # same validity filter the pipeline applies (grouping_agent.py)
    claims.loc[(claims[C.COST_COL] <= 0) | (claims[C.COST_COL] >= C.COST_CAP), C.COST_COL] = np.nan
    claims = claims.dropna(subset=[C.COST_COL])
    print(f"joined valid-cost claims:   {len(claims):,}")

    g = claims.groupby("signature_id")[C.COST_COL]
    rows = []
    for q in LEVELS:
        base = g.transform(lambda s, q=q: s.quantile(q))
        over = (claims[C.COST_COL] - base).clip(lower=0)
        tot = float(over.sum())
        n_above = int((over > 0).sum())
        rows.append({
            "baseline": f"p{int(q*100)}",
            "total_raw": tot,
            "n_claims_above": n_above,
            "pct_claims_above": 100.0 * n_above / len(claims),
        })

    base_total = rows[0]["total_raw"]          # our own p50 reconstruction
    total_spend_raw = float(claims[C.COST_COL].sum())

    # Validation: does our p50 reconstruction reproduce the shipped share of spend?
    our_share = base_total / total_spend_raw
    pub_share = PUBLISHED_TOTAL_OVERSPEND / PUBLISHED_TOTAL_SPEND
    print(f"\nVALIDATION  reconstructed p50 share of spend = {our_share:.4f}")
    print(f"            published    p50 share of spend = {pub_share:.4f}"
          f"   (delta {abs(our_share-pub_share)*100:.2f} pp)")

    print(f"\n{'baseline':>9}  {'share of spend':>15}  {'ratio to p50':>13}"
          f"  {'public-basis $':>15}  {'% claims above':>15}")
    for r in rows:
        ratio = r["total_raw"] / base_total
        share = r["total_raw"] / total_spend_raw
        pub_dollars = PUBLISHED_TOTAL_OVERSPEND * ratio     # stays on the public basis
        print(f"{r['baseline']:>9}  {share*100:>14.1f}%  {ratio:>13.3f}"
              f"  ${pub_dollars/1e6:>13.1f}M  {r['pct_claims_above']:>14.1f}%")

    print("\n(dollar column is the PUBLIC basis, derived from the published $96.8M by ratio;"
          "\n no real-basis dollar value is printed.)")

    # Artifact for the item-H fact-check: every number printed in Sec. 5.8 / Sec. 7 must trace
    # to a file under eval/out/. NEW filename -- touches no existing artifact and no _DATA_ROOT.
    out = pd.DataFrame([{
        "baseline": r["baseline"],
        "share_of_spend": r["total_raw"] / total_spend_raw,
        "ratio_to_p50": r["total_raw"] / base_total,
        "public_basis_overspend": PUBLISHED_TOTAL_OVERSPEND * (r["total_raw"] / base_total),
        "pct_claims_above_pooled": r["pct_claims_above"],
    } for r in rows])
    dest = f"{REPO}/eval/out/h4_overspend_curve.csv"
    out.to_csv(dest, index=False)
    print(f"\nwrote {dest}")
    print("  (basis-independent columns plus public-basis dollars derived by ratio;"
          " no real-basis value stored)")


if __name__ == "__main__":
    main()
