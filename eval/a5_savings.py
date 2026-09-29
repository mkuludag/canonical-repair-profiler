#!/usr/bin/env python3
"""
a5 - Is the measured overspend broad-based, or is it concentrated tail mass?

The headline savings figure sums, over every single-repair canonical group,
    avoidable_overspend = SUM over claims of max(0, claim_cost - consensus_median)
(cost_savings_agent.py:34-42). Because the baseline is the *median*, roughly half of each group's
claims sit above it by construction; what the total actually measures is the mass of the right tail.
That makes two properties worth reporting alongside the total:

  - CONCENTRATION: what share of the total comes from the largest groups. If a handful of groups
    carry most of it, the figure is a short, actionable target list rather than a diffuse tax.
  - BREADTH: the distribution of `pct_claims_above`, which should centre near 50% for a median
    baseline. Confirming that is a sanity check on the statistic, not a finding.

NOT COMPUTABLE HERE: the counterfactual of moving the baseline to an upper-quartile consensus. That
needs per-claim costs, which live only in the proprietary claim-level table. It is deferred.

    python eval/a5_savings.py --data-root ~/canonical-repair-profiler-public/grouping/out
"""
import numpy as np
import pandas as pd

import common as C

TOP_SHARES = [0.01, 0.05, 0.10, 0.25, 0.50]


def main() -> None:
    args = C.parse_args(__doc__)
    s = C.load_group_savings(args.data_root).copy()
    for c in ("canonical_cost", "n_claims", "n_above", "avoidable_overspend",
              "savings_per_claim", "pct_claims_above", "total_spend"):
        s[c] = pd.to_numeric(s[c], errors="coerce")

    total_overspend = float(s["avoidable_overspend"].sum())
    total_spend = float(s["total_spend"].sum())
    total_claims = int(s["n_claims"].sum())

    ranked = s.sort_values("avoidable_overspend", ascending=False).reset_index(drop=True)
    ranked["cum_share"] = ranked["avoidable_overspend"].cumsum() / total_overspend
    conc_rows = []
    for q in TOP_SHARES:
        k = max(1, int(np.ceil(q * len(ranked))))
        conc_rows.append({
            "top_fraction_of_groups": q, "n_groups": k,
            "share_of_overspend": float(ranked.loc[:k - 1, "avoidable_overspend"].sum() / total_overspend),
            "share_of_claims": float(ranked.loc[:k - 1, "n_claims"].sum() / total_claims),
        })
    conc = pd.DataFrame(conc_rows)

    def gini(values) -> float:
        """0 = every group contributes equally; 1 = one group carries everything."""
        v = np.sort(np.asarray(values, dtype=float))
        i = np.arange(1, len(v) + 1)
        return float((2 * (i * v).sum()) / (len(v) * v.sum()) - (len(v) + 1) / len(v))

    gini_overspend = gini(ranked["avoidable_overspend"])
    gini_claims = gini(ranked["n_claims"])
    gini_spend = gini(ranked["total_spend"])
    gini_severity = gini(ranked["savings_per_claim"])

    # Is the concentration severity, or merely group size? Spearman via ranks (no scipy dependency).
    rank_corr = lambda a, b: float(a.rank().corr(b.rank()))
    rc_size_total = rank_corr(s["n_claims"], s["avoidable_overspend"])
    rc_size_severity = rank_corr(s["n_claims"], s["savings_per_claim"])
    s["overspend_rate"] = s["avoidable_overspend"] / s["total_spend"]
    rc_size_rate = rank_corr(s["n_claims"], s["overspend_rate"])
    top_k = max(1, int(np.ceil(0.01 * len(ranked))))
    sev_top = float(ranked.head(top_k)["savings_per_claim"].median())
    sev_all = float(s["savings_per_claim"].median())

    C.write_csv(conc, "a5_savings_concentration.csv")
    C.write_csv(ranked.head(25), "a5_savings_top25_groups.csv")

    summary = {
        "n_groups": int(len(s)), "n_claims": total_claims,
        "total_spend": total_spend, "total_overspend": total_overspend,
        "overspend_share_of_spend": total_overspend / total_spend,
        "overspend_per_claim": total_overspend / total_claims,
        "gini_overspend": gini_overspend, "gini_claims": gini_claims,
        "gini_total_spend": gini_spend, "gini_savings_per_claim": gini_severity,
        "rank_corr_size_vs_overspend": rc_size_total,
        "rank_corr_size_vs_severity": rc_size_severity,
        "rank_corr_size_vs_overspend_rate": rc_size_rate,
        "median_severity_top1pct": sev_top, "median_severity_all": sev_all,
        "median_overspend_rate": float(s["overspend_rate"].median()),
        "median_pct_claims_above": float(s["pct_claims_above"].median()),
        "median_savings_per_claim": float(s["savings_per_claim"].median()),
        "top_group_overspend": float(ranked.loc[0, "avoidable_overspend"]),
        "top_group_name": str(ranked.loc[0, "repair_name"]),
        "note": "dollar values follow the data root; the public release rescales all dollars",
    }
    C.write_json(summary, "a5_savings_summary.json")

    conc_lines = "\n".join(
        f"- top {C.pct(r['top_fraction_of_groups'], 0)} of groups ({int(r['n_groups']):,}) carry "
        f"**{C.pct(r['share_of_overspend'])}** of the overspend and {C.pct(r['share_of_claims'])} of claims"
        for _, r in conc.iterrows())

    body = f"""
Across {len(s):,} single-repair canonical groups and {total_claims:,} claims totalling
${total_spend / 1e6:,.1f}M, measured spend above consensus is **${total_overspend / 1e6:,.1f}M**
({C.pct(total_overspend / total_spend)} of spend, ${total_overspend / total_claims:,.0f} per claim).

**Concentration.**

{conc_lines}

The Gini coefficient over group-level overspend is **{gini_overspend:.3f}** and the largest single
group contributes ${summary['top_group_overspend'] / 1e6:,.2f}M, so the total is far from evenly spread.

**But the concentration is group size, not severity.** Overspend is in fact *less* concentrated than
spend itself (Gini {gini_overspend:.3f} vs **{gini_spend:.3f}** for total spend), and only marginally
more concentrated than claim volume ({gini_claims:.3f}). A group's overspend is therefore close to a
fixed fraction of what flows through it. The top 1%
of groups carry {C.pct(conc.iloc[0]['share_of_overspend'])} of the overspend while also carrying
{C.pct(conc.iloc[0]['share_of_claims'])} of the claims. Group size predicts total overspend strongly
(rank correlation {rc_size_total:.3f}) but per-claim severity barely at all ({rc_size_severity:.3f}),
and each group's overspend *as a share of its own spend* is essentially size-independent
({rc_size_rate:.3f}, median {C.pct(summary['median_overspend_rate'])}). The largest groups are only
modestly worse per claim (median ${sev_top:,.0f} vs ${sev_all:,.0f} library-wide).

The honest reading: the ranking is a **volume-weighted work queue** - chase the biggest groups first
because that is where the dollars are - not evidence that a minority of repair types is pathologically
overspent. Per-claim severity is broadly distributed (Gini {gini_severity:.3f}).

**Breadth sanity check.** The median group has {C.pct(summary['median_pct_claims_above'] / 100)} of its
claims above consensus, exactly what a *median* baseline mechanically implies. The total therefore
measures right-tail mass, and its magnitude depends on tail shape rather than on the count of claims
above the line - the framing the draft already adopts.

**Not computed.** Re-basing to an upper-quartile consensus requires per-claim costs and is deferred;
it cannot be derived from the group-level artifact.

Dollar amounts follow the data root. Artifacts: `eval/out/a5_savings_concentration.csv`,
`eval/out/a5_savings_top25_groups.csv`, `eval/out/a5_savings_summary.json`.
"""
    C.append_results("a5 - Savings concentration", body, args.data_root, not args.no_append)

    print(f"\n  overspend ${total_overspend/1e6:,.1f}M of ${total_spend/1e6:,.1f}M "
          f"({C.pct(total_overspend/total_spend)}) | gini {gini_overspend:.3f} "
          f"(claims {gini_claims:.3f}) | size-vs-severity rank corr {rc_size_severity:.3f}")


if __name__ == "__main__":
    main()
