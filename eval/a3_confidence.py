#!/usr/bin/env python3
"""
a3 - Decompose the confidence score into its consensus and support-weight factors.

Two questions, one script.

(1) VERIFICATION. Does the released library actually implement the confidence equation the paper
    prints? stats_tool.py:54-62 computes kappa = consensus * min(1, log10(n+1)/log10(31)), rounded to
    3 dp. Recomputing that from the emitted consensus and support columns and comparing against the
    emitted confidence column is a direct artifact-vs-code check, of exactly the kind a reviewer who
    opens the repo would run.

(2) CORRECTION. The draft states the support weight is "saturated for all but the smallest tiers",
    which would make kappa effectively equal to the consensus fraction. That is testable: the weight
    reaches 1.0 at n >= 30. The GOLD filter requires n >= 10 at the *signature* level, but an honored
    split halves those claims across two tiers, so per-tier support is materially smaller.

    python eval/a3_confidence.py --data-root ~/canonical-repair-profiler-public/grouping/out
"""
import numpy as np
import pandas as pd

import common as C

SATURATION_N = 30  # log10(31)/log10(31) = 1.0


def main() -> None:
    args = C.parse_args(__doc__)
    g = C.load_golden(args.data_root).copy()

    g["support_w"] = C.support_weight(g["support"])
    g["kappa_recomputed"] = (g["consensus"].fillna(0) * g["support_w"]).round(3)
    # stats_tool rounds to 3 dp, so half a unit in the last place is the entire tolerance budget.
    # Anything looser would let a real disagreement pass as a match.
    g["matches_code"] = np.isclose(g["kappa_recomputed"], g["confidence"].fillna(0), atol=5.1e-4)

    # Rows with a null consensus match only because fillna(0) coerces both sides to zero - a vacuous
    # agreement. Report the verification on rows where both quantities actually exist.
    comparable = g["consensus"].notna() & g["confidence"].notna()
    n_comparable = int(comparable.sum())
    n_match_comparable = int((g["matches_code"] & comparable).sum())
    null_rows = g.loc[~comparable, ["signature_id", "cost_tier", "support", "consensus", "confidence"]]

    n = len(g)
    n_match = int(g["matches_code"].sum())
    saturated = g["support"].fillna(0) >= SATURATION_N
    n_sat = int(saturated.sum())

    support = pd.to_numeric(g["support"], errors="coerce")
    by_class = g.assign(split=np.where(g["n_repairs_detected"] == 2, "split tier", "single repair"))
    class_tbl = by_class.groupby("split").agg(
        n_rows=("support", "size"),
        median_support=("support", "median"),
        share_saturated=("support", lambda s: float((pd.to_numeric(s, errors="coerce") >= SATURATION_N).mean())),
        mean_support_w=("support_w", "mean"),
    ).reset_index()

    C.write_csv(g[["signature_id", "cost_tier", "support", "consensus", "confidence",
                   "support_w", "kappa_recomputed", "matches_code"]], "a3_confidence_rows.csv")
    C.write_csv(class_tbl, "a3_confidence_by_split_class.csv")

    summary = {
        "n_rows": n,
        "n_comparable": n_comparable,
        "n_matching_stats_tool_comparable": n_match_comparable,
        "n_null_consensus_or_confidence": int(n - n_comparable),
        "null_signatures": [int(s) for s in null_rows["signature_id"]],
        "n_matching_stats_tool": n_match,
        "n_mismatched": n - n_match,
        "tolerance_atol": 5.1e-4,
        "saturation_threshold_n": SATURATION_N,
        "n_saturated": n_sat,
        "share_saturated": n_sat / n,
        "median_support": float(support.median()),
        "p05_support": float(support.quantile(0.05)),
        "p25_support": float(support.quantile(0.25)),
        "mean_support_w": float(g["support_w"].mean()),
        "min_support_w": float(g["support_w"].min()),
        "median_confidence": float(g["confidence"].median()),
        "median_consensus": float(g["consensus"].median()),
    }
    C.write_json(summary, "a3_confidence_summary.json")

    verdict = ("exactly matches" if n_comparable - n_match_comparable == 0
               else f"MISMATCHES on {n_comparable - n_match_comparable} rows")
    cls = "\n".join(
        f"- {r['split']}: median support {r['median_support']:.0f}, "
        f"{C.pct(r['share_saturated'])} saturated, mean support weight {r['mean_support_w']:.3f}"
        for _, r in class_tbl.iterrows())

    body = f"""
**Verification.** Recomputing kappa = consensus x min(1, log10(n+1)/log10(31)) from the emitted
`consensus` and `support` columns reproduces the emitted `confidence` column on
**{n_match_comparable:,} of the {n_comparable:,} rows where both quantities are non-null**
(tolerance 5e-4, half a unit in stats_tool's last rounded place) - the released library {verdict}
`stats_tool.py:54-62`. The remaining {n - n_comparable} row(s) carry a null consensus *and* a null
confidence (signatures {', '.join(str(s) for s in summary['null_signatures'])}), so they are excluded
rather than counted as agreements: coercing both sides to zero would make them match vacuously.

**Correction to the saturation claim.** The support weight reaches 1.0 only at n >= {SATURATION_N}, and
just **{n_sat:,} of {n:,} emitted repairs ({C.pct(n_sat / n)})** clear that bar. Median support is
**{summary['median_support']:.0f}** claims, the lower quartile is {summary['p25_support']:.0f}, and the
mean support weight is **{summary['mean_support_w']:.3f}** (minimum {summary['min_support_w']:.3f}).
So the support term is not inert - it actively discounts roughly three quarters of the library, and
kappa is *not* interchangeable with the consensus fraction outside the high-support tail.

{cls}

The mechanism is the split: GOLD requires n >= 10 at the *signature* level, but an honored two-way
split partitions those claims across two tiers, so per-tier support is roughly halved.

Artifacts: `eval/out/a3_confidence_rows.csv`, `eval/out/a3_confidence_by_split_class.csv`,
`eval/out/a3_confidence_summary.json`.
"""
    C.append_results("a3 - Confidence decomposition", body, args.data_root, not args.no_append)

    print(f"\n  code match: {n_match_comparable:,}/{n_comparable:,} comparable "
          f"({n - n_comparable} null)   saturated: {n_sat:,} ({C.pct(n_sat / n)})   "
          f"median support: {summary['median_support']:.0f}")


if __name__ == "__main__":
    main()
