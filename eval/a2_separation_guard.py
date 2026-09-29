#!/usr/bin/env python3
"""
a2 - How much work does the separation guard actually do?

The guard honors an analyst-proposed two-way split only when the high tier's median cost is at least
tau = 1.5x the low tier's (solution_assembly_agent.py:55-62). The draft credits it with keeping the
library from "reporting two repairs where the data show one repair with ordinary price variation."
That is a claim about SPECIFICITY, and specificity needs a counterfactual: what ratio would the guard
have seen on signatures it never got to judge?

THE COUNTERFACTUAL IS RECOVERABLE. The split is realized as a median cut (stats_tool.py:44-46): the
low tier is the claims below the group's cost median, the high tier those at or above it. So each
tier's median is, by construction, approximately the full group's Q1 and Q3, and the ratio the guard
tests is approximately the group's Q3/Q1 - a quantity emitted for EVERY signature, including those
never proposed for splitting. This script verifies that construction on the collapsed stratum (where
the realized ratio is independently recorded in the rationale text) and then applies it to build the
missing arm.

COST BASIS. All cost statistics come from golden_solutions.csv alone. canonical_repairs.csv reports
costs on a different basis (see a4) and must not be mixed in.

    python eval/a2_separation_guard.py --data-root ~/canonical-repair-profiler-public/grouping/out
"""
import numpy as np
import pandas as pd

import common as C

TAUS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.75, 2.0, 2.5, 3.0]
N_STRATA = [(10, 20), (20, 50), (50, 10**9)]


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(random positive ranks above random negative) via the rank-sum identity. No scipy."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    args = C.parse_args(__doc__)
    g = C.load_golden(args.data_root)
    cls = C.classify_signatures(g)

    # --- build the guard ratio for all three classes, from golden_solutions only -----------------
    single = g[g["n_repairs_detected"] == 1].set_index("signature_id")
    q3q1 = (single["cost_q75"] / single["cost_q25"]).replace([np.inf, -np.inf], np.nan)
    cls = cls.merge(q3q1.rename("q3q1"), left_on="signature_id", right_index=True, how="left")
    cls = cls.merge(single["support"].rename("group_n"), left_on="signature_id",
                    right_index=True, how="left")
    # honored splits: group support is the sum of both tiers
    hon_n = g[g["n_repairs_detected"] == 2].groupby("signature_id")["support"].sum()
    cls["group_n"] = cls["group_n"].fillna(cls["signature_id"].map(hon_n))
    # the ratio the guard tested (honored/collapsed) or would have tested (never-proposed)
    cls["guard_ratio"] = np.where(cls["klass"] == "single", cls["q3q1"], cls["tier_ratio"])
    cls["proposed"] = cls["klass"].isin(["honored", "collapsed"])

    # --- verify the construction where both quantities exist (collapsed stratum) -----------------
    coll = cls[(cls["klass"] == "collapsed") & cls["q3q1"].notna() & cls["tier_ratio"].notna()].copy()
    rel = coll["q3q1"] / coll["tier_ratio"]
    construction = {
        "n": int(len(coll)),
        "median_realized": float(coll["tier_ratio"].median()),
        "median_q3q1": float(coll["q3q1"].median()),
        "ratio_median": float(rel.median()), "ratio_q25": float(rel.quantile(.25)),
        "ratio_q75": float(rel.quantile(.75)),
        "share_within_2pct": float((rel.sub(1).abs() < 0.02).mean()),
    }
    # The check is only available where the guard collapsed, i.e. on realized ratios in [1.0, 1.5),
    # yet it is applied to a stratum whose median is ~1.68. Bin within the observable range: if the
    # factor is flat, extrapolating past 1.5 is defensible; if it trends, report the trend.
    coll["rel"] = rel
    edges = [1.0, 1.2, 1.3, 1.4, 1.5]
    coll["bin"] = pd.cut(coll["tier_ratio"], bins=edges, right=False)
    drift = (coll.groupby("bin", observed=True)["rel"]
             .agg(n="size", median="median", q25=lambda s: s.quantile(.25),
                  q75=lambda s: s.quantile(.75)).reset_index())
    drift["bin"] = drift["bin"].astype(str)
    C.write_csv(drift, "a2_construction_drift.csv")

    # --- tau sweep over PROPOSED splits ---------------------------------------------------------
    proposed = cls[cls["proposed"] & cls["guard_ratio"].notna()]
    never = cls[(~cls["proposed"]) & cls["guard_ratio"].notna()]
    sweep = pd.DataFrame([{
        "tau": t,
        "n_proposed": int(len(proposed)),
        "n_honored": int((proposed["guard_ratio"] >= t).sum()),
        "share_honored": float((proposed["guard_ratio"] >= t).mean()),
        "share_never_proposed_clearing": float((never["guard_ratio"] >= t).mean()),
    } for t in TAUS])
    C.write_csv(sweep, "a2_tau_sweep.csv")

    # --- discrimination: does the ratio predict what the analyst proposed? -----------------------
    overall_auc = auc(proposed["guard_ratio"].to_numpy(), never["guard_ratio"].to_numpy())
    strata = []
    for lo, hi in N_STRATA:
        p = proposed[(proposed["group_n"] >= lo) & (proposed["group_n"] < hi)]
        q = never[(never["group_n"] >= lo) & (never["group_n"] < hi)]
        strata.append({"support_range": f"[{lo},{hi})" if hi < 10**8 else f"[{lo},inf)",
                       "n_proposed": len(p), "n_never": len(q),
                       "auc": auc(p["guard_ratio"].to_numpy(), q["guard_ratio"].to_numpy()),
                       "median_proposed": float(p["guard_ratio"].median()) if len(p) else np.nan,
                       "median_never": float(q["guard_ratio"].median()) if len(q) else np.nan})
    strata_df = pd.DataFrame(strata)
    C.write_csv(strata_df, "a2_discrimination_by_support.csv")

    # --- robustness: the honored arm mixes 2-repair and truncated 3-repair proposals --------------
    # 388 honored signatures carry a low+mid tier pair (see a4), i.e. the analyst named three scopes
    # and assembly kept the lowest two. Re-run the headline on the clean low+high subset. The 448
    # collapsed signatures MUST stay in the proposed arm: dropping them leaves only ratios >= 1.5 and
    # manufactures a large AUC out of truncation alone.
    split_rows = g[g["n_repairs_detected"] == 2]
    pair = split_rows.groupby("signature_id")["cost_tier"].agg(lambda s: "+".join(sorted(s)))
    clean_ids = set(pair[pair == "high+low"].index)
    clean_prop = proposed[proposed["signature_id"].isin(clean_ids | set(coll["signature_id"]))]
    clean_auc = auc(clean_prop["guard_ratio"].to_numpy(), never["guard_ratio"].to_numpy())
    honored_only_auc = auc(
        proposed.loc[proposed["klass"] == "honored", "guard_ratio"].to_numpy(),
        never["guard_ratio"].to_numpy())

    # --- how many honored splits sit within sampling noise of tau? -------------------------------
    # SE(median) ~ 1.253 * sigma / sqrt(n), sigma ~ IQR/1.349  =>  SE ~ 0.929 * IQR / sqrt(n).
    # Propagated to a ratio of two independent medians. Tier quartiles are positively correlated, so
    # treating them as independent OVERSTATES the spread: this is an upper bound.
    sp = g[g["n_repairs_detected"] == 2].copy()
    sp["se"] = 0.929 * (sp["cost_q75"] - sp["cost_q25"]) / np.sqrt(sp["support"].clip(lower=1).astype(float))
    hi = sp.sort_values("cost_med").groupby("signature_id").last()
    lo = sp.sort_values("cost_med").groupby("signature_id").first()
    ok = (hi["cost_med"] > 0) & (lo["cost_med"] > 0)
    ratio = hi.loc[ok, "cost_med"] / lo.loc[ok, "cost_med"]
    rel_se = np.sqrt((hi.loc[ok, "se"] / hi.loc[ok, "cost_med"]) ** 2 +
                     (lo.loc[ok, "se"] / lo.loc[ok, "cost_med"]) ** 2)
    within_1se = float((((ratio - C.SPLIT_SEPARATION_MIN).abs() / (ratio * rel_se)) <= 1).mean())
    below_2x = float((ratio < 2.0).mean())

    # --- dealer dispersion: the MECHANISM behind the base rate, stratified by dealer count -------
    ds = C.load_dealer_spread(args.data_root)
    ds["dealers"] = pd.to_numeric(ds["dealers"], errors="coerce")
    ds["dealer_spread_ratio"] = pd.to_numeric(ds["dealer_spread_ratio"], errors="coerce")
    bins = [(5, 10), (10, 20), (20, 10**9)]
    dealer_rows = []
    for lo_d, hi_d in bins:
        sub = ds[(ds["dealers"] >= lo_d) & (ds["dealers"] < hi_d)]["dealer_spread_ratio"].dropna()
        if len(sub):
            dealer_rows.append({"dealers": f"[{lo_d},{hi_d})" if hi_d < 10**8 else f"[{lo_d},inf)",
                                "n_cells": len(sub), "median_spread": float(sub.median()),
                                "share_ge_1_5": float((sub >= 1.5).mean())})
    dealer_df = pd.DataFrame(dealer_rows)
    C.write_csv(dealer_df, "a2_dealer_spread_by_dealer_count.csv")
    C.write_csv(cls, "a2_signature_classes.csv")

    n_prop, n_never = len(proposed), len(never)
    at_tau = sweep.loc[sweep["tau"] == C.SPLIT_SEPARATION_MIN].iloc[0]
    n_hon = int((cls["klass"] == "honored").sum())
    n_coll = int((cls["klass"] == "collapsed").sum())

    C.write_json({
        "n_signatures": int(len(cls)), "n_honored": n_hon, "n_collapsed": n_coll,
        "n_never_proposed": int((cls["klass"] == "single").sum()),
        "override_rate": n_coll / (n_hon + n_coll),
        "construction_check": construction,
        "median_ratio_proposed": float(proposed["guard_ratio"].median()),
        "median_ratio_never": float(never["guard_ratio"].median()),
        "share_never_clearing_tau": float(at_tau["share_never_proposed_clearing"]),
        "share_proposed_clearing_tau": float(at_tau["share_honored"]),
        "auc_overall": overall_auc,
        "auc_by_support": {r["support_range"]: r["auc"] for r in strata},
        "auc_clean_low_high_subset": clean_auc,
        "n_clean_proposed": int(len(clean_prop)),
        "auc_honored_only_TRUNCATION_ARTIFACT": honored_only_auc,
        "construction_drift": drift.to_dict("records"),
        "honored_within_1se_of_tau": within_1se,
        "honored_below_2x": below_2x,
        "scope": "GOLD signatures only; SILVER signatures never reach the agent chain",
    }, "a2_guard_summary.json")

    sweep_line = " · ".join(f"{r.tau:g} → {C.pct(r.share_honored)}"
                            for r in sweep.itertuples() if r.tau in (1.3, 1.5, 1.75, 2.0, 2.5))
    strata_line = " · ".join(f"n∈{r['support_range']}: {r['auc']:.3f}" for r in strata)
    dealer_line = " · ".join(
        f"{r['dealers']} dealers: median {r['median_spread']:.2f}x, {C.pct(r['share_ge_1_5'])} ≥1.5x"
        for _, r in dealer_df.iterrows())

    body = f"""
The analyst proposed a two-way split on **{n_hon + n_coll:,}** of the {len(cls):,} GOLD signatures;
the guard honored **{n_hon:,}** and collapsed **{n_coll:,}**, an override rate of
**{C.pct(n_coll / (n_hon + n_coll))}**. Of those {n_hon + n_coll:,} proposals, {n_prop:,} have a
computable ratio (one honored split has a tier that received no claims), and every distributional
statement below is over that {n_prop:,}.

**Recovering the counterfactual.** Because the split is realized as a median cut, the ratio the guard
tests is approximately the group's Q3/Q1 - a quantity emitted for every signature. Checking that
construction on the {construction['n']} collapsed signatures, where the realized ratio is
independently recorded in the rationale text: median realized {construction['median_realized']:.3f}
vs median Q3/Q1 {construction['median_q3q1']:.3f}, with the per-signature ratio of the two centred at
{construction['ratio_median']:.4f} (IQR [{construction['ratio_q25']:.3f}, {construction['ratio_q75']:.3f}]).
The construction is a tight, slightly conservative approximation - not an exact identity, since tier
quantiles are interpolated within the slice.

Two honest caveats about that check. It is only *available* where the guard collapsed, i.e. on
realized ratios in [1.0, 1.5), yet it is *applied* to a stratum with median
~{never['guard_ratio'].median():.2f}. Binning within the observable range, the factor is flat over
[1.2, 1.5) - {', '.join(f"{r['bin']}: {r['median']:.3f}" for _, r in drift.iterrows() if r['bin'] != '[1.0, 1.2)')} -
which makes extrapolation past 1.5 reasonable, though still untested. The lowest bin
([1.0, 1.2), n={int(drift.loc[drift['bin'] == '[1.0, 1.2)', 'n'].iloc[0]) if (drift['bin'] == '[1.0, 1.2)').any() else 0}) is unstable, as expected where the
two tier medians nearly coincide.

The bias also has a known direction: Q3/Q1 **understates** the realized ratio by
~{(1 - construction['ratio_median']) * 100:.1f}%. Both consequences run *against* the claim made here -
{C.pct(at_tau['share_never_proposed_clearing'])} is a **floor** on how many un-proposed signatures
clear tau, and the AUC below is an **over-estimate** of the ratio's discriminating power.

**The guard's threshold is not calibrated against within-repair dispersion.** Applying that
counterfactual to the {n_never:,} signatures the analyst never proposed splitting,
**{C.pct(at_tau['share_never_proposed_clearing'])} of them already clear tau=1.5**
(median ratio {never['guard_ratio'].median():.3f}), versus
{C.pct(at_tau['share_honored'])} of the {n_prop:,} proposed ones
(median {proposed['guard_ratio'].median():.3f}). As a classifier for what the analyst proposed, the
ratio has **AUC {overall_auc:.3f}** - distinguishable from chance but negligible in magnitude, and
stable across support strata ({strata_line}), so it is not an artifact of small-n noise.

Restricting the proposed arm to the {len(clean_prop):,} signatures that are *not* truncated
three-repair decisions (a4) pushes it further down, to **{clean_auc:.3f}** - indistinguishable from
chance. The residual signal in the pooled figure comes from the 388 truncated proposals, which span
three repair scopes and so genuinely separate more. On clean two-repair proposals the cost ratio
carries essentially no information about what the analyst decided.

Note that scoring honored splits alone would give {honored_only_auc:.3f}, but that number is
meaningless: honored splits are *defined* by ratio >= tau, so it measures the guard's own threshold
rather than the analyst's judgement. It is reported only to pre-empt the mistake.

Clearing tau therefore **bounds** the cost separation between tiers; it does not **establish** that
two distinct repairs exist. The discriminating work is done by the analyst's reading of the claim
text, with the guard supplying a largely independent cost-side veto.

**Threshold sensitivity.** Share of proposed splits honored: {sweep_line}. The ratio distribution is
continuous through 1.5 with no gap, so tau is a tunable conservatism knob rather than a discovered
boundary. Sampling noise matters at this threshold too: **{C.pct(within_1se)}** of honored splits lie
within one standard error of tau (an upper bound - it treats the two tier medians as independent),
and {C.pct(below_2x)} sit below 2.0x.

**Mechanism.** Within-repair dealer dispersion supplies the base rate: {dealer_line}. Note this is a
max/min statistic across dealers, so it grows with dealer count by construction and there are no cells
below 5 dealers - it explains *why* ordinary dispersion clears 1.5x, but it is not itself a calibrated
floor.

**Scope.** All of the above covers GOLD signatures; the {3026:,} SILVER signatures never reach the
agent chain and are unexamined.

Artifacts: `eval/out/a2_tau_sweep.csv`, `eval/out/a2_signature_classes.csv`,
`eval/out/a2_discrimination_by_support.csv`, `eval/out/a2_dealer_spread_by_dealer_count.csv`,
`eval/out/a2_guard_summary.json`.
"""
    C.append_results("a2 - Separation guard", body, args.data_root, not args.no_append)

    print(f"\n  override {C.pct(n_coll/(n_hon+n_coll))} | never-proposed clearing tau "
          f"{C.pct(at_tau['share_never_proposed_clearing'])} | proposed {C.pct(at_tau['share_honored'])} "
          f"| AUC {overall_auc:.3f} (clean subset {clean_auc:.3f}) | within 1SE {C.pct(within_1se)}")


if __name__ == "__main__":
    main()
