#!/usr/bin/env python3
"""
a4 - Audit the emitted library for degenerate rows and structural artifacts.

The paper reports 99.98% of Canonical Repairs as "cost-backed" and describes every split as a
two-repair decision. Both deserve an audit, because the pipeline degrades gracefully rather than
crashing, and graceful degradation leaves traces that a coverage statistic hides:

  - fallback shells      - rows whose LLM text is empty (repair_analyst_agent.py:61-63)
  - unusable rows        - rows with no cost median (usable = bool(cost_med))
  - empty tiers          - rows whose cost-tier slice received zero claims, which happens when a
                           median cut lands on a mass of tied costs (stats_tool.py:44-46)
  - truncated splits     - the batch prompt behind the released run allowed up to THREE repairs, but
                           SolutionAssemblyAgent.assemble() keeps repairs[:2]. A surviving "mid" tier
                           label is the fingerprint: the LLM ranked three scopes, the pipeline kept
                           the lowest two and dropped the most expensive one, then recorded
                           n_repairs_detected = 2.
  - mislabeled tiers     - rows carrying a "low"/"high" tier label while reporting a single repair.
                           cost_tier_partition returns the whole group when n_repairs <= 1, so their
                           statistics are full-group and correct; only the label is stale.
  - cost-basis drift     - canonical_repairs.csv and golden_solutions.csv do not report costs on the
                           same basis. Anything comparing the two must account for it.

It also extracts the single empty-text row in full, because it turns out not to be a degraded call.

    python eval/a4_library_audit.py --data-root ~/canonical-repair-profiler-public/grouping/out
"""
import numpy as np
import pandas as pd

import common as C


def main() -> None:
    args = C.parse_args(__doc__)
    g = C.load_golden(args.data_root)
    cls = C.classify_signatures(g)

    diag = g["unified_diagnosis"].str.strip()
    corr = g["suggested_correction"].str.strip()
    empty_text = g[(diag == "") | (corr == "")]
    both_empty = g[(diag == "") & (corr == "")]
    unusable = g[~g["usable"].astype(bool)]
    empty_tier = g[g["tier_n_claims"].fillna(0) == 0]

    tier_counts = g["cost_tier"].value_counts().to_dict()
    split_rows = g[g["n_repairs_detected"] == 2]
    tier_pairs = (split_rows.groupby("signature_id")["cost_tier"]
                  .agg(lambda s: "+".join(sorted(s))).value_counts().to_dict())
    n_split_sigs = int(split_rows["signature_id"].nunique())
    truncated = int(sum(v for k, v in tier_pairs.items() if "mid" in k))

    honored = cls[cls["klass"] == "honored"]
    n_ratio = int(honored["tier_ratio"].notna().sum())

    # Tier labels that survive on single-repair rows: statistics are full-group (correct), label stale.
    single_rows = g[g["n_repairs_detected"] == 1]
    mislabeled = single_rows[single_rows["cost_tier"] != "all"]
    tier_by_nrep = pd.crosstab(g["cost_tier"], g["n_repairs_detected"])

    # Cost-basis check: do the two artifacts agree on a single-repair signature's median cost?
    crl = C.load_crl(args.data_root).set_index("signature_id")
    joined = (single_rows.set_index("signature_id")[["cost_med"]]
              .join(crl[["cost_med"]], rsuffix="_crl", how="inner").dropna())
    same_basis = int(np.isclose(joined["cost_med"], joined["cost_med_crl"], rtol=1e-6).sum())
    basis_ratio = (joined["cost_med_crl"] / joined["cost_med"]).replace([np.inf, -np.inf], np.nan).dropna()

    audit = pd.DataFrame([
        {"finding": "golden solutions emitted", "count": len(g)},
        {"finding": "distinct GOLD signatures", "count": int(g["signature_id"].nunique())},
        {"finding": "empty diagnosis OR correction text", "count": len(empty_text)},
        {"finding": "empty diagnosis AND correction text", "count": len(both_empty)},
        {"finding": "unusable (no cost median)", "count": len(unusable)},
        {"finding": "cost tier received zero claims", "count": len(empty_tier)},
        {"finding": "split signatures (n_repairs_detected=2)", "count": n_split_sigs},
        {"finding": "  of which low+mid (truncated 3-repair decision)", "count": truncated},
        {"finding": "  of which tier ratio computable", "count": n_ratio},
        {"finding": "single-repair rows carrying a stale tier label", "count": len(mislabeled)},
        {"finding": "single-repair signatures where CRL and golden agree on cost_med",
         "count": same_basis},
    ])
    C.write_csv(audit, "a4_library_audit.csv")

    detail_cols = ["signature_id", "vehicle_line", "causal_part", "cost_tier",
                   "n_repairs_detected", "tier_n_claims", "cost_med", "consensus", "confidence",
                   "usable", "repair_name", "unified_diagnosis", "suggested_correction",
                   "split_rationale"]
    degenerate = pd.concat([empty_text, unusable, empty_tier]).drop_duplicates(subset=None)
    C.write_csv(degenerate[detail_cols], "a4_degenerate_rows.csv")

    C.write_json({
        "cost_tier_value_counts": {str(k): int(v) for k, v in tier_counts.items()},
        "cost_tier_by_n_repairs": {str(k): {str(c): int(x) for c, x in v.items()}
                                   for k, v in tier_by_nrep.to_dict("index").items()},
        "split_tier_pairs": {str(k): int(v) for k, v in tier_pairs.items()},
        "n_split_signatures": n_split_sigs,
        "n_truncated_three_repair": truncated,
        "n_tier_ratio_computable": n_ratio,
        "n_single_rows_with_stale_tier_label": int(len(mislabeled)),
        "cost_basis": {
            "n_compared": int(len(joined)), "n_equal": same_basis,
            "share_equal": same_basis / len(joined) if len(joined) else None,
            "crl_over_golden_median": float(basis_ratio.median()),
            "crl_over_golden_p10": float(basis_ratio.quantile(0.10)),
            "crl_over_golden_p90": float(basis_ratio.quantile(0.90)),
        },
    }, "a4_tier_structure.json")

    # The one empty-text row, quoted for the qualitative section.
    abstention = ""
    if len(empty_text) == 1:
        r = empty_text.iloc[0]
        abstention = (
            f"\n**The single empty-text row is an abstention, not a failure.** Signature "
            f"{int(r['signature_id'])} ({r['vehicle_line']}, causal part {r['causal_part']}, "
            f"{int(r['tier_n_claims'])} claims) emitted no diagnosis or correction, but its recorded "
            f"rationale reads: *“{r['split_rationale'].strip()}”* The model declined to invent a "
            f"repair for a group of administrative, zero-cost claims. It is flagged by the critic for "
            f"missing text, which is the correct outcome - but it is a refusal to fabricate, not a "
            f"degraded call.\n")

    pair_lines = "\n".join(f"- `{k}`: {v:,} signatures" for k, v in sorted(tier_pairs.items()))

    body = f"""
Of {len(g):,} emitted Canonical Repairs, **{len(empty_text)}** has empty diagnosis or correction text,
**{len(unusable)}** is unusable for lack of a cost median, and **{len(empty_tier)}** received zero
claims in its cost tier. The empty-text row and the unusable row are *different* rows, so the
degenerate set is smaller than a single statistic implies but touches two distinct failure modes.
{abstention}
**Split structure.** {n_split_sigs:,} signatures were emitted as two repairs, with tier pairs:

{pair_lines}

The **{truncated:,}** `low+mid` signatures ({C.pct(truncated / n_split_sigs)} of splits) are the
fingerprint of a truncated three-repair decision: the batch prompt behind the released run permitted
up to three repairs, `SolutionAssemblyAgent.assemble()` keeps `repairs[:2]`, so the analyst's
highest-cost scope was dropped and the remaining pair recorded as `n_repairs_detected = 2`. The
numbers on those rows are still a clean median-cut band - `cost_tier_partition` routes anything not
`"low"` to the upper half - but the prose attached to the upper band is the analyst's *middle*-scope
text. The agent-side prompt (`src/prompts/split_and_diagnose.txt`) caps at two repairs, so a re-run
through the agent chain does not reproduce this.

**Reconciliation.** {n_split_sigs:,} split signatures but only {n_ratio:,} have a computable tier
ratio; the difference is the signature whose low tier received no claims, leaving one tier median
undefined.

**Stale tier labels.** {len(mislabeled):,} single-repair rows still carry a `low`/`high` tier label.
Their statistics are unaffected - `cost_tier_partition` returns the whole group whenever
`n_repairs <= 1` - so these are full-group numbers under a stale label, not mis-sliced data.

**The two artifacts do not share a cost basis.** Across {len(joined):,} single-repair signatures
present in both files, `canonical_repairs.cost_med` equals `golden_solutions.cost_med` in
**{same_basis} case(s)** ({C.pct(same_basis / len(joined), 2)}); the CRL value is
**{basis_ratio.median():.2f}x** the golden-solution value at the median (p10
{basis_ratio.quantile(0.10):.2f}x, p90 {basis_ratio.quantile(0.90):.2f}x). The agent chain uses
`gsar_tot_cost_gross` (`config.COST_COL`), which `11_reassemble_from_cache.py` documents as the
corrected basis; the CRL predates that correction. **Consequence: any cost statistic quoted for the
emitted library must come from `golden_solutions.csv`.** Mixing the two silently inflates costs -
which is exactly the error that would corrupt a dispersion analysis built on CRL quartiles.

Artifacts: `eval/out/a4_library_audit.csv`, `eval/out/a4_degenerate_rows.csv`,
`eval/out/a4_tier_structure.json`.
"""
    C.append_results("a4 - Library audit", body, args.data_root, not args.no_append)

    print(f"\n  empty-text {len(empty_text)} | unusable {len(unusable)} | empty-tier {len(empty_tier)} "
          f"| truncated 3-repair {truncated}")


if __name__ == "__main__":
    main()
