#!/usr/bin/env python3
"""
a7 - v1 -> v2 churn: what changed when the real agent chain replaced the June batch output?

The v2 library (grouping/out_v2/) differs from v1 (grouping/out/) in exactly three intended ways:
the 2-repair agent prompt (v1's batch prompt allowed 3), the in-loop critic (v1 had none), and a
slightly different input population (the chain does not apply v1's comment-length>15 filter). This
script quantifies the churn those changes produced, so the paper swap is reviewed with eyes open.

GATE (from the plan): if split agreement < ~85%, or a headline count moves in a way that changes the
paper's story, STOP and review before prose locks.

    python eval/a7_rerun_churn.py          # compares grouping/out (v1) vs grouping/out_v2 (v2)
"""
import json

import numpy as np
import pandas as pd

import common as C

V1_ROOT = C.REPO_ROOT / "grouping" / "out"
V2_ROOT = C.REPO_ROOT / "grouping" / "out_v2"


def per_signature(df: pd.DataFrame) -> pd.DataFrame:
    """One row per signature: split decision, tier pair, total support, cost span."""
    g = df.groupby("signature_id")
    out = pd.DataFrame({
        "n_repairs": g["n_repairs_detected"].first(),
        "tier_pair": g["cost_tier"].agg(lambda s: "+".join(sorted(s))),
        "support": g["support"].sum(),
        "cost_med_min": g["cost_med"].min(),
        "cost_med_max": g["cost_med"].max(),
    })
    return out


def main() -> None:
    args = C.parse_args(__doc__)
    v1 = C.load_golden(V1_ROOT)
    v2 = C.load_golden(V2_ROOT)
    p1, p2 = per_signature(v1), per_signature(v2)
    j = p1.join(p2, how="inner", lsuffix="_v1", rsuffix="_v2")
    print(f"signatures: v1={len(p1)}  v2={len(p2)}  common={len(j)}")

    # --- split decision agreement -------------------------------------------------------------
    agree = (j["n_repairs_v1"] == j["n_repairs_v2"])
    split_matrix = pd.crosstab(j["n_repairs_v1"], j["n_repairs_v2"])
    # the 388 truncated three-repair signatures (v1 tier pair low+mid): how did v2 resolve them?
    truncated = j[j["tier_pair_v1"] == "low+mid"]
    trunc_resolution = truncated["n_repairs_v2"].value_counts().to_dict()

    # --- cost-band stability where the split decision matches ----------------------------------
    same_single = j[(j["n_repairs_v1"] == 1) & (j["n_repairs_v2"] == 1)]
    rel_delta = ((same_single["cost_med_max_v2"] - same_single["cost_med_max_v1"]).abs()
                 / same_single["cost_med_max_v1"]).dropna()

    # --- critic / degradation profile of v2 ----------------------------------------------------
    manifest = json.loads((V2_ROOT / "rerun_manifest.json").read_text())
    verdicts_v2 = v2["critic_verdict"].value_counts().to_dict()
    revised_sigs = int(v2.groupby("signature_id")["revised"].first().sum())
    shells_v2 = int(((v2["unified_diagnosis"].str.strip() == "")
                     | (v2["suggested_correction"].str.strip() == "")).sum())
    mid_v2 = int((v2["cost_tier"] == "mid").sum())

    summary = {
        "n_signatures_common": int(len(j)),
        "split_agreement_rate": float(agree.mean()),
        "split_matrix": {f"v1={a}": {f"v2={b}": int(split_matrix.loc[a, b])
                                     for b in split_matrix.columns} for a in split_matrix.index},
        "truncated_388_resolution_in_v2": {f"v2={k}": int(v) for k, v in trunc_resolution.items()},
        "cost_med_rel_delta_same_single": {
            "median": float(rel_delta.median()), "p90": float(rel_delta.quantile(0.9)),
            "share_gt_1pct": float((rel_delta > 0.01).mean()),
        },
        "v2_solutions": int(len(v2)), "v1_solutions": int(len(v1)),
        "v2_verdicts": verdicts_v2,
        "v2_revised_signatures": revised_sigs,
        "v2_empty_text_shells": shells_v2,
        "v2_mid_tiers (must be 0)": mid_v2,
        "v2_llm_degraded_calls": manifest.get("n_llm_degraded_calls"),
        "gate": {"split_agreement_ok": bool(agree.mean() >= 0.85), "mid_tiers_ok": bool(mid_v2 == 0)},
    }
    C.write_json(summary, "a7_churn.json")
    j.reset_index().to_csv(C.OUT_DIR / "a7_churn_per_signature.csv", index=False)
    print(f"  wrote eval/out/a7_churn_per_signature.csv ({len(j)} rows)")

    body = f"""
Re-running the full agent chain over the {len(j):,} GOLD signatures (manifest:
`grouping/out_v2/rerun_manifest.json`) reproduces the v1 split decision on
**{C.pct(agree.mean())}** of signatures. The {len(truncated):,} v1 signatures that were truncated
three-repair decisions resolved in v2 as {trunc_resolution}. Where both versions agree a signature is
a single repair, the consensus cost median moves by a relative
{rel_delta.median() * 100:.2f}% at the median (p90 {rel_delta.quantile(0.9) * 100:.2f}%) - the
deterministic numeric path is stable; churn is confined to the LLM's structural decision.

v2 profile: {len(v2):,} solutions (v1: {len(v1):,}); critic verdicts {verdicts_v2}; the bounded
revision fired on **{revised_sigs}** signatures; {shells_v2} empty-text shell(s);
{manifest.get('n_llm_degraded_calls')} degraded LLM call(s); `mid` tiers: {mid_v2} (2-repair prompt
cap holds). Gate: split agreement {'PASSES' if agree.mean() >= 0.85 else '**FAILS**'} the 85% bar.

Artifacts: `eval/out/a7_churn.json`, `eval/out/a7_churn_per_signature.csv`.
"""
    C.append_results("a7 - v1 to v2 rerun churn", body, args.data_root, not args.no_append)
    print(json.dumps({k: summary[k] for k in
                      ("split_agreement_rate", "truncated_388_resolution_in_v2",
                       "v2_revised_signatures", "v2_empty_text_shells")}, indent=2))


if __name__ == "__main__":
    main()
