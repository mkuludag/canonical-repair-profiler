#!/usr/bin/env python3
"""
b2 - the deterministic-numeric-path ablation: can the LLM produce the numbers itself?

ACCORD's separation principle routes every numeric quantity through StatsTool and lets the LLM own
only structure and prose. This study removes that separation on the frozen panel: the production
analyst prompt is rebuilt WITHOUT the injected group statistics and extended to ask the model for
the very numbers the deterministic path computes (per-repair cost median/IQR and labor median, plus
a group-level cost median). Two evidence arms: sample_size=8 (exactly the production evidence) and
sample_size=40 (5x more claim comments, testing whether more context closes the gap).

Grading is against the shipped StatsTool over ALL claims of the group - the quantity the released
library actually carries. A 2-way decision is graded against its own median-cut tiers, so the model
is never penalized for choosing a different structure than v2.

`--model` repeats the study through the LLMGateway gateway. Every artifact and cache path then carries
the model tag, so a second model can never overwrite the shipped Gemini run (the JSONL caches dedup
last-write-wins on signature_id, which would make a collision silent). Only the s8 arm is run for
additional models: s40 exists to show that 5x evidence does not close the gap, and the claim the
paper rests on is the production-evidence arm.

Usage:
    .venv/bin/python eval/b2_llm_numbers.py --run s8  [--limit 10]   # 400 LLM calls, resumable
    .venv/bin/python eval/b2_llm_numbers.py --run s40
    .venv/bin/python eval/b2_llm_numbers.py --analyze
    .venv/bin/python eval/b2_llm_numbers.py --run s8 --model claude-sonnet-5
    .venv/bin/python eval/b2_llm_numbers.py --analyze --model claude-sonnet-5
"""
import argparse
import json

import numpy as np
import pandas as pd

import common as C
import b_runner as R
from llm_gateway_tool import slug, suffixed

ARMS = {"s8": 8, "s40": 40}
CACHE_REL = "eval/out/ablation/llmnum_{arm}.jsonl"
PROMPT = C.EVAL_DIR / "prompts" / "split_and_diagnose_llmnum.txt"
V2_ROOT = C.REPO_ROOT / "grouping" / "out_v2"


def cache_path(arm: str, tag: str) -> str:
    return suffixed(CACHE_REL.format(arm=arm), tag)


def analyze_arm(arm: str, panel: pd.DataFrame, weights: dict, v2_all_med: pd.Series,
                tag: str = "") -> tuple:
    recs = R.load_records(cache_path(arm, tag))
    strata = panel.set_index("signature_id")["stratum"]
    ok = [r for r in recs if not r.get("schema_error")]

    # group-level median estimate (structure-independent primary metric)
    grp = pd.DataFrame({
        "signature_id": [r["signature_id"] for r in ok],
        "est": [R._num(r["decision"].get("est_group_cost_med")) for r in ok],
        "true": [R._num(r["truth"]["all"]["cost"]["median"]) for r in ok],
        "n_repairs": [r["decision"].get("n_repairs") for r in ok],
    }).set_index("signature_id")
    grp["stratum"] = strata

    # invariant: for signatures v2 emitted with full-group stats (single + collapsed strata), the
    # run-time truth must equal the released golden cost_med exactly (same claims, same StatsTool).
    inv = grp.join(v2_all_med, how="inner")
    inv_delta = (inv["true"] - inv["cost_med"]).abs()
    invariant = {"n_checked": int(len(inv)), "max_abs_delta": float(inv_delta.max()) if len(inv) else None}

    # per-(signature, tier) rows: cost median / IQR width / labor median
    tiers = pd.DataFrame([row for r in ok for row in R.est_rows(r)])
    metrics = {
        "group_cost_med": R.err_summary(R.rel_err(grp["est"], grp["true"])),
        "tier_cost_med": R.err_summary(R.rel_err(tiers["est_cost_med"], tiers["true_cost_med"])),
        "tier_iqr_width": R.err_summary(R.rel_err(tiers["est_cost_q75"] - tiers["est_cost_q25"],
                                                  tiers["true_cost_q75"] - tiers["true_cost_q25"])),
        "tier_labor_med": R.err_summary(R.rel_err(tiers["est_labor_med"], tiers["true_labor_med"])),
    }
    band_ok = tiers.dropna(subset=["est_cost_q25", "est_cost_med", "est_cost_q75"])
    band_violation = float((~((band_ok["est_cost_q25"] <= band_ok["est_cost_med"])
                              & (band_ok["est_cost_med"] <= band_ok["est_cost_q75"]))).mean())

    # context metric: split behavior under this (different) prompt vs the v2 analyst's proposal
    proposed_v2 = grp["stratum"].isin(["honored", "collapsed"])
    proposes = grp["n_repairs"] == 2
    prop_agree = {}
    w_agree = 0.0
    for s, g in grp.groupby("stratum"):
        rate = float(((g["n_repairs"] == 2) == g["stratum"].isin(["honored", "collapsed"])).mean())
        prop_agree[s] = round(rate, 4)
        w_agree += weights[s] * rate

    summary = {
        "arm": arm, "sample_size": ARMS[arm], "model": tag or "gemini-2.5-flash",
        "n_ok": len(recs), "n_schema_invalid": len(recs) - len(ok),
        "invariant_truth_vs_v2_golden": invariant,
        "rel_error": metrics,
        "band_order_violation_rate": round(band_violation, 4),
        "split_proposal_rate": round(float(proposes.mean()), 4),
        "v2_proposal_rate_panel": round(float(proposed_v2.mean()), 4),
        "proposal_agreement_with_v2_weighted": round(w_agree, 4),
        "proposal_agreement_per_stratum": prop_agree,
        "jittered_retries": int(sum(1 for r in recs if (r.get("llm_attempts") or 1) > 1)),
    }
    tiers.insert(0, "arm", arm)
    return summary, tiers


def analyze(no_append: bool, model: str = "") -> None:
    tag = slug(model) if model else ""
    panel = C.load_panel()
    weights = C.panel_weights()
    v2 = C.load_golden(V2_ROOT)
    v2_all = v2[v2["cost_tier"] == "all"].set_index("signature_id")["cost_med"]

    present = [a for a in ARMS if (C.REPO_ROOT / cache_path(a, tag)).exists()]
    if not present:
        raise SystemExit(f"no b2 cache for model '{model or 'gemini'}' - run --run s8 first")
    summaries, frames = [], []
    for arm in present:
        s, t = analyze_arm(arm, panel, weights, v2_all, tag)
        summaries.append(s)
        frames.append(t)
    out = {"arms": summaries}
    C.write_json(out, suffixed("b2_llm_numbers.json", tag))
    C.write_csv(pd.concat(frames, ignore_index=True), suffixed("b2_llm_numbers_per_tier.csv", tag))

    label = f"b2 - LLM-numbers ablation (panel)" + (f" - {model}" if tag else "")
    if len(summaries) == 1:
        one = summaries[0]
        e = one["rel_error"]
        body = f"""
The same ablation under **{one['model']}** through the LLMGateway gateway, {one['arm']} arm
(sample_size={one['sample_size']}), same frozen panel, same prompt, temperature 0, graded by the same
shipped StatsTool. Group-level cost median: **{C.pct(e['group_cost_med']['median_abs'])}** median
absolute relative error (p90 {C.pct(e['group_cost_med']['p90_abs'])}). Per-tier cost medians, the
number the library ships: **{C.pct(e['tier_cost_med']['median_abs'])}** (p90
{C.pct(e['tier_cost_med']['p90_abs'])}, within 10% on
{C.pct(e['tier_cost_med']['share_within_10pct'])}). Bands: {C.pct(e['tier_iqr_width']['median_abs'])}
(signed bias {C.pct(e['tier_iqr_width']['median_signed'])}). Labor:
{C.pct(e['tier_labor_med']['median_abs'])} (signed bias
{C.pct(e['tier_labor_med']['median_signed'])}). Band-order violations of the critic's hard gate:
**{C.pct(one['band_order_violation_rate'])}**. Schema-invalid replies
{one['n_schema_invalid']}/{one['n_ok']}; jittered parse retries {one['jittered_retries']}.
Split-proposal rate under this variant prompt {C.pct(one['split_proposal_rate'])} (v2 analyst on the
same panel {C.pct(one['v2_proposal_rate_panel'])}; context only, the prompt differs).

Artifacts: `eval/out/{suffixed('b2_llm_numbers.json', tag)}`,
`eval/out/{suffixed('b2_llm_numbers_per_tier.csv', tag)}`.
"""
        C.append_results(label, body, V2_ROOT, not no_append)
        print(json.dumps({one["model"]: {"tier_cost_med_abs": e["tier_cost_med"].get("median_abs"),
                                         "band_violations": one["band_order_violation_rate"]}},
                         indent=2))
        return

    s8, s40 = summaries
    body = f"""
The production analyst prompt, stripped of its injected StatsTool numbers and asked to estimate them
instead (grading: shipped StatsTool over all claims of the structure the model itself chose; panel of
400, temperature 0). With the production evidence (8 cost-prefixed claim comments), the group-level
cost median comes back with median absolute relative error
**{C.pct(s8['rel_error']['group_cost_med']['median_abs'])}** (p90
{C.pct(s8['rel_error']['group_cost_med']['p90_abs'])}; within 10% on
{C.pct(s8['rel_error']['group_cost_med']['share_within_10pct'])} of signatures); per-tier cost
medians err by {C.pct(s8['rel_error']['tier_cost_med']['median_abs'])} at the median, IQR width by
{C.pct(s8['rel_error']['tier_iqr_width']['median_abs'])}, and labor-hours medians - for which the
sample carries NO per-claim evidence - by {C.pct(s8['rel_error']['tier_labor_med']['median_abs'])}
(p90 {C.pct(s8['rel_error']['tier_labor_med']['p90_abs'])}). {C.pct(s8['band_order_violation_rate'])}
of estimated bands violate q25<=median<=q75 (the critic's hard gate). Five times the evidence
(sample_size=40) moves the group-median error to
{C.pct(s40['rel_error']['group_cost_med']['median_abs'])} (within 10%:
{C.pct(s40['rel_error']['group_cost_med']['share_within_10pct'])}), IQR width to
{C.pct(s40['rel_error']['tier_iqr_width']['median_abs'])}, labor to
{C.pct(s40['rel_error']['tier_labor_med']['median_abs'])}.

Schema-invalid replies: s8 {s8['n_schema_invalid']}/400, s40 {s40['n_schema_invalid']}/400.
Invariant (run-time truth == released v2 golden full-group median on single/collapsed signatures):
max abs delta s8 {s8['invariant_truth_vs_v2_golden']['max_abs_delta']}, s40
{s40['invariant_truth_vs_v2_golden']['max_abs_delta']} over
{s8['invariant_truth_vs_v2_golden']['n_checked']}/{s40['invariant_truth_vs_v2_golden']['n_checked']}
checked. Split-proposal rate under this variant prompt: s8 {C.pct(s8['split_proposal_rate'])}, s40
{C.pct(s40['split_proposal_rate'])} (v2 analyst on the same panel:
{C.pct(s8['v2_proposal_rate_panel'])}; context only - the prompt differs, so this is not a stability
measurement).

Artifacts: `eval/out/b2_llm_numbers.json`, `eval/out/b2_llm_numbers_per_tier.csv`.
"""
    C.append_results(label, body, V2_ROOT, not no_append)
    print(json.dumps({a["arm"]: {"group_med_abs": a["rel_error"]["group_cost_med"].get("median_abs"),
                                 "labor_med_abs": a["rel_error"]["tier_labor_med"].get("median_abs"),
                                 "band_violations": a["band_order_violation_rate"]}
                      for a in summaries}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", choices=list(ARMS))
    g.add_argument("--analyze", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-append", action="store_true")
    ap.add_argument("--model", default="",
                    help="LLMGateway gateway model id; omit for the shipped Vertex gemini-2.5-flash")
    args = ap.parse_args()
    if args.run:
        tag = slug(args.model) if args.model else ""
        R.run_panel(cache_path(args.run, tag), str(PROMPT), ARMS[args.run],
                    workers=args.workers, limit=args.limit,
                    label=f"b2:{args.run}{':' + tag if tag else ''}", model=args.model)
    else:
        analyze(args.no_append, args.model)


if __name__ == "__main__":
    main()
