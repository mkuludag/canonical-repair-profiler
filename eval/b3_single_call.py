#!/usr/bin/env python3
"""
b3 - single-call baseline: one prompt replacing the whole chain, graded by the chain's own gates.

One Gemini call per panel signature receives the raw evidence (40 cost-prefixed claim comments,
no injected statistics) and must produce the complete Canonical Repair record end to end: split
decision, prose, cost/labor statistics, and a self-reported confidence. No StatsTool, no separation
guard, no critic. The deterministic machinery then grades what came back:

  - the shipped SolutionAssemblyAgent runs AT CALL TIME on the baseline's own split decision
    (real claims in hand), recording whether the separation guard would have collapsed it;
  - the shipped SolutionCriticAgent's gates run post hoc on the baseline's OWN numbers and text
    (band ordering q25<=med<=q75, cost presence, text presence);
  - numeric estimates are graded against StatsTool over all claims, exactly like b2.

What no grading can restore: the consensus fraction, the support-weighted kappa, and the auditable
numeric trail are structurally unavailable to a single call - there is nothing deterministic to
recompute. That observation belongs in prose, not in a metric.

`--model` repeats the baseline through the LLMGateway gateway; every cache and artifact then carries the
model tag so the shipped Gemini run is never overwritten.

Usage:
    .venv/bin/python eval/b3_single_call.py --run [--limit 10]    # 400 LLM calls, resumable
    .venv/bin/python eval/b3_single_call.py --analyze
    .venv/bin/python eval/b3_single_call.py --run --model claude-sonnet-5
    .venv/bin/python eval/b3_single_call.py --analyze --model claude-sonnet-5
"""
import argparse
import json

import numpy as np
import pandas as pd

import common as C
import b_runner as R
from llm_gateway_tool import slug, suffixed
from src.agents.repair_analyst_agent import RepairAnalystAgent
from src.agents.solution_assembly_agent import SolutionAssemblyAgent
from src.agents.solution_critic_agent import SolutionCriticAgent
from src.tools import StatsTool

CACHE_REL = "eval/out/ablation/single_call.jsonl"
PROMPT = C.EVAL_DIR / "prompts" / "single_call.txt"
V2_ROOT = C.REPO_ROOT / "grouping" / "out_v2"


def run(workers: int, limit: int, model: str = "") -> None:
    assembly = SolutionAssemblyAgent(StatsTool())
    tag = slug(model) if model else ""

    def guard_check(ctx, decision) -> dict:
        """Run the shipped separation guard on the baseline's own 2-way decision."""
        if decision.get("n_repairs") != 2 or RepairAnalystAgent._schema_error(decision):
            return {}
        ctx.llm_decision = decision
        rows = assembly.assemble(ctx).solutions
        out = {"guard_collapsed": len(rows) == 1}
        if len(rows) == 2:
            meds = sorted(r["cost_med"] for r in rows if r["cost_med"])
            if len(meds) == 2 and meds[0]:
                out["guard_ratio"] = round(meds[1] / meds[0], 4)
        else:
            out["guard_rationale"] = str(rows[0].get("split_rationale", ""))[:200]
        return out

    R.run_panel(suffixed(CACHE_REL, tag), str(PROMPT), sample_size=40, workers=workers, limit=limit,
                label=f"b3{':' + tag if tag else ''}", extra_per_record=guard_check, model=model)


def critic_gates(rec: dict) -> list:
    """Apply the shipped critic's applicable gates to the baseline's OWN numbers and text.

    Support/consensus floors are skipped: the baseline emits no support or consensus (there is
    nothing deterministic behind it), so those gates would fire vacuously on every row."""
    critic = SolutionCriticAgent()
    rows = []
    for rep in rec["decision"].get("repairs") or []:
        if not isinstance(rep, dict):
            continue
        sol = {"cost_med": R._num(rep.get("est_cost_med")),
               "cost_q25": R._num(rep.get("est_cost_q25")),
               "cost_q75": R._num(rep.get("est_cost_q75")),
               "unified_diagnosis": rep.get("unified_diagnosis"),
               "suggested_correction": rep.get("suggested_correction"),
               "support": None, "consensus": None}
        # NaN -> None so the critic sees "missing" the way it does in production rows
        for k in ("cost_med", "cost_q25", "cost_q75"):
            if isinstance(sol[k], float) and np.isnan(sol[k]):
                sol[k] = None
        _verdict, problems, flags = critic._verdict(sol)
        rows.append({
            "signature_id": rec["signature_id"],
            "hard_fail": bool(problems),
            "missing_cost": any("no cost consensus" in p for p in problems),
            "band_disorder": any("out of order" in p for p in problems),
            "missing_text": any("text" in f for f in flags),
            "confidence_0_1": R._num(rep.get("confidence_0_1")),
        })
    return rows


def analyze(no_append: bool, model: str = "") -> None:
    tag = slug(model) if model else ""
    cache_rel = suffixed(CACHE_REL, tag)
    if not (C.REPO_ROOT / cache_rel).exists():
        raise SystemExit(f"no b3 cache for model '{model or 'gemini'}' - run --run first")
    panel = C.load_panel()
    weights = C.panel_weights()
    strata = panel.set_index("signature_id")["stratum"]

    # permanent failures: signatures whose LAST cache record is _ok=False
    last = {}
    for line in open(C.REPO_ROOT / cache_rel):
        rec = json.loads(line)
        last[rec["signature_id"]] = rec
    failed = sorted(s for s, r in last.items() if not r.get("_ok"))

    recs = R.load_records(cache_rel)
    ok = [r for r in recs if not r.get("schema_error")]
    per_sig = pd.DataFrame({
        "signature_id": [r["signature_id"] for r in ok],
        "n_repairs": [r["decision"].get("n_repairs") for r in ok],
        "guard_collapsed": [r.get("guard_collapsed") for r in ok],
        "guard_ratio": [r.get("guard_ratio") for r in ok],
    }).set_index("signature_id")
    per_sig["stratum"] = strata

    # structural agreement with the chain's analyst (proposal level, guard-independent)
    prop_agree, w_agree = {}, 0.0
    for s, g in per_sig.groupby("stratum"):
        rate = float(((g["n_repairs"] == 2) == (g["stratum"].isin(["honored", "collapsed"]))).mean())
        prop_agree[s] = round(rate, 4)
        w_agree += weights[s] * rate

    # guard verdict on its own splits
    splits = per_sig[per_sig["n_repairs"] == 2]
    guard = {
        "n_proposed_splits": int(len(splits)),
        "guard_veto_rate": round(float(splits["guard_collapsed"].mean()), 4) if len(splits) else None,
        "chain_veto_rate_reference": 0.212,
    }

    # critic gates + confidence on its own rows
    gate_rows = pd.DataFrame([row for r in ok for row in critic_gates(r)])
    gates = {
        "n_repair_rows": int(len(gate_rows)),
        "hard_fail_rate": round(float(gate_rows["hard_fail"].mean()), 4),
        "missing_cost_rate": round(float(gate_rows["missing_cost"].mean()), 4),
        "band_disorder_rate": round(float(gate_rows["band_disorder"].mean()), 4),
        "missing_text_rate": round(float(gate_rows["missing_text"].mean()), 4),
        "chain_inloop_fail_rate_reference": 0.001,
    }
    conf = gate_rows["confidence_0_1"].dropna()
    confidence = {"n": int(len(conf)), "median": round(float(conf.median()), 3) if len(conf) else None,
                  "q25": round(float(conf.quantile(0.25)), 3) if len(conf) else None,
                  "q75": round(float(conf.quantile(0.75)), 3) if len(conf) else None,
                  "share_missing": round(float(gate_rows["confidence_0_1"].isna().mean()), 4)}

    # numeric error, uniform with b2
    tiers = pd.DataFrame([row for r in ok for row in R.est_rows(r)])
    grp_est = [R._num(r["decision"].get("est_group_cost_med")) for r in ok]
    grp_true = [R._num(r["truth"]["all"]["cost"]["median"]) for r in ok]
    metrics = {
        "group_cost_med": R.err_summary(R.rel_err(grp_est, grp_true)),
        "tier_cost_med": R.err_summary(R.rel_err(tiers["est_cost_med"], tiers["true_cost_med"])),
        "tier_iqr_width": R.err_summary(R.rel_err(tiers["est_cost_q75"] - tiers["est_cost_q25"],
                                                  tiers["true_cost_q75"] - tiers["true_cost_q25"])),
        "tier_labor_med": R.err_summary(R.rel_err(tiers["est_labor_med"], tiers["true_labor_med"])),
    }

    summary = {
        "model": tag or "gemini-2.5-flash",
        "n_panel": int(len(panel)), "n_ok": len(recs), "n_permanent_failures": len(failed),
        "n_schema_invalid": len(recs) - len(ok),
        "proposal_agreement_with_v2_weighted": round(w_agree, 4),
        "proposal_agreement_per_stratum": prop_agree,
        "split_proposal_rate": round(float((per_sig["n_repairs"] == 2).mean()), 4),
        "guard": guard, "critic_gates": gates, "self_confidence": confidence,
        "rel_error": metrics,
        "jittered_retries": int(sum(1 for r in recs if (r.get("llm_attempts") or 1) > 1)),
    }
    C.write_json(summary, suffixed("b3_single_call.json", tag))
    C.write_csv(per_sig.reset_index(), suffixed("b3_single_call_per_signature.csv", tag))

    body = f"""
Model: **{summary['model']}**.
One call per signature replaces the whole chain (40 cost-prefixed comments as evidence, no
statistics injected; panel of 400, temperature 0). Validity: {len(recs)}/400 parsed,
{summary['n_schema_invalid']} schema-invalid, {len(failed)} permanent failure(s). The baseline
proposes a split on {C.pct(summary['split_proposal_rate'])} of signatures and agrees with the
chain's analyst at the proposal level on **{C.pct(w_agree)}** (population-weighted; per stratum
{prop_agree}). Of its own {guard['n_proposed_splits']} proposed splits, the shipped separation
guard would collapse **{C.pct(guard['guard_veto_rate'])}** (the chain's own operating point:
21.2%) - splits the single call would have shipped as two repairs with no cost-side veto.

Its own emitted records: **{C.pct(gates['hard_fail_rate'])}** of repair rows fail the critic's hard
cost gates outright ({C.pct(gates['band_disorder_rate'])} band-ordering violations,
{C.pct(gates['missing_cost_rate'])} missing cost; chain in-loop FAIL rate: 0.1%);
{C.pct(gates['missing_text_rate'])} missing diagnosis/correction text. Numbers: group cost median
off by {C.pct(metrics['group_cost_med']['median_abs'])} at the median (p90
{C.pct(metrics['group_cost_med']['p90_abs'])}), labor by
{C.pct(metrics['tier_labor_med']['median_abs'])}. Self-reported confidence: median
{confidence['median']} (IQR {confidence['q25']}-{confidence['q75']}) - a score with no
deterministic decomposition behind it, unlike kappa's consensus x support construction.

Artifacts: `eval/out/{suffixed('b3_single_call.json', tag)}`,
`eval/out/{suffixed('b3_single_call_per_signature.csv', tag)}`. Population-reweighted over-proposal
and guard-veto rates with SEs (the numbers the paper quotes) come from `eval/b3_reweight.py`.
"""
    C.append_results("b3 - single-call baseline (panel)" + (f" - {model}" if tag else ""),
                     body, V2_ROOT, not no_append)
    print(json.dumps({"proposal_agreement_weighted": round(w_agree, 4),
                      "guard_veto_rate": guard["guard_veto_rate"],
                      "hard_fail_rate": gates["hard_fail_rate"]}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", action="store_true")
    g.add_argument("--analyze", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-append", action="store_true")
    ap.add_argument("--model", default="",
                    help="LLMGateway gateway model id; omit for the shipped Vertex gemini-2.5-flash")
    args = ap.parse_args()
    if args.run:
        run(args.workers, args.limit, args.model)
    else:
        analyze(args.no_append, args.model)


if __name__ == "__main__":
    main()
