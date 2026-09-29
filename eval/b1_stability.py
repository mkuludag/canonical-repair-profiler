#!/usr/bin/env python3
"""
b1 - same-prompt stability: re-run the UNMODIFIED chain over the frozen panel, twice.

Section 5.5 measures 69.0% split agreement across a PROMPT REVISION but cannot say how much of that
churn is plain run-to-run sampling variance; this study is the missing control. Two fresh replicates
of the production chain (identical analyst prompt, asserted by SHA-256 against the v2 manifest;
temperature 0) run over eval/panel_400.csv, then all three libraries (v2, rep1, rep2) are compared
pairwise on a7's exact metric: per-signature agreement of the EMITTED n_repairs. Because the numeric
path is deterministic given the same claims and the same split decision, tier statistics must be
IDENTICAL wherever the structure matches - any nonzero delta is a bug, not a finding.

The `_attempts` field recorded by VertexGeminiTool tells us whether each reply came from the first
temperature-0 call or from a jittered parse-failure retry (temp >= 0.2), so the analysis can check
whether run-to-run flips concentrate in the jittered calls.

Usage:
    .venv/bin/python eval/b1_stability.py --run rep1 [--limit 10]   # ~400 LLM calls, resumable
    .venv/bin/python eval/b1_stability.py --run rep2
    .venv/bin/python eval/b1_stability.py --analyze
"""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

import common as C
from rerun_full_chain import CHUNK, PROMPT_FILE, assert_token_health, V1_COLUMNS
from src.agents.orchestrator import Orchestrator

V2_ROOT = C.REPO_ROOT / "grouping" / "out_v2"
CACHE_REL = "eval/out/ablation/stability_{rep}.jsonl"
REPS = ("rep1", "rep2")


# ------------------------------------------------------------------------------ run

def run_replicate(rep: str, workers: int, limit: int) -> None:
    manifest = json.loads((V2_ROOT / "rerun_manifest.json").read_text())
    sha = hashlib.sha256(PROMPT_FILE.read_bytes()).hexdigest()
    if sha != manifest["prompt_sha256"]:
        raise SystemExit("analyst prompt differs from the v2 manifest - this would not be a "
                         "same-prompt run. Aborting.")

    ids = C.load_panel()["signature_id"].tolist()
    if limit:
        ids = ids[:limit]
    cache_rel = CACHE_REL.format(rep=rep)
    orch = Orchestrator(root=str(C.REPO_ROOT))
    print(f"[{rep}] preloading claims for {len(ids)} panel signatures ...")
    orch.grouping.preload(ids)
    done = orch.datastore.load_cache(cache_rel, ok_only=True)
    todo = [s for s in ids if s not in done]
    print(f"[{rep}] cache: {len(ids) - len(todo)} done, {len(todo)} to run")
    for i in range(0, len(todo), CHUNK):
        assert_token_health()
        orch.run_batch(todo[i:i + CHUNK], workers=workers, cache_rel=cache_rel)
        print(f"[{rep}] progress: {min(i + CHUNK, len(todo))}/{len(todo)}", flush=True)

    # heal: a degraded LLM call completes the chain as a shell and is cached _ok, so a plain resume
    # would never retry it (same semantics as rerun_full_chain.py).
    for round_no in (1, 2):
        recs = orch.datastore.load_cache(cache_rel, ok_only=False)
        degraded = sorted(s for s in ids if (recs.get(s) or {}).get("_ok") and recs[s].get("llm_error"))
        if not degraded:
            break
        print(f"[{rep}] heal round {round_no}: retrying {len(degraded)} degraded signature(s)")
        cache_path = C.REPO_ROOT / cache_rel
        keep = [l for l in open(cache_path) if l.strip()
                and json.loads(l).get("signature_id") not in set(degraded)]
        cache_path.write_text("".join(keep))
        assert_token_health()
        orch.run_batch(degraded, workers=workers, cache_rel=cache_rel)

    recs = orch.datastore.load_cache(cache_rel, ok_only=False)
    ok = [r for r in recs.values() if r.get("_ok")]
    shells = [r for r in ok if r.get("llm_error")]
    print(f"[{rep}] done: {len(ok)}/{len(ids)} ok, {len(shells)} shell(s), "
          f"{len(recs) - len(ok)} permanent failure(s)")


# ------------------------------------------------------------------------------ analyze

def cache_solutions(rep: str) -> tuple[pd.DataFrame, dict]:
    """Per-solution DataFrame (v1 schema + critic_verdict) and per-signature llm_attempts."""
    rows, attempts = [], {}
    for line in open(C.REPO_ROOT / CACHE_REL.format(rep=rep)):
        rec = json.loads(line)
        if not rec.get("_ok"):
            continue
        sid = rec["signature_id"]
        attempts[sid] = rec.get("llm_attempts")
        for sol in rec.get("solutions", []):
            row = {k: sol.get(k) for k in V1_COLUMNS}
            row["critic_verdict"] = sol.get("critic_verdict")
            rows.append(row)
    df = pd.DataFrame(rows).sort_values(["signature_id", "cost_tier"]).reset_index(drop=True)
    for c in ("cost_med", "cost_q25", "cost_q75", "labor_med_hrs"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["split_rationale"] = df["split_rationale"].fillna("").astype(str)
    return df, attempts


def per_signature(df: pd.DataFrame) -> pd.DataFrame:
    """signature_id -> emitted n_repairs + guard class (a7's units of comparison + the strata's)."""
    klass = C.classify_signatures(df).set_index("signature_id")["klass"]
    n = df.groupby("signature_id")["n_repairs_detected"].first()
    return pd.DataFrame({"n_repairs": n, "klass": klass})


def pairwise(name: str, a: pd.DataFrame, b: pd.DataFrame, panel: pd.DataFrame,
             weights: dict) -> dict:
    j = a.join(b, how="inner", lsuffix="_a", rsuffix="_b")
    j = j.join(panel.set_index("signature_id")["stratum"], how="inner")
    agree_n = (j["n_repairs_a"] == j["n_repairs_b"])
    agree_k = (j["klass_a"] == j["klass_b"])
    per_stratum, reweighted, var = {}, 0.0, 0.0
    for s, grp in j.groupby("stratum"):
        p = float((grp["n_repairs_a"] == grp["n_repairs_b"]).mean())
        per_stratum[s] = {"n": int(len(grp)), "n_repairs_agreement": round(p, 4),
                          "class_agreement": round(float((grp["klass_a"] == grp["klass_b"]).mean()), 4)}
        w = weights[s]
        reweighted += w * p
        var += w * w * p * (1 - p) / len(grp)
    return {
        "pair": name, "n_common": int(len(j)),
        "n_repairs_agreement_raw": round(float(agree_n.mean()), 4),
        "n_repairs_agreement_population_weighted": round(reweighted, 4),
        "population_weighted_se": round(float(np.sqrt(var)), 4),
        "class_agreement_raw": round(float(agree_k.mean()), 4),
        "per_stratum": per_stratum,
    }


def numeric_identity(a: pd.DataFrame, b: pd.DataFrame, sig_a: pd.DataFrame,
                     sig_b: pd.DataFrame) -> dict:
    """Where the emitted structure matches, tier stats must be identical (same claims, same
    deterministic StatsTool). Returns the max relative delta observed - anything > 0 is a bug."""
    j = sig_a.join(sig_b, how="inner", lsuffix="_a", rsuffix="_b")
    same = j[(j["n_repairs_a"] == j["n_repairs_b"]) & (j["klass_a"] == j["klass_b"])].index
    cols = ["signature_id", "cost_tier", "cost_med", "cost_q25", "cost_q75", "labor_med_hrs",
            "critic_verdict", "unified_diagnosis"]
    m = a[a["signature_id"].isin(same)][cols].merge(
        b[b["signature_id"].isin(same)][cols], on=["signature_id", "cost_tier"],
        suffixes=("_a", "_b"))
    deltas = []
    for c in ("cost_med", "cost_q25", "cost_q75", "labor_med_hrs"):
        d = ((m[f"{c}_a"] - m[f"{c}_b"]).abs() / m[f"{c}_a"].abs().replace(0, np.nan)).dropna()
        if len(d):
            deltas.append(float(d.max()))
    return {
        "n_signatures_same_structure": int(len(same)),
        "n_tier_rows_joined": int(len(m)),
        "max_relative_delta_any_stat": max(deltas) if deltas else 0.0,
        "critic_verdict_agreement": round(float((m["critic_verdict_a"] == m["critic_verdict_b"]).mean()), 4)
        if len(m) else None,
        "diagnosis_exact_match_rate": round(float((m["unified_diagnosis_a"].fillna("")
                                                   == m["unified_diagnosis_b"].fillna("")).mean()), 4)
        if len(m) else None,
    }


def flips_vs_attempts(sig_a: pd.DataFrame, sig_b: pd.DataFrame,
                      att_a: dict, att_b: dict) -> dict:
    """Do structural flips concentrate in signatures whose reply came from a jittered retry?"""
    j = sig_a.join(sig_b, how="inner", lsuffix="_a", rsuffix="_b")
    flipped = j["n_repairs_a"] != j["n_repairs_b"]
    jittered = pd.Series({sid: max(att_a.get(sid) or 1, att_b.get(sid) or 1) > 1
                          for sid in j.index})
    tab = pd.crosstab(flipped, jittered)
    return {
        "n_jittered_either_run": int(jittered.sum()),
        "flip_rate_jittered": round(float(flipped[jittered].mean()), 4) if jittered.any() else None,
        "flip_rate_first_try": round(float(flipped[~jittered].mean()), 4) if (~jittered).any() else None,
        "crosstab": {f"flipped={fi}": {f"jittered={ji}": int(tab.loc[fi, ji])
                                       for ji in tab.columns} for fi in tab.index},
    }


def analyze(no_append: bool) -> None:
    panel = C.load_panel()
    weights = C.panel_weights()
    v2 = C.load_golden(V2_ROOT)
    v2 = v2[v2["signature_id"].isin(panel["signature_id"])]
    r1, att1 = cache_solutions("rep1")
    r2, att2 = cache_solutions("rep2")
    s_v2, s_r1, s_r2 = per_signature(v2), per_signature(r1), per_signature(r2)

    pairs = [pairwise("v2 x rep1", s_v2, s_r1, panel, weights),
             pairwise("v2 x rep2", s_v2, s_r2, panel, weights),
             pairwise("rep1 x rep2", s_r1, s_r2, panel, weights)]
    mean_w = float(np.mean([p["n_repairs_agreement_population_weighted"] for p in pairs]))

    ident = {p["pair"]: numeric_identity(a, b, sa, sb) for p, (a, b, sa, sb) in zip(
        pairs, [(v2, r1, s_v2, s_r1), (v2, r2, s_v2, s_r2), (r1, r2, s_r1, s_r2)])}
    attempts_summary = {
        "rep1_jittered": int(sum(1 for v in att1.values() if (v or 1) > 1)),
        "rep2_jittered": int(sum(1 for v in att2.values() if (v or 1) > 1)),
        "n_panel": int(len(panel)),
    }
    flips = flips_vs_attempts(s_r1, s_r2, att1, att2)

    summary = {
        "pairs": pairs,
        "mean_population_weighted_agreement": round(mean_w, 4),
        "prompt_revision_agreement_for_contrast": 0.690,
        "numeric_identity": ident,
        "jittered_retries": attempts_summary,
        "flips_vs_jitter_rep1xrep2": flips,
    }
    C.write_json(summary, "b1_stability.json")
    per_sig = s_v2.join(s_r1, how="inner", rsuffix="_r1").join(s_r2, how="inner", rsuffix="_r2")
    per_sig.reset_index().rename(columns={"index": "signature_id"}).to_csv(
        C.OUT_DIR / "b1_stability_per_signature.csv", index=False)
    print(f"  wrote eval/out/b1_stability_per_signature.csv ({len(per_sig)} rows)")

    worst_ident = max(v["max_relative_delta_any_stat"] for v in ident.values())
    body = f"""
Two fresh replicates of the unmodified chain (same prompt by SHA-256, temperature 0) over the frozen
400-signature panel (`eval/panel_400.csv`; strata reweighted to population shares). Pairwise emitted
split-decision agreement (a7's metric), population-weighted:
{'; '.join(f"{p['pair']} **{C.pct(p['n_repairs_agreement_population_weighted'])}** (SE {p['population_weighted_se'] * 100:.1f}pp)" for p in pairs)};
mean **{C.pct(mean_w)}** - versus **69.0%** across the prompt revision (a7). Per-stratum agreement
(mean over pairs): {', '.join(f"{s} {C.pct(float(np.mean([p['per_stratum'][s]['n_repairs_agreement'] for p in pairs])))}" for s in ("single", "honored", "collapsed"))}.

Where the structure matches, tier statistics are identical to relative delta
{worst_ident:.2e} max across all pairs (the deterministic numeric path, verified end to end);
critic verdicts agree on {'; '.join(f"{k} {C.pct(v['critic_verdict_agreement'])}" for k, v in ident.items())}.
Diagnosis prose exact-match where structure matches: {'; '.join(f"{k} {C.pct(v['diagnosis_exact_match_rate'])}" for k, v in ident.items())}.

Jittered parse-failure retries: rep1 {attempts_summary['rep1_jittered']}/400, rep2
{attempts_summary['rep2_jittered']}/400. rep1 x rep2 flip rate on jittered signatures
{C.pct(flips['flip_rate_jittered']) if flips['flip_rate_jittered'] is not None else 'n/a'} vs
{C.pct(flips['flip_rate_first_try']) if flips['flip_rate_first_try'] is not None else 'n/a'} on
first-try signatures.

Artifacts: `eval/out/b1_stability.json`, `eval/out/b1_stability_per_signature.csv`.
"""
    C.append_results("b1 - same-prompt stability (panel)", body, V2_ROOT, not no_append)
    print(json.dumps({"mean_population_weighted_agreement": round(mean_w, 4),
                      "worst_numeric_identity_delta": worst_ident}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", choices=REPS)
    g.add_argument("--analyze", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="pilot: only the first N panel signatures")
    ap.add_argument("--no-append", action="store_true")
    args = ap.parse_args()
    if args.run:
        run_replicate(args.run, args.workers, args.limit)
    else:
        analyze(args.no_append)


if __name__ == "__main__":
    main()
