#!/usr/bin/env python3
"""
Re-run the FULL agent chain over every GOLD signature -> the v2 library of record.

Unlike the June batch (grouping/10_golden_solutions.py: standalone script, its own 3-repair prompt,
no critic), this drives the real Orchestrator end to end: GroupingAgent -> ConsensusAgent ->
RepairAnalystAgent (2-repair prompt, temperature 0) -> SolutionAssemblyAgent (separation guard) ->
SolutionCriticAgent (verdicts + the one bounded revision round) -> CostSavingsAgent. The paper's
"produced by the 7-agent chain" claim is true of this artifact by construction.

Everything is written under grouping/out_v2/; v1 files are never touched. The JSONL cache makes the
run resumable (kill it, rerun the command, completed signatures are skipped). Gemini at temperature 0
is not bit-wise deterministic; the cache is the run's reproducibility artifact.

Usage:
    .venv/bin/python eval/rerun_full_chain.py                  # full run (resumable)
    .venv/bin/python eval/rerun_full_chain.py --limit 25       # pilot slice
    .venv/bin/python eval/rerun_full_chain.py --assemble-only  # rebuild CSV from cache, no LLM calls
"""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import common as C  # noqa: F401  (sys.path bootstrap for src imports)
from src import config as CFG
from src.agents.orchestrator import Orchestrator
from src.tools.auth import fresh_access_token

REPO = C.REPO_ROOT
OUT_DIR = REPO / "grouping" / "out_v2"
CACHE_REL = "grouping/out_v2/llm_cache.jsonl"
GOLDEN_V2 = OUT_DIR / "golden_solutions.csv"
MANIFEST = OUT_DIR / "rerun_manifest.json"
PROMPT_FILE = REPO / "src" / "prompts" / "split_and_diagnose.txt"

# Library-independent artifacts copied so out_v2 is a complete --data-root for eval/a*.py.
INDEPENDENT = ["canonical_repairs.csv", "region_overall.csv", "dealer_spread_by_state_signature.csv"]

# v1 column order, kept identical so every downstream reader works unchanged; v2 appends the two
# critic columns at the end.
V1_COLUMNS = ["signature_id", "vehicle_line", "causal_part", "n_repairs_detected", "repair_name",
              "cost_tier", "tier_n_claims", "split_rationale", "unified_diagnosis",
              "suggested_correction", "typical_parts", "labor_med_hrs", "cost_med", "cost_q25",
              "cost_q75", "consensus", "support", "confidence", "usable"]
V2_COLUMNS = V1_COLUMNS + ["critic_verdict", "revised"]


def gold_signature_ids() -> list:
    crl = pd.read_csv(REPO / CFG.CANONICAL_REPAIRS, low_memory=False)
    return sorted(int(s) for s in crl.loc[crl["status"] == "GOLD", "signature_id"])


CHUNK = 200  # probe token health this often; bounds the damage of a mid-run auth loss


def assert_token_health() -> None:
    """Abort loudly if gcloud cannot mint a token.

    Ford CAA can force interactive re-authentication mid-run; when that happens every LLM call
    degrades to a shell that looks 'successful' to the resumable cache. Discovered the hard way:
    a reauth ~90 min into the first full run silently turned ~1,700 signatures into shells.
    Failing fast (and between chunks) turns that into a clean stop + resume instead.
    """
    try:
        fresh_access_token(force=True)
    except RuntimeError as e:
        raise SystemExit(
            f"\nAUTH LOST - stopping before burning LLM calls on shells.\n{e}\n"
            "Re-authenticate (gcloud auth login), then re-run this command; the cache resumes and "
            "the heal pass retries any shelled signatures.") from e


def assemble_csv() -> pd.DataFrame:
    """golden_solutions.csv (v2) from the cache: one row per (signature x emitted repair)."""
    rows = []
    n_bad = 0
    for line in open(REPO / CACHE_REL):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if not rec.get("_ok"):
            n_bad += 1
            continue
        revised = bool((rec.get("critique") or {}).get("revised"))
        for sol in rec.get("solutions", []):
            row = {k: sol.get(k) for k in V1_COLUMNS}
            row["critic_verdict"] = sol.get("critic_verdict")
            row["revised"] = revised
            rows.append(row)
    df = pd.DataFrame(rows, columns=V2_COLUMNS).sort_values(
        ["signature_id", "cost_tier"]).reset_index(drop=True)
    if n_bad:
        print(f"  note: {n_bad} cached records are permanent failures (kept in cache, not in CSV)")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="pilot: only the first N signatures")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--assemble-only", action="store_true",
                    help="rebuild the CSV + manifest from the existing cache; no LLM calls")
    ap.add_argument("--heal-rounds", type=int, default=2,
                    help="re-run signatures whose LLM call degraded to a shell (parse/transport "
                         "error), up to N extra rounds; matches the June batch's retry-until-ok "
                         "semantics. Shells surviving all rounds ship and are counted honestly.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = gold_signature_ids()
    if args.limit:
        ids = ids[: args.limit]
    print(f"GOLD signatures: {len(ids)}")

    t0 = time.time()
    if not args.assemble_only:
        orch = Orchestrator(root=str(REPO))
        print("preloading claims for all signatures (one parquet pass) ...")
        n = orch.grouping.preload(ids)
        print(f"  preloaded {n} signatures")
        done = orch.datastore.load_cache(CACHE_REL, ok_only=True)
        todo = [s for s in ids if s not in done]
        print(f"  cache: {len(ids) - len(todo)} already done, {len(todo)} to run")
        # Chunked so a mid-run auth loss stops the run within one chunk instead of shelling the rest.
        for i in range(0, len(todo), CHUNK):
            assert_token_health()
            chunk = todo[i:i + CHUNK]
            orch.run_batch(chunk, workers=args.workers, cache_rel=CACHE_REL)
            print(f"  progress: {min(i + CHUNK, len(todo))}/{len(todo)}", flush=True)

        # Heal pass: a degraded LLM call (parse/transport failure) completes the chain as a shell
        # and is cached _ok, so a plain resume would never retry it. Give those signatures up to
        # --heal-rounds fresh attempts, mirroring the June batch's only-ok-counts-as-done semantics.
        for round_no in range(1, args.heal_rounds + 1):
            recs = orch.datastore.load_cache(CACHE_REL, ok_only=False)
            degraded = sorted(s for s in ids
                              if (recs.get(s) or {}).get("_ok") and recs[s].get("llm_error"))
            if not degraded:
                break
            print(f"  heal round {round_no}: retrying {len(degraded)} degraded signature(s)", flush=True)
            cache_path = REPO / CACHE_REL
            keep = [l for l in open(cache_path) if l.strip()
                    and json.loads(l).get("signature_id") not in set(degraded)]
            cache_path.write_text("".join(keep))
            for i in range(0, len(degraded), CHUNK):
                assert_token_health()
                orch.run_batch(degraded[i:i + CHUNK], workers=args.workers, cache_rel=CACHE_REL)
    wall = time.time() - t0

    df = assemble_csv()
    df.to_csv(GOLDEN_V2, index=False)
    print(f"wrote {GOLDEN_V2.relative_to(REPO)}: {len(df)} rows over "
          f"{df['signature_id'].nunique()} signatures")

    for name in INDEPENDENT:
        src, dst = REPO / "grouping" / "out" / name, OUT_DIR / name
        dst.write_bytes(src.read_bytes())
    print(f"copied {len(INDEPENDENT)} library-independent artifacts into out_v2/")

    cache_recs = [json.loads(l) for l in open(REPO / CACHE_REL) if l.strip()]
    ok = [r for r in cache_recs if r.get("_ok")]
    revised = sum(1 for r in ok if (r.get("critique") or {}).get("revised"))
    llm_errors = sum(1 for r in ok if r.get("llm_error"))
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": CFG.GEMINI_MODEL,
        "temperature": 0.0,
        "prompt_sha256": hashlib.sha256(PROMPT_FILE.read_bytes()).hexdigest(),
        "prompt_max_repairs": 2,
        "chain": "GroupingAgent -> ConsensusAgent -> RepairAnalystAgent -> SolutionAssemblyAgent"
                 " -> SolutionCriticAgent (bounded revision) -> CostSavingsAgent",
        "n_gold_signatures": len(ids),
        "n_cached_ok": len(ok),
        "n_cached_failed": len(cache_recs) - len(ok),
        "n_solutions": int(len(df)),
        "n_revised_signatures": int(revised),
        "n_llm_degraded_calls": int(llm_errors),
        "wall_seconds_this_invocation": round(wall, 1),
        "notes": "temperature 0 pins the decision but Gemini is not bit-wise deterministic; "
                 "llm_cache.jsonl is the reproducibility artifact. Input population: GroupingAgent "
                 "does not apply the June batch's comment-length>15 filter.",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in
                      ("n_cached_ok", "n_cached_failed", "n_solutions",
                       "n_revised_signatures", "n_llm_degraded_calls")}, indent=2))


if __name__ == "__main__":
    main()
