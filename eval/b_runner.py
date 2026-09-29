"""
Shared LLM-run machinery for the b2/b3 ablation studies (NOT used by the offline a* analyses).

Both studies call the LLM directly (no orchestrator: b2 ablates the numeric path, b3 replaces the
whole chain), over the frozen panel, with the SAME evidence discipline:

  - the context line drops the StatsTool numbers the production prompt injects (else the model
    would simply echo them back and the "can the LLM produce the numbers?" question is vacuous);
  - the claim-comment sample keeps its per-claim approved costs - that IS the model's evidence,
    it is exactly what the production analyst sees, only more of it in the sample_size=40 arms.

Ground truth is computed AT RUN TIME from the loaded claims with the shipped StatsTool and stored
inside each cache record ("truth"), so every analysis afterwards is offline: full-group cost/labor
bands always, plus per-tier bands under the model's own 2-way median cut when it proposed a split
(computed with the shipped cost_tier_partition - the same construction the assembly agent uses).

Records: {"signature_id", "_ok", "decision" | "error", "schema_error", "llm_attempts", "truth"}.
Permanent failures are cached _ok=False and are therefore retried automatically on re-run
(DataStore.load_cache(ok_only=True) skips them), which is the heal semantics the June batch and
rerun_full_chain established.
"""
import concurrent.futures as cf
import json

import numpy as np

import common as C
from llm_gateway_tool import make_client
from rerun_full_chain import CHUNK
from src.agents.grouping_agent import GroupingAgent
from src.agents.repair_analyst_agent import RepairAnalystAgent
from src.config import COST_COL as COST
from src.tools import DataStore, StatsTool


# Field names both study prompts contract for. A parsed response sharing none of them did not carry
# the model's answer at all (see LLMGatewayTool.expect_keys).
DECISION_KEYS = ("n_repairs", "repairs", "split_rationale", "est_group_cost_med")


class NoStatsAnalyst(RepairAnalystAgent):
    """Production analyst with the deterministic statistics stripped from its context line."""

    def _build_prompt(self, ctx) -> str:
        g = ctx.claims.dropna(subset=[COST]).sort_values(COST)
        if len(g) == 0:
            g = ctx.claims
        idx = np.unique(np.linspace(0, len(g) - 1, num=min(self.sample_size, len(g))).astype(int))
        samp = g.iloc[idx]
        comments = "\n".join(
            f"- [${0 if (c != c) else round(c)}] {str(t)[:280]}"
            for c, t in zip(samp[COST], samp["paws_comment_trail"]))
        context = (f"vehicle_line={ctx.vehicle_line}; causal_part={ctx.causal_part}; "
                   f"symptom_theme=[{ctx.archetype_theme}]; claims={len(ctx.claims)}")
        return self.template % {"context": context, "comments": comments}


def _band(stats: StatsTool, df, col: str) -> dict:
    b = stats.band(df[col]) if df is not None and len(df) else {}
    return {"n": b.get("n"), "median": b.get("median"), "q25": b.get("q25"), "q75": b.get("q75")}


def compute_truth(stats: StatsTool, claims, n_repairs: int) -> dict:
    """Deterministic ground truth for one signature: full-group bands always; per-tier bands under
    the model's own 2-way split (median cut), when it proposed one."""
    truth = {"all": {"cost": _band(stats, claims, COST),
                     "labor": _band(stats, claims, "gsar_labor_hrs")}}
    if n_repairs == 2:
        tiers = {}
        for tier in ("low", "high"):
            part = stats.cost_tier_partition(claims, COST, tier, 2)
            tiers[tier] = {"cost": _band(stats, part, COST),
                           "labor": _band(stats, part, "gsar_labor_hrs")}
        truth["tiers"] = tiers
    return truth


def run_panel(cache_rel: str, prompt_path: str, sample_size: int, workers: int = 8,
              limit: int = 0, label: str = "", extra_per_record=None, model: str = "") -> None:
    """Run one LLM call per panel signature, resumably, with token probes between chunks.

    `extra_per_record(ctx, decision) -> dict` lets a study attach study-specific fields computed
    while the claims are in hand (b3 uses it for the run-time guard check).

    `model` selects the LLM: empty means the shipped Vertex Gemini path and the shipped gcloud
    token probe, so a run with no --model is byte-identical to what produced the paper's numbers.
    Any other value routes through the LLMGateway gateway and brings its own token probe.
    """
    ids = C.load_panel()["signature_id"].tolist()
    if limit:
        ids = ids[:limit]
    ds = DataStore(root=str(C.REPO_ROOT))
    stats = StatsTool()
    gemini, token_probe, _tag = make_client(model, expect_keys=DECISION_KEYS)
    agent = NoStatsAnalyst(gemini, prompt_path=prompt_path, sample_size=sample_size)
    grouping = GroupingAgent(ds)
    print(f"[{label}] preloading claims for {len(ids)} panel signatures ...")
    grouping.preload(ids)
    done = ds.load_cache(cache_rel, ok_only=True)
    todo = [s for s in ids if s not in done]
    print(f"[{label}] cache: {len(ids) - len(todo)} done, {len(todo)} to run")

    def _work(sid: int):
        try:
            ctx = grouping.load_signature(int(sid))
            res = gemini.generate_json(agent._build_prompt(ctx), temperature=0.0)
            if not res.get("_ok"):
                ds.append_cache(cache_rel, {"signature_id": int(sid), "_ok": False,
                                            "error": res.get("error"),
                                            "raw": res.get("_raw", "")})
                return
            decision = {k: v for k, v in res.items() if not k.startswith("_")}
            n = decision.get("n_repairs")
            rec = {
                "signature_id": int(sid), "_ok": True, "decision": decision,
                "schema_error": RepairAnalystAgent._schema_error(decision),
                "llm_attempts": res.get("_attempts"),
                "llm_envelope": bool(res.get("_envelope")),
                "llm_debris_retries": int(res.get("_debris_retries") or 0),
                "truth": compute_truth(stats, ctx.claims, n if n in (1, 2) else 1),
            }
            if extra_per_record is not None:
                rec.update(extra_per_record(ctx, decision))
            ds.append_cache(cache_rel, rec)
        except Exception as e:  # never let one signature kill the batch
            ds.append_cache(cache_rel, {"signature_id": int(sid), "_ok": False,
                                        "error": str(e)[:200]})

    for i in range(0, len(todo), CHUNK):
        token_probe()
        chunk = todo[i:i + CHUNK]
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_work, chunk))
        print(f"[{label}] progress: {min(i + CHUNK, len(todo))}/{len(todo)}", flush=True)

    recs = ds.load_cache(cache_rel, ok_only=False)
    ok = [r for r in recs.values() if r.get("_ok")]
    bad_schema = [r for r in ok if r.get("schema_error")]
    print(f"[{label}] done: {len(ok)}/{len(ids)} ok ({len(bad_schema)} schema-invalid), "
          f"{len(recs) - len(ok)} permanent failure(s) (re-run the command to retry those)")


def load_records(cache_rel: str) -> list:
    """All _ok cache records for analysis."""
    out = []
    for line in open(C.REPO_ROOT / cache_rel):
        rec = json.loads(line)
        if rec.get("_ok"):
            out.append(rec)
    # last write wins, like DataStore.load_cache
    dedup = {r["signature_id"]: r for r in out}
    return list(dedup.values())


# --------------------------------------------------------------------- est-vs-truth extraction

def _num(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def est_rows(rec: dict) -> list:
    """Per-(signature, tier) estimate-vs-truth rows from one b2/b3 cache record.

    A 1-repair decision is compared against the full-group bands; a 2-repair decision compares each
    repair against the bands of ITS OWN median-cut tier (the model chose the structure, so it is
    graded against the deterministic statistics of the structure it chose)."""
    d, t = rec["decision"], rec["truth"]
    reps = d.get("repairs") or []
    n = d.get("n_repairs")
    if n == 1 and reps:
        pairs = [(reps[0], "all", t["all"])]
    elif n == 2:
        pairs = [(r, r.get("cost_tier"), (t.get("tiers") or {}).get(r.get("cost_tier")))
                 for r in reps if isinstance(r, dict) and r.get("cost_tier") in ("low", "high")]
    else:
        pairs = []
    rows = []
    for rep, tier, tr in pairs:
        if not tr:
            continue
        rows.append({
            "signature_id": rec["signature_id"], "cost_tier": tier,
            "est_cost_med": _num(rep.get("est_cost_med")),
            "est_cost_q25": _num(rep.get("est_cost_q25")),
            "est_cost_q75": _num(rep.get("est_cost_q75")),
            "est_labor_med": _num(rep.get("est_labor_med_hrs")),
            "true_cost_med": _num(tr["cost"]["median"]),
            "true_cost_q25": _num(tr["cost"]["q25"]),
            "true_cost_q75": _num(tr["cost"]["q75"]),
            "true_labor_med": _num(tr["labor"]["median"]),
            "true_n": tr["cost"]["n"],
        })
    return rows


def rel_err(est, true) -> "np.ndarray":
    import pandas as pd
    e, t = pd.Series(est, dtype=float), pd.Series(true, dtype=float)
    return ((e - t) / t.where(t != 0)).to_numpy()


def err_summary(errors) -> dict:
    """Distribution summary of signed relative errors (NaNs = missing estimate, reported)."""
    import pandas as pd
    s = pd.Series(errors, dtype=float)
    a = s.abs().dropna()
    if not len(a):
        return {"n": 0}
    return {
        "n": int(len(s)), "n_missing": int(s.isna().sum()),
        "median_abs": round(float(a.median()), 4), "p90_abs": round(float(a.quantile(0.9)), 4),
        "share_within_10pct": round(float((a <= 0.10).mean()), 4),
        "share_within_25pct": round(float((a <= 0.25).mean()), 4),
        "median_signed": round(float(s.dropna().median()), 4),
    }
