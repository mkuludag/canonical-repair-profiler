#!/usr/bin/env python3
"""
Phase 4 (final): Unified Diagnosis + Golden Solution per GOLD signature, using Vertex Gemini.

Per gold signature we send Gemini: the group context (vehicle line, causal part, symptom theme,
cost/labor stats, modal parts) + a COST-SPREAD SAMPLE of real claim comments. Gemini returns strict
JSON that (a) decides how many distinct repairs the group really contains (1-3 -> the agentic
reseal-vs-replace split) and (b) writes a unified diagnosis + suggested correction for each. We then
partition the group's claims by cost tier to match Gemini's decision and attach numeric consensus.

Run:
  PILOT=3 python3 grouping/10_golden_solutions.py     # validate prompt on 3 groups (prints, no full write)
  python3 grouping/10_golden_solutions.py             # full run over all GOLD signatures (cached)

Outputs:
  grouping/out/llm_cache.jsonl          per-signature raw LLM JSON (cache; resumable)
  grouping/out/golden_solutions.csv     final: one row per (signature, repair-tier)
"""
import os, json, subprocess, time, concurrent.futures as cf, threading
import numpy as np
import pandas as pd
from google.oauth2.credentials import Credentials
from google import genai
from google.genai import types

PARQUET = os.environ.get("REPAIR_PARQUET", "data/repair_profile.parquet")
SIGS = "data/claim_signatures.parquet"
CRL = "grouping/out/canonical_repairs.csv"
OUT = "grouping/out"; CACHE = f"{OUT}/llm_cache.jsonl"
PROJECT, LOCATION, MODEL = "REDACTED-VERTEX-PROJECT", "us-central1", "gemini-2.5-flash"
PILOT = int(os.environ.get("PILOT", "0"))
MAX_WORKERS = int(os.environ.get("WORKERS", "8"))
_local = threading.local()


_token_lock = threading.Lock()
TOKEN_TTL = 1500  # re-mint well before gcloud's ~1h token expiry


def _fresh_token():
    with _token_lock:  # serialize gcloud subprocess calls
        return subprocess.check_output(
            ["gcloud", "auth", "print-access-token"],
            env={**os.environ, "CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT": ""}).decode().strip()


def client(force=False):
    now = time.time()
    if force or not hasattr(_local, "c") or now - getattr(_local, "ts", 0) > TOKEN_TTL:
        _local.c = genai.Client(vertexai=True, project=PROJECT, location=LOCATION,
                                credentials=Credentials(_fresh_token()),
                                http_options=types.HttpOptions(timeout=60000))  # 60s, no indefinite hangs
        _local.ts = now
    return _local.c


PROMPT = """You are a Ford warranty repair analyst. You are given a GROUP of similar warranty claims \
(same vehicle line, same causal part, same symptom theme). Your job:

1) Decide how many DISTINCT repairs this group actually contains (1, 2, or 3). Most groups are ONE \
repair. Split ONLY if the claims clearly describe different repair scopes for the same part — e.g. a \
minor repair/reseal vs a full component REPLACEMENT — which also shows up as a bimodal cost spread. \
Do not split on trivial wording differences.
2) For EACH distinct repair, write:
   - unified_diagnosis: one clear sentence a dealer would select at filing, stating the customer \
concern + the cause (what is wrong and why). Generic enough to represent every claim of that repair.
   - suggested_correction: one or two sentences stating the correct fix (the action + key parts).
   - cost_tier: which cost band this repair maps to: "all" if only one repair, else "low" / "high" \
(and "mid" if three). Lower-cost = the minor repair, higher-cost = the replacement.
   - typical_parts: short phrase naming the main part(s), or "" if unknown.

Return STRICT JSON only:
{"n_repairs": <int>, "split_rationale": "<short>", "repairs": [ {"name":"...","cost_tier":"...",\
"unified_diagnosis":"...","suggested_correction":"...","typical_parts":"..."} ]}

GROUP CONTEXT:
%s

SAMPLE CLAIM COMMENTS (each prefixed with its approved cost; spans the group's cost range):
%s
"""


def build_inputs():
    crl = pd.read_csv(CRL)
    gold = crl[crl["status"] == "GOLD"].copy()
    gold_ids = set(gold["signature_id"].tolist())
    sig = pd.read_parquet(SIGS)
    sig["request_r"] = pd.to_numeric(sig["request_r"], errors="coerce").astype("Int64")
    sig = sig[sig["signature_id"].isin(gold_ids)][["request_r", "signature_id"]]

    rp = pd.read_parquet(PARQUET, columns=["request_r", "paws_comment_trail", "gsar_tot_cost_gross", "gsar_labor_hrs"])
    rp["request_r"] = pd.to_numeric(rp["request_r"], errors="coerce").astype("Int64")
    rp["gsar_tot_cost_gross"] = pd.to_numeric(rp["gsar_tot_cost_gross"], errors="coerce")
    rp["gsar_labor_hrs"] = pd.to_numeric(rp["gsar_labor_hrs"], errors="coerce")
    rp.loc[(rp["gsar_tot_cost_gross"] <= 0) | (rp["gsar_tot_cost_gross"] >= 100000), "gsar_tot_cost_gross"] = np.nan
    m = sig.merge(rp, on="request_r", how="inner")
    m = m[m["paws_comment_trail"].notna() & (m["paws_comment_trail"].str.len() > 15)]

    stats = gold.set_index("signature_id")
    claims_by_sid = {int(sid): g[["gsar_tot_cost_gross", "gsar_labor_hrs"]].dropna(subset=["gsar_tot_cost_gross"]).sort_values("gsar_tot_cost_gross")
                     for sid, g in m.groupby("signature_id")}
    groups = {}
    for sid, g in m.groupby("signature_id"):
        g = g.dropna(subset=["gsar_tot_cost_gross"]).sort_values("gsar_tot_cost_gross")
        if len(g) == 0:
            g = m[m["signature_id"] == sid]
        # cost-spread sample: pick across quantiles to expose any bimodality
        idx = np.unique(np.linspace(0, len(g) - 1, num=min(8, len(g))).astype(int))
        samp = g.iloc[idx]
        comments = "\n".join(
            f"- [${0 if pd.isna(c) else round(c)}] {str(t)[:280]}"
            for c, t in zip(samp["gsar_tot_cost_gross"], samp["paws_comment_trail"]))
        s = stats.loc[sid]
        ctx = (f"vehicle_line={s['vehicle_line']}; causal_part={s['causal_part']}; "
               f"symptom_theme=[{s['arch_theme']}]; claims={int(s['n'])}; "
               f"cost_median=${s['cost_med']}; cost_IQR=${s['cost_q25']}-${s['cost_q75']}; "
               f"labor_median_hrs={s['labor_med']}; modal_parts={s.get('parts_modal_set','')}")
        groups[int(sid)] = {"context": ctx, "comments": comments}
    return groups, stats, claims_by_sid


def tier_band(df, cost_tier, n_repairs):
    """Slice a signature's claims to the cost band for a given repair tier."""
    if df is None or len(df) == 0 or n_repairs <= 1 or cost_tier in ("all", "", None):
        return df
    c = df["gsar_tot_cost_gross"]
    if n_repairs == 2:
        q = c.quantile(0.5)
        return df[c < q] if cost_tier == "low" else df[c >= q]
    q1, q2 = c.quantile(1/3), c.quantile(2/3)
    if cost_tier == "low": return df[c < q1]
    if cost_tier == "high": return df[c >= q2]
    return df[(c >= q1) & (c < q2)]


def band_stats(df):
    if df is None or len(df) == 0:
        return {"n": 0, "cost_med": np.nan, "cost_q25": np.nan, "cost_q75": np.nan, "labor_med": np.nan}
    c, l = df["gsar_tot_cost_gross"], df["gsar_labor_hrs"]
    return {"n": int(len(df)), "cost_med": round(float(c.median()), 2),
            "cost_q25": round(float(c.quantile(.25)), 2), "cost_q75": round(float(c.quantile(.75)), 2),
            "labor_med": round(float(l.median()), 2) if l.notna().any() else np.nan}


def call_llm(sid, payload):
    prompt = PROMPT % (payload["context"], payload["comments"])
    cfg = types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json")
    for attempt in range(3):
        try:
            r = client(force=attempt > 0).models.generate_content(model=MODEL, contents=prompt, config=cfg)
            return {"signature_id": sid, "ok": True, **json.loads(r.text)}
        except Exception as e:
            msg = str(e)
            if ("401" in msg or "UNAUTHENTICATED" in msg or "429" in msg) and attempt < 2:
                time.sleep(2 + attempt * 3); continue  # refresh token / back off, then retry
            return {"signature_id": sid, "ok": False, "error": msg[:200]}


def main():
    groups, stats, claims_by_sid = build_inputs()
    ids = list(groups)
    if PILOT:
        ids = ids[:PILOT]
    print(f"GOLD signatures to process: {len(ids)} (pilot={PILOT or 'no'})")

    done = {}
    if os.path.exists(CACHE) and not PILOT:
        for line in open(CACHE):
            try:
                d = json.loads(line)
                if d.get("ok"):  # only successes count as done; failures get retried
                    done[d["signature_id"]] = d
            except Exception: pass
    todo = [i for i in ids if i not in done]
    print(f"cached: {len(done)} | to call: {len(todo)}")

    results = dict(done)
    lock = threading.Lock()
    cache_fh = None if PILOT else open(CACHE, "a")
    with cf.ThreadPoolExecutor(max_workers=1 if PILOT else MAX_WORKERS) as ex:
        futs = {ex.submit(call_llm, i, groups[i]): i for i in todo}
        for k, fut in enumerate(cf.as_completed(futs), 1):
            res = fut.result(); results[res["signature_id"]] = res
            if cache_fh:
                with lock:
                    cache_fh.write(json.dumps(res) + "\n"); cache_fh.flush()
            if PILOT or k % 200 == 0:
                print(f"  {k}/{len(todo)} done")
    if cache_fh: cache_fh.close()

    if PILOT:
        for i in ids:
            print("\n" + "=" * 80)
            print(groups[i]["context"])
            print("-- LLM --")
            print(json.dumps(results[i], indent=2)[:1600])
        return

    # assemble golden_solutions.csv (one row per signature x repair-tier)
    rows = []
    for sid, res in results.items():
        if not res.get("ok") or sid not in stats.index:
            continue
        s = stats.loc[sid]
        nrep = int(res.get("n_repairs", 1) or 1)
        reps = res.get("repairs", [])[:3] or [{}]
        cdf = claims_by_sid.get(sid)
        for rep in reps:
            band = band_stats(tier_band(cdf, rep.get("cost_tier", "all"), nrep))
            rows.append({
                "signature_id": sid, "vehicle_line": s["vehicle_line"], "causal_part": s["causal_part"],
                "n_claims_group": int(s["n"]), "n_repairs_detected": nrep,
                "repair_name": rep.get("name", ""), "cost_tier": rep.get("cost_tier", "all"),
                "tier_n_claims": band["n"], "split_rationale": res.get("split_rationale", ""),
                "unified_diagnosis": rep.get("unified_diagnosis", ""),
                "suggested_correction": rep.get("suggested_correction", ""),
                "typical_parts": rep.get("typical_parts", ""),
                "labor_med_hrs": band["labor_med"], "cost_med": band["cost_med"],
                "cost_q25": band["cost_q25"], "cost_q75": band["cost_q75"],
                "parts_modal_set": s.get("parts_modal_set", ""), "confidence": s["confidence"],
            })
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/golden_solutions.csv", index=False)
    print(f"\nWrote {OUT}/golden_solutions.csv : {len(out)} solution rows over {out['signature_id'].nunique()} signatures")
    print(f"signatures split into >1 repair: {(out.groupby('signature_id')['n_repairs_detected'].first()>1).sum()}")


if __name__ == "__main__":
    main()
