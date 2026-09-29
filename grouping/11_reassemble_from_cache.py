#!/usr/bin/env python3
"""
Re-assemble grouping/out/golden_solutions.csv from the cached LLM decisions using the SAME
SolutionAssemblyAgent the live pipeline uses (correct cost basis = gsar_tot_cost_gross + the
separation guard). No Gemini calls — costs/structure are deterministic, only cached text is reused.
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.tools import StatsTool
from src.agents.context import RepairContext
from src.agents.solution_assembly_agent import SolutionAssemblyAgent

CRL = "grouping/out/canonical_repairs.csv"
CACHE = "grouping/out/llm_cache.jsonl"
SIGS = "data/claim_signatures.parquet"
RP = "data/repair_profile.parquet"
OUT = "grouping/out/golden_solutions.csv"

crl = pd.read_csv(CRL); crl = crl[crl["status"] == "GOLD"].set_index("signature_id")
gold_ids = set(crl.index)

cache = {}
for line in open(CACHE):
    try:
        d = json.loads(line)
    except Exception:
        continue
    if d.get("ok") or d.get("_ok"):
        cache[d.get("signature_id")] = d

print("loading claims ...")
sig = pd.read_parquet(SIGS, columns=["request_r", "signature_id"])
sig["request_r"] = pd.to_numeric(sig["request_r"], errors="coerce").astype("Int64")
sig = sig[sig["signature_id"].isin(gold_ids)]
rp = pd.read_parquet(RP, columns=["request_r", "gsar_tot_cost_gross", "gsar_labor_hrs"])
rp["request_r"] = pd.to_numeric(rp["request_r"], errors="coerce").astype("Int64")
for c in ("gsar_tot_cost_gross", "gsar_labor_hrs"):
    rp[c] = pd.to_numeric(rp[c], errors="coerce")
rp.loc[(rp["gsar_tot_cost_gross"] <= 0) | (rp["gsar_tot_cost_gross"] >= 100000), "gsar_tot_cost_gross"] = np.nan
claims = sig.merge(rp, on="request_r", how="inner")
by_sid = {int(s): g for s, g in claims.groupby("signature_id")}

assembler = SolutionAssemblyAgent(StatsTool())
rows = []
for sid in gold_ids:
    dec = cache.get(sid)
    if dec is None or sid not in by_sid:
        continue
    r = crl.loc[sid]
    ctx = RepairContext(signature_id=int(sid), vehicle_line=r["vehicle_line"],
                        causal_part=str(r["causal_part"]), archetype=int(r["arch"]),
                        archetype_theme=str(r.get("arch_theme", "")))
    ctx.claims = by_sid[sid]
    ctx.llm_decision = {"n_repairs": dec.get("n_repairs", 1),
                        "split_rationale": dec.get("split_rationale", ""),
                        "repairs": dec.get("repairs", [{}])}
    assembler.assemble(ctx)
    rows.extend(ctx.solutions)

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(f"wrote {OUT}: {len(out)} rows over {out['signature_id'].nunique()} signatures")
print(f"  split (>1 repair): {(out.groupby('signature_id')['n_repairs_detected'].first()>1).sum()} signatures")
c = pd.to_numeric(out['cost_med'], errors='coerce')
print(f"  cost_med  median ${c.median():,.0f}  p90 ${c.quantile(.9):,.0f}  max ${c.max():,.0f}")
