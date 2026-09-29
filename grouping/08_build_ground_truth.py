#!/usr/bin/env python3
"""
Build the FULL grouping + emit the Canonical Repair Library (CRL).  See METHODOLOGY.md.

Steps:
  1. assign every claim a Repair Signature = vehicle_line x causal_part x symptom_archetype
  2. for signatures with n>=10, derive a Canonical Repair (consensus solution + confidence + status)
  3. persist the full grouping and the CRL

Outputs:
  data/claim_signatures.parquet            every claim -> signature keys + id   (gitignored)
  grouping/out/canonical_repairs.csv       the CRL (one row per signature, n>=10)
  grouping/out/canonical_repairs_gold_sample.csv
  grouping/out/crl_status_summary.csv
Memory-aware: only needed columns; vectorized consensus.
"""
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

PARQUET = os.environ.get("REPAIR_PARQUET", "data/repair_profile.parquet")
OUT = "grouping/out"; os.makedirs(OUT, exist_ok=True); os.makedirs("data", exist_ok=True)
ARCH_K = int(os.environ.get("ARCH_K", "60")); FIT_SAMPLE = 150_000
KEYS = ["gsar_veh_line_desc", "paws_causal_part", "arch"]
COST, LAB, TXT = "det_approved_amt", "gsar_labor_hrs", "paws_comment_trail"
MIN_SUPPORT = 10


def consensus(df, keys, col, tol=0.25):
    med = df.groupby(keys, observed=True)[col].transform("median")
    valid = df[col].notna() & (med > 0)
    within = (df[col].sub(med).abs() <= tol * med) & valid
    t = pd.DataFrame({"_v": valid.astype(int), "_w": within.astype(int)})
    for k in keys:
        t[k] = df[k].values
    a = t.groupby(keys, observed=True)[["_v", "_w"]].sum()
    return a["_w"] / a["_v"].where(a["_v"] > 0)


def main():
    cols = ["request_r", "gsar_veh_line_desc", "paws_causal_part", COST, LAB, TXT,
            "part_numbers", "gsar_camp_recall_flag"]
    print("loading ...")
    df = pd.read_parquet(PARQUET, columns=cols)
    df[COST] = pd.to_numeric(df[COST], errors="coerce"); df.loc[(df[COST] <= 0) | (df[COST] >= 100000), COST] = np.nan
    df[LAB] = pd.to_numeric(df[LAB], errors="coerce");  df.loc[(df[LAB] <= 0) | (df[LAB] >= 200), LAB] = np.nan
    print(f"  {len(df):,} claims")

    # --- 1. symptom archetypes ---
    has = df[TXT].notna() & (df[TXT].str.len() > 15)
    texts = df.loc[has, TXT]
    vec = TfidfVectorizer(max_features=4000, stop_words="english", ngram_range=(1, 2),
                          min_df=5, max_df=0.7, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
    Xfit = vec.fit_transform(texts.sample(min(FIT_SAMPLE, len(texts)), random_state=42))
    km = MiniBatchKMeans(n_clusters=ARCH_K, random_state=42, n_init=3, batch_size=2048).fit(Xfit)
    df["arch"] = np.nan
    df.loc[has, "arch"] = km.predict(vec.transform(texts))
    terms = np.array(vec.get_feature_names_out()); order = km.cluster_centers_.argsort()[:, ::-1]
    arch_theme = {k: ", ".join(terms[order[k, :8]]) for k in range(ARCH_K)}
    print(f"  archetypes assigned to {has.sum():,} claims")

    # drop claims missing any signature key (need all 3 to be in a signature)
    g = df.dropna(subset=KEYS).copy()
    g["arch"] = g["arch"].astype(int)
    g["signature_id"] = g.groupby(KEYS, observed=True).ngroup()
    print(f"  {g['signature_id'].nunique():,} signatures over {len(g):,} claims")

    # --- persist full grouping ---
    g[["request_r"] + KEYS + ["signature_id"]].to_parquet("data/claim_signatures.parquet", index=False)

    # --- 2. per-signature consensus + solution ---
    grp = g.groupby(KEYS, observed=True)
    agg = grp.agg(
        n=("request_r", "size"),
        cost_med=(COST, "median"), cost_q25=(COST, lambda s: s.quantile(.25)), cost_q75=(COST, lambda s: s.quantile(.75)),
        labor_med=(LAB, "median"), labor_q25=(LAB, lambda s: s.quantile(.25)), labor_q75=(LAB, lambda s: s.quantile(.75)),
        recall_share=("gsar_camp_recall_flag", lambda s: (s == "Y").mean()),
        signature_id=("signature_id", "first"),
    )
    agg["cost_consensus"] = consensus(g, KEYS, COST)
    agg["labor_consensus"] = consensus(g, KEYS, LAB)
    # text cohesion is 1.0 by construction (archetype is a key); track solution_consensus separately (stricter)
    agg["text_cohesion"] = 1.0
    agg["solution_consensus"] = agg[["cost_consensus", "labor_consensus"]].mean(axis=1, skipna=True).round(3)
    agg["gt_score"] = agg[["text_cohesion", "cost_consensus", "labor_consensus"]].mean(axis=1, skipna=True).round(3)
    support_w = np.minimum(1.0, np.log10(agg["n"].clip(lower=1)) / 2.0)
    agg["confidence"] = (agg["gt_score"] * support_w).round(3)
    agg["arch_theme"] = agg.index.get_level_values("arch").map(arch_theme)

    def status(r):
        if r["n"] < MIN_SUPPORT: return "SPARSE"
        if r["gt_score"] >= 0.6: return "GOLD"
        if r["n"] >= 20 and r["gt_score"] < 0.4: return "HOTSPOT"
        return "SILVER"
    agg["status"] = agg.apply(status, axis=1)

    emitted = agg[agg["n"] >= MIN_SUPPORT].copy()

    # --- 3. solution text + parts for emitted signatures ---
    em_keys = set(map(tuple, emitted.index.tolist()))
    sub = g[g.set_index(KEYS).index.isin(em_keys)].copy()
    # NOTE: a per-signature `correction_example` (verbatim technician/customer narrative) was
    # intentionally REMOVED for the anonymized/academic export -- raw comment text carries PII
    # (customer names, VINs, dealer emails). Downstream consumers use only the numeric consensus.
    # modal parts among parts-present claims
    pp = sub.dropna(subset=["part_numbers"])
    if len(pp):
        pm = (pp.groupby(KEYS + ["part_numbers"], observed=True).size().rename("c").reset_index())
        idx = pm.groupby(KEYS, observed=True)["c"].idxmax()
        modal = pm.loc[idx].set_index(KEYS)
        tot = pp.groupby(KEYS, observed=True).size().rename("pp_n")
        parts = modal.join(tot)
        parts["parts_modal_support"] = (parts["c"] / parts["pp_n"]).round(2)
        parts = parts.rename(columns={"part_numbers": "parts_modal_set"})[["parts_modal_set", "parts_modal_support"]]
    else:
        parts = pd.DataFrame(columns=["parts_modal_set", "parts_modal_support"])

    crl = (emitted.join(parts).reset_index()
           .rename(columns={"gsar_veh_line_desc": "vehicle_line", "paws_causal_part": "causal_part"}))
    for c in ["cost_med", "cost_q25", "cost_q75", "labor_med", "labor_q25", "labor_q75", "recall_share"]:
        crl[c] = pd.to_numeric(crl[c], errors="coerce").round(2)
    col_order = ["signature_id", "vehicle_line", "causal_part", "arch", "arch_theme", "n", "status",
                 "confidence", "gt_score", "solution_consensus", "cost_consensus", "labor_consensus",
                 "cost_med", "cost_q25", "cost_q75", "labor_med", "labor_q25", "labor_q75",
                 "parts_modal_set", "parts_modal_support", "recall_share"]
    crl = crl[[c for c in col_order if c in crl.columns]].sort_values(["status", "confidence"], ascending=[True, False])
    crl.to_csv(f"{OUT}/canonical_repairs.csv", index=False)

    summ = crl.groupby("status").agg(signatures=("n", "size"), claims=("n", "sum"),
                                     med_confidence=("confidence", "median")).reset_index()
    summ.to_csv(f"{OUT}/crl_status_summary.csv", index=False)
    crl[crl["status"] == "GOLD"].head(60).to_csv(f"{OUT}/canonical_repairs_gold_sample.csv", index=False)

    print("\n=== Canonical Repair Library status ===")
    print(summ.to_string(index=False))
    print(f"\nemitted signatures (n>=10): {len(crl):,} | claims covered: {crl['n'].sum():,} "
          f"({100*crl['n'].sum()/len(g):.1f}% of grouped claims)")
    print(f"GOLD: {(crl['status']=='GOLD').sum():,}  SILVER: {(crl['status']=='SILVER').sum():,}  "
          f"HOTSPOT: {(crl['status']=='HOTSPOT').sum():,}")
    print("\nWrote: data/claim_signatures.parquet, grouping/out/canonical_repairs.csv (+ gold sample, status summary)")


if __name__ == "__main__":
    main()
