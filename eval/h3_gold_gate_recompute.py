#!/usr/bin/env python3
"""
H3 - GOLD gate cost-basis robustness.

grouping/08_build_ground_truth.py gates GOLD signatures on `det_approved_amt` (its COST constant),
while the shipped pipeline's cost column is `gsar_tot_cost_gross` (src/config.py COST_COL). This
script asks how much of the GOLD population is an artifact of that cost-basis choice.

Steps:
  1. Reproduce the shipped gate exactly on the det_approved_amt basis from claim-level data
     (data/claim_signatures.parquet joined to paws_discovery/repair_profile.parquet on request_r,
     which is unique in both). The replicated GOLD ID set must equal the shipped 3,623-signature
     set in grouping/out_v2/canonical_repairs.csv (status == GOLD); on any mismatch the diff is
     written to the JSON and the script exits non-zero without touching RESULTS.md.
  2. Recompute the identical gate with the cost basis switched to gsar_tot_cost_gross (labor
     column, validity rules, n >= 10 support floor, 0.6 threshold and skipna behaviour unchanged).
  3. Count one-component passes (NaN cost or NaN labor consensus, so gt_score leans on the
     remaining component(s) under skipna) on both bases.

Gate replication mirrors 08 exactly:
  cost validity 0 < x < 100,000 (else NaN); labor validity 0 < x < 200 (else NaN);
  consensus = share of valid claims within +/-25% of the signature median of valid claims
  (NaN when a signature has zero valid claims); text_cohesion == 1.0 by construction;
  gt_score = mean(text, cost_consensus, labor_consensus) with skipna, rounded to 3 dp;
  GOLD = n >= 10 and gt_score >= 0.6. Grouping by signature_id is equivalent to 08's groupby
  over (vehicle_line, causal_part, arch) because signature_id = ngroup over those keys.

Writes eval/out/h3_gold_gate.json and replaces the "H3 GOLD gate cost-basis robustness"
section of eval/RESULTS.md. Reads claim-level data; run from the repo root:

  PYTHONPATH=eval .venv/bin/python eval/h3_gold_gate_recompute.py [--no-append]
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

REPAIR_PROFILE = C.REPO_ROOT / "paws_discovery" / "repair_profile.parquet"
CLAIM_SIGNATURES = C.REPO_ROOT / "data" / "claim_signatures.parquet"
SHIPPED_CRL = C.REPO_ROOT / "grouping" / "out_v2" / "canonical_repairs.csv"

DET, GSAR, LAB = "det_approved_amt", "gsar_tot_cost_gross", "gsar_labor_hrs"
COST_LO, COST_HI = 0, 100_000      # 08: (x <= 0) | (x >= 100000) -> NaN
LAB_LO, LAB_HI = 0, 200            # 08: (x <= 0) | (x >= 200)    -> NaN
TOL, MIN_SUPPORT, GOLD_THRESHOLD = 0.25, 10, 0.6

SECTION = "H3 GOLD gate cost-basis robustness"
DATA_ROOT_NOTE = ("claim-level: data/claim_signatures.parquet + paws_discovery/repair_profile.parquet; "
                  "shipped GOLD set: grouping/out_v2/canonical_repairs.csv")


def consensus(df: pd.DataFrame, col: str, tol: float = TOL) -> pd.Series:
    """08's consensus(), grouped by signature_id: share of valid claims within tol of the median."""
    med = df.groupby("signature_id")[col].transform("median")
    valid = df[col].notna() & (med > 0)
    within = (df[col].sub(med).abs() <= tol * med) & valid
    t = pd.DataFrame({"_v": valid.astype(int), "_w": within.astype(int),
                      "signature_id": df["signature_id"].values})
    a = t.groupby("signature_id")[["_v", "_w"]].sum()
    return a["_w"] / a["_v"].where(a["_v"] > 0)


def gate(g: pd.DataFrame, n: pd.Series, labor_cons: pd.Series, cost_col: str) -> pd.DataFrame:
    """Per-signature gt_score and GOLD flag under the given cost basis (labor shared, per 08)."""
    agg = pd.DataFrame({"n": n, "cost_consensus": consensus(g, cost_col),
                        "labor_consensus": labor_cons, "text_cohesion": 1.0})
    agg["gt_score"] = (agg[["text_cohesion", "cost_consensus", "labor_consensus"]]
                       .mean(axis=1, skipna=True).round(3))
    agg["gold"] = (agg["n"] >= MIN_SUPPORT) & (agg["gt_score"] >= GOLD_THRESHOLD)
    return agg


def one_component(agg: pd.DataFrame, gold_ids: set) -> dict:
    """GOLD rows whose gate leaned on skipna: NaN cost and/or NaN labor consensus."""
    a = agg.loc[sorted(gold_ids)]
    cn, ln = a["cost_consensus"].isna(), a["labor_consensus"].isna()
    return {"nan_cost_only": int((cn & ~ln).sum()), "nan_labor_only": int((ln & ~cn).sum()),
            "nan_both": int((cn & ln).sum()), "any_nan_component": int((cn | ln).sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--no-append", action="store_true",
                    help="Compute and print, but do not touch eval/RESULTS.md.")
    args = ap.parse_args()

    sig = pd.read_parquet(CLAIM_SIGNATURES, columns=["request_r", "signature_id"])
    rp = pd.read_parquet(REPAIR_PROFILE, columns=["request_r", DET, GSAR, LAB])
    if sig["request_r"].duplicated().any() or rp["request_r"].duplicated().any():
        raise AssertionError("request_r is not unique; the 1:1 join assumption is broken")
    g = sig.merge(rp, on="request_r", how="left")
    if len(g) != len(sig):
        raise AssertionError("join changed row count; membership no longer matches 08's grouping")

    for c in (DET, GSAR):
        g[c] = pd.to_numeric(g[c], errors="coerce")
        g.loc[(g[c] <= COST_LO) | (g[c] >= COST_HI), c] = np.nan
    g[LAB] = pd.to_numeric(g[LAB], errors="coerce")
    g.loc[(g[LAB] <= LAB_LO) | (g[LAB] >= LAB_HI), LAB] = np.nan

    n = g.groupby("signature_id").size()
    labor_cons = consensus(g, LAB)  # labor basis is shared by both gates

    # ---- step 1: replicate the shipped (det) gate exactly --------------------------------------
    agg_det = gate(g, n, labor_cons, DET)
    gold_det = set(agg_det.index[agg_det["gold"]])
    crl = pd.read_csv(SHIPPED_CRL, low_memory=False)
    shipped = set(crl.loc[crl["status"] == "GOLD", "signature_id"].astype(int))

    missing = sorted(shipped - gold_det)   # shipped GOLD the replication fails to recover
    extra = sorted(gold_det - shipped)     # replicated GOLD not in the shipped set
    replication = {
        "shipped_gold": len(shipped), "replicated_gold": len(gold_det),
        "missing_from_replication": len(missing), "extra_in_replication": len(extra),
        "exact": not missing and not extra,
    }
    if missing or extra:
        replication["missing_ids"] = missing
        replication["extra_ids"] = extra
        C.write_json({"replication": replication, "status": "REPLICATION_FAILED"},
                     "h3_gold_gate.json")
        print(f"REPLICATION FAILED: {len(missing)} missing, {len(extra)} extra "
              f"(vs shipped {len(shipped)}). Diff written; not proceeding to the gsar gate.")
        return 1
    print(f"step 1: exact replication of the shipped GOLD set ({len(shipped):,} signatures)")

    # ---- step 2: identical gate on the gsar basis ----------------------------------------------
    agg_gsar = gate(g, n, labor_cons, GSAR)
    gold_gsar = set(agg_gsar.index[agg_gsar["gold"]])
    survivors = gold_det & gold_gsar
    dropouts = gold_det - gold_gsar
    entrants = gold_gsar - gold_det
    jaccard = len(survivors) / len(gold_det | gold_gsar)
    survivor_share = len(survivors) / len(gold_det)

    # ---- step 3: one-component (skipna) passes on both bases -----------------------------------
    oc_det = one_component(agg_det, gold_det)
    oc_gsar = one_component(agg_gsar, gold_gsar)

    out = {
        "gate": {"cost_validity": "0 < x < 100000", "labor_validity": "0 < x < 200",
                 "consensus_tol": TOL, "min_support": MIN_SUPPORT,
                 "gt_score": "mean(text=1.0, cost_consensus, labor_consensus), skipna, round(3)",
                 "gold_threshold": GOLD_THRESHOLD},
        "inputs": {"claim_signatures": str(CLAIM_SIGNATURES.relative_to(C.REPO_ROOT)),
                   "repair_profile": str(REPAIR_PROFILE.relative_to(C.REPO_ROOT)),
                   "shipped_gold_artifact": str(SHIPPED_CRL.relative_to(C.REPO_ROOT)),
                   "claims_in_grouping": int(len(g)), "signatures": int(n.size)},
        "replication": replication,
        "det_basis": {"gold": len(gold_det), "one_component_gate": oc_det},
        "gsar_basis": {"gold": len(gold_gsar),
                       "survivors": len(survivors), "dropouts": len(dropouts),
                       "entrants": len(entrants), "jaccard": round(jaccard, 4),
                       "survivor_share_of_shipped": round(survivor_share, 4),
                       "one_component_gate": oc_gsar,
                       "dropout_ids": sorted(int(s) for s in dropouts),
                       "entrant_ids": sorted(int(s) for s in entrants)},
    }
    C.write_json(out, "h3_gold_gate.json")

    body = textwrap.fill(
        f"The shipped GOLD gate (grouping/08_build_ground_truth.py, cost basis `det_approved_amt`) "
        f"reproduces exactly from claim-level data: {len(gold_det):,} GOLD signatures, an identical "
        f"ID set to `grouping/out_v2/canonical_repairs.csv` (0 missing, 0 extra). Rerunning the "
        f"identical gate with the cost basis switched to `gsar_tot_cost_gross` (the shipped "
        f"pipeline's COST_COL; labor, validity rules, n>=10 floor and 0.6 threshold unchanged) "
        f"yields {len(gold_gsar):,} GOLD signatures. Of the shipped {len(gold_det):,}, "
        f"{len(survivors):,} survive ({C.pct(survivor_share)}) and {len(dropouts):,} drop out "
        f"({C.pct(len(dropouts) / len(gold_det))}); {len(entrants):,} signatures newly enter. "
        f"Jaccard overlap between the two GOLD sets is {jaccard:.3f}. One-component passes "
        f"(NaN cost or NaN labor consensus, so the skipna mean leans on the remaining components): "
        f"{oc_det['any_nan_component']} of {len(gold_det):,} on the det basis "
        f"({oc_det['nan_cost_only']} cost-NaN, {oc_det['nan_labor_only']} labor-NaN, "
        f"{oc_det['nan_both']} both) and {oc_gsar['any_nan_component']} of {len(gold_gsar):,} on "
        f"the gsar basis ({oc_gsar['nan_cost_only']} cost-NaN, {oc_gsar['nan_labor_only']} "
        f"labor-NaN, {oc_gsar['nan_both']} both). Full ID sets: `eval/out/h3_gold_gate.json`.",
        width=100, break_on_hyphens=False)
    C.append_results(SECTION, body, DATA_ROOT_NOTE, enabled=not args.no_append)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
