#!/usr/bin/env python3
"""
H addenda - artifact backing for two numbers the paper states in prose.

(a) Truncation-excluded rerun agreement. eval/out/a7_churn.json reports split agreement
    (n_repairs_v1 == n_repairs_v2) over all 3,623 rerun signatures. 388 of those carry
    tier_pair_v1 == "low+mid": the v1 analyst proposed 3 repairs and the library emitted 2, so
    v1's own decision was truncated and agreement against it is not a like-for-like comparison.
    This recomputes agreement excluding those 388 rows from eval/out/a7_churn_per_signature.csv.

(b) The 2,550 -> 2,546 derivation. golden_solutions.csv holds 2,550 single-repair GOLD
    signatures, but grouping/out_v2/group_savings.csv holds 2,546 rows. The four dropped
    signature_ids are materialized here with their member-claim counts and the reason: zero of
    their claims carry a valid gsar_tot_cost_gross, so grouping/12_group_savings.py has neither
    a canonical cost (cost_med is NaN in golden_solutions.csv) nor any claim surviving its
    cost filter. They are GOLD under the shipped det-basis gate via the skipna one-component
    path (text + labor consensus only).

Writes eval/out/h_addenda.json and replaces the "H addenda: artifact backing for prose claims"
section of eval/RESULTS.md. Run from the repo root:

  PYTHONPATH=eval .venv/bin/python eval/h_addenda.py [--no-append]
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

A7_CSV = C.OUT_DIR / "a7_churn_per_signature.csv"
GOLDEN = C.REPO_ROOT / "grouping" / "out_v2" / "golden_solutions.csv"
GROUP_SAVINGS = C.REPO_ROOT / "grouping" / "out_v2" / "group_savings.csv"
CLAIM_SIGNATURES = C.REPO_ROOT / "data" / "claim_signatures.parquet"
REPAIR_PROFILE = C.REPO_ROOT / "paws_discovery" / "repair_profile.parquet"

GSAR = "gsar_tot_cost_gross"
EXPECTED_DROPPED = {5799, 10355, 14626, 32910}

SECTION = "H addenda: artifact backing for prose claims"
DATA_ROOT_NOTE = ("eval/out/a7_churn_per_signature.csv + grouping/out_v2 + "
                  "claim-level parquet (data/, paws_discovery/)")


def addendum_a() -> dict:
    """Split agreement with the 388 truncated (tier_pair_v1 == low+mid) signatures excluded."""
    a7 = pd.read_csv(A7_CSV)
    agree = a7["n_repairs_v1"] == a7["n_repairs_v2"]
    truncated = a7["tier_pair_v1"] == "low+mid"
    if int(truncated.sum()) != 388:
        raise AssertionError(f"expected 388 low+mid rows, found {int(truncated.sum())}")
    kept_agree = int((agree & ~truncated).sum())
    kept_total = int((~truncated).sum())
    return {
        "source": str(A7_CSV.relative_to(C.REPO_ROOT)),
        "agreement_definition": "n_repairs_v1 == n_repairs_v2",
        "n_signatures": int(len(a7)),
        "agree_all": int(agree.sum()),
        "agreement_all": round(float(agree.mean()), 4),
        "excluded_low_mid": int(truncated.sum()),
        "agree_among_excluded": int((agree & truncated).sum()),
        "agree_excluding_truncated": kept_agree,
        "n_excluding_truncated": kept_total,
        "agreement_excluding_truncated": round(kept_agree / kept_total, 4),
    }


def addendum_b() -> dict:
    """The four single-repair GOLD signatures absent from group_savings.csv, with the reason."""
    gs = pd.read_csv(GOLDEN, low_memory=False)
    single = gs[pd.to_numeric(gs["n_repairs_detected"], errors="coerce") == 1]
    single_ids = set(single["signature_id"].astype(int))
    sv_ids = set(pd.read_csv(GROUP_SAVINGS)["signature_id"].astype(int))
    dropped = sorted(single_ids - sv_ids)
    if set(dropped) != EXPECTED_DROPPED:
        raise AssertionError(f"dropped set {dropped} != expected {sorted(EXPECTED_DROPPED)}")

    sig = pd.read_parquet(CLAIM_SIGNATURES, columns=["request_r", "signature_id"])
    sig = sig[sig["signature_id"].isin(dropped)]
    rp = pd.read_parquet(REPAIR_PROFILE, columns=["request_r", GSAR])
    m = sig.merge(rp, on="request_r", how="left")
    m[GSAR] = pd.to_numeric(m[GSAR], errors="coerce")
    m["valid"] = (m[GSAR] > 0) & (m[GSAR] < 100_000)

    reason = (f"zero member claims have a valid {GSAR} (0 < x < 100,000), so cost_med is NaN in "
              "golden_solutions.csv and grouping/12_group_savings.py drops the signature (no "
              "canonical cost, and no claim survives its cost filter); GOLD status came from the "
              "det-basis gate via the skipna one-component path (text + labor consensus only)")
    per_sig = []
    for sid, grp in m.groupby("signature_id"):
        n_valid = int(grp["valid"].sum())
        if n_valid != 0:
            raise AssertionError(f"signature {sid} has {n_valid} valid {GSAR} claims; reason wrong")
        per_sig.append({"signature_id": int(sid), "n_claims": int(len(grp)),
                        "n_valid_gsar_cost": n_valid, "reason": reason})
    return {
        "single_repair_gold_signatures": len(single_ids),
        "group_savings_rows": len(sv_ids),
        "dropped_signatures": per_sig,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--no-append", action="store_true",
                    help="Compute and print, but do not touch eval/RESULTS.md.")
    args = ap.parse_args()

    a = addendum_a()
    b = addendum_b()
    C.write_json({"a_truncation_excluded_churn": a, "b_group_savings_2550_to_2546": b},
                 "h_addenda.json")

    drop_ids = ", ".join(str(d["signature_id"]) for d in b["dropped_signatures"])
    claim_counts = ", ".join(str(d["n_claims"]) for d in b["dropped_signatures"])
    para_a = textwrap.fill(
        f"(a) Truncation-excluded rerun agreement. Over all {a['n_signatures']:,} rerun "
        f"signatures, split agreement (n_repairs_v1 == n_repairs_v2) is "
        f"{a['agree_all']:,}/{a['n_signatures']:,} ({C.pct(a['agreement_all'])}). Excluding the "
        f"{a['excluded_low_mid']} signatures with tier_pair_v1 == \"low+mid\" (v1 analyst proposed "
        f"3 repairs, library emitted 2, so the v1 decision was truncated), of which "
        f"{a['agree_among_excluded']} agreed, agreement is "
        f"({a['agree_all']:,}-{a['agree_among_excluded']})/"
        f"({a['n_signatures']:,}-{a['excluded_low_mid']}) = "
        f"{a['agree_excluding_truncated']:,}/{a['n_excluding_truncated']:,} = "
        f"{C.pct(a['agreement_excluding_truncated'])} "
        f"({a['agreement_excluding_truncated']:.4f}).",
        width=100, break_on_hyphens=False)
    para_b = textwrap.fill(
        f"(b) The 2,550 to 2,546 derivation. golden_solutions.csv holds "
        f"{b['single_repair_gold_signatures']:,} single-repair GOLD signatures; group_savings.csv "
        f"holds {b['group_savings_rows']:,} rows. The four dropped signature_ids are {drop_ids} "
        f"({claim_counts} member claims respectively). None of their claims carry a valid "
        f"gsar_tot_cost_gross, so grouping/12_group_savings.py has no canonical cost for them "
        f"(cost_med is NaN) and no claim survives its cost filter; their GOLD status came from the "
        f"det-basis gate via the skipna one-component path (text + labor consensus only). "
        f"Details: `eval/out/h_addenda.json`.",
        width=100, break_on_hyphens=False)
    body = f"{para_a}\n\n{para_b}"
    C.append_results(SECTION, body, DATA_ROOT_NOTE, enabled=not args.no_append)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
