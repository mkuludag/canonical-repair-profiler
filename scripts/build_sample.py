"""
Build the bundled OFFLINE SAMPLE under data/sample/.

The sample is a tiny, masked slice of the real warranty data for a curated set of repair signatures.
It lets the dashboard, the inference CLI, the orchestrator and the test suite run WITH NO BigQuery /
Vertex access (CRP_USE_SAMPLE=1) - so a reviewer can reproduce the demo straight from the ZIP.

What it writes (all CSV, human-readable, so they double as Sample I/O documentation):
  data/sample/canonical_repairs.csv  - CRL rows for the sample signatures
  data/sample/golden_solutions.csv   - precomputed golden solutions (REAL numbers) for the sample
  data/sample/claim_signatures.csv   - request_r -> signature_id membership (capped per signature)
  data/sample/sample_claims.csv      - per-claim cost/labor + truncated technician text

Governance: VINs are never included; technician comment text is truncated to COMMENT_MAX chars.
Run:  python scripts/build_sample.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C  # noqa: E402

# Curated signatures: span multiple vehicle lines / parts and include a real LLM split (40109, 33030).
SAMPLE_SIGS = [
    32954,   # F-150 [15-20] Long Block - single repair, high consensus
    40109,   # Super Duty Oil Pan - LLM SPLIT: reseal vs replace
    71778,   # Transit Connect Transmission - single
    53967,   # Explorer Water Pump - very high support
    14641,   # Edge Engine Long Block
    35069,   # F-150 Transmission Assembly
    3936,    # Focus Clutch System
    33030,   # F-150 6256 - VCT phaser vs long block (split)
]
CAP_PER_SIG = 500       # cap member claims per signature to keep the sample small


def main():
    os.makedirs(C.SAMPLE_DIR, exist_ok=True)
    sigs = set(SAMPLE_SIGS)

    crl = pd.read_csv(C.CANONICAL_REPAIRS)
    crl[crl["signature_id"].isin(sigs)].to_csv(C.SAMPLE_CANONICAL_REPAIRS, index=False)

    gs = pd.read_csv(C.GOLDEN_SOLUTIONS)
    gs[gs["signature_id"].isin(sigs)].to_csv(C.SAMPLE_GOLDEN_SOLUTIONS, index=False)

    sig = pd.read_parquet(C.CLAIM_SIGNATURES, columns=["request_r", "signature_id"])
    sig["request_r"] = pd.to_numeric(sig["request_r"], errors="coerce").astype("Int64")
    sig = sig[sig["signature_id"].isin(sigs)].dropna(subset=["request_r"])
    sig = sig.sample(frac=1, random_state=7)  # shuffle, then cap per signature
    sig = sig.groupby("signature_id", group_keys=False).head(CAP_PER_SIG)
    sig.to_csv(C.SAMPLE_CLAIM_SIGNATURES, index=False)

    members = set(sig["request_r"].tolist())
    rp = pd.read_parquet(
        C.REPAIR_PROFILE,
        columns=["request_r", "paws_comment_trail", C.COST_COL, C.LABOR_COL])
    rp["request_r"] = pd.to_numeric(rp["request_r"], errors="coerce").astype("Int64")
    claims = rp[rp["request_r"].isin(members)].copy()
    for c in (C.COST_COL, C.LABOR_COL):
        claims[c] = pd.to_numeric(claims[c], errors="coerce")
    claims.loc[(claims[C.COST_COL] <= 0) | (claims[C.COST_COL] >= C.COST_CAP), C.COST_COL] = np.nan
    claims = claims.merge(sig, on="request_r", how="inner")
    # Synthesize a PII-free technician narrative. Real comment text carries customer names,
    # VINs, and dealer emails, so it is NOT exported. The synthetic text is theme-flavored and
    # cost-tier aware so the offline demo's symptom clustering / bimodality signal still works.
    theme = crl.set_index("signature_id")[["arch_theme", "vehicle_line"]].to_dict("index")
    med = claims.groupby("signature_id")[C.COST_COL].median()
    def _synth(r):
        t = theme.get(r["signature_id"], {})
        toks = [x.strip() for x in str(t.get("arch_theme", "")).split(",") if x.strip()][:3]
        symptom = ", ".join(toks) or "a driveability concern"
        hi = r[C.COST_COL] >= med.get(r["signature_id"], r[C.COST_COL])
        action = ("Diagnosis confirmed component failure; full assembly replacement completed"
                  if hi else "Diagnosis performed; minor repair/reseal completed")
        return (f"Customer reports concern related to {symptom}. {action} under warranty per "
                f"applicable TSB. Vehicle line {t.get('vehicle_line', '')}. "
                f"[synthetic sanitized narrative]")
    claims["paws_comment_trail"] = claims.apply(_synth, axis=1)
    claims.to_csv(C.SAMPLE_CLAIMS, index=False)

    print(f"Sample written to {C.SAMPLE_DIR}/")
    print(f"  signatures      : {len(sigs)}")
    print(f"  canonical rows  : {crl['signature_id'].isin(sigs).sum()}")
    print(f"  golden rows     : {gs['signature_id'].isin(sigs).sum()}")
    print(f"  membership rows : {len(sig)}")
    print(f"  claim rows      : {len(claims)}")
    tot = sum(os.path.getsize(p) for p in [
        C.SAMPLE_CANONICAL_REPAIRS, C.SAMPLE_GOLDEN_SOLUTIONS,
        C.SAMPLE_CLAIM_SIGNATURES, C.SAMPLE_CLAIMS])
    print(f"  total size      : {tot/1e6:.2f} MB")


if __name__ == "__main__":
    main()
