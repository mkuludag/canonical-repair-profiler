#!/usr/bin/env python3
"""
b0 - freeze the fixed 400-signature evaluation panel shared by all B-studies (b1/b2/b3).

Stratified by what the v2 library of record says happened to each GOLD signature
(eval/common.classify_signatures): the analyst never proposed a split (`single`), proposed and the
guard honored it (`honored`), or proposed and the guard collapsed it (`collapsed`). `collapsed` is
oversampled (its population share would give ~32 signatures); population-level numbers are recovered
by reweighting with the population shares written to b0_panel.json.

Deterministic by construction: sorted stratum populations + a fixed-seed Generator, so re-running
reproduces eval/panel_400.csv byte-identically. The panel file carries only signature ids + strata
(no claim data), so it is committed as the study's artifact of record.

Usage:
    .venv/bin/python eval/b0_panel.py                      # default: grouping/out_v2
    .venv/bin/python eval/b0_panel.py --check              # verify the frozen panel, write nothing
"""
import argparse

import numpy as np
import pandas as pd

import common as C

SEED = 20260731
STRATA = {"single": 250, "honored": 100, "collapsed": 50}
PANEL_CSV = C.EVAL_DIR / "panel_400.csv"


def build_panel(data_root) -> tuple[pd.DataFrame, dict]:
    classes = C.classify_signatures(C.load_golden(data_root))
    rng = np.random.default_rng(SEED)
    parts, strata_info = [], {}
    for klass in ("single", "honored", "collapsed"):   # fixed order -> fixed rng stream
        pop = sorted(int(s) for s in classes.loc[classes["klass"] == klass, "signature_id"])
        take = STRATA[klass]
        if take > len(pop):
            raise SystemExit(f"stratum '{klass}' has only {len(pop)} signatures, need {take}")
        chosen = sorted(rng.choice(pop, size=take, replace=False).tolist())
        parts.append(pd.DataFrame({"signature_id": chosen, "stratum": klass}))
        strata_info[klass] = {
            "population": len(pop),
            "sampled": take,
            # weight for population-level estimates: the stratum's share of all GOLD signatures
            "population_share": round(len(pop) / len(classes), 6),
        }
    panel = pd.concat(parts).sort_values("signature_id").reset_index(drop=True)
    assert panel["signature_id"].is_unique and len(panel) == sum(STRATA.values())
    meta = {"seed": SEED, "n_panel": int(len(panel)),
            "n_gold_population": int(len(classes)), "strata": strata_info,
            "data_root": str(data_root)}
    return panel, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(C.REPO_ROOT / "grouping" / "out_v2"),
                    help="library of record the strata are read from (default: grouping/out_v2)")
    ap.add_argument("--check", action="store_true",
                    help="rebuild in memory and verify it matches the frozen panel_400.csv")
    args = ap.parse_args()

    panel, meta = build_panel(args.data_root)
    if args.check:
        frozen = pd.read_csv(PANEL_CSV)
        pd.testing.assert_frame_equal(frozen, panel)
        print(f"OK: frozen panel matches a fresh rebuild ({len(panel)} signatures)")
        return

    panel.to_csv(PANEL_CSV, index=False)
    print(f"wrote {PANEL_CSV.relative_to(C.REPO_ROOT)}: "
          + ", ".join(f"{v} {k}" for k, v in STRATA.items()))
    C.write_json(meta, "b0_panel.json")


if __name__ == "__main__":
    main()
