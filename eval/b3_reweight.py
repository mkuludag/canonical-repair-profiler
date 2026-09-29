#!/usr/bin/env python3
"""
b3 population reweighting: the two single-call numbers the paper actually quotes.

b3_single_call.py reports panel-RAW rates. The panel oversamples the collapsed stratum, so the
population-level over-proposal and guard-veto rates need reweighting by stratum share, with
standard errors. For the shipped Gemini run that computation lives inside h_stats_bundle.py
(`ses_and_tests.b3_weighted`); this script lifts exactly that arithmetic out so it can run
per model without re-running - or overwriting - the rest of the bundle.

Quoting the raw rate next to the paper's weighted one is the same-denominator error class the
fact-check pass exists to catch: for Gemini the guard veto is 30.6% raw and 23.5% weighted.

    .venv/bin/python eval/b3_reweight.py                              # reproduces the shipped run
    .venv/bin/python eval/b3_reweight.py --model claude-sonnet-5
    .venv/bin/python eval/b3_reweight.py --all                        # every model with a b3 CSV

Writes eval/out/b3_weighted[_<model>].json. With no --model it also asserts that its output matches
h_stats_bundle.json's b3_weighted block, so a drift between the two implementations is loud.
"""
import argparse
import json
import math

import pandas as pd

import common as C
from llm_gateway_tool import slug, suffixed

PER_SIG = "b3_single_call_per_signature.csv"


def reweight(tag: str = "") -> dict:
    """Population-weighted over-proposal and guard-veto rates, with SEs, for one model's b3 run."""
    path = C.OUT_DIR / suffixed(PER_SIG, tag)
    if not path.exists():
        raise SystemExit(f"missing {path.relative_to(C.REPO_ROOT)} - run b3 --analyze first")
    b3 = pd.read_csv(path)
    w = C.panel_weights()
    strata = sorted(w)
    b3["prop"] = (pd.to_numeric(b3["n_repairs"]) >= 2).astype(int)

    over_terms, veto_terms = {}, {}
    for s in strata:
        sub = b3[b3["stratum"] == s]
        splits = sub[sub["prop"] == 1]
        over_terms[s] = {"w": w[s], "p": float(sub["prop"].mean()), "n": int(len(sub))}
        veto_terms[s] = {"p": float(splits["guard_collapsed"].astype(float).mean()),
                         "n": int(len(splits))}
    over_rate = sum(t["w"] * t["p"] for t in over_terms.values())
    se_over = math.sqrt(sum(t["w"] ** 2 * t["p"] * (1 - t["p"]) / t["n"]
                            for t in over_terms.values()))
    # veto rate among proposed splits: splits weighted by population incidence w_s * p_s
    wsplit = {s: over_terms[s]["w"] * over_terms[s]["p"] / over_rate for s in strata}
    veto_rate = sum(wsplit[s] * veto_terms[s]["p"] for s in strata)
    se_veto = math.sqrt(sum(wsplit[s] ** 2 * veto_terms[s]["p"] * (1 - veto_terms[s]["p"])
                            / veto_terms[s]["n"] for s in strata))

    return {
        "model": tag or "gemini-2.5-flash",
        "over_proposal": {"rate": round(over_rate, 4), "se": round(se_over, 4),
                          "rate_panel_raw": round(float(b3["prop"].mean()), 4),
                          "per_stratum": over_terms},
        "guard_veto_among_splits": {
            "rate": round(veto_rate, 4), "se": round(se_veto, 4),
            "rate_panel_raw": round(float(b3[b3["prop"] == 1]["guard_collapsed"]
                                          .astype(float).mean()), 4),
            "per_stratum": veto_terms, "split_incidence_weights": wsplit,
            "note": ("splits weighted by population incidence w_s*p_s; p_s treated as fixed. A "
                     "stratum whose veto rate is 0 or 1 contributes zero variance, so the SE is "
                     "slightly understated."),
        },
    }


def _cross_check(out: dict) -> None:
    """The shipped run must agree with h_stats_bundle to the digit, or one of them has drifted."""
    bundle = C.OUT_DIR / "h_stats_bundle.json"
    if not bundle.exists():
        return
    ref = json.loads(bundle.read_text())["ses_and_tests"]["b3_weighted"]
    for key in ("over_proposal", "guard_veto_among_splits"):
        for field in ("rate", "se"):
            a, b = out[key][field], ref[key][field]
            if abs(a - b) > 5e-4:
                raise SystemExit(f"DRIFT: b3_weighted.{key}.{field} = {a} here vs {b} in "
                                 "h_stats_bundle.json - do not quote either until this is resolved")
    print("  cross-check vs h_stats_bundle.json: identical")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="", help="LLMGateway model id; omit for the shipped Gemini run")
    ap.add_argument("--all", action="store_true", help="every model that has a b3 per-signature CSV")
    args = ap.parse_args()

    tags = [""]
    if args.all:
        stem = PER_SIG[:-len(".csv")]
        tags = [""] + sorted(p.name[len(stem) + 1:-len(".csv")]
                             for p in C.OUT_DIR.glob(f"{stem}_*.csv"))
    elif args.model:
        tags = [slug(args.model)]

    for tag in tags:
        if not (C.OUT_DIR / suffixed(PER_SIG, tag)).exists():
            continue
        out = reweight(tag)
        C.write_json(out, suffixed("b3_weighted.json", tag))
        if not tag:
            _cross_check(out)
        o, v = out["over_proposal"], out["guard_veto_among_splits"]
        print(f"  {out['model']}: over-proposal {C.pct(o['rate'])} +/- {o['se'] * 100:.1f}pp "
              f"(raw {C.pct(o['rate_panel_raw'])}); guard veto {C.pct(v['rate'])} "
              f"+/- {v['se'] * 100:.1f}pp (raw {C.pct(v['rate_panel_raw'])})")


if __name__ == "__main__":
    main()
