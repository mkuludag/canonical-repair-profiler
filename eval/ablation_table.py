#!/usr/bin/env python3
"""
Emit the paper's ablation table (tab:ablation) straight from the eval/out artifacts.

Hand-transcribing twenty numbers out of JSON into LaTeX is where a fact-check finds errors, so the
table body is generated instead. Every cell here traces to eval/out/b2_llm_numbers*.json or
eval/out/b3_single_call*.json; nothing is typed twice.

    .venv/bin/python eval/ablation_table.py            # LaTeX body for the paper
    .venv/bin/python eval/ablation_table.py --json     # the same numbers as a dict, for checking

Layout is models-as-rows in two blocks (B2 production-evidence arm, B3 single call). The shipped
gemini-2.5-flash row leads each block. B2's 40-comment arm is deliberately absent: it is an evidence
arm rather than a model, it was not repeated on the gateway models, and its one job (5x evidence
does not close the gap) is a prose sentence.
"""
import argparse
import json

import common as C

# display name -> artifact tag ("" = the shipped Vertex run)
MODELS = [
    ("gemini-2.5-flash (shipped)", ""),
    ("claude-sonnet-5", "claude-sonnet-5"),
    ("gpt-5.4", "gpt-5.4-2026-03-05"),
    ("deepseek-v4-flash", "deepseekv4-flash"),
]
ARM = "s8"


def _pct(x, digits=1):
    return "--" if x is None else f"${x * 100:.{digits}f}\\%$"


def _signed(x, digits=1):
    return "--" if x is None else f"${x * 100:+.{digits}f}\\%$"


def _load(name):
    path = C.OUT_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


def b2_row(tag: str) -> dict:
    from llm_gateway_tool import suffixed
    doc = _load(suffixed("b2_llm_numbers.json", tag))
    if not doc:
        return {}
    arm = next((a for a in doc["arms"] if a["arm"] == ARM), None)
    if not arm:
        return {}
    e = arm["rel_error"]
    return {
        "group_med": e["group_cost_med"]["median_abs"], "group_p90": e["group_cost_med"]["p90_abs"],
        "tier_med": e["tier_cost_med"]["median_abs"], "tier_p90": e["tier_cost_med"]["p90_abs"],
        "iqr_med": e["tier_iqr_width"]["median_abs"], "iqr_bias": e["tier_iqr_width"]["median_signed"],
        "labor_med": e["tier_labor_med"]["median_abs"],
        "labor_bias": e["tier_labor_med"]["median_signed"],
        "band": arm["band_order_violation_rate"], "n_invalid": arm["n_schema_invalid"],
    }


def b3_row(tag: str) -> dict:
    from llm_gateway_tool import suffixed
    doc = _load(suffixed("b3_single_call.json", tag))
    if not doc:
        return {}
    e = doc["rel_error"]
    w = _load(suffixed("b3_weighted.json", tag)) or {}
    return {
        "group_med": e["group_cost_med"]["median_abs"], "group_p90": e["group_cost_med"]["p90_abs"],
        "tier_med": e["tier_cost_med"]["median_abs"], "tier_p90": e["tier_cost_med"]["p90_abs"],
        "iqr_med": e["tier_iqr_width"]["median_abs"], "iqr_bias": e["tier_iqr_width"]["median_signed"],
        "labor_med": e["tier_labor_med"]["median_abs"],
        "labor_bias": e["tier_labor_med"]["median_signed"],
        "band": doc["critic_gates"]["band_disorder_rate"], "n_invalid": doc["n_schema_invalid"],
        "over_proposal": (w.get("over_proposal") or {}).get("rate"),
        "over_proposal_se": (w.get("over_proposal") or {}).get("se"),
        "guard_veto": (w.get("guard_veto_among_splits") or {}).get("rate"),
        "guard_veto_se": (w.get("guard_veto_among_splits") or {}).get("se"),
    }


def latex() -> str:
    lines = [
        r"\begin{table}[t]",
        r"  \centering\footnotesize",
        r"  \caption{Ablations on the $400$-signature panel, relative error against the shipped",
        r"  \texttt{StatsTool} over the structure each variant chose. The group median is easy for",
        r"  every model; the per-tier numbers the library ships are not, the bands come back wide",
        r"  and the labor low in every case, and none of it fails the critic's gate.}",
        r"  \label{tab:ablation}",
        # six columns at the default 6pt tabcolsep overflows the NeurIPS 5.5in block by ~24pt
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{lccccc}",
        r"    \toprule",
        r"    & Group cost & Tier cost & Tier IQR & Tier labor & Band-order \\",
        r"    & med.\ (p90) & med.\ (p90) & med.\ (bias) & med.\ (bias) & violations \\",
        r"    \midrule",
        r"    \multicolumn{6}{l}{\emph{B2: the model writes the numbers} "
        r"($8$ comments, the production evidence)} \\",
    ]
    for block, fn in ((None, b2_row), ("B3", b3_row)):
        if block:
            lines += [r"    \midrule",
                      r"    \multicolumn{6}{l}{\emph{B3: one call instead of seven agents}} \\"]
        for name, tag in MODELS:
            r = fn(tag)
            if not r:
                lines.append(f"    \\quad {name} & \\multicolumn{{5}}{{c}}{{not run}} \\\\")
                continue
            lines.append(
                f"    \\quad {name} & {_pct(r['group_med'], 2 if r['group_med'] < 0.01 else 1)} "
                f"({_pct(r['group_p90'])}) & {_pct(r['tier_med'])} ({_pct(r['tier_p90'])}) & "
                f"{_pct(r['iqr_med'])} ({_signed(r['iqr_bias'])}) & "
                f"{_pct(r['labor_med'])} ({_signed(r['labor_bias'])}) & {_pct(r['band'])} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="dump the numbers instead of the LaTeX")
    args = ap.parse_args()
    if args.json:
        print(json.dumps({name or "gemini": {"b2": b2_row(tag), "b3": b3_row(tag)}
                          for name, tag in MODELS}, indent=2))
    else:
        print(latex())


if __name__ == "__main__":
    main()
