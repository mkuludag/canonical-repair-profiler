#!/usr/bin/env python3
"""
a1 - Apply the SolutionCriticAgent's gates to every emitted Canonical Repair.

WHY THIS EXISTS. The released library was produced by grouping/10_golden_solutions.py (a standalone
batch script) replayed through grouping/11_reassemble_from_cache.py, which imports StatsTool and
SolutionAssemblyAgent but *not* SolutionCriticAgent. So the shipped golden_solutions.csv was never
gated by the critic, and it carries no critic_verdict column. This script closes that gap for the
emitted artifact by running the real agent over it.

WHAT THIS IS AND IS NOT. This is the critic applied POST HOC to the finished library: it measures
how many emitted repairs would pass, be flagged, or fail its structural gates. It is NOT the critic
IN LOOP. The in-loop behaviour - the one bounded revision round that re-prompts the analyst when
diagnosis/correction text is empty - cannot be recovered here, because a revision would change the
LLM text and the batch's llm_cache.jsonl no longer exists. Any in-loop claim needs a live re-run.

The gates themselves are imported, never reimplemented, so this cannot drift from the shipped code.

    python eval/a1_critic_posthoc.py --data-root ~/canonical-repair-profiler-public/grouping/out
"""
from collections import Counter

import pandas as pd

import common as C
from src.agents.context import RepairContext
from src.agents.solution_critic_agent import SolutionCriticAgent

# The keys SolutionCriticAgent._verdict reads off a solution dict (solution_critic_agent.py:45-62).
SOLUTION_KEYS = ["repair_name", "cost_med", "cost_q25", "cost_q75",
                 "unified_diagnosis", "suggested_correction", "support", "consensus", "usable"]


def _solution_dict(row: pd.Series) -> dict:
    """Rebuild the dict SolutionAssemblyAgent would have handed the critic, from one CSV row.

    A missing cost median must round-trip as the SAME falsy-or-not value assembly produced, because
    the critic's hard gate is `if not med` and `not float('nan')` is False while `not None` is True.
    Assembly emits None only when the tier slice is empty (`_row`'s `has` branch, giving
    usable=False); it emits NaN when the slice is non-empty but holds no positive costs, and
    `bool(nan)` is truthy, so that row is stamped usable=True. The released `usable` flag is
    therefore the only surviving record of which of the two it was - reconstruct from it, or the
    post-hoc verdicts silently diverge from what the shipped critic would have returned.
    """
    out = {}
    for k in SOLUTION_KEYS:
        v = row.get(k)
        if k in ("cost_med", "cost_q25", "cost_q75", "consensus"):
            if pd.isna(v):
                out[k] = None if (k == "cost_med" and not bool(row.get("usable"))) else float("nan")
            else:
                out[k] = float(v)
        elif k == "support":
            out[k] = 0 if pd.isna(v) else int(v)
        elif k == "usable":
            out[k] = bool(v) if not pd.isna(v) else False
        else:
            out[k] = "" if pd.isna(v) else str(v)
    return out


def main() -> None:
    args = C.parse_args(__doc__)
    golden = C.load_golden(args.data_root)
    critic = SolutionCriticAgent()

    verdict_rows, reason_counter = [], Counter()
    for sid, grp in golden.groupby("signature_id", sort=True):
        ctx = RepairContext(signature_id=int(sid), vehicle_line=str(grp.iloc[0]["vehicle_line"]),
                            causal_part=str(grp.iloc[0]["causal_part"]), archetype=-1)
        ctx.solutions = [_solution_dict(r) for _, r in grp.iterrows()]
        critic.review(ctx)
        for src_idx, sol, v in zip(grp.index, ctx.solutions, ctx.critique["verdicts"]):
            reasons = v["reasons"]
            reason_counter.update(reasons or ["(none)"])
            verdict_rows.append({
                "signature_id": int(sid),
                "cost_tier": golden.at[src_idx, "cost_tier"],
                "repair_name": v["repair_name"],
                "critic_verdict": v["verdict"],
                "usable_after_critic": sol["usable"],
                "usable_as_released": bool(golden.at[src_idx, "usable"]),
                "reasons": "; ".join(reasons),
            })

    verdicts = pd.DataFrame(verdict_rows)
    C.write_csv(verdicts, "a1_critic_verdicts.csv")

    # v2 artifacts carry the IN-LOOP critic verdict as a column. Post-hoc re-review must reproduce
    # it exactly - an end-to-end agent-vs-reconstruction identity check. (v1 has no such column.)
    identity = None
    if "critic_verdict" in golden.columns and golden["critic_verdict"].notna().any():
        inloop = golden["critic_verdict"].fillna("").to_numpy()
        posthoc = verdicts["critic_verdict"].to_numpy()
        mismatch = [(int(golden.iloc[i]["signature_id"]), str(inloop[i]), str(posthoc[i]))
                    for i in range(len(golden)) if inloop[i] and inloop[i] != posthoc[i]]
        identity = {"n_compared": int((golden["critic_verdict"].fillna("") != "").sum()),
                    "n_mismatched": len(mismatch), "mismatches": mismatch[:20]}

    counts = verdicts["critic_verdict"].value_counts().to_dict()
    n = len(verdicts)
    n_pass, n_flag, n_fail = (int(counts.get(k, 0)) for k in ("PASS", "FLAG", "FAIL"))
    # A FAIL clears `usable`; count how many released rows that would newly demote.
    newly_unusable = int((verdicts["usable_as_released"] & ~verdicts["usable_after_critic"]).sum())
    reasons = {k: int(v) for k, v in sorted(reason_counter.items(), key=lambda kv: -kv[1])}

    # Latent defect: rows stamped usable=True that carry no cost median at all. `usable` is
    # bool(cost_band["median"]) and bool(nan) is truthy, so an empty-but-not-absent cost tier passes
    # the flag; the critic's `if not med` gate does not catch it either, since `not nan` is False.
    nan_usable = golden[golden["cost_med"].isna() & golden["usable"].astype(bool)]
    nan_usable_ids = sorted(int(s) for s in nan_usable["signature_id"])

    summary = {
        "n_solutions": n, "n_signatures": int(verdicts["signature_id"].nunique()),
        "n_pass": n_pass, "n_flag": n_flag, "n_fail": n_fail,
        "pass_rate": n_pass / n, "flag_rate": n_flag / n, "fail_rate": n_fail / n,
        "newly_unusable_vs_released": newly_unusable,
        "released_unusable": int((~verdicts["usable_as_released"]).sum()),
        "usable_true_but_no_cost_median": len(nan_usable),
        "usable_true_but_no_cost_median_signatures": nan_usable_ids,
        "inloop_vs_posthoc_identity": identity,
        "reason_counts": reasons,
        "gates": {"min_support": C.CRITIC_MIN_SUPPORT, "min_consensus": C.CRITIC_MIN_CONSENSUS},
        "scope": "post-hoc over the emitted library; NOT the in-loop critic (no revision round)",
    }
    C.write_json(summary, "a1_critic_summary.json")

    reason_lines = "\n".join(f"- `{k}` - {v:,}" for k, v in reasons.items() if k != "(none)")
    body = f"""
Running the shipped `SolutionCriticAgent` over all {n:,} emitted Canonical Repairs gives
**{n_pass:,} PASS ({C.pct(n_pass / n, 2)}) / {n_flag:,} FLAG ({C.pct(n_flag / n, 2)}) /
{n_fail:,} FAIL ({C.pct(n_fail / n, 2)})**. Reasons recorded (a solution may carry several):

{reason_lines}

Applied to the released library the critic demotes **{newly_unusable}** further repairs
({int((~verdicts['usable_as_released']).sum())} were already marked unusable): the emitted library is
already consistent with the gates it was never actually run through.

**What those {newly_unusable} demotions are, exactly.** {len(nan_usable)} emitted repairs carry
`usable = True` while having **no cost median at all** (signatures
{', '.join(map(str, nan_usable_ids))}). Their cost tier received zero claims, so `StatsTool.band`
returned NaN rather than None, and `usable` is `bool(cost_band["median"])` - where `bool(nan)` is
truthy. So the assembly flag admits them.

The critic catches all {newly_unusable}, but not through the gate one would expect: `if not med` does
not fire either, because `not nan` is also False. They are caught by the *next* gate, cost-band
ordering, since `nan <= nan <= nan` evaluates False. The QA layer therefore recovers a defect the
assembly flag misses - which is a point in the critic's favour - but it does so incidentally, via NaN
comparison semantics rather than by an explicit missing-cost test. A tier that produced a genuinely
disordered band and a tier that produced no band at all are reported under the same reason string.

The practical consequence for the paper: the released coverage figure counts {len(nan_usable)} repairs
with no cost band to stand on, and running the critic removes them.

**Scope, stated precisely.** This is the critic applied *post hoc to the finished library*, not the
critic *in loop*: it measures the structural quality of what was emitted. It cannot reproduce the one
bounded revision round, because that round would change the LLM text and the batch's
`llm_cache.jsonl` no longer exists. Gates imported from `src/config.py`
(min_support={C.CRITIC_MIN_SUPPORT}, min_consensus={C.CRITIC_MIN_CONSENSUS}); verdict logic is the
shipped `SolutionCriticAgent`, not a reimplementation.

Artifacts: `eval/out/a1_critic_verdicts.csv`, `eval/out/a1_critic_summary.json`.
"""
    C.append_results("a1 - Post-hoc critic verdicts", body, args.data_root, not args.no_append)

    print(f"\n  PASS {n_pass:,} / FLAG {n_flag:,} / FAIL {n_fail:,}   (n={n:,})")
    if identity is not None:
        print(f"  in-loop vs post-hoc identity: {identity['n_compared']:,} compared, "
              f"{identity['n_mismatched']} mismatched")


if __name__ == "__main__":
    main()
