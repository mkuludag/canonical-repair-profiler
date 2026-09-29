#!/usr/bin/env python3
"""
Verifies the full agent hand-off chain end-to-end for the demo signatures, headless (no Streamlit).
Runs the SAME Orchestrator the app and CLI use:
    GroupingAgent -> ConsensusAgent -> RepairAnalystAgent (Vertex Gemini) -> SolutionAssemblyAgent
    -> SolutionCriticAgent (QA gate + bounded revision loop) -> CostSavingsAgent -> RegionDelegatorAgent

This calls Vertex Gemini live, so it doubles as a smoke test of the cloud path. Add a state to also
verify regional localization.

Run:
    python3 scripts/verify_app_pipeline.py                 # live cloud path
    python3 scripts/verify_app_pipeline.py --state CA      # also localize cost
    CRP_USE_SAMPLE=1 python3 scripts/verify_app_pipeline.py  # claims from bundled sample
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from src.agents.orchestrator import Orchestrator  # noqa: E402

DEMO = [32954, 40109, 71778]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="", help="optional US state to localize cost, e.g. CA")
    args = ap.parse_args()

    orch = Orchestrator()
    all_ok = True
    for sid in DEMO:
        try:
            sols = orch.run_one(sid, state=args.state)
            ok = len(sols) >= 1 and all(s["repair_name"] for s in sols)
        except Exception as e:  # surface, but keep checking the rest
            ok, sols = False, []
            print(f"[FAIL] sig {sid}: {e}")
        all_ok &= ok
        if sols:
            s0 = sols[0]
            print(f"\n[{'OK' if ok else 'FAIL'}] sig {sid}: {s0['vehicle_line']}/{s0['causal_part']} "
                  f"-> {len(sols)} solution(s)")
            for s in sols:
                line = (f"    {s['repair_name']:<38} cost=${s['cost_med']}  labor={s['labor_med_hrs']}h  "
                        f"consensus={s['consensus']}  support={s['support']}")
                if s.get("critic_verdict"):
                    line += f"  critic={s['critic_verdict']}"
                if s.get("savings_per_claim"):
                    line += f"  savings/claim=${s['savings_per_claim']}"
                if s.get("cost_med_regional") is not None:
                    line += f"  {s['region_state']}=${s['cost_med_regional']}"
                print(line)
    print("\nRESULT:", "ALL DEMO SIGNATURES OK" if all_ok else "FAILURE")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
