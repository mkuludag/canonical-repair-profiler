"""
ClaimMatchAgent / inference demo — route an INCOMING claim to its Canonical Repair.

This is the product use case (automated claim adjudication + the dealer "unified diagnosis" dropdown):
given a new claim (vehicle line + causal part + the technician/customer text), we
  1. retrieve the candidate Canonical Repairs for that vehicle + causal part from the library, and
  2. use a Vertex Gemini agent to pick the best-matching repair (the dealer's dropdown selection),
then return its golden solution: unified diagnosis, suggested correction, labor band, cost band.

Run:
  python3 -m src.agents.infer --vehicle "F-150" --part 6148 \
      --concern "engine knock on cold start, oil consumption, misfire on cyl 1 and 4"
"""
import argparse
import json
import os
import textwrap

import pandas as pd

from ..tools import VertexGeminiTool, DataStore
from .. import config as C

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "match_incoming_claim.txt")


def _load_prompt() -> str:
    with open(os.path.abspath(PROMPT_PATH)) as fh:
        return fh.read()


class ClaimMatchAgent:
    def __init__(self, datastore: DataStore = None, gemini: VertexGeminiTool = None):
        self.ds = datastore or DataStore()
        self.llm = gemini or VertexGeminiTool()
        self.library = C.SAMPLE_GOLDEN_SOLUTIONS if C.use_sample() else C.GOLDEN_SOLUTIONS
        self.prompt_template = _load_prompt()

    def candidates(self, vehicle: str, part: str) -> pd.DataFrame:
        lib = self.ds.read_csv(self.library)
        lib = lib[lib["usable"] == True] if "usable" in lib.columns else lib
        # regex=False: vehicle lines contain regex specials like "[15-20]" (a bad character range)
        m = lib[lib["vehicle_line"].str.contains(vehicle, case=False, na=False, regex=False)
                & (lib["causal_part"].astype(str) == str(part))]
        return m.reset_index(drop=True)

    def match(self, vehicle: str, part: str, concern: str) -> dict:
        cand = self.candidates(vehicle, part)
        if len(cand) == 0:
            return {"status": "NO_MATCH",
                    "message": "No canonical repair exists for this vehicle+causal part yet "
                               "(claim would be routed to manual review / new-signature creation)."}
        if len(cand) == 1:
            return {"status": "MATCHED", "match_index": 0, "confidence_0_1": 1.0,
                    "reason": "only one canonical repair for this vehicle+part", "solution": cand.iloc[0].to_dict()}
        listing = "\n".join(
            f"[{i}] diagnosis: {r['unified_diagnosis']} | correction: {r['suggested_correction']} "
            f"| ~{r['labor_med_hrs']}h ~${r['cost_med']}"
            for i, r in cand.iterrows())
        res = self.llm.generate_json(self.prompt_template % {"vehicle": vehicle, "part": part,
                                                             "concern": concern, "candidates": listing})
        if not res.get("_ok"):
            # the matching agent could not run (auth/timeout/API) - surface it, don't disguise as no-match
            return {"status": "LLM_ERROR",
                    "message": "The matching agent could not reach Vertex Gemini. "
                               f"{str(res.get('error', ''))[:200]}",
                    "reason": str(res.get("error", ""))}
        idx = int(res.get("match_index", -1))
        if idx < 0 or idx >= len(cand):
            return {"status": "NO_MATCH", "message": "Agent found no good match among candidates.",
                    "reason": res.get("reason", "")}
        return {"status": "MATCHED", "match_index": idx,
                "confidence_0_1": res.get("confidence_0_1"), "reason": res.get("reason", ""),
                "solution": cand.iloc[idx].to_dict()}


def _print(result: dict):
    print("\n" + "=" * 70)
    if result["status"] != "MATCHED":
        print("RESULT:", result["status"], "\n ", result.get("message", "")); return
    s = result["solution"]
    print("MATCHED CANONICAL REPAIR  (dealer dropdown selection)")
    print(f"  match confidence: {result.get('confidence_0_1')}  | why: {result.get('reason','')}")
    print(f"\n  REPAIR        : {s.get('repair_name','')}")
    print("  UNIFIED DIAGNOSIS:\n   ", "\n    ".join(textwrap.wrap(str(s.get('unified_diagnosis','')), 90)))
    print("  SUGGESTED CORRECTION:\n   ", "\n    ".join(textwrap.wrap(str(s.get('suggested_correction','')), 90)))
    print(f"\n  LABOR (median) : {s.get('labor_med_hrs')} hrs")
    print(f"  COST  (median) : ${s.get('cost_med')}   (IQR ${s.get('cost_q25')}-${s.get('cost_q75')})")
    print(f"  TYPICAL PARTS  : {s.get('typical_parts','')}")
    print(f"  CONFIDENCE     : {s.get('confidence')}  | from {s.get('tier_n_claims')} historical claims")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description="Route an incoming claim to its Canonical Repair (inference demo)")
    ap.add_argument("--vehicle", required=True, help="vehicle line substring, e.g. 'F-150'")
    ap.add_argument("--part", required=True, help="causal part code, e.g. 6148")
    ap.add_argument("--concern", required=True, help="incoming claim concern/cause text")
    args = ap.parse_args()
    print(f"\nINCOMING CLAIM -> vehicle~'{args.vehicle}' part={args.part}\n  \"{args.concern}\"")
    _print(ClaimMatchAgent().match(args.vehicle, args.part, args.concern))


if __name__ == "__main__":
    main()
