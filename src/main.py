"""
crp - the Canonical Repair Profiler entry point (Code Track: src/main.py).

Sub-commands:
  infer    Route an incoming claim to its Canonical Repair (the product / inference path).
  run      Run the full agent hand-off chain on one repair group (signature) and print the solution(s).
  dash     Launch the Streamlit dashboard.

Examples:
  python src/main.py infer --vehicle F-150 --part 6148 --concern "engine knock, oil consumption, misfire"
  python src/main.py run --signature 32954 --state CA
  CRP_USE_SAMPLE=1 python src/main.py infer --vehicle F-150 --part 6019 --concern "oil leak front cover"
  python src/main.py dash

The root-level `cli.py` is a thin alias so the shorter `python cli.py ...` form used throughout
the docs works identically. Set CRP_USE_SAMPLE=1 to run fully offline against the bundled sample
(no BigQuery/Vertex needed for the parts that read data; inference still calls Gemini unless a
single candidate matches).
"""
import argparse
import json
import sys
from pathlib import Path

# Allow `python src/main.py ...` from anywhere: the repo root must be importable as the
# package base for `src.*` (and for the dashboard launched by `dash`).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _cmd_infer(args):
    from src.agents.infer import ClaimMatchAgent, _print
    print(f"\nINCOMING CLAIM -> vehicle~'{args.vehicle}' part={args.part}\n  \"{args.concern}\"")
    _print(ClaimMatchAgent().match(args.vehicle, args.part, args.concern))


def _cmd_run(args):
    from src.agents.orchestrator import Orchestrator
    sols = Orchestrator().run_one(args.signature, verbose=False, state=args.state)
    print(json.dumps(sols, indent=2, default=str))


def _cmd_dash(_args):
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py")], check=False)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="crp", description="Canonical Repair Profiler CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_infer = sub.add_parser("infer", help="route an incoming claim to its Canonical Repair")
    p_infer.add_argument("--vehicle", required=True, help="vehicle line substring, e.g. 'F-150'")
    p_infer.add_argument("--part", required=True, help="causal part code, e.g. 6148")
    p_infer.add_argument("--concern", required=True, help="incoming claim concern/cause text")
    p_infer.set_defaults(func=_cmd_infer)

    p_run = sub.add_parser("run", help="run the agent pipeline on one repair group")
    p_run.add_argument("--signature", type=int, required=True, help="signature_id to process")
    p_run.add_argument("--state", default="", help="optional US state to localize cost, e.g. CA")
    p_run.set_defaults(func=_cmd_run)

    p_dash = sub.add_parser("dash", help="launch the Streamlit dashboard")
    p_dash.set_defaults(func=_cmd_dash)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
