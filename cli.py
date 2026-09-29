"""Root-level alias for the Canonical Repair Profiler CLI.

The actual entry point lives at `src/main.py` (hackathon Code Track layout); this file keeps
the shorter `python cli.py infer|run|dash` form used throughout the docs working identically.
"""
from src.main import build_parser, main  # noqa: F401  (re-exported for importers)

if __name__ == "__main__":
    main()
