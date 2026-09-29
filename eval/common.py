"""
Shared helpers for the ACCORD re-derivation analyses (eval/a1..a6).

These analyses re-derive, from the *committed* Canonical Repair Library artifacts, quantities the
paper previously asserted without measuring. They make no external calls and consume no claim-level
data, so they run unchanged against the public anonymized release.

Dollar values in the public release are scaled by an undisclosed constant; every ratio, percentage,
count, labor hour and consensus/confidence score is exact and unscaled. All analyses here are
therefore identical on both data roots except where a raw dollar amount is printed.

Conventions:
  - `--data-root` selects the artifact directory (default `grouping/out`), so the same script can be
    pointed at the internal copy or the public mirror.
  - Results are written to eval/out/ as CSV/JSON; each script appends a prose paragraph to
    eval/RESULTS.md so the paper can quote exact numbers with a file path.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval"
OUT_DIR = EVAL_DIR / "out"
RESULTS_MD = EVAL_DIR / "RESULTS.md"

# Make `src` importable so analyses can reuse the shipped agents/config rather than reimplementing
# their thresholds (a re-implementation could silently drift from the code under study).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (  # noqa: E402
    CONSENSUS_TOL,
    CRITIC_MIN_CONSENSUS,
    CRITIC_MIN_SUPPORT,
    SPLIT_SEPARATION_MIN,
)

__all__ = [
    "CONSENSUS_TOL", "CRITIC_MIN_CONSENSUS", "CRITIC_MIN_SUPPORT", "SPLIT_SEPARATION_MIN",
    "parse_args", "load_golden", "load_crl", "load_group_savings", "load_region",
    "load_dealer_spread", "classify_signatures", "support_weight", "share_at_least",
    "write_csv", "write_json", "append_results", "pct", "setup_plot_style", "save_fig",
    "load_panel", "panel_weights",
]

# The separation guard writes its collapse rationale as
#   "LLM proposed 2 tiers but they were not separated ($3644/$2154 < 1.5x) -> one repair."
# (solution_assembly_agent.py:59-61). Group 1 is the HIGH tier median, group 2 the LOW tier median.
# Both are formatted with :.0f, so a parsed ratio carries ~1/low rounding error - negligible at these
# magnitudes, but it is why collapsed ratios are reported to 3 dp and never to more.
COLLAPSE_MARKER = "not separated"
COLLAPSE_RE = re.compile(r"\$([0-9.]+)/\$([0-9.]+)\s*<")


# --------------------------------------------------------------------------- CLI + IO

def parse_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--data-root",
        default=os.environ.get("CRP_EVAL_DATA_ROOT", str(REPO_ROOT / "grouping" / "out")),
        help="Directory holding the CRL artifacts (default: the internal grouping/out). "
             "Point at a clone of the public release to reproduce the published numbers.",
    )
    p.add_argument("--no-append", action="store_true",
                   help="Compute and print, but do not append to eval/RESULTS.md.")
    args = p.parse_args()
    args.data_root = Path(args.data_root).expanduser().resolve()
    if not args.data_root.is_dir():
        p.error(f"--data-root does not exist: {args.data_root}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Stamp the provenance of everything in out/. Dollar values follow the data root, so an unlabelled
    # directory of results is ambiguous between the internal basis and the rescaled public one.
    (OUT_DIR / "_DATA_ROOT.txt").write_text(
        f"{args.data_root}\n\nAll dollar values in this directory follow this data root.\n"
        "Ratios, percentages, counts, labor hours and consensus/confidence scores are basis-independent.\n")
    return args


def _read(data_root: Path, name: str) -> pd.DataFrame:
    path = Path(data_root) / name
    if not path.exists():
        raise FileNotFoundError(f"missing artifact: {path}")
    return pd.read_csv(path, low_memory=False)


def load_golden(data_root: Path) -> pd.DataFrame:
    """golden_solutions.csv - one row per (signature x emitted cost tier). 5,070 rows."""
    df = _read(data_root, "golden_solutions.csv")
    for c in ("cost_med", "cost_q25", "cost_q75", "consensus", "confidence", "labor_med_hrs"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("support", "tier_n_claims", "n_repairs_detected"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ("unified_diagnosis", "suggested_correction", "repair_name", "typical_parts",
              "split_rationale", "cost_tier"):
        df[c] = df[c].fillna("").astype(str)
    return df


def load_crl(data_root: Path) -> pd.DataFrame:
    """canonical_repairs.csv - one row per repair signature. 6,654 rows."""
    df = _read(data_root, "canonical_repairs.csv")
    for c in ("n", "cost_med", "cost_q25", "cost_q75", "labor_med", "confidence", "gt_score",
              "solution_consensus", "cost_consensus", "labor_consensus"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_group_savings(data_root: Path) -> pd.DataFrame:
    return _read(data_root, "group_savings.csv")


def load_region(data_root: Path) -> pd.DataFrame:
    return _read(data_root, "region_overall.csv")


def load_dealer_spread(data_root: Path) -> pd.DataFrame:
    return _read(data_root, "dealer_spread_by_state_signature.csv")


def load_panel() -> pd.DataFrame:
    """The frozen 400-signature evaluation panel (eval/b0_panel.py). Columns: signature_id, stratum.

    Strata name what the v2 library of record did with each signature (single/honored/collapsed);
    `collapsed` is oversampled, so population-level rates must reweight with panel_weights().
    """
    df = pd.read_csv(EVAL_DIR / "panel_400.csv")
    df["signature_id"] = df["signature_id"].astype(int)
    return df


def panel_weights() -> dict:
    """Population share of each panel stratum among all GOLD signatures (from b0_panel.json)."""
    meta = json.loads((OUT_DIR / "b0_panel.json").read_text())
    return {k: v["population_share"] for k, v in meta["strata"].items()}


# --------------------------------------------------------------------------- domain logic

def support_weight(n) -> np.ndarray:
    """The saturating support weight from stats_tool.py:61 - min(1, log10(n+1)/log10(31))."""
    n = np.asarray(pd.to_numeric(pd.Series(np.ravel(n)), errors="coerce").fillna(0), dtype=float)
    return np.minimum(1.0, np.log10(np.maximum(n, 1) + 1) / np.log10(31))


def classify_signatures(golden: pd.DataFrame) -> pd.DataFrame:
    """Partition GOLD signatures by what the analyst proposed and what the separation guard did.

    Three mutually exclusive classes, keyed by signature_id:
      honored   - analyst proposed 2 repairs and the guard honored the split (2 emitted rows)
      collapsed - analyst proposed 2 repairs, guard collapsed them to 1 (rationale records the ratio)
      single    - analyst reported a single repair; the guard never ran

    `tier_ratio` is the high/low tier median cost ratio the guard tested. For honored splits it is
    recomputed from the two emitted rows; for collapsed ones it is parsed from the rationale (the
    only surviving record, since the pre-collapse tiers are not emitted). It is NaN for `single`
    (no counterfactual exists - see the survivorship note in a2) and for honored splits where a tier
    received no claims.
    """
    g = golden
    rows: list[dict] = []

    split = g[g["n_repairs_detected"] == 2]
    agg = split.groupby("signature_id")["cost_med"].agg(["min", "max", "count", "size"])
    for sid, r in agg.iterrows():
        ratio = float(r["max"] / r["min"]) if r["count"] == 2 and r["min"] > 0 else np.nan
        rows.append({"signature_id": sid, "klass": "honored", "tier_ratio": ratio,
                     "n_emitted_rows": int(r["size"])})

    one = g[g["n_repairs_detected"] == 1]
    for sid, rationale in zip(one["signature_id"], one["split_rationale"]):
        if COLLAPSE_MARKER in rationale.lower():
            m = COLLAPSE_RE.search(rationale)
            ratio = float(m.group(1)) / float(m.group(2)) if m and float(m.group(2)) > 0 else np.nan
            rows.append({"signature_id": sid, "klass": "collapsed", "tier_ratio": ratio,
                         "n_emitted_rows": 1})
        else:
            rows.append({"signature_id": sid, "klass": "single", "tier_ratio": np.nan,
                         "n_emitted_rows": 1})

    out = pd.DataFrame(rows).sort_values("signature_id").reset_index(drop=True)
    dupes = out["signature_id"].duplicated().sum()
    if dupes:
        raise AssertionError(f"signature classified into >1 class ({dupes} duplicates)")
    return out


def share_at_least(values: pd.Series, thresholds) -> pd.DataFrame:
    """For each threshold, the count and share of `values` at or above it."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return pd.DataFrame([
        {"threshold": t, "n_at_or_above": int((v >= t).sum()), "n_total": int(len(v)),
         "share": float((v >= t).mean()) if len(v) else np.nan}
        for t in thresholds
    ])


def pct(x: float, digits: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{digits}f}%"


# --------------------------------------------------------------------------- output

def write_csv(df: pd.DataFrame, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    df.to_csv(path, index=False)
    print(f"  wrote {path.relative_to(REPO_ROOT)}  ({len(df)} rows)")
    return path


def write_json(obj: dict, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    print(f"  wrote {path.relative_to(REPO_ROOT)}")
    return path


def append_results(section: str, body: str, data_root: Path, enabled: bool = True) -> None:
    """Replace (or append) a named section in eval/RESULTS.md so re-runs stay idempotent."""
    if not enabled:
        print(f"\n--- {section} (not written to RESULTS.md) ---\n{body.strip()}\n")
        return
    header = f"## {section}"
    entry = f"{header}\n\n_Data root: `{data_root}`_\n\n{body.strip()}\n"
    if RESULTS_MD.exists():
        text = RESULTS_MD.read_text()
    else:
        text = ("# ACCORD - re-derived results\n\nEvery number below is computed by the scripts in "
                "`eval/` from the committed Canonical Repair Library artifacts. No external calls, "
                "no claim-level data. Re-running a script replaces its section in place.\n")
    if header in text:
        start = text.index(header)
        rest = text[start + len(header):]
        nxt = rest.find("\n## ")
        end = len(text) if nxt == -1 else start + len(header) + nxt + 1
        text = text[:start] + entry + ("\n" if nxt != -1 else "") + text[end:]
    else:
        text = text.rstrip() + "\n\n" + entry
    RESULTS_MD.write_text(text)
    print(f"  updated {RESULTS_MD.relative_to(REPO_ROOT)} -> '{section}'")


# --------------------------------------------------------------------------- plotting

PALETTE = {
    "honored": "#2a9d8f", "collapsed": "#e9c46a", "single": "#264653",
    "accent": "#e76f51", "muted": "#8d99ae", "guide": "#6c757d",
}


def setup_plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "legend.frameon": False,
    })
    return plt


def save_fig(fig, name: str, figures_dir: Path | None = None) -> Path:
    # CRP_FIG_OUT mirrors figures/make_figures.py, so both figure sets can be rendered side by side
    # from different data roots (e.g. internal vs the rescaled public release).
    figures_dir = Path(figures_dir or os.environ.get("CRP_FIG_OUT") or (REPO_ROOT / "figures"))
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"  wrote {path}")
    return path
