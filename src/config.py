"""
Central configuration for the Canonical Repair Profiler.

Every hardcoded project id, model name, credential account, threshold and on-disk path used by the
tools and agents is defined ONCE here, and can be overridden via environment variables. This keeps
the codebase portable (a reviewer can point it at their own GCP project / data) and is the single
source of truth referenced by the README and docs/DATA_DICTIONARY.md.

Override any value at runtime, e.g.:
    export CRP_GEMINI_MODEL=gemini-2.5-pro
    export GCLOUD_ACCOUNT=you@example.com
"""
import os

# ----- Vertex AI / Gemini (LLM brain) -----
VERTEX_PROJECT = os.environ.get("CRP_VERTEX_PROJECT", "REDACTED-VERTEX-PROJECT")
VERTEX_LOCATION = os.environ.get("CRP_VERTEX_LOCATION", "us-central1")
GEMINI_MODEL = os.environ.get("CRP_GEMINI_MODEL", "gemini-2.5-flash")
REQUEST_TIMEOUT_MS = int(os.environ.get("CRP_REQUEST_TIMEOUT_MS", "60000"))

# ----- BigQuery (warranty data source) -----
BILLING_PROJECT = os.environ.get("CRP_BILLING_PROJECT", "REDACTED-BILLING-PROJECT")
SOURCE_PROJECT = os.environ.get("CRP_SOURCE_PROJECT", "REDACTED-SOURCE-PROJECT")

# ----- Auth (gcloud user OAuth token; ADC is blocked by Ford CAA) -----
GCLOUD_ACCOUNT = os.environ.get("GCLOUD_ACCOUNT", "")  # blank -> use the active gcloud account
TOKEN_TTL_SEC = int(os.environ.get("CRP_TOKEN_TTL_SEC", "1500"))  # re-mint at ~25 min

# ----- Retry / resilience defaults -----
MAX_RETRIES = int(os.environ.get("CRP_MAX_RETRIES", "3"))
RETRYABLE_MARKERS = ("401", "UNAUTHENTICATED", "403", "429", "500", "503", "timeout", "deadline")


def is_retryable(error_msg: str) -> bool:
    """True when an external-call error is transient (auth/quota/5xx/timeout) and worth retrying."""
    return any(s in error_msg for s in RETRYABLE_MARKERS)

# ----- Domain constants -----
COST_COL = "gsar_tot_cost_gross"      # realistic warranty claim cost (NOT inflated PAWS approved amount)
LABOR_COL = "gsar_labor_hrs"
COST_CAP = 100_000                     # claims above this are data errors -> dropped
SPLIT_SEPARATION_MIN = 1.5             # honor a 2-way LLM split only if high tier >= 1.5x low tier
CONSENSUS_TOL = 0.25                   # +/-25% of median counts as "agreement"

# ----- SolutionCriticAgent QA gates (soft flags; hard gates are structural) -----
CRITIC_MIN_SUPPORT = int(os.environ.get("CRP_CRITIC_MIN_SUPPORT", "5"))        # claims backing a repair
CRITIC_MIN_CONSENSUS = float(os.environ.get("CRP_CRITIC_MIN_CONSENSUS", "0.3"))  # cost/labor agreement

# ----- On-disk artifacts (relative to repo root / DataStore root) -----
# Env-overridable (as the module docstring promises) so a rebuild can write a versioned artifact set
# (e.g. grouping/out_v2/) without touching the released files.
REPAIR_PROFILE = os.environ.get("CRP_REPAIR_PROFILE", "data/repair_profile.parquet")
CLAIM_SIGNATURES = os.environ.get("CRP_CLAIM_SIGNATURES", "data/claim_signatures.parquet")
CANONICAL_REPAIRS = os.environ.get("CRP_CANONICAL_REPAIRS", "grouping/out/canonical_repairs.csv")
GOLDEN_SOLUTIONS = os.environ.get("CRP_GOLDEN_SOLUTIONS", "grouping/out/golden_solutions.csv")
REGION_OVERALL = os.environ.get("CRP_REGION_OVERALL", "grouping/out/region_overall.csv")
GROUP_SAVINGS = os.environ.get("CRP_GROUP_SAVINGS", "grouping/out/group_savings.csv")
LLM_CACHE = os.environ.get("CRP_LLM_CACHE", "grouping/out/llm_cache.jsonl")

# ----- Bundled offline sample (lets the demo + tests run with no BigQuery/Vertex) -----
SAMPLE_DIR = "data/sample"
SAMPLE_CANONICAL_REPAIRS = f"{SAMPLE_DIR}/canonical_repairs.csv"
SAMPLE_GOLDEN_SOLUTIONS = f"{SAMPLE_DIR}/golden_solutions.csv"
SAMPLE_CLAIM_SIGNATURES = f"{SAMPLE_DIR}/claim_signatures.csv"
SAMPLE_CLAIMS = f"{SAMPLE_DIR}/sample_claims.csv"


def use_sample() -> bool:
    """True when CRP_USE_SAMPLE is set, routing tools to the bundled offline sample."""
    return os.environ.get("CRP_USE_SAMPLE", "").lower() in ("1", "true", "yes", "on")
