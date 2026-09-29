# Testing

The suite is designed to run **fully offline**: every external dependency (Vertex Gemini, BigQuery,
the gcloud CLI) is mocked, and data comes from the bundled sample (`data/sample/`). No credentials or
network are required, so it runs the same locally and in CI.

## Run

```bash
make test                 # pytest -q
make cov                  # with coverage report
pytest tests/test_stats_tool.py -q     # a single file
```

CI runs the same on every push across Python 3.10-3.12 (`.github/workflows/ci.yml`).

## What is covered (99 tests, 96% source coverage)

| File | Focus |
|---|---|
| `test_config.py` | defaults, env overrides, `CRP_USE_SAMPLE` toggle |
| `test_stats_tool.py` | consensus fraction, bands, 2/3-tier partition, confidence monotonicity |
| `test_consensus_agent.py` | numeric enrichment + agreement in range |
| `test_solution_assembly.py` | **separation guard** (collapse <1.5x, keep >=1.5x), name fallback, 2-repair cap |
| `test_repair_analyst_agent.py` | prompt construction + graceful LLM-failure fallback |
| `test_cost_savings.py` | avoidable-overspend math + per-solution fields |
| `test_region.py` | per-state index, unknown-state/missing-file degradation, delegator adjustment |
| `test_grouping_agent.py` | sample-mode signature load + cost cap + unknown-signature error |
| `test_claim_match_agent.py` | 0 / 1 / many candidates, LLM out-of-range / negative index |
| `test_vertex_gemini_tool.py` | retry on 429/5xx, no-retry on bad JSON / 400, graceful text failure |
| `test_bigquery_tool.py` | Decimal->float coercion, retry then succeed, raise after retries |
| `test_auth.py` | token caching, force refresh, actionable errors (reauth / missing gcloud) |
| `test_datastore.py` | CSV/parquet roundtrip, resumable JSONL cache, malformed-line skipping |
| `test_orchestrator.py` | full hand-off chain end-to-end (single, split, +region/+savings, critic revision loop) |
| `test_solution_critic.py` | PASS/FLAG/FAIL verdict gates, revision-request gating, bounded loop |
| `test_entrypoints_and_branches.py` | cli/src.main entrypoints + remaining error branches across agents/tools |

## Conventions
- Mock the **transport boundary**, not business logic: `VertexGeminiTool._raw_generate` and
  `BigQueryTool._client_fresh` are the monkeypatch points; `auth.subprocess.check_output` for gcloud.
- Use the `sample_env` fixture (sets `CRP_USE_SAMPLE=1`) for anything that reads data.
- Use the `fake_llm` fixture (a `FakeLLM` returning a canned JSON/text payload) to drive LLM agents.
- `synthetic_claims` (bimodal) and `tight_claims` (single-mode) fixtures exercise the split guard.

## Live smoke test (calls Vertex Gemini)
`python scripts/verify_app_pipeline.py [--state CA]` runs the real cloud path for the demo signatures
(requires `gcloud auth login`). This is intentionally **not** part of `pytest`.
