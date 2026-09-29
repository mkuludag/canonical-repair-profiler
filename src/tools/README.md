# Tool Layer

The AI Judge evaluates `src/tools/` for tool selection and API integration logic.
CRP keeps external capability access out of agents and centralizes it here.

## Ford-Approved / External Tools

- `vertex_gemini_tool.py` wraps Vertex AI Gemini. It handles auth refresh, 60s
  request timeouts, retry/backoff for transient failures, strict JSON mode, and
  safe malformed-JSON handling. Agents call `generate_json()` or `generate_text()`
  instead of talking to the SDK directly.
- `bigquery_tool.py` wraps BigQuery reads for Ford warranty data. It handles fresh
  gcloud user tokens, retry/backoff, billing project configuration, and BigQuery
  NUMERIC-to-float conversion.
- `auth.py` is the single credential boundary. It shells out to gcloud for a fresh
  access token and returns actionable errors for missing gcloud or reauthentication.

## Deterministic Local Tools

- `datastore.py` reads/writes CSV and Parquet artifacts and provides the resumable
  JSONL cache used by batch LLM runs.
- `stats_tool.py` computes cost/labor bands, consensus fractions, confidence, and
  cost-tier partitions. Numeric outputs come from this deterministic tool, not the
  LLM.
- `region_index_tool.py` provides per-state labor/cost localization. If the index
  is missing or a state is unknown, it degrades to a neutral 1.0x adjustment.

## Selection Pattern

Agents choose tools by responsibility:

- Group loading and caches: `DataStore`.
- Consensus math and split validation: `StatsTool`.
- Split/diagnosis/correction text: `VertexGeminiTool`.
- Ford data reads and discovery lineage: `BigQueryTool`.
- Regional cost localization: `RegionIndexTool`.

All project IDs, model names, paths, retry settings, and thresholds are centralized
in `src/config.py` and can be overridden with environment variables.
