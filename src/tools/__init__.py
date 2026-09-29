"""
src/tools — external capability layer for the Canonical Repair Profiling agents.

Each tool wraps ONE external system with a clean, defensive interface (auth, timeouts,
retries, graceful failure). Agents in src/agents/ select and call these tools; they never
talk to external APIs directly. This separation is what makes the agentic architecture
testable and robust.

Tools:
  - BigQueryTool      : read Ford warranty/PAWS data from BigQuery (Vertex-adjacent auth).
  - VertexGeminiTool  : call Vertex AI Gemini (LLM) with token auto-refresh + retry + timeout.
  - DataStore         : local Parquet/CSV persistence (the agents' shared memory on disk).
  - StatsTool         : deterministic consensus / IQR / cost-tiering / confidence math.
  - RegionIndexTool   : deterministic per-state cost/labor index (regional localization).
"""
from .bigquery_tool import BigQueryTool
from .vertex_gemini_tool import VertexGeminiTool
from .datastore import DataStore
from .stats_tool import StatsTool
from .region_index_tool import RegionIndexTool

__all__ = ["BigQueryTool", "VertexGeminiTool", "DataStore", "StatsTool", "RegionIndexTool"]
