"""
BigQueryTool - defensive wrapper around Google BigQuery for reading Ford warranty/PAWS data.

Why a tool: agents must not embed SQL auth/retry logic. This tool centralizes:
  - authentication (fresh gcloud user token; ADC is blocked by Ford CAA),
  - a configurable billing project,
  - retries with backoff on transient errors + a 60s per-query timeout,
  - safe dataframe conversion (BigQuery NUMERIC -> float).

Used by the GroupingAgent / data-prep to pull claims, and by the region/discrepancy tooling.

The google-cloud-bigquery SDK is imported lazily inside `_client_fresh` so the tool (and its
tests) can be imported and exercised with a mocked client on a machine with no SDK installed.
"""
import time
from decimal import Decimal

import pandas as pd

from .auth import fresh_access_token
from ..config import BILLING_PROJECT, SOURCE_PROJECT, MAX_RETRIES, REQUEST_TIMEOUT_MS, is_retryable


class BigQueryTool:
    """Read-only BigQuery access with auth refresh, retries, and Decimal->float coercion."""

    def __init__(self, billing_project: str = BILLING_PROJECT, max_retries: int = MAX_RETRIES):
        self.billing_project = billing_project
        self.max_retries = max_retries
        self._client = None
        self._token = None

    def _client_fresh(self, force: bool = False):
        from google.cloud import bigquery
        from google.oauth2.credentials import Credentials
        token = fresh_access_token(force=force)
        if self._client is None or token != self._token:
            self._client = bigquery.Client(
                project=self.billing_project, credentials=Credentials(token)
            )
            self._token = token
        return self._client

    @staticmethod
    def _coerce(df: pd.DataFrame) -> pd.DataFrame:
        """BigQuery NUMERIC columns arrive as Decimal objects; cast them to float for math."""
        for col in df.columns:
            if df[col].dtype == object and df[col].map(lambda x: isinstance(x, Decimal)).any():
                df[col] = df[col].astype(float)
        return df

    def query(self, sql: str) -> pd.DataFrame:
        """Run a query and return a DataFrame.

        Retries transient/auth errors with backoff; a per-query timeout (same 60s budget as the
        LLM tool) guarantees a stuck query can never hang the pipeline.
        """
        last = None
        for attempt in range(self.max_retries):
            try:
                client = self._client_fresh(force=attempt > 0)
                job = client.query(sql)
                df = job.result(timeout=REQUEST_TIMEOUT_MS / 1000).to_dataframe(create_bqstorage_client=False)
                return self._coerce(df)
            except Exception as e:  # noqa: BLE001 - defensive: retry transient, surface fatal
                last = e
                if is_retryable(str(e)) and attempt < self.max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                break
        raise RuntimeError(f"BigQuery query failed after {self.max_retries} attempts: {str(last)[:300]}")
