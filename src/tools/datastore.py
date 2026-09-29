"""
DataStore — the agents' shared on-disk memory (Parquet/CSV) + a resumable JSONL cache.

Keeps large intermediate artifacts (the unified dataset, the claim->signature grouping) and a
line-oriented LLM cache so an interrupted multi-thousand-call run resumes instead of restarting.
Defensive IO: missing files raise clear errors; the JSONL reader skips malformed lines.
"""
import os
import json
import logging
import threading

import pandas as pd

log = logging.getLogger(__name__)


class DataStore:
    def __init__(self, root: str = "."):
        self.root = root
        self._cache_lock = threading.Lock()

    def _path(self, rel: str) -> str:
        return os.path.join(self.root, rel)

    @staticmethod
    def _ensure_parent(path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def read_parquet(self, rel: str, columns=None) -> pd.DataFrame:
        p = self._path(rel)
        if not os.path.exists(p):
            raise FileNotFoundError(f"DataStore: parquet not found: {p}")
        return pd.read_parquet(p, columns=columns)

    def write_parquet(self, df: pd.DataFrame, rel: str) -> str:
        p = self._path(rel)
        self._ensure_parent(p)
        df.to_parquet(p, index=False)
        return p

    def read_csv(self, rel: str) -> pd.DataFrame:
        return pd.read_csv(self._path(rel))

    def write_csv(self, df: pd.DataFrame, rel: str) -> str:
        p = self._path(rel)
        self._ensure_parent(p)
        df.to_csv(p, index=False)
        return p

    # --- resumable cache for LLM results (one JSON object per line) ---
    def load_cache(self, rel: str, ok_only: bool = True) -> dict:
        p = self._path(rel)
        out = {}
        if not os.path.exists(p):
            return out
        skipped = 0
        for line in open(p):
            try:
                d = json.loads(line)
            except ValueError:  # skip malformed/partial lines defensively (e.g. a killed run)
                skipped += 1
                continue
            if (not ok_only) or d.get("_ok") or d.get("ok"):
                out[d.get("signature_id")] = d
        if skipped:
            log.warning("DataStore: skipped %d malformed line(s) in %s", skipped, rel)
        return out

    def append_cache(self, rel: str, record: dict) -> None:
        """Thread-safe: concurrent batch workers must not interleave partial JSONL lines."""
        p = self._path(rel)
        self._ensure_parent(p)
        with self._cache_lock, open(p, "a") as fh:
            fh.write(json.dumps(record) + "\n")
