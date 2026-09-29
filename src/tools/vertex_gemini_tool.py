"""
VertexGeminiTool - robust wrapper around Vertex AI Gemini (the LLM brain of our agents).

Hardening built in (these directly address real failures we hit in a 3,600-call batch):
  - token auto-refresh (per-thread client re-created when the gcloud token nears expiry),
  - 60s request timeout so a stuck call can never hang the pipeline,
  - up to 3 retries with backoff on 401 / 403 / 429 / 5xx,
  - strict-JSON response mode + safe JSON parsing with a structured error payload.

This is the only place the system talks to the LLM. Agents call `generate_json(prompt)`.

The google-genai SDK is imported lazily inside `_client` so the tool (and its tests) can be
imported and exercised with a mocked transport on a machine that has no SDK installed.
"""
import json
import time
import threading

from .auth import fresh_access_token
from ..config import (
    VERTEX_PROJECT, VERTEX_LOCATION, GEMINI_MODEL, REQUEST_TIMEOUT_MS,
    MAX_RETRIES, TOKEN_TTL_SEC, is_retryable,
)

# back-compat aliases (older scripts import these names)
PROJECT = VERTEX_PROJECT
LOCATION = VERTEX_LOCATION
MODEL = GEMINI_MODEL


class VertexGeminiTool:
    """Thread-safe Gemini client with auth refresh, timeout, retries, and JSON parsing."""

    def __init__(self, model: str = GEMINI_MODEL, project: str = VERTEX_PROJECT,
                 location: str = VERTEX_LOCATION, max_retries: int = MAX_RETRIES,
                 token_ttl_sec: int = TOKEN_TTL_SEC):
        self.model, self.project, self.location = model, project, location
        self.max_retries, self.token_ttl_sec = max_retries, token_ttl_sec
        self._local = threading.local()

    def _client(self, force: bool = False):
        from google import genai
        from google.genai import types
        from google.oauth2.credentials import Credentials
        now = time.time()
        if force or not hasattr(self._local, "c") or now - getattr(self._local, "ts", 0) > self.token_ttl_sec:
            self._local.c = genai.Client(
                vertexai=True, project=self.project, location=self.location,
                credentials=Credentials(fresh_access_token(force=force)),
                http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
            )
            self._local.ts = now
        return self._local.c

    def _raw_generate(self, prompt: str, temperature: float, json_mode: bool, force: bool = False) -> str:
        """Single transport call. Isolated so tests can monkeypatch it without the SDK."""
        from google.genai import types
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
        )
        resp = self._client(force=force).models.generate_content(
            model=self.model, contents=prompt, config=cfg)
        return resp.text

    @staticmethod
    def _is_retryable(msg: str) -> bool:
        return is_retryable(msg)

    # Parse-failure retries sample at >= this temperature: at temperature 0 a malformed response
    # (e.g. a dropped closing brace between array items - observed deterministically on
    # gemini-2.5-flash) reproduces identically on every attempt, so only a jittered retry can
    # recover. 0.2 matches the original batch run's setting.
    JSON_RETRY_MIN_TEMPERATURE = 0.2

    def generate_json(self, prompt: str, temperature: float = 0.2) -> dict:
        """Call Gemini in strict-JSON mode and return a parsed dict.

        Returns {"_ok": True, ...payload} on success, or {"_ok": False, "error": ...} on
        permanent failure - never raises, so a single bad group cannot crash a batch.
        Malformed JSON is retried like a transient error, but with the temperature floored at
        JSON_RETRY_MIN_TEMPERATURE so a deterministic glitch cannot simply repeat.
        "_attempts" records how many transport calls the payload took: attempts > 1 means the
        response was sampled at the jittered retry temperature rather than the requested one.
        """
        last = None
        for attempt in range(self.max_retries):
            t = temperature if attempt == 0 else max(temperature, self.JSON_RETRY_MIN_TEMPERATURE)
            try:
                text = self._raw_generate(prompt, t, json_mode=True, force=attempt > 0)
                out = {"_ok": True, **json.loads(text)}
                out["_attempts"] = attempt + 1
                return out
            except json.JSONDecodeError as e:
                last = f"JSON parse failed: {str(e)[:200]}"
                continue  # jittered retry (see JSON_RETRY_MIN_TEMPERATURE)
            except Exception as e:  # noqa: BLE001
                last = str(e)
                if self._is_retryable(last) and attempt < self.max_retries - 1:
                    time.sleep(2 + attempt * 3)
                    continue
                break
        return {"_ok": False, "error": str(last)[:300]}

    def generate_text(self, prompt: str, temperature: float = 0.2) -> str:
        last = None
        for attempt in range(self.max_retries):
            try:
                return self._raw_generate(prompt, temperature, json_mode=False, force=attempt > 0)
            except Exception as e:  # noqa: BLE001
                last = str(e)
                if self._is_retryable(last) and attempt < self.max_retries - 1:
                    time.sleep(2 + attempt * 3)
                    continue
                break
        return f"[LLM_ERROR] {str(last)[:200]}"
