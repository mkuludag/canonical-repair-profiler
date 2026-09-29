"""
LLMGateway gateway client for the second-model ablation runs (eval-side only; src/ is untouched).

A drop-in for VertexGeminiTool inside b1/b2/b3: same `generate_json` contract, same `_attempts`
semantics, same jittered parse-retry, so every analysis downstream is unchanged and the only thing
that varies between runs is which model wrote the JSON.

    tool = LLMGatewayTool("claude-sonnet-5")
    res  = tool.generate_json(prompt, temperature=0.0)
    # {"_ok": True, **payload, "_attempts": n}   |   {"_ok": False, "error": "..."}

Gateway behaviour below is MEASURED (probe run 2026-08-04), not documented:

  - it speaks OpenAI: POST {base}/chat/completions, GET {base}/models;
  - `temperature: 0` is REJECTED by the reasoning tier - gpt-5.6-sol, gpt-5.5-2026-04-23,
    gpt-5-mini-2025-08-07 and gpt-5.3-chat-latest all return 400 "does not support 0 with this
    model". Those models cannot hold the study's protocol fixed, so they are excluded rather than
    run at a different temperature;
  - `response_format: {"type": "json_object"}` is rejected by deepseekv4-flash ("This model does
    not support response_format argument"), which therefore runs prompt-enforced JSON plus the
    tolerant extraction in `extract_json`. See RESPONSE_FORMAT_UNSUPPORTED.

Auth: the bearer comes from CRP_LLM_CLIENT_ID/SECRET when those are set (client credentials, real
auto-refresh), otherwise from the token opencode holds. The opencode token is re-read from disk on
every mint, so a mid-batch `opencode auth login -p llm_gateway` is picked up without restarting the run;
combined with the resumable JSONL caches, an expiry is a clean stop-and-resume rather than a loss.
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import MAX_RETRIES, REQUEST_TIMEOUT_MS, is_retryable  # noqa: E402

# The gateway is internal to the corporate network; set the base URL via env.
BASE_URL = os.environ.get("CRP_LLM_BASE_URL", "")
TENANT = os.environ.get("CRP_LLM_TENANT_ID", "")
AUDIENCE = os.environ.get("CRP_LLM_AUDIENCE", "")
TOKEN_URL = f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/token"

OPENCODE_AUTH = Path.home() / ".local" / "share" / "opencode" / "auth.json"
SECRETS_FILE = Path(os.environ.get("CRP_LLM_SECRETS_FILE", str(Path.home() / ".llm_gateway.env")))

RELOGIN_HINT = "~/bin/opencode auth login -p llm_gateway"

# Models whose compat shim rejects response_format; they run prompt-enforced JSON instead.
RESPONSE_FORMAT_UNSUPPORTED = {"deepseekv4-flash"}


# Per-model request ceiling. gpt-5.4-2026-03-05 answers 429 with "limited to 100 per 60 seconds";
# 8 workers at ~1.4s/call is ~340/min, which cost 294 of 400 signatures before this existed.
# The default sits under the one limit the gateway has stated, and penalize() adapts downward for
# any model whose (undocumented) ceiling turns out to be lower.
DEFAULT_RATE_PER_MIN = 90


class _RateLimiter:
    """Sliding-window limiter shared by every worker thread on one client."""

    def __init__(self, per_min: int):
        self.per_min = per_min
        self._lock = threading.Lock()
        self._times = collections.deque()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] > 60:
                    self._times.popleft()
                if not self.per_min or len(self._times) < self.per_min:
                    self._times.append(now)
                    return
                wait = 60 - (now - self._times[0]) + 0.05
            time.sleep(min(wait, 60))  # sleep OUTSIDE the lock

    def penalize(self) -> None:
        """A 429 got through anyway: this model's real ceiling is lower than we assumed."""
        with self._lock:
            if self.per_min:
                self.per_min = max(20, int(self.per_min * 0.75))


class TokenError(RuntimeError):
    """No usable LLMGateway bearer. Carries the exact command that fixes it."""


def _read_secrets_env(key: str) -> str:
    """Pull one key out of ~/setup/secrets.env without sourcing it."""
    if not SECRETS_FILE.exists():
        return ""
    for raw in SECRETS_FILE.read_text().splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return ""


_FENCE_OPEN = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\s*```\s*$")

# Keys a tool-call envelope is allowed to carry. Anything else means it is the payload itself.
# The gateway is not consistent about which one it uses: `parameters`, `arguments` and `params` were
# all observed from claude-sonnet-5 on the same prompt across calls (finish_reason "tool_calls").
_PAYLOAD_KEYS = ("parameters", "arguments", "params", "args", "input")
_ENVELOPE_KEYS = {"name", "type", "id", "index", "function", *_PAYLOAD_KEYS}


def _as_dict(v):
    if isinstance(v, str):  # OpenAI delivers tool arguments as a JSON string
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            return None
    return v if isinstance(v, dict) and v else None


def unwrap_tool_call(obj: dict, expect_keys=()) -> tuple:
    """(payload, was_wrapped). Undo the tool-call envelope some compat shims put JSON mode in.

    Measured on claude-sonnet-5 through this gateway, which implements response_format as a forced
    tool call and returns finish_reason "tool_calls" rather than "stop": the payload arrives nested
    under a wrapper key, and the wrapper key is NOT stable. Observed across one 800-call sweep:
    `parameters`, `arguments`, `params`, `json`, `$PARAMETERS`, and once the literal string `{}`.
    Chasing that by allowlist is a losing game, so there are two rules:

      1. the known tool-call shape (envelope carries only tool-call metadata), and
      2. with `expect_keys` supplied: a wrapper carrying NONE of the contract's field names, with a
         nested object that carries at least one, under any key name at all.

    Rule 2 is what makes this robust to the next unseen wrapper key, and it stays safe because the
    nested object must look like the contract while the outer one must not - a genuinely malformed
    decision is never rescued into looking valid.
    """
    if not isinstance(obj, dict):
        return obj, False
    expect = frozenset(expect_keys)
    if expect and (expect & set(obj)):
        return obj, False  # the top level IS the payload
    if not (set(obj) - _ENVELOPE_KEYS):
        for key in _PAYLOAD_KEYS:
            inner = _as_dict(obj.get(key))
            if inner is not None:
                return inner, True
    if expect:
        for value in obj.values():
            inner = _as_dict(value)
            if inner is not None and (expect & set(inner)):
                return inner, True
    return obj, False


def extract_json(text: str) -> dict:
    """Parse one JSON object out of a model response.

    Strict parse first, so a model in real JSON mode costs nothing. The fallbacks handle exactly the
    two shapes the gateway produces when response_format is unavailable: a fenced code block, and a
    JSON object padded with prose. Anything else raises JSONDecodeError, which routes into the same
    jittered retry a malformed Gemini response takes - a deterministic formatting glitch must not
    simply repeat at the same temperature.
    """
    t = (text or "").strip()
    for candidate in (t, _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", t)).strip()):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            i, j = candidate.find("{"), candidate.rfind("}")
            if i < 0 or j <= i:
                continue
            try:
                obj = json.loads(candidate[i:j + 1])
            except json.JSONDecodeError:
                continue
        if isinstance(obj, dict):
            return obj
    raise json.JSONDecodeError(f"no JSON object in response: {t[:200]!r}", t or "", 0)


class LLMGatewayTool:
    """Thread-safe LLMGateway client with token refresh, timeout, retries, and JSON parsing.

    Deliberately mirrors VertexGeminiTool's public surface: `generate_json`, `generate_text`, the
    `_raw_generate` seam that tests monkeypatch, and JSON_RETRY_MIN_TEMPERATURE.
    """

    JSON_RETRY_MIN_TEMPERATURE = 0.2

    def __init__(self, model: str, base_url: str = BASE_URL, max_retries: int = MAX_RETRIES,
                 timeout_s: float = REQUEST_TIMEOUT_MS / 1000.0, json_mode: bool | None = None,
                 expect_keys: tuple = (), debris_extra: int = 5,
                 rate_per_min: int = DEFAULT_RATE_PER_MIN, rate_extra: int = 6):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        # Names the caller's payload is known to contain. A parsed object sharing NONE of them did
        # not carry the model's answer at all (a stray tool-call fragment, an empty object), so it
        # is transport debris and gets re-requested. A payload that HAS some of them and is still
        # wrong is the model's own failure and is returned untouched - re-rolling that would inflate
        # the model's apparent validity, which is exactly the quantity under measurement.
        self.expect_keys = frozenset(expect_keys)
        # Debris re-requests get their OWN budget rather than eating max_retries. A dropped payload
        # is a transport event, and spending the model-error budget on it would abandon signatures
        # the model never actually failed - 14 of claude-sonnet-5's 21 b3 losses were bare `{}`.
        self.debris_extra = debris_extra
        # Throttling gets its own budget for the same reason debris does: a 429 says nothing about
        # the model's answer, so it must not consume the retries reserved for real failures.
        self.rate_extra = rate_extra
        self.limiter = _RateLimiter(rate_per_min)
        # None = decide from the measured support table; explicit bool overrides it.
        self.json_mode = (model not in RESPONSE_FORMAT_UNSUPPORTED) if json_mode is None else json_mode
        self._local = threading.local()
        self._tok_lock = threading.Lock()
        self._tok = ""
        self._tok_exp = 0.0

    # ----------------------------------------------------------------- auth

    def _mint(self) -> tuple:
        cid = os.environ.get("CRP_LLM_CLIENT_ID") or _read_secrets_env("CRP_LLM_CLIENT_ID")
        sec = os.environ.get("CRP_LLM_CLIENT_SECRET") or _read_secrets_env("CRP_LLM_CLIENT_SECRET")
        if cid and sec:
            # login.microsoftonline.com is external and DOES need the Ford proxy, unlike the API host.
            r = requests.post(TOKEN_URL, timeout=30, data={
                "client_id": cid, "client_secret": sec, "grant_type": "client_credentials",
                "scope": f"api://{AUDIENCE}/.default"})
            if r.status_code != 200:
                raise TokenError(f"client-credentials token request failed: {r.status_code} "
                                 f"{r.text[:200]}")
            body = r.json()
            return body["access_token"], time.time() + float(body.get("expires_in", 3600))

        if OPENCODE_AUTH.exists():
            entry = (json.loads(OPENCODE_AUTH.read_text()) or {}).get("llm_gateway") or {}
            tok, exp = entry.get("access") or "", float(entry.get("expires") or 0) / 1000.0
            if tok and exp > time.time():
                return tok, exp
            if tok:
                raise TokenError(f"opencode's LLMGateway token expired "
                                 f"{int((time.time() - exp) / 60)}m ago")
        raise TokenError("no LLMGateway credentials and no usable opencode token")

    def _token(self, force: bool = False) -> str:
        with self._tok_lock:
            if not force and self._tok and time.time() < self._tok_exp - 60:
                return self._tok
            self._tok, self._tok_exp = self._mint()
            return self._tok

    def assert_token_health(self) -> None:
        """Abort loudly between chunks rather than burning calls on 401s.

        Same contract as rerun_full_chain.assert_token_health, which probes gcloud: a mid-run auth
        loss must become a clean stop, because the resumable cache would otherwise fill with shells
        that look successful.
        """
        try:
            self._token(force=True)
        except TokenError as e:
            raise SystemExit(
                f"\nAUTH LOST - stopping before burning LLM calls on shells.\n{e}\n"
                f"Re-authenticate ({RELOGIN_HINT}), then re-run this command; the cache resumes "
                "where it stopped.") from e

    # ------------------------------------------------------------- transport

    def _session(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None:
            s = requests.Session()
            # The API host is internal to the corporate network; routing it through the corporate
            # proxy fails. trust_env=False makes that independent of the caller's environment.
            s.trust_env = False
            self._local.s = s
        return s

    def _raw_generate(self, prompt: str, temperature: float, json_mode: bool,
                      force: bool = False) -> str:
        """Single transport call. Isolated so tests can monkeypatch it without a network."""
        body = {"model": self.model, "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]}
        if json_mode and self.json_mode:
            body["response_format"] = {"type": "json_object"}
        self.limiter.acquire()
        r = self._session().post(
            f"{self.base_url}/chat/completions", json=body, timeout=self.timeout_s,
            headers={"Authorization": f"Bearer {self._token(force=force)}",
                     "Content-Type": "application/json"})
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code} {r.text[:300]}")
        choices = (r.json() or {}).get("choices") or []
        if not choices:
            raise RuntimeError(f"no choices in response: {r.text[:300]}")
        return choices[0].get("message", {}).get("content") or ""

    @staticmethod
    def _is_retryable(msg: str) -> bool:
        return is_retryable(msg)

    # ---------------------------------------------------------------- public

    def generate_json(self, prompt: str, temperature: float = 0.2) -> dict:
        """Call the gateway and return a parsed dict.

        {"_ok": True, ...payload, "_attempts": n} on success, {"_ok": False, "error": ...} on
        permanent failure - never raises, so one bad signature cannot crash a 400-call batch.
        Malformed JSON is retried like a transient error with the temperature floored at
        JSON_RETRY_MIN_TEMPERATURE; "_attempts" > 1 therefore means the payload was sampled at the
        jittered temperature rather than the requested one.
        """
        last, raw, wrapped_any = None, "", False
        attempt, debris, hard, throttled = 0, 0, 0, 0
        while attempt < self.max_retries + self.debris_extra + self.rate_extra:
            # Only a CONTENT retry floors the temperature. A 429 says nothing about what came back,
            # so re-requesting after one must not silently move the sampling protocol.
            t = (temperature if debris + hard == 0
                 else max(temperature, self.JSON_RETRY_MIN_TEMPERATURE))
            attempt += 1
            try:
                raw = self._raw_generate(prompt, t, json_mode=True, force=attempt > 1)
                payload, wrapped = unwrap_tool_call(extract_json(raw), self.expect_keys)
                wrapped_any = wrapped_any or wrapped
                if self.expect_keys and not (self.expect_keys & set(payload)):
                    debris += 1
                    last = f"transport debris, no payload keys: {json.dumps(payload)[:150]}"
                    if debris > self.debris_extra:
                        break
                    continue  # the answer never arrived; re-request it
                # _attempts counts CONTENT attempts, matching VertexGeminiTool's meaning: >1 says
                # the payload was sampled at the jittered temperature, which is what the stability
                # and reliability analyses read. Throttle round-trips are reported separately.
                out = {"_ok": True, **payload, "_attempts": 1 + debris + hard}
                if wrapped_any:
                    out["_envelope"] = True
                if debris:
                    out["_debris_retries"] = debris
                if throttled:
                    out["_throttled"] = throttled
                return out
            except json.JSONDecodeError as e:
                last = f"JSON parse failed: {str(e)[:200]}"
                hard += 1
            except Exception as e:  # noqa: BLE001
                last = str(e)
                if "429" in last or "rate_limit" in last:
                    throttled += 1
                    self.limiter.penalize()
                    if throttled > self.rate_extra:
                        break
                    time.sleep(5 + throttled * 5)
                    continue
                hard += 1
                if not self._is_retryable(last):
                    break
                time.sleep(2 + hard * 3)
            if hard >= self.max_retries:
                break
        return {"_ok": False, "error": str(last)[:300], "_raw": (raw or "")[:500],
                "_debris_retries": debris, "_throttled": throttled}

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

    def list_models(self) -> list:
        r = self._session().get(f"{self.base_url}/models", timeout=45,
                                headers={"Authorization": f"Bearer {self._token()}"})
        r.raise_for_status()
        return sorted(m["id"] for m in (r.json() or {}).get("data", []))


# --------------------------------------------------------------- client selection

def slug(model: str) -> str:
    """Filename-safe model tag for cache paths and artifact names."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")


def make_client(model: str = "", expect_keys: tuple = ()) -> tuple:
    """(client, token_probe, tag) for a study run.

    An empty/absent model means the shipped Vertex Gemini path with the shipped gcloud probe, so
    every cache path, artifact name and number stays byte-identical to what the paper reports.
    `expect_keys` only applies to the gateway client (Vertex has never produced envelope debris).
    """
    if not model or model == "gemini":
        from src.tools import VertexGeminiTool
        from rerun_full_chain import assert_token_health
        return VertexGeminiTool(), assert_token_health, ""
    client = LLMGatewayTool(model, expect_keys=expect_keys)
    return client, client.assert_token_health, slug(model)


def suffixed(name: str, tag: str) -> str:
    """`b3_single_call.json` + tag `claude-sonnet-5` -> `b3_single_call_claude-sonnet-5.json`.

    An empty tag returns the name untouched. This is what keeps a second model from overwriting
    the shipped Gemini artifacts: the JSONL caches dedup last-write-wins on signature_id, so a
    collision would be silent.
    """
    if not tag:
        return name
    stem, dot, ext = name.rpartition(".")
    return f"{stem}_{tag}{dot}{ext}" if dot else f"{name}_{tag}"
