"""
Shared authentication helper.

Ford's context-aware access blocks long-lived Application Default Credentials (ADC) for the
warranty data project, but the interactive gcloud user credential is allowed. So every tool
authenticates with a freshly-minted gcloud user OAuth token, and re-mints it well before the
~1 hour expiry. This single helper is the reason a 3,600-call LLM batch can run unattended
without auth failures (a challenge we hit and solved - see README "Challenges Overcome").
"""
import os
import time
import threading
import subprocess

from ..config import GCLOUD_ACCOUNT, TOKEN_TTL_SEC

_LOCK = threading.Lock()
_cache = {"token": None, "ts": 0.0}
ACCOUNT = GCLOUD_ACCOUNT  # back-compat alias


def fresh_access_token(force: bool = False) -> str:
    """Return a valid gcloud user OAuth access token, re-minting if stale.

    Thread-safe and serialized so concurrent agents don't spawn a gcloud subprocess storm.
    Raises RuntimeError with a clear, actionable message if the user needs to re-authenticate.
    """
    with _LOCK:
        now = time.time()
        if force or _cache["token"] is None or (now - _cache["ts"]) > TOKEN_TTL_SEC:
            # use an explicit account only if one is configured (GCLOUD_ACCOUNT); else the active one
            cmd = ["gcloud", "auth", "print-access-token"]
            if ACCOUNT:
                cmd += ["--account", ACCOUNT]
            login_hint = f"gcloud auth login{' --account ' + ACCOUNT if ACCOUNT else ''}"
            try:
                token = subprocess.check_output(
                    cmd,
                    env={**os.environ, "CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT": ""},
                    stderr=subprocess.PIPE,
                ).decode().strip()
            except subprocess.CalledProcessError as e:  # graceful, actionable failure
                raise RuntimeError(
                    f"Could not mint a gcloud access token. Run `{login_hint}` and retry. "
                    f"Original error: {e.stderr.decode()[:200] if e.stderr else e}"
                ) from e
            except FileNotFoundError as e:  # gcloud not installed / not on PATH
                raise RuntimeError(
                    f"gcloud CLI not found on PATH. Install the Google Cloud SDK and run `{login_hint}`."
                ) from e
            _cache["token"] = token
            _cache["ts"] = now
        return _cache["token"]
