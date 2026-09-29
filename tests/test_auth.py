"""Tests for the gcloud token helper: caching, refresh, and graceful failure."""
import subprocess

import pytest

import src.tools.auth as auth


@pytest.fixture(autouse=True)
def _reset_cache():
    auth._cache["token"] = None
    auth._cache["ts"] = 0.0
    yield


def test_returns_token(monkeypatch):
    monkeypatch.setattr(auth.subprocess, "check_output", lambda *a, **k: b"abc123\n")
    assert auth.fresh_access_token() == "abc123"


def test_caches_token(monkeypatch):
    calls = {"n": 0}

    def fake(*a, **k):
        calls["n"] += 1
        return b"tok\n"

    monkeypatch.setattr(auth.subprocess, "check_output", fake)
    auth.fresh_access_token()
    auth.fresh_access_token()
    assert calls["n"] == 1  # second call served from cache


def test_force_refresh(monkeypatch):
    calls = {"n": 0}

    def fake(*a, **k):
        calls["n"] += 1
        return f"tok{calls['n']}".encode()

    monkeypatch.setattr(auth.subprocess, "check_output", fake)
    assert auth.fresh_access_token() == "tok1"
    assert auth.fresh_access_token(force=True) == "tok2"


def test_called_process_error_is_actionable(monkeypatch):
    def boom(*a, **k):
        raise subprocess.CalledProcessError(1, "gcloud", stderr=b"reauth required")

    monkeypatch.setattr(auth.subprocess, "check_output", boom)
    with pytest.raises(RuntimeError, match="gcloud auth login"):
        auth.fresh_access_token()


def test_missing_gcloud_is_actionable(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("gcloud")

    monkeypatch.setattr(auth.subprocess, "check_output", boom)
    with pytest.raises(RuntimeError, match="gcloud CLI not found"):
        auth.fresh_access_token()
