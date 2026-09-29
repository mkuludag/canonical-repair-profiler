"""Tests for DataStore IO + the resumable JSONL cache."""
import pandas as pd
import pytest

from src.tools import DataStore


def test_csv_roundtrip(tmp_path):
    ds = DataStore(root=str(tmp_path))
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    ds.write_csv(df, "sub/out.csv")
    back = ds.read_csv("sub/out.csv")
    pd.testing.assert_frame_equal(df, back)


def test_parquet_roundtrip(tmp_path):
    ds = DataStore(root=str(tmp_path))
    df = pd.DataFrame({"a": [1.0, 2.0]})
    ds.write_parquet(df, "out.parquet")
    back = ds.read_parquet("out.parquet", columns=["a"])
    assert list(back["a"]) == [1.0, 2.0]


def test_missing_parquet_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DataStore(root=str(tmp_path)).read_parquet("nope.parquet")


def test_cache_append_and_load(tmp_path):
    ds = DataStore(root=str(tmp_path))
    ds.append_cache("c.jsonl", {"signature_id": 1, "_ok": True, "solutions": []})
    ds.append_cache("c.jsonl", {"signature_id": 2, "_ok": False, "error": "x"})
    ok = ds.load_cache("c.jsonl", ok_only=True)
    assert set(ok.keys()) == {1}
    every = ds.load_cache("c.jsonl", ok_only=False)
    assert set(every.keys()) == {1, 2}


def test_cache_skips_malformed_lines(tmp_path):
    ds = DataStore(root=str(tmp_path))
    p = tmp_path / "c.jsonl"
    p.write_text('{"signature_id": 1, "_ok": true}\nNOT JSON\n{"signature_id": 3, "_ok": true}\n')
    out = ds.load_cache("c.jsonl")
    assert set(out.keys()) == {1, 3}


def test_load_cache_missing_file_returns_empty(tmp_path):
    assert DataStore(root=str(tmp_path)).load_cache("missing.jsonl") == {}
