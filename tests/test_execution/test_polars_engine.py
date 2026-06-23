"""Polars backend behaviour: ingestion, streaming, pushdown, fallback."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.execution import EngineConfig

pl = pytest.importorskip("polars")


def test_accepts_lazy_frame(small_df, native_config):
    lf = pl.from_pandas(small_df).lazy()
    out = fd.clean(lf, config=native_config, engine="polars", output_format="polars")
    assert isinstance(out, pl.DataFrame)


def test_accepts_polars_frame_returns_pandas_by_default(small_df, native_config):
    pf = pl.from_pandas(small_df)
    out = fd.clean(pf, config=native_config, engine="polars")
    assert isinstance(out, pd.DataFrame)  # output_format defaults to pandas


def test_accepts_parquet_path(parquet_10k, native_config):
    out = fd.clean(parquet_10k, config=native_config, engine="polars")
    assert isinstance(out, pd.DataFrame)
    assert len(out) > 0


def test_streaming_true_and_false(small_df, native_config):
    pf = pl.from_pandas(small_df)
    for streaming in (True, False):
        ec = EngineConfig(engine="polars", streaming=streaming)
        out = fd.clean(pf, config=native_config, engine_config=ec)
        assert isinstance(out, (pd.DataFrame,))


def test_strip_whitespace_and_sentinels(native_config):
    # id column keeps the sentinel row from becoming all-null (and thus dropped)
    df = pd.DataFrame({"id": [1, 2, 3, 4], "name": [" alice", "bob ", "N/A", "carol"]})
    out = fd.clean(df, config=native_config, engine="polars", output_format="polars")
    vals = out.sort("id")["name"].to_list()
    assert vals[0] == "alice" and vals[1] == "bob"
    assert vals[2] is None  # sentinel -> null


def test_drop_empty_column(native_config):
    df = pd.DataFrame({"a": [1, 2, 3], "empty": [None, None, None]})
    out = fd.clean(df, config=native_config, engine="polars", output_format="polars")
    assert "empty" not in out.columns


def test_fallback_on_knn_like_config_warns(small_df, caplog):
    """A config needing the decision engine falls back to pandas with a warning."""
    import logging

    with caplog.at_level(logging.WARNING, logger="freshdata.execution.polars"):
        out = fd.clean(small_df.copy(), engine="polars")  # default balanced -> fallback
    assert isinstance(out, pd.DataFrame)
    assert any("falling back to pandas" in r.message for r in caplog.records)


def test_thread_config_does_not_raise(small_df, native_config):
    ec = EngineConfig(engine="polars", polars_n_threads=2)
    out = fd.clean(small_df.copy(), config=native_config, engine_config=ec)
    assert isinstance(out, pd.DataFrame)


def test_projection_pushdown_drops_empty_before_collect(native_config):
    """Empty columns are removed; result has fewer columns than the input."""
    df = pd.DataFrame({"keep": [1, 2], "gone": [None, None]})
    out = fd.clean(df, config=native_config, engine="polars", output_format="polars")
    assert list(out.columns) == ["keep"]
