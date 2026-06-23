"""The pandas backend must be identical to the legacy fd.clean path."""

from __future__ import annotations

import pandas as pd

import freshdata as fd


def test_pandas_engine_matches_legacy(small_df):
    """engine='pandas' output equals the default (no-engine) output exactly."""
    legacy, legacy_rep = fd.clean(small_df.copy(), return_report=True)
    engined, engined_rep = fd.clean(small_df.copy(), engine="pandas", return_report=True)
    pd.testing.assert_frame_equal(legacy, engined)
    assert [(a.step, a.column, a.count) for a in legacy_rep.actions] == \
           [(a.step, a.column, a.count) for a in engined_rep.actions]


def test_default_call_unchanged(small_df):
    """fd.clean(df) with no engine kwarg still returns a pandas DataFrame."""
    out = fd.clean(small_df.copy())
    assert isinstance(out, pd.DataFrame)


def test_pandas_engine_reads_parquet_path(tmp_path, small_df, native_config):
    path = str(tmp_path / "p.parquet")
    small_df.to_parquet(path)
    out = fd.clean(path, config=native_config, engine="pandas")
    assert isinstance(out, pd.DataFrame)
    assert len(out) > 0


def test_output_format_polars_from_pandas_engine(small_df, native_config):
    pl = __import__("pytest").importorskip("polars")
    out = fd.clean(small_df.copy(), config=native_config,
                   engine="pandas", output_format="polars")
    assert isinstance(out, pl.DataFrame)
