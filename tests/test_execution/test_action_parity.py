"""The critical parity tests: all three engines agree on the native subset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import freshdata as fd

pytest.importorskip("polars")
pytest.importorskip("duckdb")


def _clean(df, config, engine):
    out, rep = fd.clean(df.copy(), config=config, engine=engine, return_report=True)
    out = out if isinstance(out, pd.DataFrame) else out.to_pandas()
    return out, rep


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(sorted(df.columns), axis=1)
    # Coerce every null representation (None / pd.NA / NaN) to np.nan so the
    # comparison is independent of which backend produced the frame.
    df = df.where(pd.notna(df), np.nan)
    return df.sort_values(by=list(df.columns)).reset_index(drop=True)


def _reports(small_df, native_config):
    return {
        eng: _clean(small_df, native_config, eng)
        for eng in ("pandas", "polars", "duckdb")
    }


def test_action_step_tuples_identical(small_df, native_config):
    reps = _reports(small_df, native_config)
    pandas_actions = [(a.step, a.column, a.count) for a in reps["pandas"][1].actions]
    for eng in ("polars", "duckdb"):
        eng_actions = [(a.step, a.column, a.count) for a in reps[eng][1].actions]
        assert eng_actions == pandas_actions, f"{eng} action mismatch"


def test_action_risk_identical(small_df, native_config):
    reps = _reports(small_df, native_config)
    pandas_risk = [(a.step, a.risk) for a in reps["pandas"][1].actions]
    for eng in ("polars", "duckdb"):
        eng_risk = [(a.step, a.risk) for a in reps[eng][1].actions]
        assert eng_risk == pandas_risk, f"{eng} risk mismatch"


def test_cleaned_values_identical(small_df, native_config):
    reps = _reports(small_df, native_config)
    base = _normalize(reps["pandas"][0])
    for eng in ("polars", "duckdb"):
        out = _normalize(reps[eng][0])
        assert list(base.columns) == list(out.columns), f"{eng} column mismatch"
        pd.testing.assert_frame_equal(base, out, check_dtype=False)


def test_trust_score_within_tolerance(small_df, native_config):
    reps = _reports(small_df, native_config)
    base = fd.compute_trust_score(reps["pandas"][0]).overall
    for eng in ("polars", "duckdb"):
        score = fd.compute_trust_score(reps[eng][0]).overall
        assert abs(base - score) < 0.1, f"{eng} trust {score} vs pandas {base}"


def test_shapes_identical(small_df, native_config):
    reps = _reports(small_df, native_config)
    base = reps["pandas"][0].shape
    for eng in ("polars", "duckdb"):
        assert reps[eng][0].shape == base, f"{eng} shape {reps[eng][0].shape} != {base}"


def test_parity_on_parquet_path(parquet_10k, native_config):
    """Path-based input: polars and duckdb agree with pandas at 10k rows."""
    pandas_out, _ = _clean(pd.read_parquet(parquet_10k), native_config, "pandas")
    base = _normalize(pandas_out)
    for eng in ("polars", "duckdb"):
        out, rep = fd.clean(parquet_10k, config=native_config, engine=eng, return_report=True)
        out = out if isinstance(out, pd.DataFrame) else out.to_pandas()
        assert len(rep.actions) > 0
        pd.testing.assert_frame_equal(base, _normalize(out), check_dtype=False)
