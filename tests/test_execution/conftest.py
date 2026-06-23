"""Shared fixtures for the execution-engine tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freshdata.config import CleanConfig


@pytest.fixture
def native_config() -> CleanConfig:
    """A config fully inside the native (out-of-core) subset.

    ``strategy="conservative"`` disables the decision engine and
    ``fix_dtypes=False`` skips the sampled dtype heuristics, so all three engines
    execute natively and must agree byte-for-byte.
    """
    return CleanConfig(strategy="conservative", fix_dtypes=False, verbose=False)


@pytest.fixture
def small_df() -> pd.DataFrame:
    """20 rows exercising rename, whitespace, sentinels, empties, and dedup."""
    return pd.DataFrame(
        {
            "patient_id": ["P001", "P002", None, "P004"] * 5,
            "age": [34.0, np.nan, 52.0, 89.0] * 5,
            "revenue": [1200.0, np.nan, 3400.0, np.nan] * 5,
            "category": ["A", None, "B", "A"] * 5,
            "diagnosis": ["X10", "Y20", None, "Z30"] * 5,
            " Name ": ["alice", " bob", "carol ", None] * 5,
            "n/a_col": ["N/A", None, "-", "#REF!"] * 5,
            "empty_col": [None] * 20,
        }
    )


@pytest.fixture
def parquet_10k(tmp_path) -> str:
    from freshdata.benchmarks._data_gen import generate_parquet

    path = str(tmp_path / "bench_10k.parquet")
    generate_parquet(10_000, path, batch_size=5_000)
    return path
