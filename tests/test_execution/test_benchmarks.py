"""Smoke tests for the benchmark harness (CI-safe at 10k rows)."""

from __future__ import annotations

import os

import pytest


def test_data_generator_creates_parquet(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from freshdata.benchmarks._data_gen import generate_parquet

    path = str(tmp_path / "g.parquet")
    generate_parquet(10_000, path, batch_size=5_000)
    assert os.path.exists(path)
    assert pq.read_metadata(path).num_rows == 10_000


def test_harness_runs_pandas_10k(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("psutil")
    from freshdata.benchmarks._harness import BenchmarkHarness

    results = BenchmarkHarness().run(
        sizes=["10k"],
        engines=["pandas"],
        data_dir=str(tmp_path / "data"),
        results_dir=str(tmp_path / "results"),
    )
    assert len(results) == 1
    r = results[0]
    assert r.error is None and not r.oom
    assert r.throughput_rps > 0


def test_harness_all_engines_agree_on_actions(tmp_path):
    pytest.importorskip("polars")
    pytest.importorskip("duckdb")
    pytest.importorskip("psutil")
    from freshdata.benchmarks._harness import BenchmarkHarness

    results = BenchmarkHarness().run(
        sizes=["10k"],
        engines=["pandas", "polars", "duckdb"],
        data_dir=str(tmp_path / "data"),
        results_dir=str(tmp_path / "results"),
        write=False,
    )
    by_engine = {r.engine: r for r in results}
    assert by_engine["pandas"].actions_count == by_engine["polars"].actions_count
    assert by_engine["pandas"].actions_count == by_engine["duckdb"].actions_count
