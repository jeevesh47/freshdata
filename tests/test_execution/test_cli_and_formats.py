"""Cover the benchmark CLI, output formats, and alternate source ingestion."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd


def test_benchmark_cli_runs(tmp_path, capsys):
    pytest.importorskip("pyarrow")
    pytest.importorskip("psutil")
    from freshdata.benchmarks.run_benchmarks import main

    rc = main([
        "--sizes", "10k",
        "--engines", "pandas",
        "--data-dir", str(tmp_path / "data"),
        "--results-dir", str(tmp_path / "results"),
    ])
    assert rc == 0
    assert "completed cleanly" in capsys.readouterr().out


def test_benchmark_cli_rejects_unknown_size(tmp_path):
    from freshdata.benchmarks.run_benchmarks import main

    with pytest.raises(SystemExit):
        main(["--sizes", "999z", "--engines", "pandas"])


def test_output_format_arrow(small_df, native_config):
    pa = pytest.importorskip("pyarrow")
    out = fd.clean(small_df.copy(), config=native_config,
                   engine="duckdb", output_format="arrow")
    assert isinstance(out, pa.Table)


def test_output_format_arrow_from_polars(small_df, native_config):
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("polars")
    out = fd.clean(small_df.copy(), config=native_config,
                   engine="polars", output_format="arrow")
    assert isinstance(out, pa.Table)


def test_clean_reads_csv_path(tmp_path, native_config):
    df = pd.DataFrame({"id": [1, 2, 3], "name": [" a ", "b", "N/A"]})
    path = str(tmp_path / "in.csv")
    df.to_csv(path, index=False)
    # default engine="pandas" now also accepts a path
    out = fd.clean(path, config=native_config)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 3


def test_clean_bare_parquet_path_auto(tmp_path, native_config):
    pytest.importorskip("duckdb")
    df = pd.DataFrame({"id": [1, 2, 3], "v": [1.0, 2.0, 3.0]})
    path = str(tmp_path / "in.parquet")
    df.to_parquet(path)
    out = fd.clean(path, config=native_config, engine="auto")
    assert isinstance(out, pd.DataFrame)
