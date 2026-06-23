# freshdata out-of-core benchmark — reference run

Reproduce with::

    pip install -e ".[outofcore,bench]"
    python -m freshdata.benchmarks.run_benchmarks \
        --sizes 10k,100k,1m,10m --engines pandas,polars,duckdb

Workload: `strategy="conservative", fix_dtypes=False` — the native (out-of-core)
subset: snake_case rename, whitespace trim, sentinel→null, empty column/row
drops, and full-row dedup. Timing for **pandas** includes the in-memory Parquet
load (that is its nature); **polars** and **duckdb** receive the Parquet *path*
and stream it. Peak RSS is the process resident-memory increase during the run.

Machine: Apple Silicon laptop, Python 3.13, polars 1.40, duckdb 1.5.

| Engine | Rows | File (MB) | Wall (s) | Peak RSS (MB) | Throughput (rows/s) | Trust |
|--------|-----:|----------:|---------:|--------------:|--------------------:|------:|
| pandas | 10,000 | 1.2 | 0.13 | 29 | 74,633 | 95.0 |
| polars | 10,000 | 1.2 | 0.11 | 66 | 92,868 | 95.0 |
| duckdb | 10,000 | 1.2 | 0.30 | 107 | 33,763 | 95.0 |
| pandas | 100,000 | 11.3 | 0.82 | 128 | 122,597 | 95.1 |
| polars | 100,000 | 11.3 | 0.27 | 247 | 374,990 | 95.1 |
| duckdb | 100,000 | 11.3 | 1.67 | 120 | 59,899 | 95.1 |
| pandas | 1,000,000 | 113.2 | 5.91 | 973 | 169,302 | 95.0 |
| polars | 1,000,000 | 113.2 | 1.87 | 671 | 535,288 | 95.0 |
| duckdb | 1,000,000 | 113.2 | 5.78 | 294 | 172,915 | 95.0 |
| pandas | 10,000,000 | 1,132.1 | 84.51 | 2,996 | 118,330 | 95.0 |
| polars | 10,000,000 | 1,132.1 | 45.05 | 4,892 | 221,951 | 95.0 |
| duckdb | 10,000,000 | 1,132.1 | 68.19 | 4,370 | 146,660 | 95.0 |

## Reading the numbers

- **Parity holds at scale.** The Data Trust Score is identical across all three
  backends at every size (95.0 / 95.1) — the Polars and DuckDB engines produce
  the same cleaned data and the same `CleanReport` as pandas.
- **Throughput:** Polars is consistently the fastest (~2–3× pandas; e.g. 10M rows
  in 45s vs 85s). DuckDB beats pandas on the larger frames.
- **Memory:** DuckDB is the most memory-efficient at moderate scale (1M rows in
  **294 MB** vs pandas' **973 MB** — 3.3× less), because it streams the Parquet
  scan and only the reduced result is fetched.
- **The 10M peak-RSS caveat (honest):** at 10M rows the polars/duckdb peak RSS is
  *higher* than pandas here, for two reasons specific to this run: (1) the harness
  converts the full result to pandas to compute the Trust Score, which transiently
  doubles the result in memory, and (2) the machine has ample RAM and the default
  `memory_limit_gb=8`, so DuckDB has no reason to spill. Lowering
  `EngineConfig(memory_limit_gb=...)` forces spill-to-disk and trades wall time for
  a much smaller footprint — that is the lever for genuinely larger-than-RAM data.

## 100M / 1B rows

The harness is parametrised up to `1b`. Those sizes need hundreds of GB of
synthetic Parquet and were not generated here; run them on a box with the disk to
spare (`--sizes 100m,1b`). pandas is expected to OOM well before 1B rows while the
streaming/spilling backends complete.
