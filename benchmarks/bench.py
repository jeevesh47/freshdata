"""Quick benchmark for freshdata.clean on a realistically messy frame.

Run:  python benchmarks/bench.py [n_rows ...]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

import freshdata as fd


def make_messy(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = rng.choice([" Alice ", "Bob", "carol", "N/A", "Dave "], n)
    return pd.DataFrame(
        {
            "Customer ID": np.arange(n),
            " Name ": names,
            "Order Total": [f"${x:,.2f}" for x in rng.uniform(5, 9999, n)],
            "Quantity": rng.integers(1, 50, n).astype(str),
            "Order Date": pd.date_range("2022-01-01", periods=n, freq="min").astype(str),
            "Is Member": rng.choice(["yes", "no", "YES", "No"], n),
            "Notes": rng.choice(["", "-", "expedite", "  gift wrap  ", "null"], n),
            "Unused": [None] * n,
        }
    )


def bench(n: int) -> None:
    df = make_messy(n)
    fd.clean(df.head(1000))  # warm-up

    start = time.perf_counter()
    cleaned, report = fd.clean(df, report=True)
    elapsed = time.perf_counter() - start

    rate = n / elapsed / 1e6
    print(
        f"{n:>10,} rows  {elapsed:8.3f}s  {rate:6.2f}M rows/s  "
        f"{len(report)} actions  "
        f"memory {report.memory_before / 1e6:.1f} -> {report.memory_after / 1e6:.1f} MB"
    )


if __name__ == "__main__":
    sizes = [int(a.replace("_", "")) for a in sys.argv[1:]] or [10_000, 100_000, 1_000_000]
    print(f"freshdata {fd.__version__} | pandas {pd.__version__} | numpy {np.__version__}")
    for size in sizes:
        bench(size)
