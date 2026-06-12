"""Exact duplicate-row removal (keeps the first occurrence)."""

from __future__ import annotations

import pandas as pd

from ..config import CleanConfig
from ..report import CleanReport


def drop_duplicate_rows(df: pd.DataFrame, config: CleanConfig,
                        report: CleanReport) -> pd.DataFrame:
    """Drop rows that are exact duplicates of an earlier row.

    With ``duplicate_subset`` set, only those columns are compared (names refer
    to *post-rename* columns when ``column_names=True``). Columns holding
    unhashable values (lists, dicts) make duplicate detection impossible; the
    step is then skipped and noted in the report rather than guessing.
    """
    if df.empty:
        return df
    subset = None
    if config.duplicate_subset is not None:
        subset = list(config.duplicate_subset)
        missing = [c for c in subset if c not in df.columns]
        if missing:
            raise ValueError(
                f"duplicate_subset column(s) not found: {missing}. "
                f"Available columns: {list(df.columns)}. "
                "Note: names refer to columns *after* renaming when column_names=True."
            )
    try:
        dup = df.duplicated(subset=subset, keep="first")
    except TypeError:
        report.add("drop_duplicates",
                   "skipped: column(s) contain unhashable values (e.g. lists)")
        return df
    n = int(dup.sum())
    if n:
        df = df.loc[~dup]
        where = f" (compared on {subset})" if subset else ""
        report.add("drop_duplicates", f"dropped {n} duplicate row(s){where}", count=n)
    return df
