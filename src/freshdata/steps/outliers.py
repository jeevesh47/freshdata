"""Opt-in outlier handling for numeric columns.

Off by default. ``outliers="clip"`` winsorizes values to the detection bounds;
``outliers="flag"`` leaves data untouched and adds a boolean ``<col>_outlier``
column instead. Detection: Tukey fences (``iqr``, factor 1.5) or mean ± k
standard deviations (``zscore``, factor 3.0).
"""

from __future__ import annotations

import math

import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype

from ..config import CleanConfig
from ..report import CleanReport


def _bounds(s: pd.Series, config: CleanConfig):
    factor = config.resolved_outlier_factor
    if config.outlier_method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        spread = q3 - q1
        if pd.isna(spread) or spread == 0:
            return None
        return float(q1 - factor * spread), float(q3 + factor * spread)
    mean, std = s.mean(), s.std()
    if pd.isna(std) or std == 0:
        return None
    return float(mean - factor * std), float(mean + factor * std)


def _unique_flag_name(df: pd.DataFrame, base: str) -> str:
    name, k = base, 1
    while name in df.columns:
        k += 1
        name = f"{base}_{k}"
    return name


def handle_outliers(df: pd.DataFrame, config: CleanConfig,
                    report: CleanReport) -> pd.DataFrame:
    """Clip or flag outliers in every numeric (non-boolean) column."""
    if config.outliers is None or df.empty:
        return df
    numeric_cols = [c for c in df.columns
                    if is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])]
    for col in numeric_cols:
        s = df[col]
        bounds = _bounds(s, config)
        if bounds is None:
            continue
        lo, hi = bounds
        if is_integer_dtype(s):  # keep integer columns integer after clipping
            lo, hi = math.floor(lo), math.ceil(hi)
        mask = (s < lo) | (s > hi)
        n = int(mask.sum())
        if n == 0:
            continue
        label = (f"{config.outlier_method}, factor "
                 f"{config.resolved_outlier_factor:g}")
        if config.outliers == "clip":
            df[col] = s.clip(lo, hi)
            report.add("outliers", f"clipped {n} outlier(s) to [{lo:g}, {hi:g}] ({label})",
                       column=str(col), count=n)
        else:
            flag = _unique_flag_name(df, f"{col}_outlier")
            df[flag] = mask.fillna(False).astype(bool)
            report.add("outliers", f"flagged {n} outlier(s) in new column {flag!r} ({label})",
                       column=str(col), count=n)
    return df
