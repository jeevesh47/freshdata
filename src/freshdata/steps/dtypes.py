"""Type inference for text columns: boolean, numeric, datetime.

Strategy per column, in priority order (first hit wins):

1. **boolean** — every non-missing value is in a small true/false vocabulary.
2. **numeric** — plain ``to_numeric``, with a second pass that understands
   thousands separators and currency symbols (``"$1,234.56"``).
3. **datetime** — guarded by a cheap "looks like a date" heuristic so we never
   pay datetime parsing on obviously non-date columns.

Conversions are *validated*: they only stick if at least ``*_threshold`` of
the non-missing values parse, and every value coerced to missing is counted
and reported. Large columns are pre-screened on a sample so hopeless
conversions are rejected at O(sample) instead of O(n).
"""

from __future__ import annotations

import re
import warnings

import pandas as pd
from pandas.api.types import infer_dtype

from .._util import PANDAS_MAJOR, sample_series, stringlike_columns
from ..config import CleanConfig
from ..report import CleanReport

_TRUE_WORDS = frozenset({"true", "t", "yes", "y"})
_FALSE_WORDS = frozenset({"false", "f", "no", "n"})
_BOOL_WORDS = _TRUE_WORDS | _FALSE_WORDS

# Numbers dressed up as text: optional sign/currency, comma groups, decimals.
_FORMATTED_NUMBER = re.compile(
    r"\s*[+-]?\s*[$€£₹]?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*[$€£₹]?\s*"
)
_NUMERIC_NOISE = re.compile(r"[,\s$€£₹]")

_DATEISH = re.compile(
    r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"  # 2021-01-05, 1/5/21, 05.01.2021
    r"|\d{1,2}:\d{2}"  # 14:30
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[ .,-]*\d{1,4}"
    r"|\d{1,2}[ .,-]*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
    re.IGNORECASE,
)

# Largest float that safely round-trips through int64.
_INT64_SAFE = float(2**63 - 1024)


def _finalize_numeric(parsed: pd.Series) -> pd.Series:
    """Normalize a parsed numeric column to int64 / Int64 / float64."""
    nonnull = parsed.dropna()
    is_integral = (
        len(nonnull) > 0
        and bool((nonnull % 1 == 0).all())
        and float(nonnull.abs().max()) < _INT64_SAFE
    )
    if is_integral:
        return parsed.astype("int64") if not parsed.isna().any() else parsed.astype("Int64")
    return parsed.astype("float64")


def _try_boolean(s: pd.Series, nonnull: pd.Series) -> pd.Series | None:
    """Convert true/false-vocabulary text (or raw Python bools) to boolean."""
    try:
        uniques = nonnull.unique()[:10]
    except TypeError:  # unhashable values (lists/dicts) — clearly not boolean
        return None
    if len(uniques) > 8:  # vocabulary has at most 8 spellings
        return None
    if all(isinstance(v, bool) for v in uniques):
        converted = s.astype("boolean")
    elif all(isinstance(v, str) for v in uniques) and {
        v.casefold() for v in uniques
    } <= _BOOL_WORDS:
        mapping = dict.fromkeys(_TRUE_WORDS, True)
        mapping.update(dict.fromkeys(_FALSE_WORDS, False))
        converted = s.str.casefold().map(mapping).astype("boolean")
    else:
        return None
    if not converted.isna().any():
        return converted.astype(bool)
    return converted


def _to_numeric_or_none(values: pd.Series) -> pd.Series | None:
    """``to_numeric`` that tolerates non-scalar cells (lists raise even with
    ``errors="coerce"``)."""
    try:
        return pd.to_numeric(values, errors="coerce")
    except (TypeError, ValueError):
        return None


def _try_numeric(
    s: pd.Series, nonnull: pd.Series, config: CleanConfig
) -> tuple[pd.Series | None, int]:
    """Parse as numeric. Returns ``(converted, n_coerced)`` or ``(None, 0)``."""
    threshold = config.numeric_threshold
    n = len(nonnull)
    sample = sample_series(nonnull, config.sample_size, config.random_state)

    # Cheap rejection on the sample before paying for a full-column parse.
    sample_parsed = _to_numeric_or_none(sample)
    if sample_parsed is None:
        return None, 0
    parsed = None
    if sample_parsed.notna().mean() >= threshold * 0.8:
        candidate = _to_numeric_or_none(s)
        if candidate is not None and candidate.notna().sum() / n >= threshold:
            parsed = candidate

    if parsed is None:
        # Second chance: values like "$1,234.56". Only worth attempting if the
        # sample actually contains separator/currency characters.
        has_noise = sample.astype("string").str.contains(r"[,$€£₹]", regex=True, na=False)
        if not bool(has_noise.any()):
            return None, 0
        matches = s.str.fullmatch(_FORMATTED_NUMBER).eq(True)
        if matches.dtype != bool:  # BooleanDtype (string input) keeps NA through eq()
            matches = matches.fillna(False).astype(bool)
        if not matches.any():
            return None, 0
        cleaned = s.str.replace(_NUMERIC_NOISE, "", regex=True).where(matches, s)
        candidate = _to_numeric_or_none(cleaned)
        if candidate is None or candidate.notna().sum() / n < threshold:
            return None, 0
        parsed = candidate

    n_coerced = int((s.notna() & parsed.isna()).sum())
    return _finalize_numeric(parsed), n_coerced


def _looks_dateish(sample: pd.Series) -> bool:
    """Cheap pre-screen: do most sampled values resemble dates/times at all?"""
    values = [v for v in sample.head(20) if isinstance(v, str) and len(v) <= 40]
    if not values:
        return False
    hits = sum(1 for v in values if _DATEISH.search(v))
    return hits / len(values) >= 0.6


def _parse_datetime(s: pd.Series, mixed_formats: bool) -> pd.Series | None:
    kwargs = {"errors": "coerce"}
    if mixed_formats:
        kwargs["format"] = "mixed"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # format-inference chatter; report covers it
        try:
            return pd.to_datetime(s, **kwargs)
        except (TypeError, ValueError, OverflowError):
            return None  # non-scalar cells can raise even with errors="coerce"


def _try_datetime(
    s: pd.Series, nonnull: pd.Series, kind: str, config: CleanConfig
) -> tuple[pd.Series | None, int]:
    """Parse as datetime. Returns ``(converted, n_coerced)`` or ``(None, 0)``."""
    threshold = config.datetime_threshold
    n = len(nonnull)

    if kind in ("date", "datetime", "datetime64"):
        # Column already holds date/datetime objects — just normalize the dtype.
        parsed = _parse_datetime(s, mixed_formats=False)
    else:
        sample = sample_series(nonnull, config.sample_size, config.random_state)
        if not _looks_dateish(sample):
            return None, 0
        parsed = None
        plain_sample = _parse_datetime(sample, mixed_formats=False)
        if plain_sample is not None and plain_sample.notna().mean() >= threshold * 0.8:
            parsed = _parse_datetime(s, mixed_formats=False)
        if parsed is None and PANDAS_MAJOR >= 2:
            mixed_sample = _parse_datetime(sample, mixed_formats=True)
            if mixed_sample is not None and mixed_sample.notna().mean() >= threshold * 0.8:
                parsed = _parse_datetime(s, mixed_formats=True)
        if parsed is None:
            return None, 0
        if parsed.notna().sum() / n < threshold and PANDAS_MAJOR >= 2:
            # Single-format inference came up short; mixed-format parsing is
            # slower but handles heterogeneous date styles in one column.
            mixed = _parse_datetime(s, mixed_formats=True)
            if mixed is not None and mixed.notna().sum() > parsed.notna().sum():
                parsed = mixed

    if parsed is None or parsed.notna().sum() / n < threshold:
        return None, 0
    n_coerced = int((s.notna() & parsed.isna()).sum())
    return parsed, n_coerced


def suggest_conversion(
    s: pd.Series, config: CleanConfig
) -> tuple[str, pd.Series | None, int]:
    """Infer the best target type for one text-capable column.

    Returns ``(target, converted, n_coerced)`` where *target* is one of
    ``"boolean"``, ``"numeric"``, ``"datetime"``, ``"none"``. Shared by both
    the cleaning pipeline and :func:`freshdata.profile` so the preview always
    matches what cleaning would actually do.
    """
    nonnull = s.dropna()
    if nonnull.empty:
        return "none", None, 0
    kind = infer_dtype(s, skipna=True)
    if kind not in ("string", "mixed", "mixed-integer", "boolean", "date", "datetime",
                    "datetime64", "mixed-integer-float"):
        return "none", None, 0

    if kind in ("string", "boolean", "mixed"):
        converted = _try_boolean(s, nonnull)
        if converted is not None:
            return "boolean", converted, 0

    if kind not in ("date", "datetime", "datetime64"):
        converted, n_coerced = _try_numeric(s, nonnull, config)
        if converted is not None:
            return "numeric", converted, n_coerced

    converted, n_coerced = _try_datetime(s, nonnull, kind, config)
    if converted is not None:
        return "datetime", converted, n_coerced

    return "none", None, 0


def fix_dtypes(df: pd.DataFrame, config: CleanConfig, report: CleanReport) -> pd.DataFrame:
    """Apply :func:`suggest_conversion` to every object/string column."""
    for col in stringlike_columns(df):
        target, converted, n_coerced = suggest_conversion(df[col], config)
        if converted is None:
            continue
        description = f"converted to {converted.dtype}"
        if n_coerced:
            description += f" ({n_coerced} unparseable value(s) set to missing)"
        report.add("fix_dtypes", description, column=str(col),
                   count=int(converted.notna().sum()) + n_coerced)
        df[col] = converted
    return df
