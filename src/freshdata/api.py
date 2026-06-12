"""Top-level convenience functions: ``fd.clean(df)`` and ``fd.profile(df)``."""

from __future__ import annotations

import pandas as pd

from .cleaner import Cleaner
from .config import CleanConfig, merge_options
from .profile import Profile, build_profile
from .report import CleanReport


def clean(
    df: pd.DataFrame,
    *,
    report: bool = False,
    config: CleanConfig | None = None,
    **options: object,
) -> pd.DataFrame | tuple[pd.DataFrame, CleanReport]:
    """Clean a DataFrame and return a new, repaired one.

    The input is never mutated. By default only *representation* problems are
    fixed; anything that would change the statistics of your data (imputation,
    outlier handling, lossy downcasting) is opt-in.

    Default steps, in order:

    1.  ``column_names`` — snake_case column names, deduplicate collisions.
    2.  ``strip_whitespace`` — trim surrounding whitespace in text cells.
    3.  ``normalize_sentinels`` — turn "N/A", "null", "-", "" … into missing.
    4.  ``drop_empty_columns`` / ``drop_empty_rows`` — remove all-missing ones.
    5.  ``fix_dtypes`` — text that is really numeric / datetime / boolean gets
        the right dtype (validated; ``numeric_threshold`` of values must parse).
    6.  ``drop_duplicates`` — drop exact duplicate rows (keep first).

    Opt-in steps: ``drop_constant_columns``, ``impute`` ("auto", "mean",
    "median", "mode"), ``outliers`` ("clip" or "flag", method "iqr"/"zscore"),
    ``optimize_memory`` (downcast numerics, categorize low-cardinality text),
    ``reset_index``. See :class:`freshdata.CleanConfig` for every option and
    its default.

    Parameters
    ----------
    df:
        The DataFrame to clean.
    report:
        If True, return ``(cleaned_df, CleanReport)`` — the report lists every
        action taken with affected counts.
    config:
        A prebuilt :class:`~freshdata.CleanConfig` to start from.
    **options:
        Any :class:`~freshdata.CleanConfig` field as a keyword override.
        Unknown names raise :class:`TypeError` immediately.

    Examples
    --------
    >>> import freshdata as fd
    >>> cleaned = fd.clean(df)
    >>> cleaned, rep = fd.clean(df, report=True)
    >>> print(rep.summary())

    >>> fd.clean(df, impute="median", outliers="clip", optimize_memory=True)
    """
    return Cleaner(config=config, **options).clean(df, report=report)


def profile(
    df: pd.DataFrame,
    *,
    config: CleanConfig | None = None,
    **options: object,
) -> Profile:
    """Inspect a DataFrame without changing it.

    Returns a :class:`~freshdata.Profile` describing shape, memory, missing
    data, duplicates, and per-column issues — including a faithful preview of
    the dtype conversions :func:`clean` would perform, computed by the same
    inference code.

    Examples
    --------
    >>> import freshdata as fd
    >>> p = fd.profile(df)
    >>> print(p)             # human-readable issue table
    >>> p.to_frame()         # one row per column, sortable in a notebook
    >>> p.to_dict()          # JSON-friendly
    """
    return build_profile(df, merge_options(config, **options))
