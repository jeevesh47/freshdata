"""Top-level convenience functions: ``fd.clean(df)`` and ``fd.profile(df)``."""

from __future__ import annotations

import pandas as pd

from .adapters.polars import from_pandas, to_pandas
from .cleaner import Cleaner
from .config import CleanConfig, merge_options
from .engine.context import build_contexts
from .engine.model_select import EngineMode, rank_missing_models
from .plan import suggest_plan
from .profile import Profile, build_profile
from .report import CleanReport


def clean(
    df: pd.DataFrame,
    *,
    config: CleanConfig | None = None,
    return_report: bool = False,
    **options: object,
) -> pd.DataFrame | tuple[pd.DataFrame, CleanReport]:
    """Clean a DataFrame and return a new, repaired one.

    Two layers run in order. **Representation repair** always happens first:

    1.  ``column_names`` — snake_case column names, deduplicate collisions.
    2.  ``strip_whitespace`` — trim surrounding whitespace in text cells.
    3.  ``normalize_sentinels`` — turn "N/A", "null", "-", "" … into missing.
    4.  ``drop_empty_columns`` / ``drop_empty_rows`` — remove all-missing ones.
    5.  ``fix_dtypes`` — text that is really numeric / datetime / boolean gets
        the right dtype (validated; ``numeric_threshold`` of values must parse).
    6.  ``drop_duplicates`` — resolve duplicate rows (``duplicate_keep``
        chooses first/last/drop/aggregate; time-indexed frames are protected).

    Then, with ``strategy="auto"`` (the default), the **decision engine**
    profiles every column — missing ratio, dtype, skewness, cardinality,
    inferred role (id / target / datetime / text / categorical), whether
    missingness looks informative — and applies threshold rules for missing
    values and outliers. Nothing is done silently: every action (including
    deliberately preserving a column) is logged with a rationale, a risk
    level, and a confidence score. ``strategy="conservative"`` disables the
    engine; imputation and outlier handling are then opt-in via ``impute=`` /
    ``outliers=``.

    Parameters
    ----------
    df:
        The DataFrame to clean.
    config:
        A prebuilt :class:`~freshdata.CleanConfig` to start from.
    return_report:
        If True, return ``(cleaned_df, CleanReport)``. The report carries
        per-action rationale/risk/confidence, missing counts before/after,
        warnings, and recommendations for manual review.
    **options:
        Any :class:`~freshdata.CleanConfig` field as a keyword override — e.g.
        ``strategy`` (``"balanced"`` default / ``"aggressive"`` / ``"conservative"``),
        ``missing_threshold_low``/``_medium``/``_high``, ``duplicate_threshold``,
        ``outlier_method``, ``outlier_action``, ``preserve_original``, ``verbose``,
        ``preserve_columns``, ``target_column``, ``duplicate_keep``, ``impute``,
        ``outliers``. Unknown names raise :class:`TypeError`.

    Examples
    --------
    >>> import freshdata as fd
    >>> cleaned = fd.clean(df)
    >>> cleaned, rep = fd.clean(df, return_report=True)
    >>> print(rep.summary())

    >>> fd.clean(df, outlier_action="flag", target_column="churn",
    ...          preserve_columns=("notes",), verbose=False)
    """
    cleaner = Cleaner(config=config, **options)
    result = cleaner.clean(df, report=return_report)
    if return_report:
        cleaned, rep = result
        return from_pandas(cleaned, df), rep
    return from_pandas(result, df)


def _engine_mode(cfg: CleanConfig) -> EngineMode:
    mode = cfg.engine_mode or "balanced"
    return "balanced" if mode == "balanced" else "aggressive"


def infer_roles(
    df: pd.DataFrame,
    *,
    strategy: str = "balanced",
    config: CleanConfig | None = None,
    **options: object,
) -> pd.DataFrame:
    """Infer column roles and primary missing models without mutating data."""
    cfg = merge_options(config, strategy=strategy, **options)
    frame = to_pandas(df)
    contexts = build_contexts(frame, cfg)
    mode = _engine_mode(cfg)
    rows = []
    for col, ctx in sorted(contexts.items()):
        primary = None
        if ctx.missing_ratio > 0:
            primary = rank_missing_models(frame, col, ctx, cfg, mode=mode).primary
        rows.append({
            "column": col,
            "role": ctx.role,
            "missing_pct": round(ctx.missing_ratio * 100, 2),
            "cardinality": ctx.nunique,
            "skew": ctx.skew,
            "domain_sensitive": ctx.domain_sensitive,
            "primary_missing_model": primary.model_id if primary else None,
        })
    return pd.DataFrame(rows)


def profile(
    df: pd.DataFrame,
    *,
    config: CleanConfig | None = None,
    include_plan: bool = False,
    **options: object,
) -> Profile:
    """Inspect a DataFrame without changing it.

    Returns a :class:`~freshdata.Profile` describing shape, memory, missing
    data, duplicates, and per-column issues — including a faithful preview of
    the dtype conversions :func:`clean` would perform, computed by the same
    inference code.

    With ``include_plan=True``, attaches a :class:`~freshdata.CleanPlan` at
    ``profile.plan`` previewing engine model choices.

    Examples
    --------
    >>> import freshdata as fd
    >>> p = fd.profile(df)
    >>> print(p)             # human-readable issue table
    >>> p.to_frame()         # one row per column, sortable in a notebook
    >>> p.to_dict()          # JSON-friendly
    """
    cfg = merge_options(config, **options)
    prof = build_profile(to_pandas(df), cfg)
    if include_plan:
        object.__setattr__(prof, "plan", suggest_plan(to_pandas(df), config=cfg))
    return prof
