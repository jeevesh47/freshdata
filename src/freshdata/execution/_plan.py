"""Decide, without touching data, which stages a backend can run natively.

The native backends reproduce freshdata's deterministic "representation repair"
subset plus simple imputation/outlier handling. Anything that needs the
accuracy-first decision engine or heuristic dtype inference is delegated to the
pandas pipeline. :class:`PlanGenerator` makes that split from the
:class:`~freshdata.CleanConfig` alone, so it is pure and cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..steps.columns import normalized_column_labels

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import CleanConfig

#: Ordered native stages, matching ``cleaner.run_pipeline`` execution order.
#:
#: The native subset is intentionally the *deterministic* part of the pipeline
#: (representation repair + structural reduction + full-row dedup), which the
#: Polars/DuckDB backends reproduce bit-for-bit. The accuracy-first decision
#: engine and the statistically nuanced opt-in steps (``impute``/``outliers`` —
#: which depend on quantile interpolation, mode tie-breaking, and skew) are
#: evaluated by the pandas backend so results stay byte-identical to ``fd.clean``.
NATIVE_STAGE_ORDER = (
    "column_names",
    "clean_strings",
    "drop_empty_columns",
    "drop_empty_rows",
    "drop_duplicates",
    "reset_index",
)

#: Duplicate keep-policies the native full-row dedup can express exactly.
_NATIVE_DUPLICATE_KEEP = ("first", "last")


@dataclass
class NativePlan:
    """The result of planning a clean for a native backend."""

    rename_map: dict[str, str]
    stages: list[str]
    fallback_reason: str | None = None
    extra_sentinels: tuple[str, ...] = field(default_factory=tuple)

    @property
    def needs_fallback(self) -> bool:
        return self.fallback_reason is not None


class PlanGenerator:
    """Build a :class:`NativePlan` for a config (no data access)."""

    def __init__(self, config: CleanConfig) -> None:
        self.config = config

    def fallback_reason(self) -> str | None:
        """Return why this config needs the pandas fallback, or ``None``."""
        c = self.config
        if c.engine_mode is not None:
            return (
                f"strategy={c.strategy!r} runs the accuracy-first decision engine, "
                "which is evaluated by the pandas backend"
            )
        if c.fix_dtypes:
            return "fix_dtypes uses sampled heuristics evaluated by the pandas backend"
        if c.drop_constant_columns:
            return "drop_constant_columns is evaluated by the pandas backend"
        if c.optimize_memory:
            return "optimize_memory downcasting is evaluated by the pandas backend"
        if c.impute is not None:
            return f"impute={c.impute!r} is evaluated by the pandas backend"
        if c.outliers is not None:
            return f"outliers={c.outliers!r} is evaluated by the pandas backend"
        if c.duplicate_subset is not None:
            return "drop_duplicates with a subset is evaluated by the pandas backend"
        if c.duplicate_keep not in _NATIVE_DUPLICATE_KEEP:
            return (
                f"duplicate_keep={c.duplicate_keep!r} is evaluated by the pandas backend"
            )
        return None

    def _enabled_stages(self) -> list[str]:
        c = self.config
        enabled = {
            "column_names": c.column_names,
            "clean_strings": c.strip_whitespace or c.normalize_sentinels,
            "drop_empty_columns": c.drop_empty_columns,
            "drop_empty_rows": c.drop_empty_rows,
            "drop_duplicates": c.drop_duplicates,
            "impute": c.impute is not None,
            "outliers": c.outliers is not None,
            "reset_index": c.reset_index,
        }
        return [s for s in NATIVE_STAGE_ORDER if enabled.get(s, False)]

    def plan(self, columns: list[object]) -> NativePlan:
        """Plan a clean for a frame with the given *columns*."""
        renamed = normalized_column_labels(columns) if self.config.column_names else list(columns)
        rename_map = {
            str(old): str(new)
            for old, new in zip(columns, renamed)
            if isinstance(old, str) and old != new
        }
        return NativePlan(
            rename_map=rename_map,
            stages=self._enabled_stages(),
            fallback_reason=self.fallback_reason(),
            extra_sentinels=tuple(self.config.extra_sentinels),
        )
