"""Polars backend — native out-of-core cleaning on LazyFrames.

Reproduces freshdata's deterministic representation-repair + structural-reduction
subset (column rename, whitespace/sentinel normalization, empty column/row drops,
full-row dedup) directly on a ``pl.LazyFrame`` with projection/predicate pushdown
and streaming collection. Steps outside that subset (the decision engine,
heuristic dtype repair, opt-in impute/outliers) transparently fall back to the
pandas pipeline, so output stays identical to ``fd.clean``.

Action records use the *same* ``step`` names and ``count`` semantics the pandas
``steps/`` modules emit, so ``CleanReport`` consumers are unaffected.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from .._base import ExecutionEngine
from .._lazy import has_polars, require_polars
from .._metadata import MetadataScanner
from .._plan import NativePlan, PlanGenerator
from .._report import finalize_report, init_report
from ._pandas import materialize_to_pandas

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...config import CleanConfig
    from ...report import CleanReport
    from .._config import EngineConfig

log = logging.getLogger("freshdata.execution.polars")


class PolarsEngine(ExecutionEngine):
    name = "polars"

    def supports_source(self, source: Any) -> bool:
        if isinstance(source, str):
            return True
        import pandas as pd

        if isinstance(source, pd.DataFrame):
            return True
        if has_polars():
            import polars as pl

            return isinstance(source, (pl.DataFrame, pl.LazyFrame))
        return False

    # -- source ingestion ---------------------------------------------------

    def _to_lazy(self, source: Any, pl: Any) -> tuple[Any, int]:
        """Return ``(LazyFrame, memory_before_bytes)`` for *source*."""
        if isinstance(source, pl.LazyFrame):
            return source, 0
        if isinstance(source, pl.DataFrame):
            return source.lazy(), int(source.estimated_size())
        if isinstance(source, str):
            low = source.lower()
            if low.endswith((".parquet", ".pq")):
                return pl.scan_parquet(source), 0
            if low.endswith(".csv"):
                return pl.scan_csv(source), 0
            if low.endswith((".ipc", ".feather", ".arrow")):
                return pl.scan_ipc(source), 0
            raise ValueError(f"PolarsEngine: unsupported file type for path {source!r}")
        import pandas as pd

        if isinstance(source, pd.DataFrame):
            return pl.from_pandas(source).lazy(), int(source.memory_usage(deep=True).sum())
        raise TypeError(f"PolarsEngine: unsupported source type {type(source).__name__}")

    def _pandas_index_forces_fallback(self, source: Any) -> bool:
        """A non-default pandas index (e.g. DatetimeIndex) has index-aware
        dedup semantics that a polars frame cannot carry — fall back."""
        import pandas as pd

        return isinstance(source, pd.DataFrame) and not isinstance(source.index, pd.RangeIndex)

    # -- execution ----------------------------------------------------------

    def execute(
        self,
        source: Any,
        config: CleanConfig,
        engine_config: EngineConfig,
    ) -> tuple[Any, CleanReport]:
        pl = require_polars()
        self._configure_threads(engine_config)
        started = time.perf_counter()

        lf, memory_before = self._to_lazy(source, pl)
        names = list(lf.collect_schema().names())
        plan = PlanGenerator(config).plan(names)

        if plan.needs_fallback or self._pandas_index_forces_fallback(source):
            reason = plan.fallback_reason or "pandas index semantics"
            log.warning("freshdata PolarsEngine: falling back to pandas (%s)", reason)
            return self._fallback(source, config)

        meta = MetadataScanner.from_polars_lazy(lf)
        report = init_report(meta, memory_before)
        lf = self._apply_native(lf, plan, config, report, pl)
        cleaned = self._collect(lf, engine_config, pl)
        finalize_report(report, cleaned, started)
        return cleaned, report

    def _fallback(self, source: Any, config: CleanConfig) -> tuple[Any, CleanReport]:
        from ...cleaner import run_pipeline

        df = materialize_to_pandas(source)
        return run_pipeline(df, config)

    # -- native stages (mirror cleaner.run_pipeline order) ------------------

    def _apply_native(
        self, lf: Any, plan: NativePlan, config: CleanConfig, report: CleanReport, pl: Any
    ) -> Any:
        rows_before = report.rows_before
        for stage in plan.stages:
            if stage == "column_names":
                lf = self._stage_rename(lf, plan, report)
            elif stage == "clean_strings":
                lf = self._stage_clean_strings(lf, config, report, pl)
            elif stage == "drop_empty_columns":
                if rows_before > 0:
                    lf = self._stage_drop_empty_columns(lf, report, pl)
            elif stage == "drop_empty_rows":
                if rows_before > 0:
                    lf = self._stage_drop_empty_rows(lf, report, pl)
            elif stage == "drop_duplicates":
                lf = self._stage_drop_duplicates(lf, config, report, pl)
            elif stage == "reset_index":
                pass  # polars frames carry no index
        return lf

    def _stage_rename(self, lf: Any, plan: NativePlan, report: CleanReport) -> Any:
        if not plan.rename_map:
            return lf
        changes = list(plan.rename_map.items())
        preview = ", ".join(f"{o!r}->{n!r}" for o, n in changes[:4])
        if len(changes) > 4:
            preview += f", … (+{len(changes) - 4} more)"
        report.add("column_names", f"renamed {len(changes)} column(s): {preview}",
                   count=len(changes))
        return lf.rename(plan.rename_map)

    def _stage_clean_strings(
        self, lf: Any, config: CleanConfig, report: CleanReport, pl: Any
    ) -> Any:
        from ...steps.strings import active_sentinels

        sentinels = list(active_sentinels(config))
        schema = lf.collect_schema()
        string_cols = [n for n in schema.names() if schema[n] == pl.Utf8]
        if not string_cols:
            return lf

        # One aggregate pass for all strip / sentinel counts.
        count_exprs: list[Any] = []
        for c in string_cols:
            col = pl.col(c)
            stripped = col.str.strip_chars()
            base = stripped if config.strip_whitespace else col
            if config.strip_whitespace:
                count_exprs.append(
                    ((stripped != col) & col.is_not_null()).sum().alias(f"__strip__{c}")
                )
            if config.normalize_sentinels:
                count_exprs.append(
                    (base.str.to_lowercase().is_in(sentinels) & base.is_not_null())
                    .sum()
                    .alias(f"__sent__{c}")
                )
        counts = lf.select(count_exprs).collect().row(0, named=True) if count_exprs else {}

        transforms: list[Any] = []
        for c in string_cols:
            col = pl.col(c)
            stripped = col.str.strip_chars()
            base = stripped if config.strip_whitespace else col
            n_strip = int(counts.get(f"__strip__{c}", 0) or 0)
            n_sent = int(counts.get(f"__sent__{c}", 0) or 0)
            if config.strip_whitespace and n_strip:
                report.add("strip_whitespace", "trimmed surrounding whitespace",
                           column=c, count=n_strip)
            if config.normalize_sentinels and n_sent:
                report.add("normalize_sentinels",
                           'replaced sentinel strings ("N/A", "-", "", …) with missing',
                           column=c, count=n_sent)
            if config.normalize_sentinels:
                expr = (
                    pl.when(base.str.to_lowercase().is_in(sentinels))
                    .then(None)
                    .otherwise(base)
                    .alias(c)
                )
            else:
                expr = base.alias(c)
            transforms.append(expr)
        return lf.with_columns(transforms)

    def _stage_drop_empty_columns(self, lf: Any, report: CleanReport, pl: Any) -> Any:
        schema = lf.collect_schema()
        names = list(schema.names())
        stats = lf.select(
            [pl.len().alias("__h__")] + [pl.col(c).null_count().alias(c) for c in names]
        ).collect().row(0, named=True)
        height = int(stats["__h__"])
        dropped = [c for c in names if int(stats[c]) == height]
        if dropped:
            report.columns_dropped.extend(dropped)
            report.add(
                "drop_empty_columns",
                f"dropped {len(dropped)} all-missing column(s): {', '.join(dropped[:6])}"
                + (" …" if len(dropped) > 6 else ""),
                count=len(dropped),
            )
            lf = lf.drop(dropped)
        return lf

    def _stage_drop_empty_rows(self, lf: Any, report: CleanReport, pl: Any) -> Any:
        if not lf.collect_schema().names():
            return lf
        all_null = pl.all_horizontal(pl.all().is_null())
        n = int(lf.select(all_null.sum().alias("__n__")).collect().item())
        if n:
            report.add("drop_empty_rows", f"dropped {n} all-missing row(s)", count=n)
            lf = lf.filter(~all_null)
        return lf

    def _stage_drop_duplicates(
        self, lf: Any, config: CleanConfig, report: CleanReport, pl: Any
    ) -> Any:
        n_before = int(lf.select(pl.len()).collect().item())
        if n_before < 1:
            return lf
        deduped = lf.unique(keep=config.duplicate_keep, maintain_order=True)
        n_after = int(deduped.select(pl.len()).collect().item())
        n_dup = n_before - n_after
        if n_dup <= 0:
            return lf
        pct = 100.0 * n_dup / n_before
        risk = "medium" if (n_dup / n_before) > config.duplicate_threshold else "low"
        report.add(
            "drop_duplicates",
            f"dropped {n_dup} duplicate row(s) "
            f"({pct:.1f}% of rows, keep={config.duplicate_keep!r})",
            count=n_dup,
            risk=risk,
        )
        report.duplicates_removed += n_dup
        if risk == "medium":
            report.add_warning(
                f"{pct:.1f}% of rows were duplicates "
                f"(> {100 * config.duplicate_threshold:.0f}%); confirm they are not legitimate"
            )
        return deduped

    # -- collection ---------------------------------------------------------

    def _collect(self, lf: Any, engine_config: EngineConfig, pl: Any) -> Any:
        if not engine_config.streaming:
            return lf.collect()
        # Polars renamed the streaming switch across versions; try the modern
        # keyword first, then the legacy one, then a plain collect.
        try:
            return lf.collect(engine="streaming")
        except TypeError:
            pass
        try:
            return lf.collect(streaming=True)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("freshdata: streaming collect failed (%s); retrying eagerly", exc)
            return lf.collect()

    def _configure_threads(self, engine_config: EngineConfig) -> None:
        if engine_config.polars_n_threads is not None:
            import os

            # Only effective if set before polars is first imported; harmless otherwise.
            os.environ.setdefault("POLARS_MAX_THREADS", str(engine_config.polars_n_threads))
