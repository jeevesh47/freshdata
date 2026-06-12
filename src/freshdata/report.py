"""Structured record of everything :func:`freshdata.clean` did.

Trust is the core feature of an auto-cleaner: every transformation is recorded
as an :class:`Action` so users can audit exactly what changed and how much.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ._util import format_bytes


@dataclass(frozen=True)
class Action:
    """One transformation applied to the data.

    Attributes
    ----------
    step:
        Machine-readable step name, e.g. ``"fix_dtypes"``.
    column:
        Column the action applied to, or ``None`` for table-level actions.
    description:
        Human-readable summary of what happened.
    count:
        Number of cells or rows affected (0 for informational notes).
    """

    step: str
    column: str | None
    description: str
    count: int = 0

    def __str__(self) -> str:
        target = f"{self.column!r}: " if self.column is not None else ""
        return f"[{self.step}] {target}{self.description}"


@dataclass
class CleanReport:
    """Everything one :func:`freshdata.clean` run did, in order.

    Iterable and sized: ``len(report)`` is the number of actions, and
    ``for action in report`` walks them in execution order. ``bool(report)``
    is ``True`` iff anything was changed.
    """

    actions: list[Action] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0
    cols_before: int = 0
    cols_after: int = 0
    memory_before: int = 0
    memory_after: int = 0
    duration_seconds: float = 0.0

    def add(self, step: str, description: str, *, column: str | None = None,
            count: int = 0) -> None:
        """Record one action (internal; called by the pipeline)."""
        self.actions.append(Action(step=step, column=column, description=description,
                                   count=int(count)))

    # -- introspection ------------------------------------------------------

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self) -> Iterator[Action]:
        return iter(self.actions)

    def __bool__(self) -> bool:
        return any(a.count for a in self.actions)

    @property
    def cells_changed(self) -> int:
        """Total affected cells/rows summed across all actions."""
        return sum(a.count for a in self.actions)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form, suitable for JSON serialization or logging."""
        return {
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "cols_before": self.cols_before,
            "cols_after": self.cols_after,
            "memory_before": self.memory_before,
            "memory_after": self.memory_after,
            "duration_seconds": self.duration_seconds,
            "actions": [
                {"step": a.step, "column": a.column, "description": a.description,
                 "count": a.count}
                for a in self.actions
            ],
        }

    def to_frame(self) -> pd.DataFrame:
        """One row per action, as a DataFrame — convenient for notebooks."""
        return pd.DataFrame(
            [(a.step, a.column, a.description, a.count) for a in self.actions],
            columns=["step", "column", "description", "count"],
        )

    def summary(self) -> str:
        """Multi-line human-readable summary."""
        d_rows = self.rows_after - self.rows_before
        d_cols = self.cols_after - self.cols_before
        lines = [
            "freshdata clean report",
            f"  rows:    {self.rows_before:,} -> {self.rows_after:,} ({d_rows:+,})",
            f"  columns: {self.cols_before:,} -> {self.cols_after:,} ({d_cols:+,})",
            f"  memory:  {format_bytes(self.memory_before)} -> "
            f"{format_bytes(self.memory_after)}",
            f"  time:    {self.duration_seconds:.3f}s",
        ]
        if self.actions:
            lines.append(f"  actions ({len(self.actions)}):")
            lines.extend(f"    - {a}" for a in self.actions)
        else:
            lines.append("  actions: none — data was already clean")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return (
            f"<CleanReport: {len(self.actions)} actions, "
            f"rows {self.rows_before:,}->{self.rows_after:,}, "
            f"cols {self.cols_before}->{self.cols_after}>"
        )
