"""Human review queue primitives for auditable repair workflows.

The review queue is intentionally modeled as data, not as a UI concern. Every
candidate change carries the required decision, confidence, reason, risk, and
approval fields needed by strict operating environments. The resulting
DataFrame can be handed to a human workflow tool, committed as an artifact, or
exported for a downstream approval system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

ReviewExportFormat = Literal["csv", "json", "parquet"]
ReviewStatus = Literal["pending", "approved", "rejected", "needs_more_context"]


@dataclass(frozen=True)
class ReviewOption:
    """One explicit choice a reviewer can make for a candidate change."""

    option_id: str
    label: str
    description: str = ""
    applies_patch: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "option_id": self.option_id,
            "label": self.label,
            "description": self.description,
            "applies_patch": self.applies_patch,
        }


@dataclass(frozen=True)
class ReviewTask:
    """One candidate change that requires an explicit human decision."""

    review_id: str
    candidate_change: str
    required_decision: str
    confidence: float
    reason: str
    risk: str = "medium"
    suggested_options: tuple[ReviewOption, ...] = field(default_factory=tuple)
    source: str = ""
    patch_id: str = ""
    row: Any | None = None
    column: str | None = None
    approval_status: ReviewStatus = "pending"
    approved: bool | None = None
    rejected: bool | None = None
    reviewer: str = ""
    reviewed_at: str = ""
    reviewer_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the stable, exportable review queue schema."""
        options = [option.to_dict() for option in self.suggested_options]
        return {
            "review_id": self.review_id,
            "source": self.source,
            "patch_id": self.patch_id,
            "row": _json_value(self.row),
            "column": self.column,
            "candidate_change": self.candidate_change,
            "required_decision": self.required_decision,
            "confidence": float(self.confidence),
            "reason": self.reason,
            "risk": self.risk,
            "suggested_options": json.dumps(options, sort_keys=True),
            "approval_status": self.approval_status,
            "approved": self.approved,
            "rejected": self.rejected,
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
            "reviewer_notes": self.reviewer_notes,
        }


@dataclass
class ReviewDataset:
    """A collection of review tasks with deterministic export helpers."""

    tasks: tuple[ReviewTask, ...] = ()
    dataset_id: str = "freshdata-review"

    def to_frame(self) -> pd.DataFrame:
        """Return the review dataset as a pandas DataFrame."""
        columns = [
            "review_id",
            "source",
            "patch_id",
            "row",
            "column",
            "candidate_change",
            "required_decision",
            "confidence",
            "reason",
            "risk",
            "suggested_options",
            "approval_status",
            "approved",
            "rejected",
            "reviewer",
            "reviewed_at",
            "reviewer_notes",
        ]
        return pd.DataFrame([task.to_dict() for task in self.tasks], columns=columns)

    def export(self, path: str | Path, *, format: ReviewExportFormat | None = None) -> Path:
        """Export the review dataset as CSV, JSON, or Parquet.

        The format is inferred from the file suffix when not supplied. Parquet
        export delegates to pandas, so projects can choose their preferred
        engine through the normal pandas optional dependencies.
        """
        out = Path(path)
        fmt = format or out.suffix.lstrip(".").lower()
        frame = self.to_frame()
        if fmt == "csv":
            frame.to_csv(out, index=False)
        elif fmt == "json":
            frame.to_json(out, orient="records", indent=2)
        elif fmt == "parquet":
            frame.to_parquet(out, index=False)
        else:
            raise ValueError("format must be one of: csv, json, parquet")
        return out


class ReviewQueue:
    """Mutable builder for review datasets used by bridge and repair modules."""

    def __init__(self, *, dataset_id: str = "freshdata-review") -> None:
        self.dataset_id = dataset_id
        self._tasks: list[ReviewTask] = []

    def add(self, task: ReviewTask) -> ReviewTask:
        """Append *task* and return it for fluent construction."""
        self._tasks.append(task)
        return task

    def add_candidate(
        self,
        *,
        review_id: str,
        candidate_change: str,
        required_decision: str,
        confidence: float,
        reason: str,
        risk: str = "medium",
        source: str = "",
        patch_id: str = "",
        row: Any | None = None,
        column: str | None = None,
        suggested_options: tuple[ReviewOption, ...] | None = None,
    ) -> ReviewTask:
        """Create and append a review task in one call."""
        task = ReviewTask(
            review_id=review_id,
            candidate_change=candidate_change,
            required_decision=required_decision,
            confidence=confidence,
            reason=reason,
            risk=risk,
            suggested_options=suggested_options or default_approval_options(),
            source=source,
            patch_id=patch_id,
            row=row,
            column=column,
        )
        return self.add(task)

    def to_dataset(self) -> ReviewDataset:
        """Freeze the queued tasks into an exportable dataset."""
        return ReviewDataset(tasks=tuple(self._tasks), dataset_id=self.dataset_id)

    def to_frame(self) -> pd.DataFrame:
        """Return the queued tasks as a DataFrame."""
        return self.to_dataset().to_frame()

    def export(self, path: str | Path, *, format: ReviewExportFormat | None = None) -> Path:
        """Export the queued tasks as CSV, JSON, or Parquet."""
        return self.to_dataset().export(path, format=format)


def default_approval_options() -> tuple[ReviewOption, ReviewOption, ReviewOption]:
    """Default decisions for strict approval workflows."""
    return (
        ReviewOption("approve", "Approve", "Apply the candidate change.", True),
        ReviewOption("reject", "Reject", "Do not apply the candidate change.", False),
        ReviewOption(
            "needs_more_context",
            "Needs more context",
            "Route to a data steward or upstream owner.",
            False,
        ),
    )


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(v) for v in value]
    return str(value)
