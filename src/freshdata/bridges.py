"""Adapter layer for validation tools FreshData should complement, not replace.

These bridges normalize external validation failures into a small FreshData
failure model. The adapters deliberately do not evaluate expectations
themselves. They consume the output of tools such as Great Expectations, dbt,
and Pandera, then turn those findings into deterministic review tasks that can
feed repair planning.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .review import ReviewDataset, ReviewQueue


@dataclass(frozen=True)
class ValidationFailure:
    """One normalized validator failure from GX, dbt, Pandera, or similar."""

    failure_id: str
    validator: str
    check: str
    column: str | None = None
    row: Any | None = None
    failure_case: Any = None
    severity: str = "medium"
    reason: str = ""
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "failure_id": self.failure_id,
            "validator": self.validator,
            "check": self.check,
            "column": self.column,
            "row": _json_value(self.row),
            "failure_case": _json_value(self.failure_case),
            "severity": self.severity,
            "reason": self.reason,
            "raw": self.raw or {},
        }


@dataclass(frozen=True)
class ValidationBridgeResult:
    """Normalized failures plus helpers for pipeline review handoff."""

    source: str
    failures: tuple[ValidationFailure, ...]

    def to_frame(self) -> pd.DataFrame:
        """Return failures as a DataFrame."""
        return pd.DataFrame([failure.to_dict() for failure in self.failures])

    def to_review_dataset(self) -> ReviewDataset:
        """Convert validation failures into explicit human review tasks."""
        queue = ReviewQueue(dataset_id=f"{self.source}-validation-review")
        for failure in self.failures:
            target = failure.column or "row"
            queue.add_candidate(
                review_id=failure.failure_id,
                source=self.source,
                patch_id=failure.failure_id,
                row=failure.row,
                column=failure.column,
                candidate_change=f"Repair {target} failing {failure.check}",
                required_decision="Approve repair, reject repair, or request context.",
                confidence=0.65,
                reason=failure.reason or f"{self.source} reported {failure.check}",
                risk=failure.severity,
            )
        return queue.to_dataset()


def from_gx(validation_result: Any) -> ValidationBridgeResult:
    """Consume a Great Expectations validation result.

    The adapter accepts the common dict form returned by GX as well as objects
    exposing ``results``/``success`` attributes. It only imports no GX symbols,
    keeping FreshData embedded safely in projects that do not depend on GX.
    """
    failures: list[ValidationFailure] = []
    for result in _get(validation_result, "results", []) or []:
        if bool(_get(result, "success", False)):
            continue
        config = _get(result, "expectation_config", {}) or {}
        result_payload = _get(result, "result", {}) or {}
        expectation_type = str(_get(config, "expectation_type", "expectation"))
        kwargs = _get(config, "kwargs", {}) or {}
        column = _first_present(kwargs, ("column", "column_A", "column_B"))
        unexpected_indexes = _get(result_payload, "unexpected_index_list", None)
        unexpected_values = _get(result_payload, "partial_unexpected_list", None)
        rows = unexpected_indexes if unexpected_indexes else [None]
        for offset, row in enumerate(rows):
            failure_case = None
            if unexpected_values and offset < len(unexpected_values):
                failure_case = unexpected_values[offset]
            failures.append(
                ValidationFailure(
                    failure_id=_failure_id("gx", expectation_type, column, row, offset),
                    validator="great_expectations",
                    check=expectation_type,
                    column=str(column) if column is not None else None,
                    row=row,
                    failure_case=failure_case,
                    severity="medium",
                    reason=f"Great Expectations check {expectation_type!r} failed.",
                    raw=_plain_dict(result),
                )
            )
    return ValidationBridgeResult(source="great_expectations", failures=tuple(failures))


def from_dbt_failures(path_or_table: str | Path | pd.DataFrame) -> ValidationBridgeResult:
    """Consume dbt test failures from a failure table or exported file.

    ``path_or_table`` may be a pandas DataFrame or a CSV, JSON, JSONL, or
    Parquet file containing failed rows. FreshData treats each row as evidence
    for review; it does not rerun the dbt test.
    """
    frame = _read_failure_table(path_or_table)
    failures = []
    for row_number, row in frame.reset_index(drop=False).iterrows():
        unique_id = row.get("unique_id", row.get("index", row_number))
        column = row.get("column_name", row.get("column"))
        check = str(row.get("test_name", row.get("check", "dbt_test_failure")))
        failures.append(
            ValidationFailure(
                failure_id=_failure_id("dbt", check, column, unique_id, row_number),
                validator="dbt",
                check=check,
                column=str(column) if pd.notna(column) else None,
                row=unique_id,
                failure_case=row.to_dict(),
                severity="high",
                reason=f"dbt test {check!r} returned a failed row.",
                raw={str(k): _json_value(v) for k, v in row.to_dict().items()},
            )
        )
    return ValidationBridgeResult(source="dbt", failures=tuple(failures))


def from_pandera_errors(schema_errors: Any) -> ValidationBridgeResult:
    """Consume Pandera ``SchemaErrors`` or its ``failure_cases`` DataFrame."""
    failure_cases = _get(schema_errors, "failure_cases", schema_errors)
    frame = _read_failure_table(failure_cases)
    failures = []
    for row_number, row in frame.reset_index(drop=False).iterrows():
        column = row.get("column")
        check = str(row.get("check", row.get("schema_context", "pandera_check")))
        case = row.get("failure_case", None)
        index = row.get("index", row_number)
        failures.append(
            ValidationFailure(
                failure_id=_failure_id("pandera", check, column, index, row_number),
                validator="pandera",
                check=check,
                column=str(column) if pd.notna(column) else None,
                row=index,
                failure_case=case,
                severity="medium",
                reason=f"Pandera schema check {check!r} failed.",
                raw={str(k): _json_value(v) for k, v in row.to_dict().items()},
            )
        )
    return ValidationBridgeResult(source="pandera", failures=tuple(failures))


def emit_gx_expectations(
    columns: dict[str, str] | None = None,
    *,
    suite_name: str = "freshdata_contract",
    non_null_columns: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Emit a minimal Great Expectations expectation suite.

    ``columns`` maps column names to logical FreshData types. The emitted suite
    is intentionally conservative: column existence and optional non-null checks
    only. Type coercion and repair remain FreshData responsibilities.
    """
    expectations: list[dict[str, Any]] = []
    for column in sorted(columns or {}):
        expectations.append({
            "expectation_type": "expect_column_to_exist",
            "kwargs": {"column": column},
        })
    for column in non_null_columns:
        expectations.append({
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {"column": column},
        })
    return {
        "expectation_suite_name": suite_name,
        "expectations": expectations,
        "meta": {"created_by": "freshdata"},
    }


def emit_dbt_tests(
    columns: dict[str, str] | None = None,
    *,
    model_name: str = "freshdata_model",
    non_null_columns: tuple[str, ...] = (),
    unique_columns: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Emit dbt schema-test YAML as a dependency-free Python dictionary."""
    column_entries = []
    for column, logical_type in sorted((columns or {}).items()):
        tests: list[Any] = []
        if column in non_null_columns:
            tests.append("not_null")
        if column in unique_columns:
            tests.append("unique")
        column_entries.append({
            "name": column,
            "description": f"FreshData logical type: {logical_type}",
            "tests": tests,
        })
    return {"version": 2, "models": [{"name": model_name, "columns": column_entries}]}


def _read_failure_table(path_or_table: str | Path | pd.DataFrame | Any) -> pd.DataFrame:
    if isinstance(path_or_table, pd.DataFrame):
        return path_or_table.copy(deep=True)
    if isinstance(path_or_table, list):
        return pd.DataFrame(path_or_table)
    if isinstance(path_or_table, dict):
        return pd.DataFrame([path_or_table])
    path = Path(path_or_table)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("dbt/Pandera failures must be a DataFrame, records, or csv/json/parquet")


def _failure_id(*parts: Any) -> str:
    payload = json.dumps([_json_value(part) for part in parts], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _plain_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return {str(k): _json_value(v) for k, v in obj.items()}
    if hasattr(obj, "to_json_dict"):
        return _plain_dict(obj.to_json_dict())
    if hasattr(obj, "__dict__"):
        return {str(k): _json_value(v) for k, v in vars(obj).items()}
    return {"value": _json_value(obj)}


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
