"""Schema drift repair and canonical mapping.

The harmonizer is a conservative execution layer for upstream schema changes.
It detects likely renames, maps compatible type changes, quarantines rows that
cannot be safely coerced, and records a migration diff so pipeline owners can
review drift before promoting a new schema contract.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

SchemaAction = Literal["exact", "alias", "renamed", "missing", "extra", "incompatible"]


@dataclass(frozen=True)
class ColumnContract:
    """Canonical definition for one column in a schema contract."""

    name: str
    dtype: str
    nullable: bool = True
    required: bool = True
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class SchemaContract:
    """Versioned canonical schema contract."""

    name: str
    version: str
    columns: tuple[ColumnContract, ...]
    description: str = ""

    @property
    def column_map(self) -> dict[str, ColumnContract]:
        """Canonical columns keyed by name."""
        return {column.name: column for column in self.columns}


@dataclass(frozen=True)
class SchemaColumnMapping:
    """How one source column maps to the canonical contract."""

    source_column: str | None
    canonical_column: str
    action: SchemaAction
    confidence: float
    reason: str
    compatible_type_change: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping row."""
        return {
            "source_column": self.source_column,
            "canonical_column": self.canonical_column,
            "action": self.action,
            "confidence": self.confidence,
            "reason": self.reason,
            "compatible_type_change": self.compatible_type_change,
        }


@dataclass(frozen=True)
class MigrationDiff:
    """Reviewable diff between observed source data and the canonical schema."""

    contract_name: str
    from_version: str | None
    to_version: str
    added_columns: tuple[str, ...] = ()
    removed_columns: tuple[str, ...] = ()
    renamed_columns: dict[str, str] = field(default_factory=dict)
    type_changes: dict[str, str] = field(default_factory=dict)
    incompatible_columns: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a migration diff dictionary for logging or PR review."""
        return {
            "contract_name": self.contract_name,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "added_columns": list(self.added_columns),
            "removed_columns": list(self.removed_columns),
            "renamed_columns": dict(self.renamed_columns),
            "type_changes": dict(self.type_changes),
            "incompatible_columns": list(self.incompatible_columns),
        }


@dataclass(frozen=True)
class QuarantineResult:
    """Accepted rows and incompatible rows with deterministic reasons."""

    accepted: pd.DataFrame
    quarantined: pd.DataFrame
    reasons: dict[Any, str]


@dataclass(frozen=True)
class SchemaHarmonizationResult:
    """End-to-end schema harmonization output."""

    canonical_frame: pd.DataFrame
    quarantine: QuarantineResult
    mapping: tuple[SchemaColumnMapping, ...]
    migration_diff: MigrationDiff

    def mapping_frame(self) -> pd.DataFrame:
        """Return the source-to-canonical mapping as a DataFrame."""
        return pd.DataFrame([mapping.to_dict() for mapping in self.mapping])


class SchemaHarmonizer:
    """Map messy upstream schemas into a versioned canonical contract."""

    def __init__(
        self,
        contract: SchemaContract,
        *,
        known_contracts: tuple[SchemaContract, ...] = (),
        rename_threshold: float = 0.82,
    ) -> None:
        self.contract = contract
        self.rename_threshold = rename_threshold
        self.contract_history: dict[str, SchemaContract] = {
            contract.version: contract,
            **{known.version: known for known in known_contracts},
        }

    def register_contract(self, contract: SchemaContract) -> None:
        """Register a historical or future contract version."""
        if contract.name != self.contract.name:
            raise ValueError("contract name must match the active harmonizer contract")
        self.contract_history[contract.version] = contract

    def detect_mappings(self, df: pd.DataFrame) -> tuple[SchemaColumnMapping, ...]:
        """Detect exact, alias, and likely rename mappings for *df*."""
        mappings: list[SchemaColumnMapping] = []
        used_sources: set[str] = set()
        source_columns = [str(column) for column in df.columns]
        normalized_sources = {_normalize_name(column): column for column in source_columns}

        for target in self.contract.columns:
            if target.name in df.columns:
                used_sources.add(target.name)
                mappings.append(SchemaColumnMapping(
                    source_column=target.name,
                    canonical_column=target.name,
                    action="exact",
                    confidence=1.0,
                    reason="source column matches canonical name",
                    compatible_type_change=_compatible_dtype(df[target.name], target.dtype),
                ))
                continue

            alias = next((alias for alias in target.aliases if alias in df.columns), None)
            if alias is not None:
                used_sources.add(alias)
                mappings.append(SchemaColumnMapping(
                    source_column=alias,
                    canonical_column=target.name,
                    action="alias",
                    confidence=0.98,
                    reason=f"source column matches alias {alias!r}",
                    compatible_type_change=_compatible_dtype(df[alias], target.dtype),
                ))
                continue

            best_source, score = self._best_rename_candidate(
                target.name,
                normalized_sources,
                used_sources,
            )
            if best_source is not None and score >= self.rename_threshold:
                used_sources.add(best_source)
                mappings.append(SchemaColumnMapping(
                    source_column=best_source,
                    canonical_column=target.name,
                    action="renamed",
                    confidence=round(score, 4),
                    reason=f"normalized names are similar ({score:.2f})",
                    compatible_type_change=_compatible_dtype(df[best_source], target.dtype),
                ))
                continue

            mappings.append(SchemaColumnMapping(
                source_column=None,
                canonical_column=target.name,
                action="missing",
                confidence=1.0,
                reason="canonical column missing from source",
            ))

        for source in source_columns:
            if source not in used_sources:
                mappings.append(SchemaColumnMapping(
                    source_column=source,
                    canonical_column=source,
                    action="extra",
                    confidence=1.0,
                    reason="source column is not in canonical contract",
                ))
        return tuple(mappings)

    def harmonize(
        self,
        df: pd.DataFrame,
        *,
        source_version: str | None = None,
    ) -> SchemaHarmonizationResult:
        """Map source data to canonical columns and quarantine incompatible rows."""
        mapping = self.detect_mappings(df)
        canonical = self.apply_mapping(df, mapping)
        quarantine = self.quarantine_incompatible_rows(canonical)
        migration = self.generate_migration_diff(df, mapping, source_version=source_version)
        return SchemaHarmonizationResult(
            canonical_frame=quarantine.accepted,
            quarantine=quarantine,
            mapping=mapping,
            migration_diff=migration,
        )

    def apply_mapping(
        self,
        df: pd.DataFrame,
        mapping: tuple[SchemaColumnMapping, ...],
    ) -> pd.DataFrame:
        """Return a canonical-column frame before row quarantine."""
        out = pd.DataFrame(index=df.index)
        by_canonical = {item.canonical_column: item for item in mapping if item.action != "extra"}
        for column in self.contract.columns:
            item = by_canonical[column.name]
            if item.source_column is None:
                out[column.name] = pd.NA
            else:
                out[column.name] = df[item.source_column]
        return out

    def quarantine_incompatible_rows(self, canonical: pd.DataFrame) -> QuarantineResult:
        """Coerce compatible values and quarantine rows that cannot be coerced."""
        accepted = canonical.copy(deep=True)
        invalid_reasons: dict[Any, list[str]] = {}
        for column in self.contract.columns:
            coerced, invalid = _coerce_series(accepted[column.name], column.dtype)
            if not column.nullable:
                invalid = invalid | coerced.isna()
            accepted[column.name] = coerced
            for row_id in accepted.index[invalid]:
                invalid_reasons.setdefault(row_id, []).append(
                    f"{column.name} incompatible with {column.dtype}"
                )

        if not invalid_reasons:
            return QuarantineResult(
                accepted=accepted,
                quarantined=accepted.iloc[0:0].copy(),
                reasons={},
            )

        bad_index = list(invalid_reasons)
        quarantined = canonical.loc[bad_index].copy()
        reasons = {row: "; ".join(parts) for row, parts in invalid_reasons.items()}
        quarantined["freshdata_quarantine_reason"] = [reasons[row] for row in bad_index]
        return QuarantineResult(
            accepted=accepted.drop(index=bad_index),
            quarantined=quarantined,
            reasons=reasons,
        )

    def generate_migration_diff(
        self,
        df: pd.DataFrame,
        mapping: tuple[SchemaColumnMapping, ...],
        *,
        source_version: str | None = None,
    ) -> MigrationDiff:
        """Generate a reviewable migration diff for upstream schema drift."""
        renamed = {
            item.source_column: item.canonical_column
            for item in mapping
            if item.action in {"alias", "renamed"} and item.source_column is not None
        }
        added = tuple(
            item.source_column or item.canonical_column
            for item in mapping
            if item.action == "extra"
        )
        removed = tuple(item.canonical_column for item in mapping if item.action == "missing")
        type_changes = {}
        incompatible = []
        for item in mapping:
            if item.source_column is None or item.action in {"missing", "extra"}:
                continue
            target = self.contract.column_map[item.canonical_column]
            observed = str(df[item.source_column].dtype)
            if _dtype_family(observed) != _dtype_family(target.dtype):
                type_changes[item.canonical_column] = f"{observed} -> {target.dtype}"
            if not item.compatible_type_change:
                incompatible.append(item.canonical_column)
        return MigrationDiff(
            contract_name=self.contract.name,
            from_version=source_version,
            to_version=self.contract.version,
            added_columns=added,
            removed_columns=removed,
            renamed_columns=renamed,
            type_changes=type_changes,
            incompatible_columns=tuple(incompatible),
        )

    def _best_rename_candidate(
        self,
        canonical_name: str,
        normalized_sources: dict[str, str],
        used_sources: set[str],
    ) -> tuple[str | None, float]:
        target = _normalize_name(canonical_name)
        best_source: str | None = None
        best_score = 0.0
        for normalized, source in normalized_sources.items():
            if source in used_sources:
                continue
            score = difflib.SequenceMatcher(None, target, normalized).ratio()
            if score > best_score:
                best_source = source
                best_score = score
        return best_source, best_score


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _dtype_family(dtype: str) -> str:
    text = str(dtype).lower()
    if any(token in text for token in ("int", "float", "double", "number", "numeric")):
        return "numeric"
    if any(token in text for token in ("datetime", "date", "timestamp")):
        return "datetime"
    if any(token in text for token in ("bool", "boolean")):
        return "bool"
    return "string"


def _compatible_dtype(series: pd.Series, target_dtype: str) -> bool:
    family = _dtype_family(target_dtype)
    if family == "string":
        return True
    _, invalid = _coerce_series(series, target_dtype)
    return not bool(invalid.any())


def _coerce_series(series: pd.Series, target_dtype: str) -> tuple[pd.Series, pd.Series]:
    family = _dtype_family(target_dtype)
    if family == "numeric":
        coerced = pd.to_numeric(series, errors="coerce")
        invalid = series.notna() & coerced.isna()
        if "int" in target_dtype.lower() and not invalid.any():
            return coerced.astype("Int64"), invalid
        return coerced, invalid
    if family == "datetime":
        coerced = pd.to_datetime(series, errors="coerce")
        invalid = series.notna() & coerced.isna()
        return coerced, invalid
    if family == "bool":
        lowered = series.map(
            lambda value: str(value).strip().lower() if pd.notna(value) else value
        )
        valid_values = {"true", "false", "1", "0", "yes", "no", True, False}
        invalid = series.notna() & ~lowered.isin(valid_values)
        mapped = lowered.map({
            "true": True,
            "1": True,
            "yes": True,
            "false": False,
            "0": False,
            "no": False,
            True: True,
            False: False,
        })
        return mapped.astype("boolean"), invalid
    return series.astype("string"), pd.Series(False, index=series.index)
