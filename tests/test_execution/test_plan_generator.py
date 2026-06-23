"""Tests for the pure-Python plan generator (native vs fallback split)."""

from __future__ import annotations

from freshdata.config import CleanConfig
from freshdata.execution import PlanGenerator


def _plan(config, columns=("a", "b")):
    return PlanGenerator(config).plan(list(columns))


def test_native_config_has_no_fallback():
    cfg = CleanConfig(strategy="conservative", fix_dtypes=False)
    assert _plan(cfg).fallback_reason is None
    assert _plan(cfg).needs_fallback is False


def test_decision_engine_forces_fallback():
    assert _plan(CleanConfig(strategy="balanced")).needs_fallback


def test_fix_dtypes_forces_fallback():
    cfg = CleanConfig(strategy="conservative", fix_dtypes=True)
    assert _plan(cfg).needs_fallback


def test_impute_forces_fallback():
    cfg = CleanConfig(strategy="conservative", fix_dtypes=False, impute="median")
    assert _plan(cfg).needs_fallback


def test_outliers_force_fallback():
    cfg = CleanConfig(strategy="conservative", fix_dtypes=False, outliers="clip")
    assert _plan(cfg).needs_fallback


def test_rename_map_uses_snake_case():
    cfg = CleanConfig(strategy="conservative", fix_dtypes=False)
    plan = _plan(cfg, columns=["Customer ID", "amount"])
    assert plan.rename_map == {"Customer ID": "customer_id"}


def test_native_stage_order():
    cfg = CleanConfig(strategy="conservative", fix_dtypes=False)
    stages = _plan(cfg).stages
    # representation repair + structural reduction, in pipeline order
    assert stages == [
        "column_names",
        "clean_strings",
        "drop_empty_columns",
        "drop_empty_rows",
        "drop_duplicates",
    ]


def test_disabled_stages_excluded():
    cfg = CleanConfig(
        strategy="conservative", fix_dtypes=False,
        column_names=False, drop_duplicates=False,
    )
    stages = _plan(cfg).stages
    assert "column_names" not in stages
    assert "drop_duplicates" not in stages
