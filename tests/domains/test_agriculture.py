"""Conformance tests for the agriculture (ADAPT) domain pack."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.domains import UnknownDomainError
from freshdata.domains.agriculture import AgricultureValidator


def _violated(rep, rule_id: str) -> bool:
    return any(f["rule_id"] == rule_id and f["status"] == "violated" for f in rep.domain_findings)


@pytest.fixture
def good_agri() -> pd.DataFrame:
    return pd.DataFrame({
        "field_id": ["F1", "F2", "F3"],
        "operation_id": ["OP1", "OP2", "OP3"],
        "operation_type": ["Harvesting", "Planting", "Tillage"],
        "operation_date": ["2024-09-15", "2024-04-01", "2024-03-20"],
        "crop_type": ["Maize", "Wheat", "Soybean"],
        "crop_code": ["56", "15", "236"],
        "area": [120.5, 80.0, 60.0],
        "area_unit": ["ACR", "HAR", "ACR"],
        "yield_value": [11000.0, None, None],
        "yield_unit": ["KGM", None, None],
        "soil_texture_class": ["Loam", "Silt loam", "Clay loam"],
        "soil_ph": [6.5, 6.8, 7.1],
        "soil_om_pct": [3.5, 4.0, 2.8],
        "equipment_id": ["E1", "E2", "E3"],
        "operator_id": ["U1", "U2", "U3"],
        "season_year": [2024, 2024, 2024],
    })


def test_happy_path_passes(good_agri):
    out, rep = fd.clean(good_agri, domain="agriculture", return_report=True, verbose=False)
    assert rep.domain == "agriculture"
    assert rep.domain_trust_score >= 0.95
    assert not [f for f in rep.domain_findings
                if f["status"] == "violated" and f["severity"] == "error"]
    assert out.shape[0] == 3


def test_required_field_missing(good_agri):
    validator = AgricultureValidator()
    actual = validator.detect_columns(good_agri).actual("field_id")
    report = validator.validate(good_agri.drop(columns=[actual]))
    assert "field_id" in report.mapping.unmapped_required
    assert any("MISSING_REQUIRED_FIELD" in r.message and r.violated for r in report.results)


def test_format_violations(good_agri):
    df = good_agri.copy()
    df.loc[1, "operation_date"] = "April 2024"   # AG-002: not ISO 8601
    df.loc[0, "area"] = -5.0                      # AG-006: not positive
    df.loc[0, "yield_value"] = 0.0               # AG-009: not positive
    df.loc[2, "soil_ph"] = 20.0                  # AG-014: outside 0-14
    df.loc[1, "soil_om_pct"] = 150.0             # AG-016: outside 0-100
    df.loc[2, "season_year"] = 3000              # AG-018: beyond current + 1
    _, rep = fd.clean(df, domain="agriculture", return_report=True, verbose=False)
    for rule_id in ("AG-002", "AG-006", "AG-009", "AG-014", "AG-016", "AG-018"):
        assert _violated(rep, rule_id), rule_id


def test_future_operation_date(good_agri):
    df = good_agri.copy()
    df.loc[0, "operation_date"] = "2099-08-01"
    _, rep = fd.clean(df, domain="agriculture", return_report=True, verbose=False)
    assert _violated(rep, "AG-003")


def test_reference_violations(good_agri):
    df = good_agri.copy()
    df.loc[0, "operation_type"] = "Levitating"   # AG-004
    df.loc[1, "crop_code"] = "999999"            # AG-005 (warning)
    df.loc[2, "area_unit"] = "BOGUS"             # AG-007 (uncoercible)
    df.loc[0, "yield_unit"] = "ZZZ"              # AG-010 (uncoercible)
    df.loc[1, "soil_texture_class"] = "Marsh"    # AG-013
    _, rep = fd.clean(df, domain="agriculture", return_report=True, verbose=False)
    for rule_id in ("AG-004", "AG-005", "AG-007", "AG-010", "AG-013"):
        assert _violated(rep, rule_id), rule_id


def test_business_rules(good_agri):
    df = good_agri.copy()
    df.loc[1, "area_unit"] = None                # AG-008: area present, unit absent
    df.loc[2, "yield_value"] = 200.0             # AG-011: yield present, unit absent (row 2)
    df.loc[1, "yield_value"] = 300.0             # AG-012: yield on a Planting op
    df.loc[1, "yield_unit"] = "KGM"
    df.loc[0, "season_year"] = 2020              # AG-019: 2024 op vs 2020 season
    _, rep = fd.clean(df, domain="agriculture", return_report=True, verbose=False)
    for rule_id in ("AG-008", "AG-011", "AG-012", "AG-019"):
        assert _violated(rep, rule_id), rule_id


def test_semantic_plausibility_warnings(good_agri):
    df = good_agri.copy()
    df.loc[0, "soil_ph"] = 2.0       # AG-015: implausible but in [0,14]
    df.loc[1, "soil_om_pct"] = 25.0  # AG-017: very high organic matter
    _, rep = fd.clean(df, domain="agriculture", return_report=True, verbose=False)
    assert _violated(rep, "AG-015")
    assert _violated(rep, "AG-017")
    ag015 = next(f for f in rep.domain_findings if f["rule_id"] == "AG-015")
    assert ag015["severity"] == "warning"


def test_repair_audit_unit_coercion(good_agri):
    df = good_agri.copy()
    df["area_unit"] = df["area_unit"].astype(object)
    df["yield_unit"] = df["yield_unit"].astype(object)
    df.loc[0, "area_unit"] = "acres"     # -> ACR
    df.loc[0, "yield_unit"] = "bushels"  # -> BU
    out, rep = fd.clean(df, domain="agriculture", return_report=True, verbose=False)
    a007 = [a for a in rep.domain_repairs if a["rule_id"] == "AG-007" and a["status"] == "applied"]
    a010 = [a for a in rep.domain_repairs if a["rule_id"] == "AG-010" and a["status"] == "applied"]
    assert a007 and a007[0]["from"] == "acres" and a007[0]["to"] == "ACR"
    assert a007[0]["strategy"] == "coerce" and a007[0]["row"] == 0
    assert a010 and a010[0]["from"] == "bushels" and a010[0]["to"] == "BU"
    assert out.loc[0, "area_unit"] == "ACR" and out.loc[0, "yield_unit"] == "BU"


def test_uncoercible_unit_is_unresolvable(good_agri):
    df = good_agri.copy()
    df.loc[0, "area_unit"] = "BOGUS"
    out, rep = fd.clean(df, domain="agriculture", return_report=True, verbose=False)
    assert out.loc[0, "area_unit"] == "BOGUS"   # left untouched
    assert any(a["rule_id"] == "AG-007" and a["status"] == "unresolvable"
               for a in rep.domain_repairs)


def test_id_safety_null_field_id_never_filled(good_agri):
    df = good_agri.copy()
    df.loc[0, "field_id"] = None
    out, rep = fd.clean(df, domain="agriculture", return_report=True, verbose=False)
    assert pd.isna(out.loc[0, "field_id"])
    assert _violated(rep, "AG-001")
    assert not [a for a in rep.domain_repairs
                if a["column"] == "field_id" and a["status"] == "applied"]


def test_crop_code_never_inferred_from_crop_type(good_agri):
    df = good_agri.copy()
    df.loc[0, "crop_code"] = None       # missing code, crop_type='Maize' is present
    out, rep = fd.clean(df, domain="agriculture", return_report=True, verbose=False)
    assert pd.isna(out.loc[0, "crop_code"])   # never inferred from the text name
    assert not [a for a in rep.domain_repairs
                if a["column"] == "crop_code" and a["status"] == "applied"]


def test_validation_never_mutates_input(good_agri):
    before = good_agri.copy()
    AgricultureValidator().validate(good_agri)
    pd.testing.assert_frame_equal(good_agri, before)


def test_unknown_domain_lists_available(good_agri):
    with pytest.raises(UnknownDomainError) as exc:
        fd.clean(good_agri, domain="unknown_xyz")
    assert "agriculture" in exc.value.available


def test_standalone_import():
    assert AgricultureValidator().domain_name == "agriculture"
