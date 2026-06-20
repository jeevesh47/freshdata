"""Conformance tests for the education (Ed-Fi) domain pack."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.domains import UnknownDomainError
from freshdata.domains._common import redact_phi_actions
from freshdata.domains.base import RepairAction, RepairLog
from freshdata.domains.education import EducationValidator


def _violated(rep, rule_id: str) -> bool:
    return any(f["rule_id"] == rule_id and f["status"] == "violated" for f in rep.domain_findings)


@pytest.fixture
def good_education() -> pd.DataFrame:
    return pd.DataFrame({
        "student_unique_id": ["S1", "S2", "S3"],
        "school_id": ["SCH01", "SCH01", "SCH02"],
        "school_year": [2024, 2024, 2023],
        "grade_level": ["First grade", "Kindergarten", "Twelfth grade"],
        "enrollment_date": ["2023-08-15", "2023-08-20", "2022-09-01"],
        "exit_withdrawal_date": ["2024-05-30", None, "2023-06-10"],
        "exit_withdrawal_type": ["Transferred", None, "Graduated"],
        "assessment_title": ["Math", "Reading", "Math"],
        "assessment_score": [85.0, 92.0, 78.0],
        "assessment_score_type": ["Scale score", "Scale score", "Raw score"],
        "staff_unique_id": ["T1", "T1", "T2"],
        "staff_classification": ["Teachers", "Teachers", "Principals/Assistant principals"],
    })


def test_happy_path_passes(good_education):
    out, rep = fd.clean(good_education, domain="education", return_report=True, verbose=False)
    assert rep.domain == "education"
    assert rep.domain_trust_score >= 0.95
    assert not [f for f in rep.domain_findings
                if f["status"] == "violated" and f["severity"] == "error"]
    assert out.shape[0] == 3


@pytest.mark.parametrize("field", ["student_unique_id", "school_id", "school_year"])
def test_each_required_field_missing(good_education, field):
    validator = EducationValidator()
    actual = validator.detect_columns(good_education).actual(field)
    report = validator.validate(good_education.drop(columns=[actual]))
    assert field in report.mapping.unmapped_required
    assert any("MISSING_REQUIRED_FIELD" in r.message and r.violated for r in report.results)


def test_format_violations(good_education):
    df = good_education.copy()
    df["assessment_score"] = df["assessment_score"].astype(object)
    df.loc[0, "school_year"] = 3000            # ED-003: beyond current year + 2
    df.loc[1, "enrollment_date"] = "08/2023"   # ED-005: not ISO 8601
    df.loc[2, "exit_withdrawal_date"] = "soon"  # ED-006: not ISO 8601
    df.loc[0, "assessment_score"] = "n/a-text"  # ED-012: not numeric
    _, rep = fd.clean(df, domain="education", return_report=True, verbose=False)
    assert _violated(rep, "ED-003")
    assert _violated(rep, "ED-005")
    assert _violated(rep, "ED-006")
    assert _violated(rep, "ED-012")


def test_reference_violations(good_education):
    df = good_education.copy()
    df.loc[0, "grade_level"] = "Grade 99"           # ED-004
    df.loc[1, "exit_withdrawal_type"] = "Vanished"  # ED-009
    df.loc[0, "assessment_score_type"] = "Vibes"    # ED-013 (warning)
    df.loc[0, "staff_classification"] = "Wizard"    # ED-016
    _, rep = fd.clean(df, domain="education", return_report=True, verbose=False)
    assert _violated(rep, "ED-004")
    assert _violated(rep, "ED-009")
    assert _violated(rep, "ED-013")
    assert _violated(rep, "ED-016")


def test_business_rules(good_education):
    df = good_education.copy()
    df.loc[2, "exit_withdrawal_date"] = "2021-01-01"   # ED-007: before enrollment
    df.loc[0, "enrollment_date"] = "2010-01-01"        # ED-008: outside year window
    df.loc[1, "exit_withdrawal_date"] = "2024-06-01"   # ED-010: date present, type absent
    _, rep = fd.clean(df, domain="education", return_report=True, verbose=False)
    assert _violated(rep, "ED-007")
    assert _violated(rep, "ED-008")
    assert _violated(rep, "ED-010")


def test_assessment_subschema_both_present(good_education):
    df = good_education.copy()
    df.loc[0, "assessment_score"] = None    # title present, score absent -> ED-011
    _, rep = fd.clean(df, domain="education", return_report=True, verbose=False)
    assert _violated(rep, "ED-011")


def test_negative_score_is_warning_not_rejected(good_education):
    df = good_education.copy()
    df.loc[0, "assessment_score"] = -3.0    # growth/delta scores can be negative
    out, rep = fd.clean(df, domain="education", return_report=True, verbose=False)
    assert _violated(rep, "ED-014")
    ed014 = next(f for f in rep.domain_findings if f["rule_id"] == "ED-014")
    assert ed014["severity"] == "warning"
    assert out.loc[0, "assessment_score"] == -3.0   # never altered


def test_repairs_are_flag_only_and_audited(good_education):
    df = good_education.copy()
    df.loc[0, "grade_level"] = "Grade 99"   # ED-004 violated -> flagged, not repaired
    _, rep = fd.clean(df, domain="education", return_report=True, verbose=False)
    flagged = [a for a in rep.domain_repairs if a["rule_id"] == "ED-004"]
    assert flagged and all(a["status"] == "flagged" for a in flagged)
    assert all(a["status"] != "applied" for a in rep.domain_repairs)


def test_id_safety_null_student_id_never_filled(good_education):
    df = good_education.copy()
    df.loc[0, "student_unique_id"] = None
    out, rep = fd.clean(df, domain="education", return_report=True, verbose=False)
    assert pd.isna(out.loc[0, "student_unique_id"])
    assert _violated(rep, "ED-001")
    assert not [a for a in rep.domain_repairs
                if a["column"] == "student_unique_id" and a["status"] == "applied"]


def test_id_safety_null_staff_id_never_filled(good_education):
    df = good_education.copy()
    df.loc[1, "staff_unique_id"] = None
    out, _ = fd.clean(df, domain="education", return_report=True, verbose=False)
    assert pd.isna(out.loc[1, "staff_unique_id"])


def test_phi_redaction_masks_by_default(good_education):
    validator = EducationValidator()
    mapping = validator.detect_columns(good_education)
    log = RepairLog()
    log.add(RepairAction("ED-001", "flag_only", "student_unique_id", 0, None, None, "flagged"))
    redact_phi_actions(good_education, log, mapping, validator.PHI_FIELDS, include_phi=False)
    serialized = log.actions[0].to_dict()
    assert serialized["from"] == "[PHI]"            # the real id 'S1' is masked
    assert "S1" not in str(serialized)


def test_phi_redaction_opt_in_shows_value(good_education):
    validator = EducationValidator(audit_include_phi=True)
    assert validator._audit_include_phi is True
    mapping = validator.detect_columns(good_education)
    log = RepairLog()
    log.add(RepairAction("ED-001", "flag_only", "student_unique_id", 0, None, None, "flagged"))
    redact_phi_actions(good_education, log, mapping, validator.PHI_FIELDS, include_phi=True)
    assert log.actions[0].to_dict()["from"] == "S1"


def test_validation_never_mutates_input(good_education):
    before = good_education.copy()
    EducationValidator().validate(good_education)
    pd.testing.assert_frame_equal(good_education, before)


def test_messy_columns_detected(good_education):
    renamed = good_education.rename(columns={
        "student_unique_id": "Student ID", "school_year": "Academic Year",
    })
    out, rep = fd.clean(renamed, domain="education", return_report=True, verbose=False)
    assert rep.domain_trust_score >= 0.95
    assert not [f for f in rep.domain_findings
                if f["status"] == "violated" and f["severity"] == "error"]
    assert out.shape[0] == 3


def test_unknown_domain_lists_available(good_education):
    with pytest.raises(UnknownDomainError) as exc:
        fd.clean(good_education, domain="unknown_xyz")
    assert "education" in exc.value.available


def test_standalone_import():
    assert EducationValidator().domain_name == "education"
