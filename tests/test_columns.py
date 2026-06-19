import pandas as pd
import pytest

import freshdata as fd
from freshdata.steps.columns import snake_case


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" First Name ", "first_name"),
        ("AGE", "age"),
        ("CustomerID", "customer_id"),
        ("HTTPResponseCode", "http_response_code"),
        ("Salary($)", "salary"),
        ("already_snake", "already_snake"),
        ("Total %", "total"),
        ("a  b", "a_b"),
        ("Café Größe", "café_größe"),  # unicode letters survive
    ],
)
def test_snake_case(raw, expected):
    assert snake_case(raw) == expected


def test_unnameable_columns_get_positional_names():
    df = pd.DataFrame([[1, 2]], columns=["###", "$$$"])
    out = fd.clean(df)
    assert list(out.columns) == ["column_0", "column_1"]


def test_collisions_deduplicated_without_new_clashes():
    df = pd.DataFrame([[1, 2, 3]], columns=["a b", "a_b", "a_b_2"])
    out = fd.clean(df)
    assert len(set(out.columns)) == 3
    assert "a_b" in out.columns


def test_non_string_labels_untouched():
    df = pd.DataFrame([[1, 2]], columns=[0, 1])
    out = fd.clean(df)
    assert list(out.columns) == [0, 1]


def test_renaming_can_be_disabled():
    df = pd.DataFrame({"Mixed Case": [1, 2]})
    out = fd.clean(df, column_names=False)
    assert list(out.columns) == ["Mixed Case"]


def test_rename_recorded_in_report():
    df = pd.DataFrame({"First Name": [1]})
    _, report = fd.clean(df, return_report=True)
    actions = [a for a in report if a.step == "column_names"]
    assert len(actions) == 1
    assert "first_name" in actions[0].description
