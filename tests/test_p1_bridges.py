import pandas as pd

import freshdata as fd


def test_from_gx_normalizes_failed_expectations_to_review_dataset():
    result = {
        "results": [
            {
                "success": False,
                "expectation_config": {
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {"column": "email"},
                },
                "result": {
                    "unexpected_index_list": [2],
                    "partial_unexpected_list": [None],
                },
            },
            {
                "success": True,
                "expectation_config": {"expectation_type": "expect_table_row_count_to_be_between"},
            },
        ]
    }

    bridged = fd.from_gx(result)

    assert isinstance(bridged, fd.ValidationBridgeResult)
    assert len(bridged.failures) == 1
    assert bridged.failures[0].validator == "great_expectations"
    assert bridged.failures[0].column == "email"
    review = bridged.to_review_dataset().to_frame()
    assert {"candidate_change", "required_decision", "confidence", "risk"}.issubset(
        review.columns
    )


def test_from_dbt_failures_reads_failure_table(tmp_path):
    path = tmp_path / "dbt_failures.csv"
    pd.DataFrame({
        "unique_id": ["orders.not_null_order_id.1"],
        "column_name": ["order_id"],
        "test_name": ["not_null"],
        "order_id": [None],
    }).to_csv(path, index=False)

    bridged = fd.from_dbt_failures(path)

    assert len(bridged.failures) == 1
    assert bridged.failures[0].validator == "dbt"
    assert bridged.failures[0].check == "not_null"
    assert bridged.to_frame().loc[0, "column"] == "order_id"


def test_from_pandera_errors_accepts_failure_cases_dataframe():
    class FakeSchemaErrors:
        failure_cases = pd.DataFrame({
            "schema_context": ["Column"],
            "column": ["amount"],
            "check": ["greater_than_or_equal_to(0)"],
            "failure_case": [-1],
            "index": [3],
        })

    bridged = fd.from_pandera_errors(FakeSchemaErrors())

    assert len(bridged.failures) == 1
    assert bridged.failures[0].validator == "pandera"
    assert bridged.failures[0].failure_case == -1


def test_emit_validator_contracts_are_dependency_free():
    gx = fd.emit_gx_expectations(
        {"order_id": "integer", "amount": "float"},
        non_null_columns=("order_id",),
    )
    dbt = fd.emit_dbt_tests(
        {"order_id": "integer", "amount": "float"},
        model_name="orders",
        non_null_columns=("order_id",),
        unique_columns=("order_id",),
    )

    assert gx["expectation_suite_name"] == "freshdata_contract"
    assert gx["expectations"][0]["expectation_type"] == "expect_column_to_exist"
    assert dbt["models"][0]["name"] == "orders"
    order_id = next(col for col in dbt["models"][0]["columns"] if col["name"] == "order_id")
    assert order_id["tests"] == ["not_null", "unique"]
