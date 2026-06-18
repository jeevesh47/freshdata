import pandas as pd

import freshdata as fd


def test_schema_harmonizer_maps_renames_and_quarantines_bad_rows():
    contract = fd.SchemaContract(
        name="orders",
        version="2.0",
        columns=(
            fd.ColumnContract("customer_id", "int64", nullable=False),
            fd.ColumnContract("amount", "float64", nullable=False),
        ),
    )
    source = pd.DataFrame({
        "Customer ID": ["1", "bad"],
        "amount": ["10.5", "oops"],
        "source_extra": ["x", "y"],
    })

    result = fd.SchemaHarmonizer(contract).harmonize(source, source_version="1.0")

    assert list(result.canonical_frame.columns) == ["customer_id", "amount"]
    assert len(result.canonical_frame) == 1
    assert result.canonical_frame.loc[0, "customer_id"] == 1
    assert len(result.quarantine.quarantined) == 1
    assert "amount incompatible" in result.quarantine.quarantined.iloc[0][
        "freshdata_quarantine_reason"
    ]
    mapping = result.mapping_frame()
    assert mapping.loc[mapping["canonical_column"] == "customer_id", "action"].item() == "renamed"
    assert result.migration_diff.renamed_columns == {"Customer ID": "customer_id"}
    assert result.migration_diff.added_columns == ("source_extra",)
    assert "amount" in result.migration_diff.incompatible_columns


def test_schema_harmonizer_tracks_contract_versions_and_aliases():
    v1 = fd.SchemaContract(
        name="customers",
        version="1.0",
        columns=(fd.ColumnContract("customer_id", "int64"),),
    )
    v2 = fd.SchemaContract(
        name="customers",
        version="2.0",
        columns=(fd.ColumnContract("customer_id", "int64", aliases=("cust_id",)),),
    )
    harmonizer = fd.SchemaHarmonizer(v2, known_contracts=(v1,))
    source = pd.DataFrame({"cust_id": [100]})

    mapping = harmonizer.detect_mappings(source)

    assert set(harmonizer.contract_history) == {"1.0", "2.0"}
    assert mapping[0].action == "alias"
    assert mapping[0].source_column == "cust_id"
