import json

import pandas as pd
import pytest

import freshdata as fd


def test_duplicate_defense_separates_exact_near_and_replayed_batches():
    df = pd.DataFrame({
        "entity_id": [1, 1, 2],
        "name": ["Acme Inc", "Acme Inc", "Acme inc."],
        "amount": [10, 10, 11],
    })
    defense = fd.DuplicateDefense(near_threshold=0.85)
    prior = defense.build_manifest(
        df,
        source_id="crm",
        load_id="load-1",
        key_columns=("entity_id",),
    )

    report = defense.analyze(
        df,
        source_id="crm",
        load_id="load-2",
        key_columns=("entity_id",),
        entity_columns=("name",),
        prior_manifests=(prior,),
    )

    assert len(report.exact_duplicates) == 1
    assert report.exact_duplicates[0].duplicate_type == "exact_row"
    assert report.exact_duplicates[0].review_required is False
    assert report.near_duplicates
    assert all(item.review_required for item in report.near_duplicates)
    assert len(report.replayed_batches) == 1
    review = report.to_review_dataset().to_frame()
    assert {"candidate_change", "required_decision", "risk"}.issubset(review.columns)


def test_review_queue_exports_csv_json_and_parquet(tmp_path):
    queue = fd.ReviewQueue(dataset_id="unit-test-review")
    queue.add_candidate(
        review_id="review-1",
        candidate_change="Update amount from -1 to null",
        required_decision="Approve or reject the repair.",
        confidence=0.72,
        reason="Negative amount violates a validator failure.",
        risk="medium",
        source="pandera",
        patch_id="patch-1",
        row=4,
        column="amount",
    )

    csv_path = queue.export(tmp_path / "review.csv")
    json_path = queue.export(tmp_path / "review.json")

    assert pd.read_csv(csv_path).loc[0, "approval_status"] == "pending"
    assert json.loads(json_path.read_text())[0]["required_decision"].startswith("Approve")

    pytest.importorskip("pyarrow")
    parquet_path = queue.export(tmp_path / "review.parquet")
    assert pd.read_parquet(parquet_path).loc[0, "review_id"] == "review-1"
