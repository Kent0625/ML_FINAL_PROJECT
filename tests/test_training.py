from datetime import date
from pathlib import Path

import joblib

from src.predictor import TARGET_NAMES, predict_bill
from train_model import (
    CONGRESS_FILES,
    load_source_data,
    prepare_balanced_dataset,
    train_model_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_source_data_combines_all_four_congresses():
    data = load_source_data(PROJECT_ROOT)

    assert len(data) == 7506
    assert set(data["Congress_Period"]) == set(CONGRESS_FILES)


def test_prepare_balanced_dataset_maps_equal_class_counts():
    prepared = prepare_balanced_dataset(
        load_source_data(PROJECT_ROOT),
        max_per_class=12,
    )

    assert prepared["Target_Status"].value_counts().to_dict() == {
        "First Reading": 12,
        "Second Reading": 12,
        "Approved": 12,
    }
    assert prepared["Clean_Text"].notna().all()
    assert prepared["Primary_Committee"].str.upper().equals(
        prepared["Primary_Committee"]
    )


def test_trained_bundle_is_serializable_and_predicts_three_classes(tmp_path):
    data = load_source_data(PROJECT_ROOT)
    bundle = train_model_bundle(
        data,
        max_per_class=18,
        max_features=40,
        model_params={
            "n_estimators": 8,
            "max_depth": 2,
            "learning_rate": 0.2,
        },
    )
    artifact_path = tmp_path / "model.joblib"
    joblib.dump(bundle, artifact_path)
    restored = joblib.load(artifact_path)

    result = predict_bill(
        bundle=restored,
        subject="Mental health facilities",
        author=restored["authors"][0],
        committee=restored["committees"][0],
        scope="National",
        congress_period="18th",
        filing_date=date(2020, 1, 1),
    )

    assert set(result["probabilities"]) == set(TARGET_NAMES)
    assert restored["metrics"]["training_rows"] == 54
    assert restored["metrics"]["macro_f1"] >= 0
    assert restored["feature_names"][:4] == [
        "Scope_Binary",
        "Days_Since_Congress_Start",
        "Legislative_Efficacy",
        "Committee_Passage_Rate",
    ]
