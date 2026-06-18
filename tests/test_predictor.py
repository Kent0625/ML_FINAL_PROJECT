from datetime import date

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from src.predictor import (
    STATUS_MAPPING,
    TARGET_NAMES,
    calculate_days_since_congress_start,
    clean_legislative_text,
    predict_bill,
    prepare_model_input,
)


def build_test_bundle():
    vectorizer = TfidfVectorizer(max_features=20)
    vectorizer.fit(
        [
            "public health hospitals",
            "education schools teachers",
            "local road infrastructure",
        ]
    )

    feature_count = 4 + len(vectorizer.get_feature_names_out())
    model = DummyClassifier(strategy="prior")
    model.fit(
        np.zeros((6, feature_count)),
        np.array([0, 0, 0, 1, 2, 2]),
    )

    return {
        "vectorizer": vectorizer,
        "model": model,
        "author_efficacy": {"Known Author": 0.35},
        "committee_passage_rate": {"HEALTH AND DEMOGRAPHY": 0.42},
        "global_approval_rate": 0.2,
        "class_names": TARGET_NAMES,
        "congress_start_dates": {
            "18th": "2019-07-22",
        },
        "metrics": {
            "macro_f1": 0.5017,
            "balanced_accuracy": 0.5098,
        },
    }


def test_status_mapping_groups_legislative_milestones():
    assert STATUS_MAPPING["PENDING IN THE COMMITTEE"] == "First Reading"
    assert (
        STATUS_MAPPING["APPROVED ON SECOND READING, WITH AMENDMENTS"]
        == "Second Reading"
    )
    assert (
        STATUS_MAPPING["APPROVED BY THE PRESIDENT OF THE PHILIPPINES"]
        == "Approved"
    )


def test_clean_legislative_text_removes_boilerplate_and_punctuation():
    cleaned = clean_legislative_text(
        "AN ACT providing for PUBLIC HEALTH facilities, and for other purposes!"
    )

    assert cleaned == "health facilities"


def test_calculate_days_since_congress_start_clamps_early_dates():
    bundle = build_test_bundle()

    assert (
        calculate_days_since_congress_start(
            "18th",
            date(2020, 7, 22),
            bundle["congress_start_dates"],
        )
        == 366
    )
    assert (
        calculate_days_since_congress_start(
            "18th",
            date(2018, 1, 1),
            bundle["congress_start_dates"],
        )
        == 0
    )


def test_prepare_model_input_uses_global_rate_for_unknown_categories():
    bundle = build_test_bundle()

    model_input, signals = prepare_model_input(
        bundle=bundle,
        subject="Mental health facilities",
        author="Unseen Senator",
        committee="UNSEEN COMMITTEE",
        scope="National",
        congress_period="18th",
        filing_date=date(2020, 7, 22),
    )

    assert model_input.shape[0] == 1
    assert model_input.shape[1] == 4 + len(
        bundle["vectorizer"].get_feature_names_out()
    )
    assert signals["author_efficacy"] == pytest.approx(0.2)
    assert signals["committee_passage_rate"] == pytest.approx(0.2)
    assert signals["scope_binary"] == 1
    assert signals["days_since_congress_start"] == 366


def test_predict_bill_returns_three_normalized_probabilities():
    bundle = build_test_bundle()

    result = predict_bill(
        bundle=bundle,
        subject="Public health hospitals",
        author="Known Author",
        committee="HEALTH AND DEMOGRAPHY",
        scope="National",
        congress_period="18th",
        filing_date=date(2020, 7, 22),
    )

    assert result["predicted_milestone"] in TARGET_NAMES
    assert set(result["probabilities"]) == set(TARGET_NAMES)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert result["model_context"]["macro_f1"] == pytest.approx(0.5017)


def test_predict_bill_rejects_empty_subject():
    with pytest.raises(ValueError, match="subject"):
        predict_bill(
            bundle=build_test_bundle(),
            subject="   ",
            author="Known Author",
            committee="HEALTH AND DEMOGRAPHY",
            scope="National",
            congress_period="18th",
            filing_date=date(2020, 7, 22),
        )
