from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


TARGET_NAMES = ["First Reading", "Second Reading", "Approved"]

STATUS_MAPPING = {
    "PENDING IN THE COMMITTEE": "First Reading",
    "CONSOLIDATED/SUBSTITUTED IN THE COMMITTEE REPORT": "First Reading",
    "SENT TO THE ARCHIVES": "First Reading",
    "WITHDRAWN": "First Reading",
    "COMMITTEE REPORT SENT TO THE ARCHIVES": "First Reading",
    "RECOMMITTED TO THE COMMITTEE": "First Reading",
    "PENDING SECOND READING, SPECIAL ORDER": "Second Reading",
    "PENDING SECOND READING, ORDINARY BUSINESS": "Second Reading",
    "RECONSIDERATION OF APPROVAL ON SECOND READING": "Second Reading",
    "APPROVED ON SECOND READING, WITH AMENDMENTS": "Second Reading",
    "APPROVED ON SECOND READING, WITHOUT AMENDMENT": "Second Reading",
    "PENDING IN THE HOUSE OF REPRESENTATIVES": "Approved",
    "PENDING CONFERENCE COMMITTEE": "Approved",
    "WITH SENATE DESIGNATED CONFEREES": "Approved",
    "CONFERENCE COMMITTEE REPORT APPROVED BY SENATE": "Approved",
    "APPROVED BY THE PRESIDENT OF THE PHILIPPINES": "Approved",
    "CONSOLIDATED WITH APPROVED BILL": "Approved",
    "LAPSED INTO LAW": "Approved",
    "PASSED BY BOTH HOUSES": "Approved",
    "ENROLLED COPY SENT TO MALACANANG": "Approved",
    "VETOED BY THE PRESIDENT OF THE PHILIPPINES": "Approved",
}

LEGISLATIVE_STOP_WORDS = {
    "act",
    "additional",
    "amended",
    "amending",
    "appropriating",
    "batas",
    "bilang",
    "certain",
    "city",
    "code",
    "creating",
    "declaring",
    "decree",
    "establishing",
    "funds",
    "known",
    "local",
    "national",
    "otherwise",
    "pambansa",
    "philippine",
    "philippines",
    "presidential",
    "providing",
    "public",
    "purpose",
    "purposes",
    "republic",
    "revised",
    "section",
    "sections",
    "therefor",
}

ALL_STOP_WORDS = set(ENGLISH_STOP_WORDS).union(LEGISLATIVE_STOP_WORDS)
CLEAN_PATTERN = re.compile(r"[^a-z\s]")


def clean_legislative_text(text: Any) -> str:
    """Normalize policy keywords while removing common legislative boilerplate."""
    if text is None or pd.isna(text):
        return ""

    normalized = CLEAN_PATTERN.sub("", str(text).lower())
    return " ".join(
        word
        for word in normalized.split()
        if len(word) > 2 and word not in ALL_STOP_WORDS
    )


def calculate_days_since_congress_start(
    congress_period: str,
    filing_date: date | datetime | str,
    congress_start_dates: Mapping[str, str],
) -> int:
    """Return a non-negative filing offset for a configured Congress."""
    if congress_period not in congress_start_dates:
        raise ValueError(f"Unknown Congress period: {congress_period}")

    start = pd.Timestamp(congress_start_dates[congress_period]).normalize()
    filed = pd.Timestamp(filing_date).normalize()
    if pd.isna(filed):
        filed = start
    return max(0, int((filed - start).days))


def prepare_model_input(
    *,
    bundle: Mapping[str, Any],
    subject: str,
    author: str,
    committee: str,
    scope: str,
    congress_period: str,
    filing_date: date | datetime | str,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Transform one visitor scenario into the feature order used by the model."""
    cleaned_subject = clean_legislative_text(subject)
    vectorizer = bundle["vectorizer"]
    global_rate = float(bundle["global_approval_rate"])

    scope_binary = int(str(scope).strip().upper() == "NATIONAL")
    days_since_start = calculate_days_since_congress_start(
        congress_period,
        filing_date,
        bundle["congress_start_dates"],
    )
    author_efficacy = float(
        bundle["author_efficacy"].get(str(author).strip(), global_rate)
    )
    committee_passage_rate = float(
        bundle["committee_passage_rate"].get(
            str(committee).strip().upper(),
            global_rate,
        )
    )

    structured = np.array(
        [
            [
                scope_binary,
                days_since_start,
                author_efficacy,
                committee_passage_rate,
            ]
        ],
        dtype=float,
    )
    text_features = vectorizer.transform([cleaned_subject]).toarray()
    model_input = np.hstack([structured, text_features])

    signals = {
        "scope_binary": scope_binary,
        "days_since_congress_start": days_since_start,
        "author_efficacy": author_efficacy,
        "committee_passage_rate": committee_passage_rate,
    }
    return model_input, signals


def predict_bill(
    *,
    bundle: Mapping[str, Any],
    subject: str,
    author: str,
    committee: str,
    scope: str,
    congress_period: str,
    filing_date: date | datetime | str,
) -> dict[str, Any]:
    """Predict a legislative milestone and return an evaluation-friendly result."""
    if not str(subject).strip():
        raise ValueError("Enter a bill subject or policy description.")

    model_input, signals = prepare_model_input(
        bundle=bundle,
        subject=subject,
        author=author,
        committee=committee,
        scope=scope,
        congress_period=congress_period,
        filing_date=filing_date,
    )

    raw_probabilities = bundle["model"].predict_proba(model_input)[0]
    class_names = list(bundle.get("class_names", TARGET_NAMES))
    probabilities = {
        class_names[int(class_id)]: float(probability)
        for class_id, probability in zip(
            bundle["model"].classes_,
            raw_probabilities,
        )
    }
    predicted_milestone = max(probabilities, key=probabilities.get)

    return {
        "predicted_milestone": predicted_milestone,
        "probabilities": probabilities,
        "signals": signals,
        "model_context": dict(bundle.get("metrics", {})),
    }
