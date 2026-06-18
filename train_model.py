from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.predictor import STATUS_MAPPING, TARGET_NAMES, clean_legislative_text


RANDOM_STATE = 42
TARGET_MAP = {name: index for index, name in enumerate(TARGET_NAMES)}
CONGRESS_FILES = {
    "15th": "ldr_senate_bills_15th_congress.csv",
    "16th": "ldr_senate_bills_16th_congress.csv",
    "17th": "ldr_senate_bills_17th_congress.csv",
    "18th": "ldr_senate_bills_18th_congress.csv",
}
STRUCTURED_FEATURE_NAMES = [
    "Scope_Binary",
    "Days_Since_Congress_Start",
    "Legislative_Efficacy",
    "Committee_Passage_Rate",
]


def load_source_data(project_root: Path) -> pd.DataFrame:
    """Load and label the four Senate LDR datasets."""
    frames = []
    for congress_period, filename in CONGRESS_FILES.items():
        frame = pd.read_csv(project_root / filename)
        frame["Congress_Period"] = congress_period
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _prepare_mapped_data(data: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "Author",
        "Date_Filed",
        "Scope",
        "Subjects",
        "Primary_Committee",
        "Legislative_Status",
        "Congress_Period",
    ]
    prepared = data.dropna(subset=required_columns).copy()
    prepared["Author"] = prepared["Author"].astype(str).str.strip().str.title()
    prepared["Primary_Committee"] = (
        prepared["Primary_Committee"].astype(str).str.strip().str.upper()
    )
    prepared["Scope"] = prepared["Scope"].astype(str).str.strip().str.upper()
    prepared["Target_Status"] = prepared["Legislative_Status"].map(STATUS_MAPPING)
    prepared = prepared.dropna(subset=["Target_Status"]).copy()
    prepared["Target"] = prepared["Target_Status"].map(TARGET_MAP).astype(int)
    prepared["Date_Filed"] = pd.to_datetime(prepared["Date_Filed"], errors="coerce")
    prepared = prepared.dropna(subset=["Date_Filed"]).copy()
    prepared["Clean_Text"] = prepared["Subjects"].apply(clean_legislative_text)
    prepared["Scope_Binary"] = prepared["Scope"].eq("NATIONAL").astype(int)

    start_dates = prepared.groupby("Congress_Period")["Date_Filed"].transform("min")
    prepared["Days_Since_Congress_Start"] = (
        prepared["Date_Filed"] - start_dates
    ).dt.days.clip(lower=0)
    return prepared.reset_index(drop=True)


def prepare_balanced_dataset(
    data: pd.DataFrame,
    *,
    max_per_class: int | None = None,
) -> pd.DataFrame:
    """Map statuses and create a deterministic equal-size class sample."""
    prepared = _prepare_mapped_data(data)
    smallest_class = int(prepared["Target_Status"].value_counts().min())
    sample_size = min(smallest_class, max_per_class or smallest_class)

    balanced = (
        pd.concat(
            [
                prepared[prepared["Target_Status"] == target_status].sample(
                    n=sample_size,
                    random_state=RANDOM_STATE,
                )
                for target_status in TARGET_NAMES
            ],
            ignore_index=True,
        )
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    return balanced


def _build_feature_matrix(
    frame: pd.DataFrame,
    *,
    vectorizer: TfidfVectorizer,
    author_efficacy: dict[str, float],
    committee_passage_rate: dict[str, float],
    global_rate: float,
    fit_text: bool,
) -> np.ndarray:
    structured = np.column_stack(
        [
            frame["Scope_Binary"].astype(float).to_numpy(),
            frame["Days_Since_Congress_Start"].astype(float).to_numpy(),
            frame["Author"].map(author_efficacy).fillna(global_rate).to_numpy(),
            frame["Primary_Committee"]
            .map(committee_passage_rate)
            .fillna(global_rate)
            .to_numpy(),
        ]
    )
    text_features = (
        vectorizer.fit_transform(frame["Clean_Text"])
        if fit_text
        else vectorizer.transform(frame["Clean_Text"])
    ).toarray()
    return np.hstack([structured, text_features])


def train_model_bundle(
    data: pd.DataFrame,
    *,
    max_per_class: int | None = None,
    max_features: int = 1000,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train the deployable XGBoost bundle and capture evaluation context."""
    mapped_data = _prepare_mapped_data(data)
    balanced = prepare_balanced_dataset(data, max_per_class=max_per_class)

    train_frame, test_frame = train_test_split(
        balanced,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=balanced["Target"],
    )

    train_encoding = train_frame.copy()
    train_encoding["Is_Approved"] = (
        train_encoding["Target"] == TARGET_MAP["Approved"]
    ).astype(float)
    global_rate = float(train_encoding["Is_Approved"].mean())
    author_efficacy = (
        train_encoding.groupby("Author")["Is_Approved"].mean().to_dict()
    )
    committee_passage_rate = (
        train_encoding.groupby("Primary_Committee")["Is_Approved"].mean().to_dict()
    )

    min_df = 1 if len(train_frame) < 150 else 2
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=min_df,
    )
    x_train = _build_feature_matrix(
        train_frame,
        vectorizer=vectorizer,
        author_efficacy=author_efficacy,
        committee_passage_rate=committee_passage_rate,
        global_rate=global_rate,
        fit_text=True,
    )
    x_test = _build_feature_matrix(
        test_frame,
        vectorizer=vectorizer,
        author_efficacy=author_efficacy,
        committee_passage_rate=committee_passage_rate,
        global_rate=global_rate,
        fit_text=False,
    )

    parameters = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": 3,
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    parameters.update(model_params or {})
    model = XGBClassifier(**parameters)
    model.fit(x_train, train_frame["Target"])

    predictions = model.predict(x_test)
    raw_approval_rate = float(
        (mapped_data["Target"] == TARGET_MAP["Approved"]).mean()
    )
    metrics = {
        "macro_f1": float(
            f1_score(test_frame["Target"], predictions, average="macro")
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(test_frame["Target"], predictions)
        ),
        "accuracy": float(accuracy_score(test_frame["Target"], predictions)),
        "training_rows": int(len(balanced)),
        "holdout_rows": int(len(test_frame)),
        "source_rows": int(len(mapped_data)),
        "raw_approval_rate": raw_approval_rate,
    }

    congress_start_dates = (
        mapped_data.groupby("Congress_Period")["Date_Filed"]
        .min()
        .dt.strftime("%Y-%m-%d")
        .to_dict()
    )
    feature_names = STRUCTURED_FEATURE_NAMES + [
        f"word_{word}" for word in vectorizer.get_feature_names_out()
    ]

    return {
        "artifact_version": 1,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "vectorizer": vectorizer,
        "author_efficacy": author_efficacy,
        "committee_passage_rate": committee_passage_rate,
        "global_approval_rate": global_rate,
        "class_names": TARGET_NAMES,
        "feature_names": feature_names,
        "metrics": metrics,
        "congress_start_dates": congress_start_dates,
        "authors": sorted(mapped_data["Author"].unique().tolist()),
        "committees": sorted(mapped_data["Primary_Committee"].unique().tolist()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Life of a Bill model.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/life_of_a_bill.joblib"),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    source_data = load_source_data(project_root)
    bundle = train_model_bundle(source_data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output)

    metrics = bundle["metrics"]
    print(f"Saved model artifact to {args.output}")
    print(f"Balanced training rows: {metrics['training_rows']}")
    print(f"Holdout Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Holdout balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Historical raw approval rate: {metrics['raw_approval_rate']:.2%}")


if __name__ == "__main__":
    main()
