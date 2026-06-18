from __future__ import annotations

from datetime import date
from pathlib import Path

import joblib
import streamlit as st

from src.predictor import TARGET_NAMES, predict_bill


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "life_of_a_bill.joblib"

SAMPLE_SCENARIOS = {
    "Public health access": {
        "subject": "Mental health services, public hospitals, and community care",
        "author": "Marcos, Imee R.",
        "committee": "HEALTH AND DEMOGRAPHY",
        "scope": "National",
        "congress": "18th",
        "filing_date": date(2020, 7, 22),
    },
    "Local hospital expansion": {
        "subject": "Hospital conversion, bed capacity, and regional health services",
        "author": 'Zubiri, Juan Miguel "Migz" F.',
        "committee": "HEALTH AND DEMOGRAPHY",
        "scope": "Local",
        "congress": "17th",
        "filing_date": date(2018, 1, 29),
    },
    "Education modernization": {
        "subject": "Teacher development, digital classrooms, and public school access",
        "author": "Cayetano, Pia S.",
        "committee": "BASIC EDUCATION, ARTS AND CULTURE",
        "scope": "National",
        "congress": "18th",
        "filing_date": date(2020, 8, 10),
    },
}

MILESTONE_COPY = {
    "First Reading": (
        "The model sees a pattern most similar to bills that remained at the "
        "committee stage."
    ),
    "Second Reading": (
        "The model sees a pattern most similar to bills that reached floor "
        "deliberation."
    ),
    "Approved": (
        "The model sees a pattern most similar to bills that advanced beyond "
        "the Senate approval stage."
    ),
}


st.set_page_config(
    page_title="The Life of a Bill",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_bundle():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            "The trained model artifact is missing from this deployment."
        )
    return joblib.load(ARTIFACT_PATH)


def apply_sample() -> None:
    scenario_name = st.session_state.get("sample_selector")
    scenario = SAMPLE_SCENARIOS.get(scenario_name)
    if not scenario:
        return

    st.session_state["subject_input"] = scenario["subject"]
    st.session_state["author_input"] = scenario["author"]
    st.session_state["committee_input"] = scenario["committee"]
    st.session_state["scope_input"] = scenario["scope"]
    st.session_state["congress_input"] = scenario["congress"]
    st.session_state["filing_date_input"] = scenario["filing_date"]


def initialize_form(bundle) -> None:
    if "subject_input" in st.session_state:
        return

    first_sample = next(iter(SAMPLE_SCENARIOS.values()))
    for key, value in (
        ("subject_input", first_sample["subject"]),
        ("author_input", first_sample["author"]),
        ("committee_input", first_sample["committee"]),
        ("scope_input", first_sample["scope"]),
        ("congress_input", first_sample["congress"]),
        ("filing_date_input", first_sample["filing_date"]),
    ):
        st.session_state[key] = value

    if st.session_state["author_input"] not in bundle["authors"]:
        st.session_state["author_input"] = bundle["authors"][0]
    if st.session_state["committee_input"] not in bundle["committees"]:
        st.session_state["committee_input"] = bundle["committees"][0]


st.markdown(
    """
    <style>
    :root {
        --ink: #10243e;
        --muted: #5c6877;
        --line: #d8dee7;
        --paper: #ffffff;
        --soft: #f4f7fa;
        --teal: #007c83;
        --coral: #c74b50;
        --gold: #c88a16;
    }

    .stApp {
        background: #f4f7fa;
        color: var(--ink);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.25rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
    }

    h1 {
        font-size: clamp(2rem, 4vw, 3.4rem);
        line-height: 1.02;
        margin-bottom: 0.75rem;
    }

    .app-kicker {
        color: var(--teal);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        margin: 0 0 0.55rem;
        text-transform: uppercase;
    }

    .app-intro {
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.65;
        max-width: 760px;
        margin-bottom: 1rem;
    }

    .evidence-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        border: 1px solid var(--line);
        background: var(--paper);
        margin: 1.5rem 0 1.75rem;
    }

    .evidence-item {
        padding: 1rem 1.1rem;
        border-right: 1px solid var(--line);
    }

    .evidence-item:last-child {
        border-right: 0;
    }

    .evidence-item strong,
    .evidence-item span {
        display: block;
    }

    .evidence-item strong {
        color: var(--ink);
        font-size: 1.1rem;
    }

    .evidence-item span {
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: 0.2rem;
    }

    div[data-testid="stForm"] {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 1.15rem;
    }

    div[data-testid="stForm"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stTextArea"] label,
    div[data-testid="stDateInput"] label {
        color: var(--ink);
        font-weight: 700;
    }

    .stButton > button,
    div[data-testid="stFormSubmitButton"] button {
        min-height: 3rem;
        border-radius: 4px;
        border: 1px solid var(--teal);
        background: var(--teal);
        color: white;
        font-weight: 800;
    }

    .stButton > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        border-color: #005e63;
        background: #005e63;
        color: white;
    }

    .result-header {
        border-left: 5px solid var(--teal);
        background: var(--paper);
        border-top: 1px solid var(--line);
        border-right: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        padding: 1rem 1.15rem;
        margin: 0.5rem 0 1rem;
    }

    .result-header span {
        color: var(--muted);
        display: block;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
    }

    .result-header strong {
        color: var(--ink);
        display: block;
        font-size: 1.65rem;
        margin: 0.15rem 0 0.35rem;
    }

    .result-header p {
        color: var(--muted);
        margin: 0;
    }

    div[data-testid="stMetric"] {
        background: var(--paper);
        border: 1px solid var(--line);
        padding: 0.9rem;
    }

    div[data-testid="stAlert"] {
        border-radius: 4px;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-top: 1.25rem;
        }

        .evidence-strip {
            grid-template-columns: 1fr;
        }

        .evidence-item {
            border-bottom: 1px solid var(--line);
            border-right: 0;
        }

        .evidence-item:last-child {
            border-bottom: 0;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


try:
    bundle = load_bundle()
except Exception as error:
    st.error(f"Model unavailable: {error}")
    st.stop()

initialize_form(bundle)
metrics = bundle["metrics"]

st.markdown('<p class="app-kicker">Interactive model evaluation</p>', unsafe_allow_html=True)
st.title("The Life of a Bill")
st.markdown(
    """
    <p class="app-intro">
        Explore how an XGBoost model reads historical patterns in Philippine
        Senate bill subjects, authors, committees, scope, and filing timing.
    </p>
    """,
    unsafe_allow_html=True,
)
st.warning(
    "**Historical pattern, not a forecast.** This educational prototype has "
    f"a holdout macro F1 of {metrics['macro_f1']:.2f}. It should not be used "
    "for legal, political, investment, or public-policy decisions."
)

st.markdown(
    f"""
    <div class="evidence-strip">
        <div class="evidence-item">
            <strong>{metrics['source_rows']:,}</strong>
            <span>historical bill records modeled</span>
        </div>
        <div class="evidence-item">
            <strong>{metrics['raw_approval_rate']:.2%}</strong>
            <span>approval share in the source data</span>
        </div>
        <div class="evidence-item">
            <strong>{metrics['macro_f1']:.2f}</strong>
            <span>balanced holdout macro F1</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_column, right_column = st.columns([1.1, 0.9], gap="large")

with left_column:
    st.subheader("Evaluate a bill scenario")
    st.selectbox(
        "Try a sample scenario",
        options=list(SAMPLE_SCENARIOS),
        key="sample_selector",
        on_change=apply_sample,
    )

    with st.form("prediction_form"):
        st.text_area(
            "Bill subject or policy keywords",
            key="subject_input",
            height=116,
            help="The deployed model uses the Subjects feature from the research dataset.",
        )

        author_column, committee_column = st.columns(2)
        with author_column:
            st.selectbox(
                "Author",
                options=bundle["authors"],
                key="author_input",
            )
        with committee_column:
            st.selectbox(
                "Primary committee",
                options=bundle["committees"],
                key="committee_input",
            )

        scope_column, congress_column, date_column = st.columns(3)
        with scope_column:
            st.selectbox(
                "Scope",
                options=["National", "Local"],
                key="scope_input",
            )
        with congress_column:
            st.selectbox(
                "Congress",
                options=list(bundle["congress_start_dates"]),
                key="congress_input",
            )
        with date_column:
            st.date_input(
                "Filing date",
                key="filing_date_input",
                min_value=date(2010, 1, 1),
                max_value=date.today(),
            )

        submitted = st.form_submit_button(
            "Run historical pattern model",
            use_container_width=True,
        )

with right_column:
    st.subheader("Model result")
    result_container = st.container()

if submitted:
    try:
        result = predict_bill(
            bundle=bundle,
            subject=st.session_state["subject_input"],
            author=st.session_state["author_input"],
            committee=st.session_state["committee_input"],
            scope=st.session_state["scope_input"],
            congress_period=st.session_state["congress_input"],
            filing_date=st.session_state["filing_date_input"],
        )
    except ValueError as error:
        with result_container:
            st.error(str(error))
    except Exception:
        with result_container:
            st.error(
                "The model could not evaluate this scenario. Please retry or "
                "open the GitHub repository for the full technical workflow."
            )
    else:
        predicted = result["predicted_milestone"]
        with result_container:
            st.markdown(
                f"""
                <div class="result-header">
                    <span>Highest model probability</span>
                    <strong>{predicted}</strong>
                    <p>{MILESTONE_COPY[predicted]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for milestone in TARGET_NAMES:
                probability = result["probabilities"][milestone]
                st.markdown(f"**{milestone}** · {probability:.1%}")
                st.progress(probability)

            with st.expander("Structured signals used"):
                signals = result["signals"]
                signal_columns = st.columns(2)
                signal_columns[0].metric(
                    "Author efficacy lookup",
                    f"{signals['author_efficacy']:.1%}",
                )
                signal_columns[1].metric(
                    "Committee passage lookup",
                    f"{signals['committee_passage_rate']:.1%}",
                )
                st.caption(
                    "These lookup values are derived from the balanced training "
                    "sample. Unseen values use the balanced baseline."
                )
else:
    with result_container:
        st.info(
            "Choose a historical scenario and run the model to compare all "
            "three legislative milestone probabilities."
        )

st.divider()
model_column, evidence_column = st.columns(2, gap="large")
with model_column:
    st.subheader("Evaluation context")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Macro F1", f"{metrics['macro_f1']:.2f}")
    metric_columns[1].metric(
        "Balanced accuracy",
        f"{metrics['balanced_accuracy']:.2f}",
    )
    metric_columns[2].metric("Holdout bills", f"{metrics['holdout_rows']:,}")
    st.caption(
        "The deployment uses the same feature family as the technical report: "
        "TF-IDF subject terms, scope, timing, author efficacy, and committee rate."
    )

with evidence_column:
    st.subheader("Project evidence")
    st.markdown(
        "- [Technical source and notebooks](https://github.com/Kent0625/ML_FINAL_PROJECT)\n"
        "- Historical records from the Philippine Senate Legislative Document Room\n"
        "- XGBoost multiclass model with a reproducible training script"
    )
