# Life of a Bill Live Demo Design

## Goal

Turn the existing notebook-based legislative machine learning project into a public app that portfolio visitors can use and evaluate.

## Approaches Considered

### 1. Link only to the notebook

This preserves the research exactly but does not let a recruiter try the model without setting up Python. It is the lowest-effort option and the weakest product experience.

### 2. Train the model whenever the app starts

This keeps the artifact out of Git but makes cold starts slow and creates avoidable deployment failures. Hugging Face free hardware may also retrain the model after sleep.

### 3. Ship a reproducible trained artifact with a Streamlit app

This is the selected approach. A training script rebuilds the artifact from the four Senate CSV files, while the deployed app loads the saved artifact immediately. It gives visitors a fast demo and leaves the full modeling workflow auditable in GitHub.

## App Experience

The app is a focused historical-pattern explorer named **The Life of a Bill**.

Visitors provide:

- Bill subject or policy keywords
- Author
- Primary committee
- Scope: national or local
- Congress period
- Filing date

The app returns:

- Predicted milestone: First Reading, Second Reading, or Approved
- Probability for all three classes
- A short explanation of the strongest structured signals used for the request
- A clear disclaimer that the output is experimental and not legal, political, or policy advice

Sample bill scenarios let visitors evaluate the app without knowing Senate metadata.

## Modeling Architecture

The implementation follows the notebook's final feature family:

- TF-IDF features from the `Subjects` field
- National/local scope
- Days since the start of the selected Congress
- Historical author efficacy lookup
- Historical committee passage-rate lookup
- XGBoost multiclass classification

The model is trained on a balanced sample of mapped legislative milestones, matching the notebook's class-balancing strategy. Training statistics, class labels, lookup tables, vectorizer, feature names, and model are stored together in one versioned `joblib` artifact.

The training command is reproducible and separate from the deployed app. The app never retrains during startup.

## Hosting

The public deployment target is a Hugging Face Docker Space:

`Kent0625/life-of-a-bill`

The Docker image runs Streamlit on port `8501`. No API key or paid service is required.

## Portfolio Integration

The existing `projects/life-of-a-bill.html` page gains:

- A primary **Try Live Model** action
- A responsive embedded app section
- An **Open full-screen** fallback link
- Copy that explains what visitors can test and how to interpret the result

On narrow screens, the external full-screen app remains the primary usable path.

## Error Handling

- Empty subject text is rejected with a useful prompt.
- Unknown authors and committees fall back to the model's historical global approval rate.
- Invalid dates are normalized to the selected Congress start.
- Missing or incompatible model artifacts produce a clear deployment error.
- The app identifies Hugging Face cold starts without implying that the model is broken.

## Verification

Automated checks cover:

- Status mapping and text cleaning
- Date-to-congress timing conversion
- Unknown-category fallbacks
- Prediction output shape and probability normalization
- Required deployment files
- Required portfolio links and embedded-app markup

Manual browser checks cover:

- Model submission and result rendering
- Mobile, tablet, and desktop layouts
- Portfolio-to-app navigation
- Live Hugging Face and Render URLs

