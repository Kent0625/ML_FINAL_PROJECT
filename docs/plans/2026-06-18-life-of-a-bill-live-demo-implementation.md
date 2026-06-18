# Life of a Bill Live Demo Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and deploy a public Streamlit predictor for the legislative ML project, then embed it in the portfolio case-study page.

**Architecture:** A reusable Python module owns preprocessing and inference. A separate training script creates a versioned `joblib` bundle from the four historical CSV files. Streamlit loads that bundle without retraining, and the static portfolio embeds the Hugging Face app with a full-screen fallback.

**Tech Stack:** Python 3.11, pandas, scikit-learn, XGBoost, joblib, Streamlit, Docker, Hugging Face Spaces, static HTML/CSS, PowerShell verification scripts.

---

### Task 1: Create the Tested Prediction Core

**Files:**
- Create: `src/__init__.py`
- Create: `src/predictor.py`
- Create: `tests/test_predictor.py`

**Step 1: Write failing tests**

Cover status mapping, legislative text cleaning, Congress timing, unseen author and committee fallbacks, and normalized three-class prediction output.

**Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest tests/test_predictor.py -q
```

Expected: failure because `src.predictor` does not exist.

**Step 3: Implement the minimal prediction module**

Add pure preprocessing helpers, a serializable model bundle, and one inference entry point.

**Step 4: Run tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_predictor.py -q
```

Expected: all predictor tests pass.

**Step 5: Commit**

```powershell
git add src tests
git commit -m "feat: add legislative prediction core"
```

### Task 2: Add Reproducible Model Training

**Files:**
- Create: `train_model.py`
- Create: `tests/test_training.py`
- Create: `artifacts/.gitkeep`
- Create after training: `artifacts/life_of_a_bill.joblib`

**Step 1: Write failing training tests**

Test that the four datasets load, mapped classes are balanced for training, required lookup tables are present, and the artifact can produce a three-class prediction.

**Step 2: Run tests and verify RED**

```powershell
python -m pytest tests/test_training.py -q
```

Expected: failure because the training module does not exist.

**Step 3: Implement model training**

Reproduce the notebook feature family with a deterministic balanced sample and the documented XGBoost settings. Save model metadata and evaluation metrics with the artifact.

**Step 4: Train the artifact**

```powershell
python train_model.py
```

Expected: `artifacts/life_of_a_bill.joblib` is created and summary metrics are printed.

**Step 5: Verify tests**

```powershell
python -m pytest tests/test_training.py -q
```

Expected: all training tests pass.

**Step 6: Commit**

```powershell
git add train_model.py tests artifacts
git commit -m "feat: add reproducible XGBoost artifact"
```

### Task 3: Build the Streamlit Product

**Files:**
- Create: `app.py`
- Create: `requirements.txt`
- Create: `Dockerfile`
- Modify: `README.md`
- Create: `tools/verify_deployment.ps1`

**Step 1: Add a failing deployment verifier**

Require Hugging Face metadata, Docker port `8501`, the saved artifact, Streamlit entry point, disclaimer text, sample scenarios, and probability result labels.

**Step 2: Run verifier and verify RED**

```powershell
powershell -ExecutionPolicy Bypass -File tools/verify_deployment.ps1
```

Expected: failure because deployment files are missing.

**Step 3: Implement the app and deployment files**

Create a professional responsive Streamlit interface, cached artifact loading, form validation, probability display, model context, and evaluation disclaimer.

**Step 4: Run automated checks**

```powershell
python -m pytest -q
powershell -ExecutionPolicy Bypass -File tools/verify_deployment.ps1
```

Expected: all checks pass.

**Step 5: Run the app locally**

```powershell
streamlit run app.py --server.port 8502 --server.address 127.0.0.1
```

Verify one sample prediction and one custom prediction.

**Step 6: Commit**

```powershell
git add app.py requirements.txt Dockerfile README.md tools
git commit -m "feat: add interactive Streamlit demo"
```

### Task 4: Deploy the Hugging Face Space

**Files:**
- No new source files expected.

**Step 1: Create the public Space**

Create `Kent0625/life-of-a-bill` with the Docker SDK.

**Step 2: Add the Space Git remote**

```powershell
git remote add huggingface https://huggingface.co/spaces/Kent0625/life-of-a-bill
```

**Step 3: Push the app**

```powershell
git push huggingface main
```

**Step 4: Verify the deployment**

Open:

`https://kent0625-life-of-a-bill.hf.space`

Expected: the app loads, accepts a sample, and returns three probabilities.

### Task 5: Integrate the Live App into the Portfolio

**Files:**
- Modify: `../portfolio/projects/life-of-a-bill.html`
- Modify: `../portfolio/style.css`
- Modify: `../portfolio/tools/verify-project-pages.ps1`

**Step 1: Add failing portfolio checks**

Require the Hugging Face URL, a **Try Live Model** button, responsive embed markup, full-screen fallback, and accessible iframe title.

**Step 2: Run checks and verify RED**

```powershell
powershell -ExecutionPolicy Bypass -File tools/verify-project-pages.ps1
```

Expected: failure because the live app is not integrated.

**Step 3: Implement the case-study integration**

Add the live-model CTA and an unframed responsive demo band that fits the existing data.world-inspired visual system.

**Step 4: Verify locally**

Run all portfolio verifiers and inspect the page at phone, tablet, and desktop widths.

**Step 5: Commit and push**

```powershell
git add projects/life-of-a-bill.html style.css tools/verify-project-pages.ps1
git commit -m "feat: embed Life of a Bill live demo"
git push origin main
```

### Task 6: Final End-to-End Verification

**Step 1: Verify both repositories are clean**

Run `git status --short --branch` in both repositories.

**Step 2: Verify live application behavior**

Submit a sample input on the Hugging Face Space and confirm the milestone and all three probabilities render.

**Step 3: Verify live portfolio behavior**

Open the Render portfolio, navigate to The Life of a Bill, use the embedded app, and test the full-screen fallback.

**Step 4: Verify responsive layouts**

Check phone, tablet, and desktop viewports for overflow, readable controls, and usable app height.

