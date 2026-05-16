# AI-Powered Healthcare Insurance Fraud Detection System

> **BUDT751 — Information Systems Program**
> Robert H. Smith School of Business, University of Maryland
> Spring 2026 · Group 1

**Team Members:** Anirudh Patil · Anish Tumla · Arya Kadarkar · Ashley Lun · Dhvani Khatri · Rishika Methi · Venkat Gollangi

---

This project builds a practical, business-friendly fraud detection workflow for healthcare claims and providers. The main model uses the Kaggle `Healthcare Provider Fraud Detection Analysis` dataset to predict whether a provider is potentially fraudulent. A second synthetic claim-level model is included for comparison and for an easier Streamlit demo experience.

## Project Structure

```text
Healthcare Project/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── demo/
│   │   └── demo_scored_claims.csv
├── notebooks/
│   ├── 01_main_provider_fraud_model.ipynb
│   ├── 02_synthetic_claim_demo_model.ipynb
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   ├── generate_demo_data.py
├── streamlit_app/
│   ├── app.py
├── .streamlit/
│   └── config.toml
├── models/
├── outputs/
│   ├── charts/
│   ├── metrics/
├── requirements.txt
├── README.md
└── final_project_summary.md
```

## Where To Place Raw Data

Place the Kaggle CSV files in `data/raw`. The code is flexible and can also discover the files if they are still in `Dataset/` or the repo root, but `data/raw` is the intended location.

Expected source files:

- `Train_Beneficiarydata...csv`
- `Train_Inpatientdata...csv`
- `Train_Outpatientdata...csv`
- `Train...csv` with provider fraud labels
- `synthetic_health_claims.csv`

> **Note:** The large model file `models/synthetic_claim_fraud_model.joblib` (184MB) is excluded from this repo due to GitHub's file size limit. Run the training pipeline to regenerate it locally.

## What The Project Does

- Builds provider-level features from inpatient, outpatient, beneficiary, and provider-label data.
- Trains and compares Logistic Regression, Random Forest, and XGBoost if installed, with sklearn boosting fallback if not.
- Evaluates fraud detection with accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve, and precision-recall curve.
- Tests business thresholds at `0.30`, `0.50`, and `0.70`.
- Applies a Decision Automation Engine that converts fraud probabilities into five operational routing actions: auto approve, secondary rules check, human review queue, priority investigation, and payment hold + mandatory investigator review.
- Saves trained models in `models/`.
- Exports charts and metrics to `outputs/`.
- Includes a dark-themed Streamlit dashboard (Command Center) for manual and batch fraud scoring.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

If `xgboost` is not available on your machine, the training script will automatically fall back to sklearn boosting models.

## How To Run The Project

Run the full training pipeline:

```bash
python3 -m src.train_model
```

Generate the dashboard demo CSV:

```bash
python3 src/generate_demo_data.py
```

Open the notebooks:

```bash
jupyter notebook
```

Launch the Streamlit dashboard:

```bash
streamlit run streamlit_app/app.py
```

## How To Use The App

1. Launch the app with `streamlit run streamlit_app/app.py`
2. Open **Batch Scoring** first.
3. Upload one of the test files from `data/demo/`.
4. If the file already contains model scores, the dashboard will use them directly.
5. If the file is raw claim data, click the scoring button so the saved model can score it.
6. After a file is loaded, it stays active across all dashboard tabs for that session.
7. Navigate through the pages using the **Command Center** sidebar.

Recommended test files:

- `data/demo/demo_upload_scored_claims.csv` — test the already-scored upload flow (fastest)
- `data/demo/demo_upload_raw_synthetic_claims.csv` — test the raw upload + model-scoring flow
- `data/demo/demo_scored_claims.csv` — larger presentation dataset for a fuller dashboard view

## Dashboard Pages

The sidebar is split into two sections:

**Main navigation:**

| Page | Description |
|------|-------------|
| Overview | Fraud command-center with KPIs, risk distribution, decision action charts, and top risky cases |
| Batch Scoring | Upload and score new claims; filter, review, and download results |
| Decision Engine | Editable threshold routing with secondary rules for borderline claims; live routing summary |
| Case Review | Investigator-style deep-dive into a single high-risk claim with risk gauge and contributing features |
| Business Impact | Interactive calculator for fraud savings vs. review cost with monthly and annual projections |
| Responsible AI | Human-in-the-loop rules, false positive/negative costs, bias risks, and mitigation controls |

**Technical Reference** (collapsed expander at the bottom of the sidebar):

| Page | Description |
|------|-------------|
| Model Insights | Model leaderboard, confusion matrix, ROC curve, precision-recall curve, and feature importance |
| About | Business framing, datasets used, AI approach, model selection, limitations, and future roadmap |

## Dashboard Demo Walkthrough

Suggested professor/presentation flow:

1. Start on **Overview** — explain the fraud risk summary, high-risk queue, and estimated exposure.
2. Go to **Batch Scoring** — show uploading and scoring a new file.
3. Go to **Case Review** — open a suspicious claim and explain the risk driver and contributing features.
4. Go to **Decision Engine** — show threshold controls and how probabilities map to routing actions.
5. Go to **Business Impact** — walk through fraud savings vs. review cost estimates.
6. End with **Responsible AI** — explain human-review safeguards and fairness controls.
7. Expand **Technical Reference → Model Insights** for the technical model performance deep-dive.

For the cleanest walkthrough, upload `demo_scored_claims.csv` in Batch Scoring first, then navigate through the pages.

## Presentation Mode

The sidebar includes a **Presentation Mode** toggle. When enabled it reduces visible table rows, hides advanced filters, and keeps pages cleaner for video demos and live walkthroughs.

## Model Performance

| Model | ROC-AUC | Recall | Precision | F1 |
|-------|---------|--------|-----------|-----|
| Provider Fraud (Logistic Regression) | 0.951 | 0.874 | 0.442 | 0.587 |
| Synthetic Claim Fraud (Random Forest) | 0.837 | 0.540 | 0.730 | 0.621 |

## Decision Engine Routing

| Fraud Probability | Tier | Action |
|---|---|---|
| < 0.40 | Low Risk | Auto Approve |
| 0.40 – 0.60 | Borderline | Secondary Rules Check |
| 0.60 – 0.70 | Elevated | Human Review Queue |
| 0.70 – 0.90 | High Risk | Priority Investigation |
| > 0.90 | Critical | Payment Hold + Mandatory Investigator Review |

Claims over $10,000 require human review regardless of score.

## Troubleshooting

- If dashboard pages look empty, go to **Batch Scoring** and upload a dataset first.
- If a raw upload does not score, use the scoring diagnostics expander on Batch Scoring to see which model-input columns are missing.
- To test just the UI, upload `demo_upload_scored_claims.csv`.
- To test the full model-scoring path, upload `demo_upload_raw_synthetic_claims.csv`.
- If the sidebar looks dark/invisible, make sure `.streamlit/config.toml` exists in the project root.

## Limitations

- The primary Kaggle dataset predicts provider-level risk, not final legal fraud findings.
- Fraud labels are imbalanced, so precision and recall tradeoffs matter more than raw accuracy.
- The synthetic dataset is useful for demos but is not real claims data.
- A flagged claim or provider should trigger human investigation, not an autonomous payment decision.
- Results may vary if raw files use different column names or missing fields, though the code handles common schema differences gracefully.
