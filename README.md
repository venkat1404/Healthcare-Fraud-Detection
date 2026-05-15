# AI-Powered Healthcare Insurance Fraud Detection System

This project builds a practical, business-friendly fraud detection workflow for healthcare claims and providers. The main model uses the Kaggle `Healthcare Provider Fraud Detection Analysis` dataset to predict whether a provider is potentially fraudulent. A second synthetic claim-level model is included for comparison and for an easier Streamlit demo experience.

## Project Structure

```text
healthcare-fraud-ai/
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
├── models/
├── outputs/
│   ├── charts/
│   ├── metrics/
├── requirements.txt
├── README.md
└── final_project_summary.md
```

## Where To Place Raw Data

Place the Kaggle CSV files in [data/raw](/Users/amoux/Desktop/Healthcare%20Project/data/raw). The code is flexible and can also discover the files if they are still in `Dataset/` or the repo root, but `data/raw` is the intended location.

Expected source files:

- `Train_Beneficiarydata...csv`
- `Train_Inpatientdata...csv`
- `Train_Outpatientdata...csv`
- `Train...csv` with provider fraud labels
- `synthetic_health_claims.csv`

## What The Project Does

- Builds provider-level features from inpatient, outpatient, beneficiary, and provider-label data.
- Trains and compares Logistic Regression, Random Forest, and XGBoost if installed, with sklearn boosting fallback if not.
- Evaluates fraud detection with accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve, and precision-recall curve.
- Tests business thresholds at `0.30`, `0.50`, and `0.70`.
- Applies a Decision Automation Engine that converts fraud probabilities into operational routing actions such as auto approve, secondary rules check, human review queue, priority investigation, and payment hold plus mandatory investigator review.
- Saves trained models in [models](/Users/amoux/Desktop/Healthcare%20Project/models).
- Exports charts and metrics to [outputs](/Users/amoux/Desktop/Healthcare%20Project/outputs).
- Includes a Streamlit app for manual and batch fraud scoring.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
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

Launch the Streamlit app:

```bash
streamlit run streamlit_app/app.py
```

## How To Use The App

Recommended team workflow:

1. Launch the app with:

```bash
streamlit run streamlit_app/app.py
```

2. Open `Batch Scoring` first.
3. Upload one of the test files from [data/demo](/Users/amoux/Desktop/Healthcare%20Project/data/demo).
4. If the file is already scored, the dashboard will use those scores directly.
5. If the file is raw claim data, click the scoring button so the saved model can score it.
6. After a file is loaded, it stays active across the dashboard tabs during that session.
7. Move through the other pages to review fraud risk, routing decisions, case details, model performance, business impact, and responsible AI controls.

Recommended test files:

- [demo_upload_scored_claims.csv](/Users/amoux/Desktop/Healthcare%20Project/data/demo/demo_upload_scored_claims.csv)
  Use this to test the already-scored upload flow.
- [demo_upload_raw_synthetic_claims.csv](/Users/amoux/Desktop/Healthcare%20Project/data/demo/demo_upload_raw_synthetic_claims.csv)
  Use this to test the raw upload plus model-scoring flow.
- [demo_scored_claims.csv](/Users/amoux/Desktop/Healthcare%20Project/data/demo/demo_scored_claims.csv)
  Use this as the larger presentation dataset if you want a fuller dashboard view.

## Dashboard Demo Walkthrough

Suggested professor/video flow:

1. Start on `Overview` and explain the fraud risk summary, high-risk queue, and estimated exposure.
2. Go to `Batch Scoring` and show how a new file can be uploaded and scored.
3. Go to `Case Review` and open one suspicious claim.
4. Explain why the model flagged the claim using the risk driver and supporting features.
5. Go to `Model Insights` and explain model performance using recall, precision, F1, ROC-AUC, and threshold tuning.
6. Go to `Business Impact` and show estimated fraud savings versus review cost.
7. End with `Responsible AI` and explain human-review safeguards.

For the cleanest walkthrough, upload [demo_scored_claims.csv](/Users/amoux/Desktop/Healthcare%20Project/data/demo/demo_scored_claims.csv) in `Batch Scoring` first, then move through the rest of the pages.

## Presentation Mode

The sidebar includes a `Presentation Mode` toggle for cleaner professor walkthroughs.

When enabled, the dashboard:

- shows fewer rows in tables
- hides or reduces advanced filters where possible
- keeps the pages cleaner for video demos
- emphasizes the most important KPIs and visuals first

## Notebook Purpose

- `01_main_provider_fraud_model.ipynb`: main business model using provider-level fraud prediction.
- `02_synthetic_claim_demo_model.ipynb`: claim-level comparison model using the synthetic dataset.

## Streamlit Pages

- `Overview`: fraud command-center summary with KPIs, charts, and top risky cases
- `Batch Scoring`: upload and score new claims or review the built-in demo file
- `Decision Engine`: editable threshold routing layer with secondary rules for borderline claims
- `Case Review`: investigator-style view of one suspicious claim
- `Model Insights`: model metrics, charts, and threshold comparison
- `Business Impact`: interactive savings and cost calculator
- `Responsible AI`: human-in-the-loop and fairness guardrails
- `About`: business framing, datasets, approach, limitations, and future roadmap

## Upload Behavior

- The app does not auto-load a dataset on startup.
- A dataset becomes active after you upload a scored CSV, or after you upload a raw CSV and run model scoring.
- Once loaded, that dataset stays active across tabs for the rest of the session.
- The sidebar status area shows:
  - whether uploaded data is loaded
  - total records
  - human reviews required
  - payment holds

## Presentation Mode

Use the `Presentation Mode` checkbox in the sidebar when recording or presenting.

It:

- reduces visible table rows
- hides some advanced filters
- keeps the pages cleaner for a walkthrough
- emphasizes KPIs and charts over detailed data grids

## Troubleshooting

- If the dashboard pages look empty, go to `Batch Scoring` and upload a dataset first.
- If a raw upload does not score, use the scoring diagnostics section on `Batch Scoring` to see which model-input columns are missing.
- If you only want to test the UI path, upload `demo_upload_scored_claims.csv`.
- If you want to test the full model-scoring path, upload `demo_upload_raw_synthetic_claims.csv`.

## Limitations

- The primary Kaggle dataset predicts provider-level risk, not final legal fraud findings.
- Fraud labels are imbalanced, so precision and recall tradeoffs matter more than raw accuracy.
- The synthetic dataset is useful for demos, but it is not real claims data.
- A flagged claim or provider should trigger human investigation, not an autonomous payment decision.
- Results may vary if raw files use different column names or missing fields, though the code is built to handle common schema differences gracefully.
