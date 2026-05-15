from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from src.data_processing import PROCESSED_DATA_DIR, PROJECT_ROOT
from src.decision_engine import apply_decision_engine
from src.evaluate_model import recommended_action, risk_category
from src.feature_engineering import transform_synthetic_claim_features


PROVIDER_MODEL_PATH = PROJECT_ROOT / "models" / "provider_fraud_model.joblib"
SYNTHETIC_MODEL_PATH = PROJECT_ROOT / "models" / "synthetic_claim_fraud_model.joblib"


def load_model_bundle(model_type: str = "provider") -> Dict[str, object]:
    model_path = PROVIDER_MODEL_PATH if model_type == "provider" else SYNTHETIC_MODEL_PATH
    return joblib.load(model_path)


def _business_risk_category(probability: float) -> str:
    if probability < 0.30:
        return "Low Risk"
    if probability < 0.70:
        return "Medium Risk"
    return "High Risk"


def _business_recommended_action(probability: float) -> str:
    if probability < 0.30:
        return "Auto Approve"
    if probability < 0.70:
        return "Human Review"
    return "Investigate Before Payment"


def score_with_bundle(bundle: Dict[str, object], X: pd.DataFrame) -> pd.DataFrame:
    pipeline = bundle["pipeline"]
    threshold = float(bundle.get("threshold", 0.5))
    probabilities = pipeline.predict_proba(X)[:, 1]
    confidence = np.maximum(probabilities, 1 - probabilities)
    predictions = (probabilities >= threshold).astype(int)

    scored = X.copy()
    scored["fraud_probability"] = probabilities
    scored["predicted_fraud"] = predictions
    scored["risk_category"] = scored["fraud_probability"].apply(risk_category)
    scored["model_confidence"] = confidence
    scored["recommended_action"] = [
        recommended_action(prob, conf) for prob, conf in zip(probabilities, confidence)
    ]
    return scored.sort_values("fraud_probability", ascending=False)


def _prepare_synthetic_score_frame(claim_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    bundle = load_model_bundle("synthetic")
    transformed = transform_synthetic_claim_features(claim_df)
    missing_columns = [column for column in bundle["train_columns"] if column not in transformed.columns]
    model_input = transformed.copy()
    for column in bundle["train_columns"]:
        if column not in model_input.columns:
            model_input[column] = np.nan
    model_input = model_input[bundle["train_columns"]]

    training_feature_path = PROCESSED_DATA_DIR / "synthetic_claim_features.csv"
    if training_feature_path.exists():
        training_reference = pd.read_csv(training_feature_path, nrows=50)
        for column in bundle["train_columns"]:
            if column in training_reference.columns and pd.api.types.is_numeric_dtype(training_reference[column]):
                model_input[column] = pd.to_numeric(model_input[column], errors="coerce")

    return model_input, missing_columns


def score_provider_features(provider_features: pd.DataFrame) -> pd.DataFrame:
    bundle = load_model_bundle("provider")
    return score_with_bundle(bundle, provider_features[bundle["train_columns"]])


def score_synthetic_claims(claim_df: pd.DataFrame) -> pd.DataFrame:
    bundle = load_model_bundle("synthetic")
    X, _ = _prepare_synthetic_score_frame(claim_df)
    return score_with_bundle(bundle, X)


def score_synthetic_claims_with_context(claim_df: pd.DataFrame) -> pd.DataFrame:
    bundle = load_model_bundle("synthetic")
    original = claim_df.copy()
    X, _ = _prepare_synthetic_score_frame(claim_df)
    scored = score_with_bundle(bundle, X)
    output = original.reset_index(drop=True).copy()
    output["Model_Fraud_Probability"] = scored["fraud_probability"].values
    output["Predicted_Fraud"] = scored["predicted_fraud"].values
    output["Risk_Category"] = output["Model_Fraud_Probability"].apply(_business_risk_category)
    output["Recommended_Action"] = output["Model_Fraud_Probability"].apply(_business_recommended_action)
    output["Model_Confidence"] = scored["model_confidence"].values
    output = apply_decision_engine(output)
    return output.sort_values("Model_Fraud_Probability", ascending=False)


def get_missing_synthetic_scoring_columns(claim_df: pd.DataFrame) -> List[str]:
    _, missing = _prepare_synthetic_score_frame(claim_df)
    return missing
