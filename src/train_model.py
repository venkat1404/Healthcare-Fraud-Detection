from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_processing import (
    PROJECT_ROOT,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    ensure_directories,
    infer_target_column,
    load_primary_datasets,
    load_synthetic_dataset,
)
from src.evaluate_model import (
    choose_business_threshold,
    compute_metrics,
    evaluate_thresholds,
    save_confusion_matrix_chart,
    save_feature_importance_chart,
    save_metrics_report,
    save_roc_pr_charts,
)
from src.feature_engineering import build_provider_level_dataset, prepare_synthetic_claim_dataset


def _maybe_get_xgboost():
    try:
        from xgboost import XGBClassifier

        return XGBClassifier
    except Exception:
        return None


def _train_provider_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Pipeline]:
    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    models: Dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="liblinear",
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=42,
                        class_weight="balanced",
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }

    xgb_cls = _maybe_get_xgboost()
    if xgb_cls is not None:
        fraud_rate = y_train.mean()
        safe_rate = max(fraud_rate, 1e-6)
        scale_pos_weight = (1 - safe_rate) / safe_rate
        models["xgboost"] = Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "model",
                    xgb_cls(
                        n_estimators=250,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=42,
                        scale_pos_weight=scale_pos_weight,
                    ),
                ),
            ]
        )
    else:
        models["hist_gradient_boosting"] = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", HistGradientBoostingClassifier(random_state=42, max_depth=6)),
            ]
        )
        models["gradient_boosting"] = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        )

    return models


def _train_synthetic_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Pipeline]:
    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return {
        "synthetic_logistic_regression": Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="liblinear",
                    ),
                ),
            ]
        ),
        "synthetic_random_forest": Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        random_state=42,
                        class_weight="balanced",
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def _score_models(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    rows: List[Dict[str, float]] = []
    predictions: Dict[str, Dict[str, np.ndarray]] = {}

    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.50).astype(int)
        metrics = compute_metrics(y_test, y_pred, y_proba)
        rows.append({"model": model_name, **metrics})
        predictions[model_name] = {"pipeline": pipeline, "y_proba": y_proba, "y_pred": y_pred}

    leaderboard = pd.DataFrame(rows).sort_values(["roc_auc", "f1", "recall"], ascending=False)
    return leaderboard, predictions


def train_primary_provider_model() -> Dict[str, object]:
    ensure_directories()
    datasets = load_primary_datasets()
    provider_df = build_provider_level_dataset(
        datasets["beneficiary"], datasets["inpatient"], datasets["outpatient"], datasets["labels"]
    )

    provider_df.to_csv(PROCESSED_DATA_DIR / "provider_level_features.csv", index=False)

    target = provider_df["fraud"]
    features = provider_df.drop(columns=["fraud", "Provider"])

    X_train, X_test, y_train, y_test, provider_train, provider_test = train_test_split(
        features,
        target,
        provider_df["Provider"],
        test_size=0.25,
        random_state=42,
        stratify=target,
    )

    models = _train_provider_models(X_train, y_train)
    leaderboard, predictions = _score_models(models, X_train, X_test, y_train, y_test)
    leaderboard.to_csv(PROJECT_ROOT / "outputs" / "metrics" / "provider_model_leaderboard.csv", index=False)

    best_model_name = leaderboard.iloc[0]["model"]
    best_bundle = predictions[best_model_name]
    best_pipeline = best_bundle["pipeline"]
    y_proba = best_bundle["y_proba"]

    threshold_metrics = evaluate_thresholds(y_test, y_proba, thresholds=[0.30, 0.50, 0.70])
    chosen_threshold = choose_business_threshold(threshold_metrics)
    threshold_metrics.to_csv(
        PROJECT_ROOT / "outputs" / "metrics" / "provider_threshold_metrics.csv", index=False
    )

    y_pred = (y_proba >= chosen_threshold).astype(int)
    save_metrics_report(
        "provider_best_model", y_test, y_pred, y_proba, PROJECT_ROOT / "outputs" / "metrics"
    )
    save_roc_pr_charts(
        y_test,
        y_proba,
        PROJECT_ROOT / "outputs" / "charts" / "provider_roc_curve.png",
        PROJECT_ROOT / "outputs" / "charts" / "provider_precision_recall_curve.png",
        "Provider Fraud Model",
    )
    save_confusion_matrix_chart(
        y_test,
        y_pred,
        PROJECT_ROOT / "outputs" / "charts" / "provider_confusion_matrix.png",
        "Provider Fraud Confusion Matrix",
    )

    tree_candidates = ["xgboost", "random_forest", "hist_gradient_boosting", "gradient_boosting"]
    selected_tree_model_name = next(
        (name for name in leaderboard["model"].tolist() if name in tree_candidates),
        best_model_name,
    )
    tree_pipeline = predictions[selected_tree_model_name]["pipeline"]
    tree_model_step = tree_pipeline.named_steps["model"]
    tree_preprocess_step = tree_pipeline.named_steps["preprocess"]
    transformed_names = tree_preprocess_step.get_feature_names_out()
    transformed_X_test = tree_preprocess_step.transform(X_test)
    if hasattr(transformed_X_test, "toarray"):
        transformed_X_test = transformed_X_test.toarray()
    transformed_X_test_df = pd.DataFrame(transformed_X_test, columns=transformed_names)
    importance_df = save_feature_importance_chart(
        tree_model_step,
        transformed_X_test_df,
        y_test,
        PROJECT_ROOT / "outputs" / "charts" / "provider_feature_importance.png",
        f"{selected_tree_model_name} Top Feature Importance",
        feature_names=transformed_names,
    )
    importance_df.to_csv(
        PROJECT_ROOT / "outputs" / "metrics" / "provider_feature_importance.csv", index=False
    )

    bundle = {
        "model_name": best_model_name,
        "pipeline": best_pipeline,
        "tree_model_name_for_explanations": selected_tree_model_name,
        "threshold": chosen_threshold,
        "threshold_metrics": threshold_metrics.to_dict(orient="records"),
        "feature_importance": importance_df.to_dict(orient="records"),
        "train_columns": features.columns.tolist(),
        "provider_ids_test": provider_test.tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
        "notes": {
            "high_value_claim_review": "Claims above $10,000 flagged as suspicious require human review before payment action.",
            "low_confidence_review": "Predictions below 80% confidence require human verification.",
            "responsible_ai": "The model supports investigators and should not be used for autonomous claim denial.",
        },
    }
    joblib.dump(bundle, PROJECT_ROOT / "models" / "provider_fraud_model.joblib")
    return bundle


def train_synthetic_demo_model() -> Dict[str, object]:
    ensure_directories()
    synthetic_df = load_synthetic_dataset()
    target_column = infer_target_column(synthetic_df)
    X, y = prepare_synthetic_claim_dataset(synthetic_df, target_column)
    processed = X.copy()
    processed[target_column] = y
    processed.to_csv(PROCESSED_DATA_DIR / "synthetic_claim_features.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    models = _train_synthetic_models(X_train, y_train)
    leaderboard, predictions = _score_models(models, X_train, X_test, y_train, y_test)
    leaderboard.to_csv(PROJECT_ROOT / "outputs" / "metrics" / "synthetic_model_leaderboard.csv", index=False)

    best_model_name = leaderboard.iloc[0]["model"]
    best_bundle = predictions[best_model_name]
    best_pipeline = best_bundle["pipeline"]
    y_proba = best_bundle["y_proba"]
    threshold_metrics = evaluate_thresholds(y_test, y_proba, thresholds=[0.30, 0.50, 0.70])
    chosen_threshold = choose_business_threshold(threshold_metrics)
    threshold_metrics.to_csv(
        PROJECT_ROOT / "outputs" / "metrics" / "synthetic_threshold_metrics.csv", index=False
    )

    y_pred = (y_proba >= chosen_threshold).astype(int)
    save_metrics_report(
        "synthetic_best_model", y_test, y_pred, y_proba, PROJECT_ROOT / "outputs" / "metrics"
    )
    save_roc_pr_charts(
        y_test,
        y_proba,
        PROJECT_ROOT / "outputs" / "charts" / "synthetic_roc_curve.png",
        PROJECT_ROOT / "outputs" / "charts" / "synthetic_precision_recall_curve.png",
        "Synthetic Claims Fraud Model",
    )
    save_confusion_matrix_chart(
        y_test,
        y_pred,
        PROJECT_ROOT / "outputs" / "charts" / "synthetic_confusion_matrix.png",
        "Synthetic Claims Fraud Confusion Matrix",
    )

    bundle = {
        "model_name": best_model_name,
        "pipeline": best_pipeline,
        "threshold": chosen_threshold,
        "threshold_metrics": threshold_metrics.to_dict(orient="records"),
        "target_column": target_column,
        "train_columns": X.columns.tolist(),
        "notes": {
            "purpose": "Synthetic claim-level model for demo/comparison and easier manual input in Streamlit."
        },
    }
    joblib.dump(bundle, PROJECT_ROOT / "models" / "synthetic_claim_fraud_model.joblib")
    return bundle


def main() -> None:
    provider_bundle = train_primary_provider_model()
    synthetic_bundle = train_synthetic_demo_model()

    summary = {
        "provider_model": provider_bundle["model_name"],
        "provider_threshold": provider_bundle["threshold"],
        "synthetic_model": synthetic_bundle["model_name"],
        "synthetic_threshold": synthetic_bundle["threshold"],
        "raw_data_dir": str(RAW_DATA_DIR),
    }
    (PROJECT_ROOT / "outputs" / "metrics" / "training_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
