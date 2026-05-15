from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def risk_category(probability: float) -> str:
    if probability < 0.30:
        return "Low Risk / Auto Approve"
    if probability < 0.70:
        return "Medium Risk / Human Review"
    return "High Risk / Investigate Before Payment"


def recommended_action(probability: float, confidence: Optional[float] = None) -> str:
    if confidence is not None and confidence < 0.80:
        return "Human verification required because model confidence is below 80%."
    if probability < 0.30:
        return "Auto approve with routine monitoring."
    if probability < 0.70:
        return "Send to human reviewer before payment."
    return "Place on payment hold and require investigator review before payment."


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def evaluate_thresholds(y_true: pd.Series, y_proba: np.ndarray, thresholds: Iterable[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_proba)
        rows.append({"threshold": threshold, **metrics})
    frame = pd.DataFrame(rows)
    frame["business_score"] = (
        frame["recall"] * 0.45 + frame["precision"] * 0.35 + frame["f1"] * 0.20
    )
    return frame.sort_values("threshold").reset_index(drop=True)


def choose_business_threshold(threshold_metrics: pd.DataFrame) -> float:
    best_row = threshold_metrics.sort_values(
        ["business_score", "recall", "precision", "f1"], ascending=False
    ).iloc[0]
    return float(best_row["threshold"])


def top_flagged_precision(y_true: pd.Series, y_proba: np.ndarray, top_fraction: float = 0.10) -> float:
    top_n = max(1, int(len(y_proba) * top_fraction))
    top_indices = np.argsort(y_proba)[::-1][:top_n]
    return float(np.mean(np.asarray(y_true)[top_indices]))


def save_metrics_report(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    metrics_dir: Path,
) -> Dict[str, object]:
    metrics = compute_metrics(y_true, y_pred, y_proba)
    metrics["top_decile_precision"] = top_flagged_precision(y_true, y_proba, top_fraction=0.10)
    metrics["classification_report"] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_path = metrics_dir / f"{model_name}_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def save_confusion_matrix_chart(
    y_true: pd.Series, y_pred: np.ndarray, output_path: Path, title: str
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks([0, 1], labels=["Pred No Fraud", "Pred Fraud"])
    ax.set_yticks([0, 1], labels=["Actual No Fraud", "Actual Fraud"])
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_roc_pr_charts(
    y_true: pd.Series, y_proba: np.ndarray, roc_path: Path, pr_path: Path, prefix: str
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{prefix} ROC Curve")
    ax.legend()
    fig.tight_layout()
    roc_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}", color="darkorange")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{prefix} Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    pr_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pr_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance_chart(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Path,
    title: str,
    feature_names: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Save model feature importance using native or permutation importance."""
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        names = list(feature_names) if feature_names is not None else list(X.columns)
    else:
        result = permutation_importance(model, X, y, n_repeats=5, random_state=42, scoring="f1")
        values = result.importances_mean
        names = list(X.columns)

    importance_df = (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1], color="seagreen")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return importance_df
