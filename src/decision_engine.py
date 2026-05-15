from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS: Dict[str, float] = {
    "low_upper": 0.40,
    "borderline_upper": 0.60,
    "elevated_upper": 0.70,
    "high_upper": 0.90,
}


@dataclass
class DecisionThresholds:
    low_upper: float = 0.40
    borderline_upper: float = 0.60
    elevated_upper: float = 0.70
    high_upper: float = 0.90

    def as_dict(self) -> Dict[str, float]:
        return {
            "low_upper": self.low_upper,
            "borderline_upper": self.borderline_upper,
            "elevated_upper": self.elevated_upper,
            "high_upper": self.high_upper,
        }


def thresholds_are_valid(thresholds: Dict[str, float]) -> bool:
    return (
        0 <= thresholds["low_upper"] < thresholds["borderline_upper"] < thresholds["elevated_upper"] < thresholds["high_upper"] <= 1
    )


def _safe_numeric(series: pd.Series | None, default: float = 0.0) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _safe_bool(series: pd.Series | None, default: bool = False) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    if series.dtype == bool:
        return series.fillna(default)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin(["true", "1", "yes", "y"])


def _get_series(df: pd.DataFrame, column: str, default_value) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default_value] * len(df), index=df.index)


def _secondary_rule_components(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    claim_amount = _safe_numeric(_get_series(df, "Claim_Amount", 0))
    reimbursed = _safe_numeric(_get_series(df, "Reimbursed_Amount", 0))
    claim_count = _safe_numeric(_get_series(df, "Claim_Count_Last_12_Months", 0))
    unique_patients = _safe_numeric(_get_series(df, "Unique_Patients_Last_12_Months", 0))
    diagnosis_codes = _safe_numeric(_get_series(df, "Unique_Diagnosis_Codes", 0))
    procedure_codes = _safe_numeric(_get_series(df, "Unique_Procedure_Codes", 0))
    chronic_conditions = _safe_numeric(_get_series(df, "Chronic_Condition_Count", 0))
    prior_fraud = _safe_bool(_get_series(df, "Prior_Fraud_Flag", False))
    deductible = _safe_numeric(_get_series(df, "Deductible_Amount", 0))

    score = pd.Series(0, index=df.index, dtype=int)
    reason_bits: List[List[str]] = [[] for _ in range(len(df))]

    def add_rule(mask: pd.Series, points: int, reason: str) -> None:
        nonlocal score
        score = score + mask.astype(int) * points
        for idx in df.index[mask]:
            reason_bits[df.index.get_loc(idx)].append(reason)

    add_rule(claim_amount > 10000, 1, "Claim amount above $10,000")
    add_rule(reimbursed > 8000, 1, "High reimbursed amount")
    add_rule(claim_count > 50, 1, "High claim frequency")
    add_rule(unique_patients > 40, 1, "Large patient volume")
    add_rule(diagnosis_codes > 8, 1, "Many unique diagnosis codes")
    add_rule(procedure_codes > 6, 1, "Many unique procedure codes")
    add_rule(chronic_conditions > 5, 1, "Complex chronic condition profile")
    add_rule(prior_fraud, 2, "Prior provider fraud history")
    add_rule(deductible > 1000, 1, "High deductible amount")

    reasons = []
    for items in reason_bits:
        reasons.append("; ".join(items) if items else "No major secondary risk factors found")
    return score, pd.Series(reasons, index=df.index)


def apply_decision_engine(
    df: pd.DataFrame,
    thresholds: Dict[str, float] | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.copy()
    else:
        thresholds = {**DEFAULT_THRESHOLDS, **thresholds}

    if not thresholds_are_valid(thresholds):
        raise ValueError("Decision thresholds must be in increasing order between 0 and 1.")

    routed = df.copy()
    routed["Model_Fraud_Probability"] = _safe_numeric(_get_series(routed, "Model_Fraud_Probability", 0))
    routed["Claim_Amount"] = _safe_numeric(_get_series(routed, "Claim_Amount", 0))

    probability = routed["Model_Fraud_Probability"]
    secondary_score, secondary_reason = _secondary_rule_components(routed)

    routed["Automation_Risk_Tier"] = "Low Risk"
    routed["Decision_Action"] = "Auto Approve"
    routed["Secondary_Risk_Score"] = 0
    routed["Secondary_Check_Reason"] = "No secondary check required"
    routed["Human_Review_Required"] = False

    low_mask = probability < thresholds["low_upper"]
    borderline_mask = (probability >= thresholds["low_upper"]) & (probability < thresholds["borderline_upper"])
    elevated_mask = (probability >= thresholds["borderline_upper"]) & (probability < thresholds["elevated_upper"])
    high_mask = (probability >= thresholds["elevated_upper"]) & (probability < thresholds["high_upper"])
    critical_mask = probability >= thresholds["high_upper"]

    routed.loc[low_mask, "Automation_Risk_Tier"] = "Low Risk"
    routed.loc[low_mask, "Decision_Action"] = "Auto Approve"

    routed.loc[borderline_mask, "Automation_Risk_Tier"] = "Borderline Risk"
    routed.loc[borderline_mask, "Decision_Action"] = "Secondary Rules Check"
    routed.loc[borderline_mask, "Secondary_Risk_Score"] = secondary_score[borderline_mask]
    routed.loc[borderline_mask, "Secondary_Check_Reason"] = secondary_reason[borderline_mask]

    borderline_low = borderline_mask & (secondary_score <= 1)
    borderline_medium = borderline_mask & secondary_score.between(2, 3)
    borderline_high = borderline_mask & (secondary_score >= 4)
    routed.loc[borderline_low, "Automation_Risk_Tier"] = "Low Risk"
    routed.loc[borderline_low, "Decision_Action"] = "Auto Approve After Secondary Check"
    routed.loc[borderline_medium, "Automation_Risk_Tier"] = "Medium Risk"
    routed.loc[borderline_medium, "Decision_Action"] = "Human Review Queue"
    routed.loc[borderline_high, "Automation_Risk_Tier"] = "High Risk"
    routed.loc[borderline_high, "Decision_Action"] = "Priority Investigation"

    routed.loc[elevated_mask, "Automation_Risk_Tier"] = "Elevated Risk"
    routed.loc[elevated_mask, "Decision_Action"] = "Human Review Queue"

    routed.loc[high_mask, "Automation_Risk_Tier"] = "High Risk"
    routed.loc[high_mask, "Decision_Action"] = "Priority Investigation"

    routed.loc[critical_mask, "Automation_Risk_Tier"] = "Critical Risk"
    routed.loc[critical_mask, "Decision_Action"] = "Payment Hold + Mandatory Investigator Review"

    human_review = (
        (routed["Claim_Amount"] > 10000)
        | (probability >= thresholds["elevated_upper"])
        | routed["Automation_Risk_Tier"].isin(["High Risk", "Critical Risk", "Elevated Risk", "Medium Risk"])
        | routed["Decision_Action"].astype(str).str.contains("Payment Hold", case=False, na=False)
    )
    routed["Human_Review_Required"] = human_review
    return routed


def summarize_decision_actions(df: pd.DataFrame) -> Dict[str, int]:
    action_counts = df["Decision_Action"].value_counts().to_dict() if "Decision_Action" in df.columns else {}
    return {
        "Auto Approve": int(action_counts.get("Auto Approve", 0)),
        "Secondary Rules Checked": int(action_counts.get("Auto Approve After Secondary Check", 0))
        + int(action_counts.get("Secondary Rules Check", 0)),
        "Human Review Queue": int(action_counts.get("Human Review Queue", 0)),
        "Priority Investigation": int(action_counts.get("Priority Investigation", 0)),
        "Payment Holds": int(action_counts.get("Payment Hold + Mandatory Investigator Review", 0)),
        "Total Human Reviews Required": int(df.get("Human_Review_Required", pd.Series(dtype=bool)).sum()),
    }
