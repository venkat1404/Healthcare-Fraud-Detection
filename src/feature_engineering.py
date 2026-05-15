from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _warn_missing(columns: Sequence[str], dataset_name: str) -> None:
    for column in columns:
        warnings.warn(
            f"Column '{column}' not found in {dataset_name}. Skipping dependent feature.",
            stacklevel=2,
        )


def _existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    return [column for column in columns if column in df.columns]


def _safe_datetime(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def _normalize_fraud_label(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    return values.map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}).fillna(0).astype(int)


def build_provider_level_dataset(
    beneficiary_df: pd.DataFrame,
    inpatient_df: pd.DataFrame,
    outpatient_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create provider-level features from merged beneficiary and claim files."""
    inpatient = inpatient_df.copy()
    outpatient = outpatient_df.copy()
    beneficiary = beneficiary_df.copy()
    labels = labels_df.copy()

    inpatient["claim_type"] = "inpatient"
    outpatient["claim_type"] = "outpatient"

    claims = pd.concat([inpatient, outpatient], ignore_index=True, sort=False)
    claims = _safe_datetime(
        claims,
        ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt"],
    )
    beneficiary = _safe_datetime(beneficiary, ["DOB", "DOD"])

    if "ClaimStartDt" in claims.columns and "ClaimEndDt" in claims.columns:
        claims["claim_duration_days"] = (
            (claims["ClaimEndDt"] - claims["ClaimStartDt"]).dt.days.fillna(0).clip(lower=0)
        )
    else:
        _warn_missing(["ClaimStartDt", "ClaimEndDt"], "claims")
        claims["claim_duration_days"] = np.nan

    if "ClaimStartDt" in claims.columns and "DOB" in beneficiary.columns:
        merged_age = claims[["BeneID", "ClaimStartDt"]].merge(
            beneficiary[["BeneID", "DOB"]], on="BeneID", how="left"
        )
        claims["patient_age"] = (
            (merged_age["ClaimStartDt"] - merged_age["DOB"]).dt.days.div(365.25)
        ).clip(lower=0, upper=115)
    else:
        _warn_missing(["ClaimStartDt", "DOB"], "claims/beneficiary")
        claims["patient_age"] = np.nan

    chronic_columns = [column for column in beneficiary.columns if column.startswith("ChronicCond_")]
    if chronic_columns:
        chronic_numeric = beneficiary[chronic_columns].replace({2: 0})
        beneficiary["chronic_condition_count"] = chronic_numeric.fillna(0).sum(axis=1)
    else:
        _warn_missing(["ChronicCond_*"], "beneficiary")
        beneficiary["chronic_condition_count"] = np.nan

    claims = claims.merge(beneficiary, on="BeneID", how="left")
    claims = claims.merge(labels, on="Provider", how="left")
    claims["fraud"] = _normalize_fraud_label(claims["PotentialFraud"]) if "PotentialFraud" in claims.columns else 0

    if "ClmAdmitDiagnosisCode" in claims.columns:
        claims["has_admit_diagnosis"] = claims["ClmAdmitDiagnosisCode"].notna().astype(int)
    else:
        _warn_missing(["ClmAdmitDiagnosisCode"], "claims")
        claims["has_admit_diagnosis"] = 0

    physician_columns = _existing_columns(
        claims, ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]
    )
    diagnosis_columns = _existing_columns(
        claims, [f"ClmDiagnosisCode_{index}" for index in range(1, 11)]
    )
    procedure_columns = _existing_columns(
        claims, [f"ClmProcedureCode_{index}" for index in range(1, 7)]
    )

    if not physician_columns:
        _warn_missing(["AttendingPhysician", "OperatingPhysician", "OtherPhysician"], "claims")
    if not diagnosis_columns:
        _warn_missing([f"ClmDiagnosisCode_{index}" for index in range(1, 11)], "claims")
    if not procedure_columns:
        _warn_missing([f"ClmProcedureCode_{index}" for index in range(1, 7)], "claims")

    def _unique_across_columns(group: pd.DataFrame, columns: Sequence[str]) -> int:
        if not columns:
            return 0
        flat = pd.Series(group.loc[:, columns].astype("object").values.ravel())
        flat = flat.where(flat.notna(), np.nan)
        flat = flat.astype("string").replace("<NA>", pd.NA).dropna()
        return int(flat.nunique())

    def _mode_or_unknown(series: pd.Series) -> str:
        mode = series.dropna().mode()
        return mode.iloc[0] if not mode.empty else "Unknown"

    aggregation_map: Dict[str, Tuple[str, str]] = {
        "Provider": ("Provider", "first"),
        "fraud": ("fraud", "max"),
        "total_claims": ("ClaimID", "count"),
        "inpatient_claims": ("claim_type", lambda s: int((s == "inpatient").sum())),
        "outpatient_claims": ("claim_type", lambda s: int((s == "outpatient").sum())),
        "avg_claim_amount_reimbursed": ("InscClaimAmtReimbursed", "mean"),
        "total_claim_amount_reimbursed": ("InscClaimAmtReimbursed", "sum"),
        "avg_deductible_amount": ("DeductibleAmtPaid", "mean"),
        "total_deductible_amount": ("DeductibleAmtPaid", "sum"),
        "unique_beneficiaries": ("BeneID", pd.Series.nunique),
        "avg_claim_duration": ("claim_duration_days", "mean"),
        "avg_patient_age": ("patient_age", "mean"),
        "avg_chronic_condition_count": ("chronic_condition_count", "mean"),
        "pct_claims_with_admit_diagnosis": ("has_admit_diagnosis", "mean"),
    }

    available_agg = {
        new_name: pd.NamedAgg(column=source, aggfunc=aggfunc)
        for new_name, (source, aggfunc) in aggregation_map.items()
        if source in claims.columns
    }

    missing_agg_sources = [source for _, (source, _) in aggregation_map.items() if source not in claims.columns]
    if missing_agg_sources:
        _warn_missing(missing_agg_sources, "claims")

    provider_features = claims.groupby("Provider").agg(**available_agg).reset_index(drop=True)

    extra_features = []
    for provider, group in claims.groupby("Provider"):
        extra_features.append(
            {
                "Provider": provider,
                "unique_physicians": _unique_across_columns(group, physician_columns),
                "unique_diagnosis_codes": _unique_across_columns(group, diagnosis_columns),
                "unique_procedure_codes": _unique_across_columns(group, procedure_columns),
                "dominant_state": _mode_or_unknown(group["State"]) if "State" in group.columns else "Unknown",
                "dominant_gender": _mode_or_unknown(group["Gender"]) if "Gender" in group.columns else "Unknown",
                "dominant_race": _mode_or_unknown(group["Race"]) if "Race" in group.columns else "Unknown",
            }
        )

    provider_features = pd.DataFrame(extra_features).merge(provider_features, on="Provider", how="left")
    provider_features["pct_claims_with_admit_diagnosis"] = (
        provider_features["pct_claims_with_admit_diagnosis"].fillna(0) * 100
    )

    numeric_columns = provider_features.select_dtypes(include=[np.number]).columns
    provider_features[numeric_columns] = provider_features[numeric_columns].replace([np.inf, -np.inf], np.nan)
    return provider_features


def transform_synthetic_claim_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply reusable feature transformations for the synthetic claim dataset."""
    data = df.copy()

    date_columns = [
        column
        for column in data.columns
        if "date" in column.lower()
        and not column.lower().endswith(("_year", "_month", "_dayofweek"))
    ]
    data = _safe_datetime(data, date_columns)

    for column in date_columns:
        data[f"{column}_year"] = data[column].dt.year
        data[f"{column}_month"] = data[column].dt.month
        data[f"{column}_dayofweek"] = data[column].dt.dayofweek

    if {"Claim_Date", "Service_Date"}.issubset(data.columns):
        data["claim_submission_lag_days"] = (data["Claim_Date"] - data["Service_Date"]).dt.days
    if {"Policy_Expiration_Date", "Claim_Date"}.issubset(data.columns):
        data["days_until_policy_expiration"] = (
            data["Policy_Expiration_Date"] - data["Claim_Date"]
        ).dt.days

    return data


def prepare_synthetic_claim_dataset(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Clean the synthetic claim dataset and split it into X/y."""
    data = transform_synthetic_claim_features(df)
    data[target_column] = _normalize_fraud_label(data[target_column])

    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y
