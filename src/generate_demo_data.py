from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_processing import DEMO_DATA_DIR, ensure_directories
from src.decision_engine import apply_decision_engine


OUTPUT_PATH = DEMO_DATA_DIR / "demo_scored_claims.csv"


@dataclass
class ProviderProfile:
    provider_id: str
    region: str
    provider_type: str
    prior_fraud_flag: bool
    base_claim_frequency: int
    base_unique_patients: int
    diagnosis_complexity: int
    procedure_complexity: int


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + np.exp(-value))


def _risk_category(probability: float) -> str:
    if probability < 0.30:
        return "Low Risk"
    if probability < 0.70:
        return "Medium Risk"
    return "High Risk"


def _recommended_action(risk_category: str) -> str:
    return {
        "Low Risk": "Auto Approve",
        "Medium Risk": "Human Review",
        "High Risk": "Investigate Before Payment",
    }[risk_category]


def _human_review_required(risk_category: str, claim_amount: float) -> bool:
    return risk_category in {"Medium Risk", "High Risk"} or claim_amount > 10000


def _provider_profiles(rng: np.random.Generator) -> List[ProviderProfile]:
    regions = ["Northeast", "Mid-Atlantic", "Southeast", "Midwest", "Southwest", "West"]
    provider_types = ["Hospital", "Clinic", "Specialist", "Surgery Center", "Rehab Facility"]

    profiles: List[ProviderProfile] = []
    for idx in range(1, 41):
        provider_type = rng.choice(provider_types, p=[0.26, 0.24, 0.22, 0.16, 0.12])
        profiles.append(
            ProviderProfile(
                provider_id=f"PRV{idx:04d}",
                region=str(rng.choice(regions)),
                provider_type=str(provider_type),
                prior_fraud_flag=bool(rng.random() < 0.18),
                base_claim_frequency=int(rng.integers(18, 180)),
                base_unique_patients=int(rng.integers(10, 120)),
                diagnosis_complexity=int(rng.integers(2, 11)),
                procedure_complexity=int(rng.integers(1, 8)),
            )
        )
    return profiles


def _sample_claim_type(provider_type: str, rng: np.random.Generator) -> str:
    if provider_type in {"Hospital", "Rehab Facility", "Surgery Center"}:
        return str(rng.choice(["Inpatient", "Outpatient"], p=[0.58, 0.42]))
    return str(rng.choice(["Outpatient", "Inpatient"], p=[0.78, 0.22]))


def _claim_amount(provider_type: str, claim_type: str, rng: np.random.Generator) -> float:
    base_map: Dict[str, tuple] = {
        "Hospital": (9800, 4200),
        "Clinic": (2400, 1100),
        "Specialist": (3600, 1800),
        "Surgery Center": (7200, 2600),
        "Rehab Facility": (6800, 2200),
    }
    mean, std = base_map.get(provider_type, (3500, 1400))
    if claim_type == "Inpatient":
        mean *= 1.55
        std *= 1.45

    amount = max(350.0, rng.normal(mean, std))
    if rng.random() < 0.12:
        amount *= rng.uniform(1.6, 2.8)
    return float(round(amount, 2))


def _build_record(
    claim_idx: int,
    provider: ProviderProfile,
    rng: np.random.Generator,
) -> Dict[str, object]:
    claim_type = _sample_claim_type(provider.provider_type, rng)
    claim_amount = _claim_amount(provider.provider_type, claim_type, rng)

    claim_count_last_12_months = max(
        4,
        int(provider.base_claim_frequency + rng.normal(0, provider.base_claim_frequency * 0.18)),
    )
    unique_patients_last_12_months = max(
        3,
        int(provider.base_unique_patients + rng.normal(0, provider.base_unique_patients * 0.2)),
    )
    unique_diagnosis_codes = max(
        1,
        int(provider.diagnosis_complexity + rng.normal(0, 2.2)),
    )
    unique_procedure_codes = max(
        1,
        int(provider.procedure_complexity + rng.normal(0, 1.8)),
    )
    chronic_condition_count = int(np.clip(rng.normal(3.3 if claim_type == "Inpatient" else 2.4, 1.8), 0, 9))

    reimbursement_ratio = rng.uniform(0.58, 0.92)
    deductible_ratio = rng.uniform(0.03, 0.16)

    suspicious_bonus = 0.0
    if claim_amount > 10000:
        reimbursement_ratio = min(0.98, reimbursement_ratio + rng.uniform(0.04, 0.10))
        suspicious_bonus += 0.18
    if provider.prior_fraud_flag:
        suspicious_bonus += 0.22
    if claim_count_last_12_months > 130:
        suspicious_bonus += 0.17
    if unique_diagnosis_codes >= 10:
        suspicious_bonus += 0.12
    if unique_procedure_codes >= 7:
        suspicious_bonus += 0.10

    if rng.random() < 0.08:
        deductible_ratio = rng.uniform(0.18, 0.27)
        suspicious_bonus += 0.11

    reimbursed_amount = float(round(claim_amount * reimbursement_ratio, 2))
    deductible_amount = float(round(claim_amount * deductible_ratio, 2))

    driver_scores = {
        "High reimbursement amount": max(0.0, (reimbursed_amount / max(claim_amount, 1)) - 0.82) * 2.0,
        "Unusually high claim frequency": max(0.0, (claim_count_last_12_months - 100) / 75),
        "Prior provider fraud history": 1.2 if provider.prior_fraud_flag else 0.0,
        "Many unique diagnosis codes": max(0.0, (unique_diagnosis_codes - 8) / 4),
        "Many unique procedure codes": max(0.0, (unique_procedure_codes - 6) / 3),
        "High deductible amount": max(0.0, (deductible_amount / max(claim_amount, 1)) - 0.16) * 4.5,
        "Complex chronic condition profile": max(0.0, (chronic_condition_count - 5) / 3),
    }

    linear_score = (
        -3.0
        + 0.00011 * reimbursed_amount
        + 0.0053 * claim_count_last_12_months
        + 0.086 * unique_diagnosis_codes
        + 0.091 * unique_procedure_codes
        + 0.072 * chronic_condition_count
        + (0.85 if provider.prior_fraud_flag else 0.0)
        + suspicious_bonus
        - 0.004 * unique_patients_last_12_months
    )

    fraud_probability = float(np.clip(_sigmoid(linear_score), 0.02, 0.98))
    risk_category = _risk_category(fraud_probability)
    recommended_action = _recommended_action(risk_category)
    top_risk_driver = max(driver_scores, key=driver_scores.get)

    if risk_category == "Low Risk" and driver_scores[top_risk_driver] < 0.18:
        top_risk_driver = "Routine claim pattern"

    return {
        "Claim_ID": f"CLM{claim_idx:06d}",
        "Provider_ID": provider.provider_id,
        "Patient_ID": f"PAT{int(rng.integers(100000, 999999))}",
        "Claim_Type": claim_type,
        "Region": provider.region,
        "Provider_Type": provider.provider_type,
        "Claim_Amount": round(claim_amount, 2),
        "Reimbursed_Amount": reimbursed_amount,
        "Deductible_Amount": deductible_amount,
        "Claim_Count_Last_12_Months": claim_count_last_12_months,
        "Unique_Patients_Last_12_Months": unique_patients_last_12_months,
        "Unique_Diagnosis_Codes": unique_diagnosis_codes,
        "Unique_Procedure_Codes": unique_procedure_codes,
        "Chronic_Condition_Count": chronic_condition_count,
        "Prior_Fraud_Flag": provider.prior_fraud_flag,
        "Model_Fraud_Probability": round(fraud_probability, 4),
        "Risk_Category": risk_category,
        "Recommended_Action": recommended_action,
        "Top_Risk_Driver": top_risk_driver,
        "Human_Review_Required": _human_review_required(risk_category, claim_amount),
    }


def generate_demo_claims(num_records: int = 180, seed: int = 42) -> pd.DataFrame:
    ensure_directories()
    DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    providers = _provider_profiles(rng)
    records = [
        _build_record(claim_idx=index + 1, provider=providers[index % len(providers)], rng=rng)
        for index in range(num_records)
    ]
    demo_df = pd.DataFrame(records).sort_values("Model_Fraud_Probability", ascending=False).reset_index(drop=True)
    demo_df = apply_decision_engine(demo_df)
    return demo_df


def save_demo_claims(output_path: Path = OUTPUT_PATH, num_records: int = 180, seed: int = 42) -> Path:
    demo_df = generate_demo_claims(num_records=num_records, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    demo_df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    output_path = save_demo_claims()
    print(f"Saved demo claims to {output_path}")


if __name__ == "__main__":
    main()
