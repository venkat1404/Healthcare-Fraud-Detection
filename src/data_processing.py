from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEMO_DATA_DIR = PROJECT_ROOT / "data" / "demo"


def ensure_directories() -> None:
    """Create the standard project directories if they do not exist."""
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        DEMO_DATA_DIR,
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "outputs" / "charts",
        PROJECT_ROOT / "outputs" / "metrics",
        PROJECT_ROOT / "notebooks",
        PROJECT_ROOT / "streamlit_app",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def find_file(
    patterns: Iterable[str],
    search_dirs: Optional[Iterable[Path]] = None,
    required: bool = True,
) -> Optional[Path]:
    """Return the first file that matches any of the provided glob patterns."""
    if search_dirs is None:
        search_dirs = [
            RAW_DATA_DIR,
            PROJECT_ROOT / "Dataset",
            PROJECT_ROOT,
        ]

    for directory in search_dirs:
        if not directory.exists():
            continue
        for pattern in patterns:
            matches = sorted(directory.glob(pattern))
            if matches:
                return matches[0]

    if required:
        raise FileNotFoundError(f"Could not find any file matching patterns: {list(patterns)}")
    return None


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV file with a small amount of defensive configuration."""
    return pd.read_csv(path, low_memory=False, **kwargs)


def load_primary_datasets() -> Dict[str, pd.DataFrame]:
    """Load the provider fraud Kaggle dataset files using flexible filename matching."""
    paths = {
        "beneficiary": find_file(["*Train_Beneficiary*.csv", "*Beneficiary*train*.csv"]),
        "inpatient": find_file(["*Train_Inpatient*.csv", "*Inpatient*train*.csv"]),
        "outpatient": find_file(["*Train_Outpatient*.csv", "*Outpatient*train*.csv"]),
        "labels": find_file(["Train-*.csv", "*provider*fraud*train*.csv"]),
    }

    return {name: load_csv(path) for name, path in paths.items()}


def load_synthetic_dataset() -> pd.DataFrame:
    """Load the synthetic health insurance claims dataset using flexible filename matching."""
    path = find_file(
        ["*synthetic*claim*.csv", "*health*insurance*fraud*.csv", "*claims*fraud*.csv"]
    )
    return load_csv(path)


def infer_target_column(df: pd.DataFrame, candidates: Optional[Iterable[str]] = None) -> str:
    """Infer the fraud target column from a shortlist of likely names."""
    if candidates is None:
        candidates = [
            "Is_Fraudulent",
            "is_fraudulent",
            "Fraud",
            "fraud",
            "FraudLabel",
            "PotentialFraud",
            "target",
            "label",
        ]

    normalized = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    for column in df.columns:
        if "fraud" in column.lower():
            return column

    raise ValueError("Unable to infer the target fraud column from the synthetic dataset.")
