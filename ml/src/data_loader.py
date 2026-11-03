"""
data_loader.py — Loads and validates the raw UCI Cleveland Heart Disease dataset.
==================================================================================
This module is imported by both eda.py and train_pipeline.py.
It NEVER modifies the original file on disk.
"""

from pathlib import Path
import pandas as pd

# ── Column names per UCI documentation ─────────────────────────────────────
COLUMN_NAMES = [
    "age",       # numerical
    "sex",       # categorical: 1=male, 0=female
    "cp",        # categorical: 1=typical angina, 2=atypical, 3=non-anginal, 4=asymptomatic
    "trestbps",  # numerical: resting blood pressure (mmHg)
    "chol",      # numerical: serum cholesterol (mg/dl)
    "fbs",       # categorical: fasting blood sugar >120 mg/dl (1=true)
    "restecg",   # categorical: resting ECG results (0, 1, 2)
    "thalach",   # numerical: max heart rate achieved (bpm)
    "exang",     # categorical: exercise-induced angina (1=yes)
    "oldpeak",   # numerical: ST depression
    "slope",     # categorical: slope of peak exercise ST segment (1, 2, 3)
    "ca",        # numerical*: vessels coloured by fluoroscopy (0–3) — has '?' missing
    "thal",      # categorical*: thalassemia type (3=normal, 6=fixed, 7=reversible) — has '?' missing
    "target",    # target: 0=no disease; 1–4=disease (mapped to binary before training)
]

CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
NUMERICAL_FEATURES   = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]


def load_raw(path: Path | str) -> pd.DataFrame:
    """
    Load the raw processed.cleveland.data file into a DataFrame.

    - Assigns the 14 UCI column names.
    - Converts '?' to NaN.
    - Does NOT impute, drop, or transform anything.

    Parameters
    ----------
    path : Path | str
        Absolute or relative path to processed.cleveland.data.

    Returns
    -------
    pd.DataFrame with shape (303, 14).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Place 'processed.cleveland.data' in ml/data/raw/ and retry."
        )

    df = pd.read_csv(
        path,
        header=None,
        names=COLUMN_NAMES,
        na_values=["?"],
    )

    # Basic shape guard
    assert df.shape == (303, 14), (
        f"Unexpected dataset shape {df.shape}. "
        "Expected (303, 14). Verify the file is the Cleveland subset."
    )

    return df


def apply_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the original multi-class target (0–4) to a binary target.

    Mapping:
        0  →  0   (no disease)
        1  →  1   (disease present)
        2  →  1
        3  →  1
        4  →  1

    This is the standard convention for the Cleveland dataset.
    The original 'target' column is preserved; 'target_binary' is added.
    """
    df = df.copy()
    df["target_binary"] = (df["target"] >= 1).astype(int)
    return df


def validate_features(df: pd.DataFrame) -> dict:
    """
    Run basic integrity checks on the loaded DataFrame.

    Returns a dict of validation results; raises ValueError on critical failures.
    """
    results = {}

    # Check all expected columns present
    missing_cols = [c for c in COLUMN_NAMES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    results["columns_ok"] = True

    # Check original target range
    valid_targets = {0, 1, 2, 3, 4}
    actual_targets = set(df["target"].dropna().unique().astype(int))
    results["target_values_valid"] = actual_targets.issubset(valid_targets)
    results["actual_target_values"] = sorted(actual_targets)

    # Report missing values
    missing = df.isnull().sum()
    results["missing_by_column"] = missing[missing > 0].to_dict()
    results["total_missing"] = int(missing.sum())

    return results
