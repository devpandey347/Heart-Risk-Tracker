"""
test_ml_pipeline.py — Automated tests for Chunk 3 ML pipeline.
===============================================================
Run from the project root:
    python -m pytest ml/tests/test_ml_pipeline.py -v

Tests cover:
  1. Target mapping correctness
  2. Missing-value handling inside the pipeline
  3. Pipeline prediction shape and class validity
  4. Artifact loading (pkl + json)
  5. Data leakage guard (preprocessor not fitted on test rows)
  6. Feature schema consistency
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
ML_DIR    = Path(__file__).resolve().parent.parent      # ml/
SRC_DIR   = ML_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import (
    load_raw, apply_binary_target, validate_features,
    COLUMN_NAMES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
)
from preprocessing import (
    ALL_FEATURES, BINARY_FEATURES, NOMINAL_FEATURES,
    NOMINAL_CATEGORIES, build_preprocessor,
)

RAW_DATA   = ML_DIR / "data" / "raw" / "processed.cleveland.data"
ARTIFACTS  = ML_DIR / "artifacts"

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    return load_raw(RAW_DATA)


@pytest.fixture(scope="module")
def binary_df(raw_df):
    return apply_binary_target(raw_df)


@pytest.fixture(scope="module")
def X_y(binary_df):
    X = binary_df[ALL_FEATURES]
    y = binary_df["target_binary"]
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Dataset shape
# ─────────────────────────────────────────────────────────────────────────────
def test_dataset_shape(raw_df):
    assert raw_df.shape == (303, 14), \
        f"Expected (303,14), got {raw_df.shape}"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Target mapping correctness
# ─────────────────────────────────────────────────────────────────────────────
def test_target_mapping_all_zeros_stay_zero(raw_df):
    """All original 0s must map to binary 0."""
    df = apply_binary_target(raw_df)
    original_zeros = df[df["target"] == 0]["target_binary"]
    assert (original_zeros == 0).all(), "Some original-0 targets were mapped to 1"


def test_target_mapping_nonzero_become_one(raw_df):
    """All original 1–4 must map to binary 1."""
    df = apply_binary_target(raw_df)
    nonzero = df[df["target"] >= 1]["target_binary"]
    assert (nonzero == 1).all(), "Some original nonzero targets were NOT mapped to 1"


def test_binary_target_only_0_or_1(binary_df):
    """Binary target column must only contain 0 and 1."""
    vals = set(binary_df["target_binary"].unique())
    assert vals == {0, 1}, f"Unexpected binary target values: {vals}"


def test_class_counts(binary_df):
    """Verify known class counts from EDA."""
    counts = binary_df["target_binary"].value_counts()
    assert counts[0] == 164, f"Expected 164 no-disease, got {counts[0]}"
    assert counts[1] == 139, f"Expected 139 disease, got {counts[1]}"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Missing values
# ─────────────────────────────────────────────────────────────────────────────
def test_missing_values_only_in_ca_thal(raw_df):
    """Only 'ca' and 'thal' should have any missing values."""
    missing = raw_df.isnull().sum()
    cols_with_missing = set(missing[missing > 0].index.tolist())
    assert cols_with_missing == {"ca", "thal"}, \
        f"Unexpected missing columns: {cols_with_missing}"


def test_missing_counts(raw_df):
    """Exact missing counts must match EDA findings."""
    missing = raw_df.isnull().sum()
    assert missing["ca"]   == 4, f"Expected 4 missing in ca, got {missing['ca']}"
    assert missing["thal"] == 2, f"Expected 2 missing in thal, got {missing['thal']}"


def test_pipeline_handles_missing_values(X_y):
    """After fitting the preprocessor on data with missing values, no NaNs should remain."""
    X, y = X_y
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X, y)
    assert not np.isnan(X_transformed).any(), \
        "Pipeline produced NaN values — imputation failed"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Preprocessor output shape
# ─────────────────────────────────────────────────────────────────────────────
def test_preprocessor_output_shape(X_y):
    """Transformed output must have 18 columns (6 num + 3 binary + 9 OHE dummies)."""
    X, y = X_y
    preprocessor = build_preprocessor()
    X_t = preprocessor.fit_transform(X, y)
    assert X_t.shape[1] == 18, \
        f"Expected 18 transformed features (6+3+9), got {X_t.shape[1]}"
    assert X_t.shape[0] == len(X), \
        f"Row count changed after transformation: {X_t.shape[0]} vs {len(X)}"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — Feature schema consistency
# ─────────────────────────────────────────────────────────────────────────────
def test_all_features_present_in_dataset(raw_df):
    """Every feature in ALL_FEATURES must exist as a column in the raw dataset."""
    for feat in ALL_FEATURES:
        assert feat in raw_df.columns, f"Feature '{feat}' missing from dataset columns"


def test_feature_count():
    """There must be exactly 13 raw features: 6 numerical + 3 binary + 4 nominal."""
    assert len(NUMERICAL_FEATURES) == 6,  f"Expected 6 numerical, got {len(NUMERICAL_FEATURES)}"
    assert len(BINARY_FEATURES)    == 3,  f"Expected 3 binary, got {len(BINARY_FEATURES)}"
    assert len(NOMINAL_FEATURES)   == 4,  f"Expected 4 nominal, got {len(NOMINAL_FEATURES)}"
    assert len(ALL_FEATURES)       == 13, f"Expected 13 total, got {len(ALL_FEATURES)}"


def test_no_overlap_between_feature_lists():
    """Numerical, binary and nominal feature lists must be pairwise disjoint."""
    num_set    = set(NUMERICAL_FEATURES)
    binary_set = set(BINARY_FEATURES)
    nominal_set= set(NOMINAL_FEATURES)
    assert len(num_set & binary_set)  == 0, f"Num/binary overlap: {num_set & binary_set}"
    assert len(num_set & nominal_set) == 0, f"Num/nominal overlap: {num_set & nominal_set}"
    assert len(binary_set & nominal_set) == 0, f"Binary/nominal overlap: {binary_set & nominal_set}"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — Categorical categories correctness (verified from actual data)
# ─────────────────────────────────────────────────────────────────────────────
def test_categorical_categories_match_data(raw_df):
    """Known category sets for all categorical features must match actual dataset values."""
    expected = {
        "sex"    : {0.0, 1.0},
        "cp"     : {1.0, 2.0, 3.0, 4.0},
        "fbs"    : {0.0, 1.0},
        "restecg": {0.0, 1.0, 2.0},
        "exang"  : {0.0, 1.0},
        "slope"  : {1.0, 2.0, 3.0},
        "thal"   : {3.0, 6.0, 7.0},
    }
    for col, expected_vals in expected.items():
        actual_vals = set(raw_df[col].dropna().unique())
        assert actual_vals.issubset(expected_vals), \
            f"'{col}': actual values {actual_vals} not subset of expected {expected_vals}"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 — Pipeline prediction validity (requires training artifacts)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(
    not (ARTIFACTS / "best_pipeline.pkl").exists(),
    reason="best_pipeline.pkl not found — run train_pipeline.py first"
)
def test_artifact_pipeline_predictions(X_y):
    """Loaded pipeline must produce binary predictions for all test rows."""
    import joblib
    pipeline = joblib.load(ARTIFACTS / "best_pipeline.pkl")
    X, y = X_y
    preds = pipeline.predict(X)
    assert set(preds).issubset({0, 1}), f"Unexpected prediction values: {set(preds)}"
    assert len(preds) == len(X), "Prediction count mismatch"


@pytest.mark.skipif(
    not (ARTIFACTS / "best_pipeline.pkl").exists(),
    reason="best_pipeline.pkl not found — run train_pipeline.py first"
)
def test_artifact_pipeline_predict_proba(X_y):
    """Loaded pipeline must produce valid probabilities in [0,1]."""
    import joblib
    pipeline = joblib.load(ARTIFACTS / "best_pipeline.pkl")
    X, _ = X_y
    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2), f"Unexpected proba shape: {proba.shape}"
    assert np.all(proba >= 0) and np.all(proba <= 1), "Probabilities out of [0,1]"
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6), "Probabilities don't sum to 1"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 8 — model_meta.json artifact
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(
    not (ARTIFACTS / "model_meta.json").exists(),
    reason="model_meta.json not found — run train_pipeline.py first"
)
def test_model_meta_json_structure():
    """model_meta.json must contain all required keys."""
    with open(ARTIFACTS / "model_meta.json") as f:
        meta = json.load(f)

    required_keys = [
        "best_model", "best_metrics", "split_config",
        "feature_schema", "target_mapping", "dataset", "disclaimer"
    ]
    for key in required_keys:
        assert key in meta, f"Missing key '{key}' in model_meta.json"

    metric_keys = ["test_accuracy", "test_f1", "test_roc_auc", "cv_accuracy_mean"]
    for key in metric_keys:
        assert key in meta["best_metrics"], f"Missing metric '{key}' in best_metrics"

    # Metrics must be plausible floats
    acc = meta["best_metrics"]["test_accuracy"]
    assert 0.0 < acc <= 1.0, f"Implausible test_accuracy: {acc}"


@pytest.mark.skipif(
    not (ARTIFACTS / "model_meta.json").exists(),
    reason="model_meta.json not found — run train_pipeline.py first"
)
def test_model_meta_feature_schema():
    """Feature schema in model_meta.json must match preprocessing.py constants."""
    with open(ARTIFACTS / "model_meta.json") as f:
        meta = json.load(f)
    schema = meta["feature_schema"]
    assert schema["all_features"]       == ALL_FEATURES,      "all_features mismatch"
    assert schema["numerical_features"] == NUMERICAL_FEATURES, "numerical_features mismatch"
    assert schema["binary_features"]    == BINARY_FEATURES,   "binary_features mismatch"
    assert schema["nominal_features"]   == NOMINAL_FEATURES,  "nominal_features mismatch"
    assert schema["feature_count"]      == 13,                "feature_count must be 13"
    # cv_f1_mean must exist in best_metrics (model selection criterion)
    assert "cv_f1_mean" in meta["best_metrics"], "cv_f1_mean missing from best_metrics"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 9 — Single-row prediction (smoke test for the backend API shape)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(
    not (ARTIFACTS / "best_pipeline.pkl").exists(),
    reason="best_pipeline.pkl not found — run train_pipeline.py first"
)
def test_single_row_prediction():
    """Pipeline must handle a single-row DataFrame input correctly."""
    import joblib
    pipeline = joblib.load(ARTIFACTS / "best_pipeline.pkl")

    # A realistic patient profile (from the first row of the dataset)
    single_input = pd.DataFrame([{
        "age": 63.0, "trestbps": 145.0, "chol": 233.0,
        "thalach": 150.0, "oldpeak": 2.3, "ca": 0.0,
        "sex": 1.0, "cp": 1.0, "fbs": 1.0, "restecg": 2.0,
        "exang": 0.0, "slope": 3.0, "thal": 6.0,
    }])
    pred  = pipeline.predict(single_input)
    proba = pipeline.predict_proba(single_input)

    assert pred.shape  == (1,), f"Expected shape (1,), got {pred.shape}"
    assert proba.shape == (1, 2), f"Expected shape (1,2), got {proba.shape}"
    assert pred[0] in {0, 1}, f"Prediction must be 0 or 1, got {pred[0]}"
