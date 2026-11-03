"""
test_explainability.py — Unit tests for SHAP Explainability Module.
===================================================================
Run with:
    python -m pytest ml/tests/test_explainability.py -v
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loader import load_raw, apply_binary_target
from preprocessing import ALL_FEATURES
from explainability import (
    get_tree_explainer,
    compute_global_feature_importance,
    explain_patient_prediction,
    verify_shap_additivity,
    FEATURE_METADATA,
)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "processed.cleveland.data"


@pytest.fixture(scope="module")
def pipeline():
    pipe_path = ARTIFACTS_DIR / "best_pipeline.pkl"
    assert pipe_path.exists(), "best_pipeline.pkl does not exist"
    return joblib.load(pipe_path)


@pytest.fixture(scope="module")
def dataset():
    assert RAW_DATA_PATH.exists(), "raw dataset does not exist"
    df_raw = load_raw(RAW_DATA_PATH)
    return apply_binary_target(df_raw)


def test_explainer_initialization(pipeline):
    """Test that explainer extracts cleanly from the saved pipeline with 18 transformed features."""
    explainer, preprocessor, feature_names = get_tree_explainer(pipeline)
    assert explainer is not None
    assert preprocessor is not None
    assert len(feature_names) == 18


def test_global_importance_structure(pipeline, dataset):
    """Test that global feature importance returns 18 transformed rankings and 13 clinical rankings."""
    sample = dataset[ALL_FEATURES].iloc[:20]
    result = compute_global_feature_importance(pipeline, sample)

    assert "rankings" in result
    assert len(result["rankings"]) == 18
    assert result["rankings"][0]["rank"] == 1
    assert result["rankings"][-1]["rank"] == 18

    # Clinical features aggregated into the 13 raw inputs
    assert "clinical_rankings" in result
    assert len(result["clinical_rankings"]) == 13

    # Ensure ranking is sorted descending by mean_abs_shap
    shap_vals = [r["mean_abs_shap"] for r in result["rankings"]]
    assert shap_vals == sorted(shap_vals, reverse=True)


def test_mathematical_additivity(pipeline, dataset):
    """Verify that reconstructed predictions match model predictions within machine tolerance."""
    sample = dataset[ALL_FEATURES].iloc[:30]
    verif = verify_shap_additivity(pipeline, sample, tolerance=1e-5)
    assert verif["all_passed"] is True
    assert verif["max_residual"] < 1e-5


def test_single_patient_explanation_schema(pipeline):
    """Test single-patient explanation with complete raw feature inputs."""
    patient = {
        "age": 58.0,
        "sex": 1.0,
        "cp": 4.0,
        "trestbps": 140.0,
        "chol": 260.0,
        "fbs": 0.0,
        "restecg": 2.0,
        "thalach": 130.0,
        "exang": 1.0,
        "oldpeak": 2.2,
        "slope": 2.0,
        "ca": 2.0,
        "thal": 7.0,
    }
    exp = explain_patient_prediction(pipeline, patient)

    assert "prediction" in exp
    assert exp["prediction"] in (0, 1)
    assert 0.0 <= exp["disease_probability"] <= 1.0
    assert 0.0 <= exp["healthy_probability"] <= 1.0
    assert abs(exp["disease_probability"] + exp["healthy_probability"] - 1.0) < 1e-5
    assert exp["additive_verification"]["passed"] is True
    assert len(exp["all_features_ranked"]) == 18
    assert len(exp["clinical_features_ranked"]) == 13


def test_single_patient_explanation_with_missing_values(pipeline):
    """Test single-patient explanation handles NaN values in ca and thal gracefully."""
    patient_with_nans = {
        "age": 52.0,
        "sex": 0.0,
        "cp": 2.0,
        "trestbps": 120.0,
        "chol": 210.0,
        "fbs": 0.0,
        "restecg": 0.0,
        "thalach": 165.0,
        "exang": 0.0,
        "oldpeak": 0.4,
        "slope": 1.0,
        "ca": np.nan,    # Missing
        "thal": np.nan,  # Missing
    }
    exp = explain_patient_prediction(pipeline, patient_with_nans)

    assert exp["additive_verification"]["passed"] is True
    assert len(exp["all_features_ranked"]) == 18
    assert len(exp["clinical_features_ranked"]) == 13

    # Verify ca and thal received formatted imputed label in clinical rankings
    ca_item = next(item for item in exp["clinical_features_ranked"] if item["feature"] == "ca")
    assert ca_item["formatted_value"] == "Missing (Imputed)"

