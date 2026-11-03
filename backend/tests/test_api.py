"""
test_api.py — Comprehensive test suite for Heart Risk Tracker FastAPI backend.
=============================================================================
Run with:
    python -m pytest backend/tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient


# Sample test patient fixtures
@pytest.fixture
def high_risk_patient():
    return {
        "age": 63.0,
        "sex": 1.0,
        "cp": 4.0,           # Asymptomatic (highest risk)
        "trestbps": 160.0,
        "chol": 286.0,
        "fbs": 1.0,
        "restecg": 2.0,       # LV hypertrophy
        "thalach": 108.0,
        "exang": 1.0,         # Exercise angina
        "oldpeak": 3.2,
        "slope": 2.0,         # Flat
        "ca": 3.0,            # 3 vessels
        "thal": 7.0,          # Reversible defect
    }


@pytest.fixture
def low_risk_patient():
    return {
        "age": 37.0,
        "sex": 0.0,
        "cp": 3.0,           # Non-anginal
        "trestbps": 115.0,
        "chol": 190.0,
        "fbs": 0.0,
        "restecg": 0.0,       # Normal
        "thalach": 178.0,
        "exang": 0.0,         # No angina
        "oldpeak": 0.0,
        "slope": 1.0,         # Upsloping
        "ca": 0.0,            # 0 vessels
        "thal": 3.0,          # Normal perfusion
    }


# ── TEST SUITE ───────────────────────────────────────────────────────────────

def test_root_endpoint(client: TestClient):
    """Verify root endpoint availability and version display."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data


def test_health_endpoint(client: TestClient):
    """Verify health endpoint reports model readiness and expected feature schema."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_ready"] is True
    assert data["features_expected"] == 13
    assert data["champion_model"] == "SVM"


def test_models_comparison_endpoint(client: TestClient):
    """Verify model comparison endpoint returns genuine historical benchmark data."""
    response = client.get("/api/models/comparison")
    assert response.status_code == 200
    data = response.json()
    assert "champion_model" in data
    assert "selection_method" in data
    assert len(data["models"]) >= 5

    # Check required metrics for each model
    for m in data["models"]:
        assert "model" in m
        assert "cv_accuracy_mean" in m
        assert "test_accuracy" in m
        assert "test_f1" in m
        assert 0.0 <= m["test_accuracy"] <= 1.0


def test_dataset_stats_endpoint(client: TestClient):
    """Verify dataset stats endpoint accurately reflects Cleveland dataset properties."""
    response = client.get("/api/dataset/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_samples"] == 303
    assert data["total_features"] == 13
    assert data["class_distribution"]["no_disease_count"] == 164
    assert data["class_distribution"]["disease_present_count"] == 139
    assert "ca" in data["missing_values"]
    assert "thal" in data["missing_values"]


def test_predict_high_risk_patient(client: TestClient, high_risk_patient):
    """Verify inference for a clinically high-risk profile."""
    response = client.post("/api/predict", json=high_risk_patient)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["disease_probability"] <= 1.0
    assert 0.0 <= data["healthy_probability"] <= 1.0
    assert abs(data["disease_probability"] + data["healthy_probability"] - 1.0) < 1e-4
    assert data["risk_category"] in ("Moderate Risk", "High Risk")
    assert "disclaimer" in data
    assert "NOT a medical device" in data["disclaimer"]


def test_predict_low_risk_patient(client: TestClient, low_risk_patient):
    """Verify inference for a clinically low-risk profile."""
    response = client.post("/api/predict", json=low_risk_patient)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] == 0
    assert data["risk_category"] == "Low Risk"
    assert data["disease_probability"] < 0.35
    assert data["healthy_probability"] > 0.65


def test_predict_with_missing_imputed_values(client: TestClient, low_risk_patient):
    """Verify that ca=null and thal=null are handled gracefully via pipeline imputer."""
    patient_missing = low_risk_patient.copy()
    patient_missing["ca"] = None
    patient_missing["thal"] = None

    response = client.post("/api/predict", json=patient_missing)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 0
    assert "disease_probability" in data


def test_predict_invalid_numerical_range(client: TestClient, low_risk_patient):
    """Verify Pydantic rejects out-of-range numerical values (e.g. age=15 or resting BP=350)."""
    bad_patient = low_risk_patient.copy()
    bad_patient["age"] = 12.0  # Min is 18.0

    response = client.post("/api/predict", json=bad_patient)
    assert response.status_code == 422


def test_predict_invalid_categorical_codes(client: TestClient, low_risk_patient):
    """Verify Pydantic rejects invalid category codes (e.g. sex=5, cp=9, restecg=7)."""
    bad_patient = low_risk_patient.copy()
    bad_patient["cp"] = 8.0  # Valid are 1, 2, 3, 4

    response = client.post("/api/predict", json=bad_patient)
    assert response.status_code == 422


def test_predict_missing_required_fields(client: TestClient):
    """Verify Pydantic rejects payload with missing required inputs."""
    incomplete_patient = {"age": 55.0, "sex": 1.0}
    response = client.post("/api/predict", json=incomplete_patient)
    assert response.status_code == 422


def test_explain_patient_mode(client: TestClient, high_risk_patient):
    """Verify patient-level SHAP explanation returns attributions and verified additivity."""
    payload = {
        "mode": "patient",
        "patient_data": high_risk_patient,
    }
    response = client.post("/api/explain", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["mode"] == "patient"
    assert "additive_verification" in data
    assert data["additive_verification"]["passed"] is True
    assert len(data["all_features_ranked"]) == 18
    assert len(data["clinical_features_ranked"]) == 13
    assert len(data["top_risk_drivers"]) > 0


def test_explain_patient_missing_patient_data(client: TestClient):
    """Verify explain returns 422 when mode='patient' but patient_data is omitted."""
    payload = {"mode": "patient"}
    response = client.post("/api/explain", json=payload)
    assert response.status_code == 422


def test_explain_global_post(client: TestClient):
    """Verify explain returns global feature rankings when mode='global'."""
    payload = {"mode": "global"}
    response = client.post("/api/explain", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["mode"] == "global"
    assert "rankings" in data
    assert "clinical_rankings" in data
    assert len(data["clinical_rankings"]) == 13


def test_explain_global_get_convenience(client: TestClient):
    """Verify GET /api/explain/global returns global feature importance ranking."""
    response = client.get("/api/explain/global")
    assert response.status_code == 200
    data = response.json()

    assert data["mode"] == "global"
    assert len(data["clinical_rankings"]) == 13
    assert data["sample_count"] > 0


def test_prediction_and_explanation_consistency_with_champion(client: TestClient, high_risk_patient, low_risk_patient):
    """
    Prove that prediction probabilities and SHAP explanations refer to the exact same
    champion model (no silent model mismatch or surrogate drift).
    """
    for patient in [high_risk_patient, low_risk_patient]:
        pred_res = client.post("/api/predict", json=patient).json()
        exp_res = client.post("/api/explain", json={"mode": "patient", "patient_data": patient}).json()

        # 1. Exact equality of probabilities
        assert abs(pred_res["disease_probability"] - exp_res["disease_probability"]) < 1e-4
        assert abs(pred_res["healthy_probability"] - exp_res["healthy_probability"]) < 1e-4

        # 2. Exact equality of classification
        assert pred_res["prediction"] == exp_res["prediction"]

        # 3. Additive SHAP verification matches prediction probability (approximate for KernelExplainer)
        assert exp_res["additive_verification"]["passed"] is True
        assert abs(exp_res["reconstructed_probability"] - pred_res["disease_probability"]) < 0.05


def test_prediction_and_explanation_consistency_with_missing_values(client: TestClient, high_risk_patient):
    """
    Prove that prediction and explanation handle missing ca/thal identically and refer
    to the same champion pipeline with imputed values.
    """
    patient_missing = high_risk_patient.copy()
    patient_missing["ca"] = None
    patient_missing["thal"] = None

    pred_res = client.post("/api/predict", json=patient_missing).json()
    exp_res = client.post("/api/explain", json={"mode": "patient", "patient_data": patient_missing}).json()

    # Probabilities must be identical
    assert abs(pred_res["disease_probability"] - exp_res["disease_probability"]) < 1e-4
    assert exp_res["additive_verification"]["passed"] is True

    # Missing fields must be clearly labeled as imputed in clinical explanation
    ca_item = next(item for item in exp_res["clinical_features_ranked"] if item["feature"] == "ca")
    thal_item = next(item for item in exp_res["clinical_features_ranked"] if item["feature"] == "thal")
    assert ca_item["formatted_value"] == "Missing (Imputed)"
    assert thal_item["formatted_value"] == "Missing (Imputed)"


def test_champion_explainer_type_matches_model():
    """Verify that the explainer initialized for the champion model is KernelExplainer for SVM."""
    from backend.service import ml_service
    from explainability import get_explainer
    import shap
    from sklearn.svm import SVC

    assert ml_service.is_loaded is True
    clf = ml_service.pipeline.named_steps["classifier"]
    assert isinstance(clf, SVC)

    explainer, _, feature_names = get_explainer(ml_service.pipeline)
    assert isinstance(explainer, shap.KernelExplainer)
    assert len(feature_names) == 18


def test_champion_class_order(client: TestClient, high_risk_patient):
    """Verify champion classes_ maps index 1 to disease class."""
    from backend.service import ml_service
    clf = ml_service.pipeline.named_steps["classifier"]
    assert list(clf.classes_) == [0, 1]
    # Probabilities sum to 1
    response = client.post("/api/predict", json=high_risk_patient)
    data = response.json()
    assert abs(data["disease_probability"] + data["healthy_probability"] - 1.0) < 1e-4


def test_comparison_still_has_all_5_models(client: TestClient):
    """Verify model comparison endpoint returns all 5 models with SVM as champion."""
    response = client.get("/api/models/comparison")
    assert response.status_code == 200
    data = response.json()
    model_names = [m["model"] for m in data["models"]]
    assert "KNN" in model_names
    assert "Logistic Regression" in model_names
    assert "SVM" in model_names
    assert "Random Forest" in model_names
    assert "Gradient Boosting" in model_names
    assert len(data["models"]) >= 5
    assert data["champion_model"] == "SVM"


def test_explain_global_has_explainer_note(client: TestClient):
    """Verify global explanation response contains explainer notes."""
    response = client.get("/api/explain/global")
    assert response.status_code == 200
    data = response.json()
    assert "explainer_note" in data or "explainer_method" in data

