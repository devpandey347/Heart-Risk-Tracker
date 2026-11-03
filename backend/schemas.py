"""
schemas.py — Pydantic models for API request validation and response schemas.
=============================================================================
Enforces strict medical-range validation, clean typing, and unambiguous field documentation.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ── PATIENT INPUT VALIDATION SCHEMA ──────────────────────────────────────────
class PatientInput(BaseModel):
    """
    13 clinical features for cardiovascular risk evaluation.
    Matches the Cleveland dataset schema and order.
    """
    age: float = Field(
        ...,
        ge=18.0,
        le=120.0,
        description="Patient age in years (18–120)",
        examples=[58.0],
    )
    sex: float = Field(
        ...,
        description="Biological sex (0 = Female, 1 = Male)",
        examples=[1.0],
    )
    cp: float = Field(
        ...,
        description="Chest pain type: 1 = Typical Angina, 2 = Atypical Angina, 3 = Non-Anginal, 4 = Asymptomatic",
        examples=[4.0],
    )
    trestbps: float = Field(
        ...,
        ge=60.0,
        le=250.0,
        description="Resting blood pressure in mm Hg (60–250)",
        examples=[140.0],
    )
    chol: float = Field(
        ...,
        ge=80.0,
        le=600.0,
        description="Serum cholesterol in mg/dl (80–600)",
        examples=[260.0],
    )
    fbs: float = Field(
        ...,
        description="Fasting blood sugar > 120 mg/dl (0 = No, 1 = Yes)",
        examples=[0.0],
    )
    restecg: float = Field(
        ...,
        description="Resting ECG results: 0 = Normal, 1 = ST-T Wave Abnormality, 2 = Left Ventricular Hypertrophy",
        examples=[2.0],
    )
    thalach: float = Field(
        ...,
        ge=50.0,
        le=230.0,
        description="Maximum heart rate achieved during exercise stress in bpm (50–230)",
        examples=[130.0],
    )
    exang: float = Field(
        ...,
        description="Exercise-induced angina (0 = No, 1 = Yes)",
        examples=[1.0],
    )
    oldpeak: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="ST depression induced by exercise relative to rest in mm (0.0–10.0)",
        examples=[2.2],
    )
    slope: float = Field(
        ...,
        description="Slope of peak exercise ST segment: 1 = Upsloping, 2 = Flat, 3 = Downsloping",
        examples=[2.0],
    )
    ca: Optional[float] = Field(
        None,
        description="Number of major vessels (0–3) colored by fluoroscopy; null if unknown (handled via median imputation)",
        examples=[2.0],
    )
    thal: Optional[float] = Field(
        None,
        description="Thallium stress test result: 3 = Normal Perfusion, 6 = Fixed Defect, 7 = Reversible Defect; null if unknown (handled via most-frequent imputation)",
        examples=[7.0],
    )

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: float) -> float:
        if v not in (0.0, 1.0):
            raise ValueError("sex must be 0 (Female) or 1 (Male)")
        return float(v)

    @field_validator("cp")
    @classmethod
    def validate_cp(cls, v: float) -> float:
        if v not in (1.0, 2.0, 3.0, 4.0):
            raise ValueError("cp must be one of: 1 (Typical Angina), 2 (Atypical Angina), 3 (Non-Anginal), 4 (Asymptomatic)")
        return float(v)

    @field_validator("fbs")
    @classmethod
    def validate_fbs(cls, v: float) -> float:
        if v not in (0.0, 1.0):
            raise ValueError("fbs must be 0 (<= 120 mg/dl) or 1 (> 120 mg/dl)")
        return float(v)

    @field_validator("restecg")
    @classmethod
    def validate_restecg(cls, v: float) -> float:
        if v not in (0.0, 1.0, 2.0):
            raise ValueError("restecg must be 0 (Normal), 1 (ST-T abnormality), or 2 (LV hypertrophy)")
        return float(v)

    @field_validator("exang")
    @classmethod
    def validate_exang(cls, v: float) -> float:
        if v not in (0.0, 1.0):
            raise ValueError("exang must be 0 (No) or 1 (Yes)")
        return float(v)

    @field_validator("slope")
    @classmethod
    def validate_slope(cls, v: float) -> float:
        if v not in (1.0, 2.0, 3.0):
            raise ValueError("slope must be 1 (Upsloping), 2 (Flat), or 3 (Downsloping)")
        return float(v)

    @field_validator("ca")
    @classmethod
    def validate_ca(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v not in (0.0, 1.0, 2.0, 3.0):
            raise ValueError("ca must be 0, 1, 2, 3 or null")
        return float(v) if v is not None else None

    @field_validator("thal")
    @classmethod
    def validate_thal(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v not in (3.0, 6.0, 7.0):
            raise ValueError("thal must be 3 (Normal), 6 (Fixed Defect), 7 (Reversible Defect), or null")
        return float(v) if v is not None else None


# ── PREDICTION RESPONSE SCHEMA ───────────────────────────────────────────────
class PredictionResponse(BaseModel):
    """Output prediction and risk classification."""
    prediction: int = Field(..., description="Binary prediction: 0 = Low Risk, 1 = Disease Detected")
    prediction_label: str = Field(..., description="Human-readable assessment category")
    risk_category: Literal["Low Risk", "Moderate Risk", "High Risk"] = Field(
        ..., description="Clinical risk tier based on predicted probability"
    )
    disease_probability: float = Field(..., ge=0.0, le=1.0, description="Estimated probability of heart disease presence")
    healthy_probability: float = Field(..., ge=0.0, le=1.0, description="Estimated probability of absence of heart disease")
    confidence_score: float = Field(..., ge=0.5, le=1.0, description="Model confidence in the predicted class (max probability)")
    champion_model: str = Field(..., description="Name of the champion model generating the inference")
    disclaimer: str = Field(..., description="Required non-diagnostic medical disclaimer")


# ── EXPLANATION SCHEMAS ──────────────────────────────────────────────────────
class FeatureAttribution(BaseModel):
    feature: str
    base_feature: str
    display_name: str
    raw_value: Optional[Any] = None
    formatted_value: str
    shap_value: float
    abs_impact: float
    effect: Literal["increases_risk", "decreases_risk", "neutral"]
    unit: str = ""


class AdditiveVerification(BaseModel):
    passed: bool
    max_residual: float
    explanation: str


class PatientExplanationResponse(BaseModel):
    mode: Literal["patient"] = "patient"
    prediction: int
    prediction_label: str
    disease_probability: float
    healthy_probability: float
    base_risk: float
    reconstructed_probability: float
    additive_verification: AdditiveVerification
    top_risk_drivers: List[FeatureAttribution]
    top_risk_reducers: List[FeatureAttribution]
    clinical_features_ranked: List[Dict[str, Any]]
    all_features_ranked: List[FeatureAttribution]
    disclaimer: str


class GlobalRankingItem(BaseModel):
    rank: int
    feature: str
    display_name: str
    type: str
    mean_abs_shap: float
    unit: str = ""


class GlobalExplanationResponse(BaseModel):
    mode: Literal["global"] = "global"
    base_risk: float
    sample_count: int
    feature_count: int
    clinical_feature_count: int
    rankings: List[Dict[str, Any]]
    clinical_rankings: List[GlobalRankingItem]
    beeswarm_plot_available: bool
    bar_plot_available: bool
    explainer_method: Optional[str] = None
    explainer_note: Optional[str] = None


class ExplainRequest(BaseModel):
    mode: Literal["patient", "global"] = Field("patient", description="'patient' for single prediction attribution, 'global' for dataset summary")
    patient_data: Optional[PatientInput] = Field(None, description="Patient input required when mode='patient'")


# ── HEALTH & METADATA SCHEMAS ────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    api_version: str
    model_ready: bool
    champion_model: str
    selection_method: str
    metrics: Dict[str, Any]
    features_expected: int


class ModelMetricItem(BaseModel):
    model: str
    cv_accuracy_mean: float
    cv_accuracy_std: float
    cv_f1_mean: Optional[float] = None
    cv_f1_std: Optional[float] = None
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    test_roc_auc: Optional[float] = None


class ModelComparisonResponse(BaseModel):
    champion_model: str
    selection_method: str
    split_protocol: Dict[str, Any]
    models: List[Dict[str, Any]]


class DatasetStatsResponse(BaseModel):
    total_samples: int
    total_features: int
    class_distribution: Dict[str, Any]
    missing_values: Dict[str, int]
    numerical_features_summary: Dict[str, Dict[str, float]]
    categorical_categories: Dict[str, Dict[str, str]]
