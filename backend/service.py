"""
service.py — ML inference and explanation service.
==================================================
Manages the model lifecycle, caching, prediction calculations, and SHAP execution.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from backend.config import (
    ALL_RESULTS_PATH,
    BEST_PIPELINE_PATH,
    MEDICAL_DISCLAIMER,
    ML_DIR,
    MODEL_META_PATH,
    RAW_DATA_PATH,
    SHAP_GLOBAL_PATH,
)

# Ensure ml/src is importable
if str(ML_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ML_DIR / "src"))

from preprocessing import ALL_FEATURES
from data_loader import load_raw, apply_binary_target
from explainability import (
    explain_patient_prediction,
    compute_global_feature_importance,
    FEATURE_METADATA,
)

logger = logging.getLogger("backend.service")


class MLService:
    """Singleton service for model loading, prediction, and SHAP explainability."""

    def __init__(self) -> None:
        self.pipeline: Optional[Pipeline] = None
        self.model_meta: Dict[str, Any] = {}
        self.all_results: Dict[str, Any] = {}
        self.global_shap_data: Dict[str, Any] = {}
        self.dataset_stats: Dict[str, Any] = {}
        self.is_loaded: bool = False

    def load_artifacts(self) -> None:
        """Load trained pipeline and metadata from disk."""
        logger.info("Loading ML artifacts...")
        if not BEST_PIPELINE_PATH.exists():
            raise FileNotFoundError("Trained model pipeline artifact not found.")

        self.pipeline = joblib.load(BEST_PIPELINE_PATH)
        logger.info("Successfully loaded champion model pipeline.")

        if MODEL_META_PATH.exists():
            with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
                self.model_meta = json.load(f)

        if ALL_RESULTS_PATH.exists():
            with open(ALL_RESULTS_PATH, "r", encoding="utf-8") as f:
                self.all_results = json.load(f)

        if SHAP_GLOBAL_PATH.exists():
            with open(SHAP_GLOBAL_PATH, "r", encoding="utf-8") as f:
                self.global_shap_data = json.load(f)

        self._compute_dataset_stats()
        self.is_loaded = True
        logger.info("ML Service is fully initialized and ready.")

    def _compute_dataset_stats(self) -> None:
        """Compute verified summary stats directly from dataset without leakage."""
        if not RAW_DATA_PATH.exists():
            return

        df_raw = load_raw(RAW_DATA_PATH)
        df_binary = apply_binary_target(df_raw)

        counts = df_binary["target_binary"].value_counts().to_dict()
        missing = df_raw.isnull().sum().to_dict()

        num_summary: Dict[str, Dict[str, float]] = {}
        for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
            s = df_raw[col].dropna()
            num_summary[col] = {
                "mean": float(np.round(s.mean(), 2)),
                "median": float(np.round(s.median(), 2)),
                "std": float(np.round(s.std(), 2)),
                "min": float(np.round(s.min(), 2)),
                "max": float(np.round(s.max(), 2)),
            }

        # Categories mapping
        cat_map = {
            "sex": {"0.0": "Female", "1.0": "Male"},
            "cp": {
                "1.0": "Typical Angina",
                "2.0": "Atypical Angina",
                "3.0": "Non-Anginal Pain",
                "4.0": "Asymptomatic",
            },
            "fbs": {"0.0": "Normal (<= 120 mg/dl)", "1.0": "Elevated (> 120 mg/dl)"},
            "restecg": {
                "0.0": "Normal",
                "1.0": "ST-T Wave Abnormality",
                "2.0": "Left Ventricular Hypertrophy",
            },
            "exang": {"0.0": "No", "1.0": "Yes"},
            "slope": {"1.0": "Upsloping", "2.0": "Flat", "3.0": "Downsloping"},
            "thal": {
                "3.0": "Normal Perfusion",
                "6.0": "Fixed Defect",
                "7.0": "Reversible Defect",
            },
        }

        self.dataset_stats = {
            "total_samples": len(df_raw),
            "total_features": 13,
            "class_distribution": {
                "no_disease_count": int(counts.get(0, 0)),
                "no_disease_pct": float(np.round(counts.get(0, 0) / len(df_raw) * 100, 1)),
                "disease_present_count": int(counts.get(1, 0)),
                "disease_present_pct": float(np.round(counts.get(1, 0) / len(df_raw) * 100, 1)),
            },
            "missing_values": {k: int(v) for k, v in missing.items() if v > 0},
            "numerical_features_summary": num_summary,
            "categorical_categories": cat_map,
        }

    def predict(self, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference using the champion pipeline.
        Expects a dict of the 13 raw input features.
        """
        if not self.is_loaded or self.pipeline is None:
            raise RuntimeError("Model pipeline is not loaded.")

        df_in = pd.DataFrame([patient_dict])[ALL_FEATURES]

        probas = self.pipeline.predict_proba(df_in)[0]
        classifier = self.pipeline.named_steps["classifier"]
        classes = list(classifier.classes_)
        healthy_col = classes.index(0)
        disease_col = classes.index(1)
        prob_healthy = float(np.round(probas[healthy_col], 4))
        prob_disease = float(np.round(probas[disease_col], 4))

        # Risk stratification
        if prob_disease < 0.35:
            risk_cat = "Low Risk"
            pred_class = 0
            label = "Low Risk — Cardiovascular Disease Less Likely"
        elif prob_disease < 0.65:
            risk_cat = "Moderate Risk"
            pred_class = 1 if prob_disease >= 0.5 else 0
            label = "Moderate Risk — Further Clinical Evaluation Recommended"
        else:
            risk_cat = "High Risk"
            pred_class = 1
            label = "Elevated Risk — Cardiovascular Disease Indicated"

        conf_score = float(np.round(max(prob_disease, prob_healthy), 4))
        model_name = self.model_meta.get("best_model", type(self.pipeline.named_steps["classifier"]).__name__)

        return {
            "prediction": pred_class,
            "prediction_label": label,
            "risk_category": risk_cat,
            "disease_probability": prob_disease,
            "healthy_probability": prob_healthy,
            "confidence_score": conf_score,
            "champion_model": model_name,
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    def explain_patient(self, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Compute local SHAP explanation for an individual patient."""
        if not self.is_loaded or self.pipeline is None:
            raise RuntimeError("Model pipeline is not loaded.")

        explanation = explain_patient_prediction(self.pipeline, patient_dict)
        explanation["disclaimer"] = MEDICAL_DISCLAIMER
        return explanation

    def get_global_explanation(self) -> Dict[str, Any]:
        """Return dataset-level global SHAP feature importance."""
        if self.global_shap_data:
            data = dict(self.global_shap_data)
            data["mode"] = "global"
            data["beeswarm_plot_available"] = (ML_DIR / "reports" / "figures" / "shap_summary_beeswarm.png").exists()
            data["bar_plot_available"] = (ML_DIR / "reports" / "figures" / "shap_global_importance_bar.png").exists()
            return data

        # Fallback: compute on the fly if file missing
        df_raw = load_raw(RAW_DATA_PATH)
        df_clean = apply_binary_target(df_raw)
        data = compute_global_feature_importance(self.pipeline, df_clean[ALL_FEATURES])
        data["mode"] = "global"
        data["beeswarm_plot_available"] = False
        data["bar_plot_available"] = False
        return data


# Instantiate global singleton
ml_service = MLService()
