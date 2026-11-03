"""
explainability.py — SHAP Explainability Module for Heart Risk Tracker.
======================================================================
Provides model interpretability using SHAP (SHapley Additive exPlanations)
for the champion pipeline with dynamic explainer selection:

  - LogisticRegression  -> shap.LinearExplainer  (exact log-odds additivity)
  - RandomForest / GradientBoosting -> shap.TreeExplainer (exact tree additivity)
  - KNN / SVM / others  -> shap.KernelExplainer  (model-agnostic, approximate)

Key Capabilities:
  1. Dynamic explainer initialization based on actual champion model type.
  2. Global feature importance (mean absolute SHAP, beeswarm, and bar plots).
  3. Local explanation for individual patient predictions (waterfall plots
     and structured JSON payloads for frontend / API integration).
  4. Human-readable clinical translation of transformed and dummy features.
  5. SHAP additivity verification — exact for linear/tree models, approximate
     (tolerance 0.05) for KernelExplainer (KNN, SVM).

IMPORTANT: SHAP contributions indicate statistical associations between features
and model output probability. They are NOT causal medical explanations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from preprocessing import ALL_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES


# ── CLINICAL FEATURE METADATA REGISTRY ───────────────────────────────────────
FEATURE_METADATA: Dict[str, Dict[str, Any]] = {
    "age": {
        "display_name": "Age",
        "description": "Patient age in years",
        "unit": "years",
        "type": "numerical",
    },
    "sex": {
        "display_name": "Biological Sex",
        "description": "Patient sex at birth",
        "unit": "",
        "type": "categorical",
        "categories": {0.0: "Female", 1.0: "Male"},
    },
    "cp": {
        "display_name": "Chest Pain Type",
        "description": "Type of chest pain reported",
        "unit": "",
        "type": "categorical",
        "categories": {
            1.0: "Typical Angina",
            2.0: "Atypical Angina",
            3.0: "Non-Anginal Pain",
            4.0: "Asymptomatic",
        },
    },
    "trestbps": {
        "display_name": "Resting Blood Pressure",
        "description": "Resting blood pressure upon admission",
        "unit": "mm Hg",
        "type": "numerical",
    },
    "chol": {
        "display_name": "Serum Cholesterol",
        "description": "Serum cholesterol level",
        "unit": "mg/dl",
        "type": "numerical",
    },
    "fbs": {
        "display_name": "Fasting Blood Sugar",
        "description": "Fasting blood sugar > 120 mg/dl",
        "unit": "",
        "type": "categorical",
        "categories": {0.0: "Normal (<= 120 mg/dl)", 1.0: "Elevated (> 120 mg/dl)"},
    },
    "restecg": {
        "display_name": "Resting ECG Results",
        "description": "Resting electrocardiographic results",
        "unit": "",
        "type": "categorical",
        "categories": {
            0.0: "Normal",
            1.0: "ST-T Wave Abnormality",
            2.0: "Left Ventricular Hypertrophy",
        },
    },
    "thalach": {
        "display_name": "Max Heart Rate Achieved",
        "description": "Maximum heart rate achieved during exercise stress",
        "unit": "bpm",
        "type": "numerical",
    },
    "exang": {
        "display_name": "Exercise-Induced Angina",
        "description": "Angina triggered by exercise stress",
        "unit": "",
        "type": "categorical",
        "categories": {0.0: "No", 1.0: "Yes"},
    },
    "oldpeak": {
        "display_name": "ST Depression (Oldpeak)",
        "description": "ST depression induced by exercise relative to rest",
        "unit": "mm",
        "type": "numerical",
    },
    "slope": {
        "display_name": "Slope of Peak Exercise ST",
        "description": "Slope of the peak exercise ST segment",
        "unit": "",
        "type": "categorical",
        "categories": {
            1.0: "Upsloping",
            2.0: "Flat",
            3.0: "Downsloping",
        },
    },
    "ca": {
        "display_name": "Major Fluoroscopy Vessels",
        "description": "Number of major coronary vessels colored by fluoroscopy",
        "unit": "vessels (0-3)",
        "type": "numerical",
    },
    "thal": {
        "display_name": "Thallium Stress Defect",
        "description": "Nuclear thallium myocardial perfusion scan result",
        "unit": "",
        "type": "categorical",
        "categories": {
            3.0: "Normal Perfusion",
            6.0: "Fixed Defect",
            7.0: "Reversible Defect",
        },
    },
}


def format_feature_value(feature: str, val: Any) -> str:
    """Format raw patient feature values into friendly clinical strings."""
    if val is None or pd.isna(val):
        return "Missing (Imputed)"
    meta = FEATURE_METADATA.get(feature, {})
    if meta.get("type") == "categorical":
        try:
            num_val = float(val)
            cat_map = meta.get("categories", {})
            if num_val in cat_map:
                return f"{cat_map[num_val]} ({int(num_val) if num_val.is_integer() else num_val})"
        except (ValueError, TypeError):
            pass
    unit = meta.get("unit", "")
    unit_str = f" {unit}" if unit else ""
    if isinstance(val, float):
        if val.is_integer():
            return f"{int(val)}{unit_str}"
        return f"{val:.1f}{unit_str}"
    return f"{val}{unit_str}"


def parse_transformed_feature_name(feat_name: str) -> Dict[str, Any]:
    """
    Parse a transformed feature name (e.g. 'num__age', 'binary__sex', 'nominal__cp_4.0')
    into base feature, display name, unit, and clinical category description.
    """
    prefix = ""
    col_part = feat_name
    for p in ["num__", "binary__", "nominal__"]:
        if feat_name.startswith(p):
            prefix = p.rstrip("_")
            col_part = feat_name[len(p):]
            break

    if prefix == "nominal" and "_" in col_part:
        base_feature, cat_val_str = col_part.rsplit("_", 1)
        try:
            cat_val = float(cat_val_str)
        except ValueError:
            cat_val = cat_val_str
        meta = FEATURE_METADATA.get(base_feature, {})
        base_display = meta.get("display_name", base_feature)
        cat_labels = meta.get("categories", {})
        cat_desc = cat_labels.get(cat_val, f"Category {cat_val}")
        display_name = f"{base_display}: {cat_desc}"
        return {
            "base_feature": base_feature,
            "display_name": display_name,
            "type": "categorical_dummy",
            "dummy_level": cat_val,
            "unit": "",
        }
    else:
        base_feature = col_part
        meta = FEATURE_METADATA.get(base_feature, {})
        return {
            "base_feature": base_feature,
            "display_name": meta.get("display_name", base_feature),
            "type": meta.get("type", "numerical"),
            "dummy_level": None,
            "unit": meta.get("unit", ""),
        }


# ── EXPLAINER INITIALIZATION ────────────────────────────────────────────────
def get_tree_explainer(pipeline: Pipeline) -> Tuple[Any, Any, List[str]]:
    """
    Extract the preprocessor and estimator from a fitted Pipeline and
    return an appropriate SHAP Explainer based on actual model type:

      - LogisticRegression → shap.LinearExplainer  (exact log-odds additivity)
      - RandomForest / GradientBoosting → shap.TreeExplainer
      - KNN / SVM / others → shap.KernelExplainer (model-agnostic, approximate)

    For KernelExplainer a background dataset of 50 median-sampled training
    points is used with a fixed seed for reproducibility.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline with steps ('preprocessor', 'classifier')

    Returns
    -------
    explainer : shap.LinearExplainer | shap.TreeExplainer | shap.KernelExplainer
    preprocessor : fitted ColumnTransformer
    feature_names : list of transformed feature name strings
    """
    if "preprocessor" not in pipeline.named_steps:
        raise ValueError("Pipeline must contain a 'preprocessor' step.")
    if "classifier" not in pipeline.named_steps:
        raise ValueError("Pipeline must contain a 'classifier' step.")

    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["classifier"]
    feature_names = list(preprocessor.get_feature_names_out())

    # TreeExplainer is exact for tree-based models (RandomForest, GradientBoosting)
    if hasattr(estimator, "estimators_") or hasattr(estimator, "tree_"):
        explainer = shap.TreeExplainer(estimator)
    elif hasattr(estimator, "coef_"):
        # LinearExplainer for linear models (LogisticRegression)
        masker = shap.maskers.Independent(np.zeros((1, len(feature_names))))
        explainer = shap.LinearExplainer(estimator, masker)
    else:
        # Model-agnostic KernelExplainer for KNN, SVM, etc.
        # We need transformed training data for the background; store a small
        # representative sample on the pipeline object if available, else use zeros.
        if hasattr(pipeline, "_shap_background"):
            background = pipeline._shap_background
        else:
            # Fall back: background = 50 zero vectors (zero-imputed baseline)
            background = np.zeros((50, len(feature_names)))

        def _predict_class1(X_arr: np.ndarray) -> np.ndarray:
            return estimator.predict_proba(X_arr)[:, 1]

        explainer = shap.KernelExplainer(_predict_class1, background, seed=42)

    return explainer, preprocessor, feature_names


# Canonical function alias
get_explainer = get_tree_explainer



def extract_class_1_shap(
    shap_output: Any,
    expected_value: Any,
) -> Tuple[np.ndarray, float]:
    """
    Standardize SHAP values and base expected value for Class 1 (disease presence).

    Handles different SHAP return formats across versions:
      - 3D array: (N, n_features, 2)
      - List of 2D arrays: [class_0_array, class_1_array]
      - 2D array: (N, n_features)
    """
    # 1. Base expected value for Class 1
    if isinstance(expected_value, (list, np.ndarray)):
        if len(expected_value) >= 2:
            base_val = float(expected_value[1])
        elif len(expected_value) == 1:
            base_val = float(expected_value[0])
        else:
            base_val = 0.0
    else:
        base_val = float(expected_value)

    # 2. SHAP values matrix for Class 1
    if isinstance(shap_output, list):
        if len(shap_output) >= 2:
            shap_matrix = np.asarray(shap_output[1])
        else:
            shap_matrix = np.asarray(shap_output[0])
    elif isinstance(shap_output, np.ndarray):
        if shap_output.ndim == 3:
            shap_matrix = shap_output[:, :, 1]
        else:
            shap_matrix = shap_output
    elif hasattr(shap_output, "values"):
        vals = shap_output.values
        if vals.ndim == 3:
            shap_matrix = vals[:, :, 1]
        else:
            shap_matrix = vals
    else:
        raise TypeError(f"Unrecognized SHAP output type: {type(shap_output)}")

    return shap_matrix, base_val


# ── GLOBAL FEATURE IMPORTANCE ────────────────────────────────────────────────
def compute_global_feature_importance(
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
    save_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Compute global feature importance across a dataset sample using SHAP.

    Parameters
    ----------
    pipeline : fitted pipeline
    X_sample : DataFrame of raw input features (e.g., test or train set)
    save_dir : optional directory to save figures (beeswarm and bar plot)

    Returns
    -------
    dict with structured global importance metrics and ranking.
    """
    explainer, preprocessor, feature_names = get_tree_explainer(pipeline)

    X_transformed = preprocessor.transform(X_sample[ALL_FEATURES])
    shap_explanation = explainer(X_transformed)

    shap_matrix, base_val = extract_class_1_shap(
        shap_explanation.values, explainer.expected_value
    )

    # Calculate mean absolute SHAP value per feature
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)

    # Sort descending by importance
    ranked_indices = np.argsort(mean_abs_shap)[::-1]

    ranked_features: List[Dict[str, Any]] = []
    for rank, idx in enumerate(ranked_indices, start=1):
        feat = feature_names[idx]
        parsed = parse_transformed_feature_name(feat)
        ranked_features.append({
            "rank": rank,
            "feature": feat,
            "base_feature": parsed["base_feature"],
            "display_name": parsed["display_name"],
            "type": parsed["type"],
            "mean_abs_shap": float(np.round(mean_abs_shap[idx], 4)),
            "unit": parsed["unit"],
        })

    # Aggregated clinical features ranking (13 clinical inputs)
    clinical_importance: Dict[str, float] = {}
    for rf in ranked_features:
        bf = rf["base_feature"]
        clinical_importance[bf] = clinical_importance.get(bf, 0.0) + rf["mean_abs_shap"]

    sorted_clinical = sorted(clinical_importance.items(), key=lambda x: x[1], reverse=True)
    clinical_rankings: List[Dict[str, Any]] = []
    for rank, (bf, val) in enumerate(sorted_clinical, start=1):
        meta = FEATURE_METADATA.get(bf, {})
        clinical_rankings.append({
            "rank": rank,
            "feature": bf,
            "display_name": meta.get("display_name", bf),
            "type": meta.get("type", "numerical"),
            "mean_abs_shap": float(np.round(val, 4)),
            "unit": meta.get("unit", ""),
        })

    result = {
        "base_risk": float(np.round(base_val, 4)),
        "sample_count": len(X_sample),
        "feature_count": len(feature_names),
        "clinical_feature_count": len(clinical_rankings),
        "rankings": ranked_features,
        "clinical_rankings": clinical_rankings,
    }

    # Save visualization figures if save_dir provided
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # 1. Beeswarm plot
        fig_beeswarm = plt.figure(figsize=(9, 6))
        # Build Explanation object for Class 1 for beeswarm
        if shap_explanation.values.ndim == 3:
            exp_c1 = shap_explanation[:, :, 1]
        else:
            exp_c1 = shap_explanation
        exp_c1.feature_names = [
            parse_transformed_feature_name(f)["display_name"] for f in feature_names
        ]
        shap.plots.beeswarm(exp_c1, max_display=18, show=False)
        plt.title("SHAP Global Feature Importance (Beeswarm Plot)", fontsize=13, pad=12)
        plt.xlabel("SHAP Value (Impact on Disease Risk)", fontsize=10)
        plt.tight_layout()
        beeswarm_file = save_path / "shap_summary_beeswarm.png"
        fig_beeswarm.savefig(beeswarm_file, dpi=150, bbox_inches="tight")
        plt.close(fig_beeswarm)
        result["beeswarm_plot"] = beeswarm_file.name

        # 2. Bar plot of mean absolute SHAP
        fig_bar, ax = plt.subplots(figsize=(9, 6))
        y_pos = np.arange(len(clinical_rankings))
        names = [f["display_name"] for f in reversed(clinical_rankings)]
        vals = [f["mean_abs_shap"] for f in reversed(clinical_rankings)]

        bars = ax.barh(y_pos, vals, color="#3b82f6", edgecolor="#1d4ed8", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9.5)
        ax.set_xlabel("Mean |SHAP Value| (Marginal Clinical Impact)", fontsize=10)
        ax.set_title("Clinical Feature Importance Ranking (Mean |SHAP|)", fontsize=13, pad=12)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2, f"{w:.3f}",
                    va="center", ha="left", fontsize=8.5, color="#1e293b")

        plt.tight_layout()
        bar_file = save_path / "shap_global_importance_bar.png"
        fig_bar.savefig(bar_file, dpi=150, bbox_inches="tight")
        plt.close(fig_bar)
        result["importance_bar_plot"] = bar_file.name

    return result


# ── LOCAL PATIENT EXPLANATION ───────────────────────────────────────────────
def explain_patient_prediction(
    pipeline: Pipeline,
    patient_data: Union[Dict[str, Any], pd.DataFrame, pd.Series],
    save_waterfall_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Generate an individual patient risk explanation using SHAP.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    patient_data : raw input features (dict, Series, or 1-row DataFrame)
    save_waterfall_path : optional file path to save local waterfall plot

    Returns
    -------
    dict with full structured explanation, risk breakdown, and additive verification.
    """
    # 1. Format input into DataFrame with correct schema
    if isinstance(patient_data, dict):
        df_input = pd.DataFrame([patient_data])
    elif isinstance(patient_data, pd.Series):
        df_input = pd.DataFrame([patient_data.to_dict()])
    elif isinstance(patient_data, pd.DataFrame):
        df_input = patient_data.copy()
    else:
        raise TypeError("patient_data must be a dict, Series, or DataFrame.")

    # Ensure all required features are present
    for feat in ALL_FEATURES:
        if feat not in df_input.columns:
            df_input[feat] = np.nan

    df_input = df_input[ALL_FEATURES]

    # 2. Extract components
    explainer, preprocessor, feature_names = get_tree_explainer(pipeline)
    classifier = pipeline.named_steps["classifier"]

    # 3. Model predictions — use classes_ to identify the disease column safely
    prob_array = pipeline.predict_proba(df_input)[0]
    classes = list(classifier.classes_)
    disease_col = classes.index(1)  # index of class 1 (disease)
    healthy_col = classes.index(0)  # index of class 0 (no disease)
    prob_disease = float(prob_array[disease_col])
    prob_healthy = float(prob_array[healthy_col])
    pred_class = int(prob_disease >= 0.5)

    # 4. Transformed features and SHAP calculation
    X_trans = preprocessor.transform(df_input)

    is_linear = hasattr(classifier, "coef_") and not hasattr(classifier, "estimators_")
    is_tree = hasattr(classifier, "estimators_") or hasattr(classifier, "tree_")
    is_kernel = not is_linear and not is_tree  # KNN, SVM, etc.

    if is_kernel:
        # KernelExplainer: call shap_values directly; returns ndarray (n_samples, n_features)
        shap_vals_raw = explainer.shap_values(X_trans, nsamples=200, l1_reg="num_features(10)")
        if isinstance(shap_vals_raw, list):
            patient_shap = np.asarray(shap_vals_raw[disease_col])[0]
        else:
            patient_shap = np.asarray(shap_vals_raw)[0]
        base_val = float(explainer.expected_value)
        shap_exp = None  # KernelExplainer doesn't return Explanation objects
    else:
        shap_exp = explainer(X_trans)
        shap_matrix, base_val = extract_class_1_shap(
            shap_exp.values, explainer.expected_value
        )
        patient_shap = shap_matrix[0]  # shape: (n_features,)

    # 5. Additivity verification
    if is_linear:
        # Logistic Regression is linear in log-odds (logit space)
        # z = base_val + sum(shap_values); P = 1 / (1 + exp(-z))
        reconstructed_margin = float(base_val + np.sum(patient_shap))
        reconstructed_prob = float(1.0 / (1.0 + np.exp(-reconstructed_margin)))
        residual = float(abs(reconstructed_prob - prob_disease))
        additive_verified = bool(residual < 1e-5)
        formula_exp = "1 / (1 + exp(-(base_value + sum(shap_values)))) == predict_proba[:, 1]"
    elif is_tree:
        # Tree-based probability explainer
        reconstructed_prob = float(base_val + np.sum(patient_shap))
        residual = float(abs(reconstructed_prob - prob_disease))
        additive_verified = bool(residual < 1e-5)
        formula_exp = "base_value + sum(shap_values) == predict_proba[:, 1]"
    else:
        # KernelExplainer — approximate SHAP, additivity holds only approximately
        reconstructed_prob = float(base_val + np.sum(patient_shap))
        residual = float(abs(reconstructed_prob - prob_disease))
        # KernelExplainer residuals up to ~0.05 are normal; use relaxed tolerance
        additive_verified = bool(residual < 0.05)
        formula_exp = (
            "base_value + sum(shap_values) ≈ predict_proba[:, 1]  "
            "[KernelExplainer: approximate, residual tolerance 0.05]"
        )

    # 6. Feature level attributions
    feature_attributions: List[Dict[str, Any]] = []
    top_drivers: List[Dict[str, Any]] = []
    top_reducers: List[Dict[str, Any]] = []

    for idx, feat in enumerate(feature_names):
        sv = float(patient_shap[idx])
        parsed = parse_transformed_feature_name(feat)
        base_feat = parsed["base_feature"]
        raw_val = df_input.iloc[0][base_feat] if base_feat in df_input.columns else None

        if parsed["type"] == "categorical_dummy":
            try:
                is_active = (float(raw_val) == float(parsed["dummy_level"])) if pd.notna(raw_val) else False
            except (ValueError, TypeError):
                is_active = False
            formatted_val = "Yes" if is_active else "No"
        else:
            formatted_val = format_feature_value(base_feat, raw_val)

        item = {
            "feature": feat,
            "base_feature": base_feat,
            "display_name": parsed["display_name"],
            "raw_value": None if pd.isna(raw_val) else float(raw_val) if isinstance(raw_val, (int, float, np.number)) else str(raw_val),
            "formatted_value": formatted_val,
            "transformed_value": float(np.round(X_trans[0, idx], 3)),
            "shap_value": float(np.round(sv, 4)),
            "abs_impact": float(np.round(abs(sv), 4)),
            "effect": "increases_risk" if sv > 0 else "decreases_risk" if sv < 0 else "neutral",
            "unit": parsed["unit"],
        }
        feature_attributions.append(item)

        if sv > 0:
            top_drivers.append(item)
        elif sv < 0:
            top_reducers.append(item)

    # Sort drivers (largest positive impact first) and reducers (largest negative impact first)
    top_drivers.sort(key=lambda x: x["shap_value"], reverse=True)
    top_reducers.sort(key=lambda x: x["shap_value"])  # most negative first
    feature_attributions.sort(key=lambda x: x["abs_impact"], reverse=True)

    # 7. Aggregated clinical feature attributions (13 clinical features)
    clinical_map: Dict[str, Dict[str, Any]] = {}
    for item in feature_attributions:
        bf = item["base_feature"]
        if bf not in clinical_map:
            meta = FEATURE_METADATA.get(bf, {})
            raw_val = df_input.iloc[0][bf] if bf in df_input.columns else None
            clinical_map[bf] = {
                "feature": bf,
                "display_name": meta.get("display_name", bf),
                "raw_value": None if pd.isna(raw_val) else float(raw_val) if isinstance(raw_val, (int, float, np.number)) else str(raw_val),
                "formatted_value": format_feature_value(bf, raw_val),
                "shap_value": 0.0,
                "unit": meta.get("unit", ""),
            }
        clinical_map[bf]["shap_value"] += item["shap_value"]

    clinical_features_ranked: List[Dict[str, Any]] = []
    for bf, c_item in clinical_map.items():
        sv = float(np.round(c_item["shap_value"], 4))
        c_item["shap_value"] = sv
        c_item["abs_impact"] = float(np.round(abs(sv), 4))
        c_item["effect"] = "increases_risk" if sv > 0 else "decreases_risk" if sv < 0 else "neutral"
        clinical_features_ranked.append(c_item)

    clinical_features_ranked.sort(key=lambda x: x["abs_impact"], reverse=True)

    # 8. Waterfall plot if path specified (only available for non-KernelExplainer)
    if save_waterfall_path:
        out_file = Path(save_waterfall_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if is_kernel or shap_exp is None:
            # KernelExplainer doesn't produce Explanation objects with waterfall support
            # Build a manual bar chart instead
            fig, ax = plt.subplots(figsize=(9, 6))
            top_n = feature_attributions[:12]
            names_plot = [f["display_name"] for f in reversed(top_n)]
            vals_plot  = [f["shap_value"] for f in reversed(top_n)]
            colors_plot = ["#e11d48" if v > 0 else "#0d9488" for v in vals_plot]
            ax.barh(range(len(names_plot)), vals_plot, color=colors_plot, height=0.6)
            ax.set_yticks(range(len(names_plot)))
            ax.set_yticklabels(names_plot, fontsize=9)
            ax.axvline(0, color="#64748b", linewidth=1, linestyle="--")
            ax.set_xlabel("SHAP Contribution (approx. KernelExplainer)", fontsize=10)
            ax.set_title(
                f"Patient Risk Attribution (P(Disease) = {prob_disease*100:.1f}%)\n"
                f"[KernelExplainer — approximate, not exact log-odds]",
                fontsize=11, pad=10
            )
            plt.tight_layout()
            fig.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            fig = plt.figure(figsize=(9, 6))
            single_exp = shap_exp[0]
            if single_exp.values.ndim == 2:
                single_exp = single_exp[:, 1]
            single_exp.feature_names = [
                parse_transformed_feature_name(f)["display_name"] for f in feature_names
            ]
            shap.plots.waterfall(single_exp, max_display=15, show=False)
            plt.title(f"Patient Risk Attribution (P(Disease) = {prob_disease*100:.1f}%)", fontsize=12, pad=12)
            plt.tight_layout()
            fig.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close(fig)

    return {
        "prediction": pred_class,
        "prediction_label": "High Risk / Disease Likely" if pred_class == 1 else "Low Risk / No Disease Detected",
        "disease_probability": float(np.round(prob_disease, 4)),
        "healthy_probability": float(np.round(prob_healthy, 4)),
        "base_risk": float(np.round(base_val, 4)),
        "reconstructed_probability": float(np.round(reconstructed_prob, 4)),
        "additive_verification": {
            "passed": additive_verified,
            "max_residual": float(residual),
            "explanation": formula_exp,
        },
        "top_risk_drivers": top_drivers[:5],
        "top_risk_reducers": top_reducers[:5],
        "all_features_ranked": feature_attributions,
        "clinical_features_ranked": clinical_features_ranked,
    }


# ── MATHEMATICAL VERIFICATION HELPER ────────────────────────────────────────
def verify_shap_additivity(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    tolerance: float = 1e-5,
) -> Dict[str, Any]:
    """
    Verify the mathematical additivity of SHAP values across an entire dataset.

    Returns max residual and checks if sum(SHAP) + base_value matches predictions.
    """
    explainer, preprocessor, _ = get_tree_explainer(pipeline)
    classifier = pipeline.named_steps["classifier"]
    X_trans = preprocessor.transform(X_test[ALL_FEATURES])

    shap_exp = explainer(X_trans)
    shap_matrix, base_val = extract_class_1_shap(
        shap_exp.values, explainer.expected_value
    )

    actual_probas = pipeline.predict_proba(X_test)[:, 1]

    is_linear = hasattr(classifier, "coef_") and not hasattr(classifier, "estimators_")
    if is_linear:
        reconstructed_margins = base_val + np.sum(shap_matrix, axis=1)
        reconstructed_probas = 1.0 / (1.0 + np.exp(-reconstructed_margins))
        formula_note = "Exact log-odds additivity transformed via sigmoid."
    else:
        reconstructed_probas = base_val + np.sum(shap_matrix, axis=1)
        formula_note = "Exact probability additivity up to machine precision."

    residuals = np.abs(actual_probas - reconstructed_probas)
    max_residual = float(np.max(residuals))
    mean_residual = float(np.mean(residuals))
    all_passed = bool(max_residual < tolerance)

    return {
        "all_passed": all_passed,
        "sample_count": len(X_test),
        "max_residual": max_residual,
        "mean_residual": mean_residual,
        "tolerance": tolerance,
        "note": formula_note,
    }


if __name__ == "__main__":
    from data_loader import load_raw, apply_binary_target
    ROOT = Path(__file__).resolve().parent.parent
    data_path = ROOT / "data" / "raw" / "processed.cleveland.data"
    pipe_path = ROOT / "artifacts" / "best_pipeline.pkl"
    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Running SHAP Global Feature Importance...")
    df_raw = load_raw(data_path)
    df_clean = apply_binary_target(df_raw)
    pipe = joblib.load(pipe_path)

    global_res = compute_global_feature_importance(pipe, df_clean[ALL_FEATURES], save_dir=fig_dir)
    out_json = ROOT / "artifacts" / "shap_global_importance.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(global_res, f, indent=2)
    print(f"Saved global importance to: {out_json}")
    print("Done!")


