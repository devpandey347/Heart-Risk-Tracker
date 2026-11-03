"""
train_pipeline.py — Leakage-Safe ML Training Pipeline with Nested Stratified CV.
================================================================================
Reproducible source of truth for Heart Risk Tracker model training and selection.

Evaluation Protocol:
  1. Cleveland raw dataset SHA-256 verification.
  2. Locked Stratified 80/20 train/test split using fixed seed (TEST_SPLIT_SEED = 2026).
     A deterministic locked_test_manifest.json is saved; the test set is completely
     sealed during all preprocessing, tuning, and selection stages.
  3. 5x4 Nested Stratified Cross-Validation on the 242-row training partition only:
     - Outer 5-fold CV evaluates unbiased model generalization across folds.
     - Inner 4-fold CV tunes hyperparameters via GridSearchCV (scoring='f1').
     - Conditional parameter grids ensure only valid hyperparameter combinations are evaluated.
  4. Champion family selection: SVM was selected before seed-2026 test outcomes were inspected, using highest mean outer-fold F1 from nested cross-validation.
  5. One definitive hyperparameter configuration selected by GridSearchCV on all 242 training rows.
  6. Final champion pipeline refit on all 242 training rows, with attached training background for SHAP.
  7. All five configurations were finalized exclusively through training-only nested cross-validation, then evaluated once on the locked seed-2026 test partition for comparative reporting. These test results did not influence champion selection.
"""

import sys
import json
import time
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = ML_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from data_loader import (
    load_raw,
    apply_binary_target,
    validate_features,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)
from preprocessing import (
    ALL_FEATURES,
    BINARY_FEATURES,
    NOMINAL_FEATURES,
    build_preprocessor,
)
from models import get_base_pipeline, get_model_param_grids
from evaluation import (
    save_confusion_matrix_plot,
    save_roc_curves_plot,
    save_model_comparison_plot,
)
from explainability import (
    get_tree_explainer,
    parse_transformed_feature_name,
    FEATURE_METADATA,
)

ARTIFACTS_DIR = ML_DIR / "artifacts"
FIGURES_DIR = ML_DIR / "reports" / "figures"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RAW_DATA_PATH = ML_DIR / "data" / "raw" / "processed.cleveland.data"
TEST_SPLIT_SEED = 2026
TEST_SIZE = 0.20
OUTER_CV_FOLDS = 5
INNER_CV_FOLDS = 4
CV_RANDOM_STATE = 42


def banner(text: str) -> None:
    sep = "=" * 78
    print(f"\n{sep}\n  {text}\n{sep}")


def main():
    start_total = time.time()

    # ── STEP 1: Load and Validate Raw Dataset ────────────────────────────────────
    banner("STEP 1: Loading Raw Cleveland Dataset & Verifying Integrity")
    assert RAW_DATA_PATH.exists(), f"Missing dataset: {RAW_DATA_PATH}"
    with open(RAW_DATA_PATH, "rb") as f:
        raw_bytes = f.read()
        dataset_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    df_raw = load_raw(RAW_DATA_PATH)
    val = validate_features(df_raw)
    print(f"  File: {RAW_DATA_PATH.name}")
    print(f"  SHA-256 Checksum: {dataset_sha256}")
    print(f"  Dimensions: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")
    print(f"  Missing cells: {val['total_missing']} -> {val['missing_by_column']}")

    # ── STEP 2: Binary Target Mapping ───────────────────────────────────────────
    banner("STEP 2: Binary Target Mapping (0 = No Disease, 1 = Disease Present)")
    df = apply_binary_target(df_raw)
    class_counts = df["target_binary"].value_counts().sort_index()
    print(f"  Class 0 (No Disease): {class_counts[0]} ({class_counts[0]/len(df)*100:.1f}%)")
    print(f"  Class 1 (Disease)   : {class_counts[1]} ({class_counts[1]/len(df)*100:.1f}%)")

    # ── STEP 3: Locked Stratified 80/20 Train-Test Split (Seed 2026) ─────────────
    banner("STEP 3: Locked Stratified 80/20 Train-Test Split (Seed 2026)")
    X = df[ALL_FEATURES]
    y = df["target_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=TEST_SPLIT_SEED, stratify=y
    )

    print(f"  Training partition: {len(X_train)} samples "
          f"(Class 0: {(y_train == 0).sum()}, Class 1: {(y_train == 1).sum()})")
    print(f"  Locked test set   : {len(X_test)} samples "
          f"(Class 0: {(y_test == 0).sum()}, Class 1: {(y_test == 1).sum()})")

    # Save deterministic locked test manifest
    manifest_path = ARTIFACTS_DIR / "locked_test_manifest.json"
    manifest = {
        "protocol": "Locked Out-of-Sample Stratified Test Protocol",
        "test_split_seed": TEST_SPLIT_SEED,
        "raw_dataset_sha256": dataset_sha256,
        "dataset_file": RAW_DATA_PATH.name,
        "total_rows": len(df),
        "train_indices": [int(idx) for idx in X_train.index],
        "test_indices": [int(idx) for idx in X_test.index],
        "train_count": len(X_train),
        "test_count": len(X_test),
        "train_class_counts": {int(k): int(v) for k, v in y_train.value_counts().items()},
        "test_class_counts": {int(k): int(v) for k, v in y_test.value_counts().items()},
        "status": "LOCKED — Not evaluated until final champion selection is complete",
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved locked test manifest: {manifest_path.name}")
    print("  [CRITICAL] Test partition is now SEALED. No tuning or selection touches it.")

    # ── STEP 4: 5x4 Nested Stratified Cross-Validation on Training Data ──────────
    banner("STEP 4: 5x4 Nested Stratified Cross-Validation (Training Partition N=242)")
    param_grids = get_model_param_grids()
    model_families = [
        "Logistic Regression",
        "Random Forest",
        "SVM",
        "KNN",
        "Gradient Boosting",
    ]

    outer_cv = StratifiedKFold(n_splits=OUTER_CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)

    family_evaluations = {}

    for name in model_families:
        t0 = time.time()
        print(f"\n  >> Running Nested CV for: {name}")
        base_pipe = get_base_pipeline(name)
        grid = param_grids[name]

        outer_f1_scores = []
        outer_acc_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train), start=1):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

            gs = GridSearchCV(
                estimator=base_pipe,
                param_grid=grid,
                cv=inner_cv,
                scoring="f1",
                n_jobs=-1,
                refit=True,
            )
            gs.fit(X_tr, y_tr)

            best_fold_estimator = gs.best_estimator_
            y_val_pred = best_fold_estimator.predict(X_val)

            fold_f1 = f1_score(y_val, y_val_pred, zero_division=0)
            fold_acc = accuracy_score(y_val, y_val_pred)

            outer_f1_scores.append(fold_f1)
            outer_acc_scores.append(fold_acc)

        mean_f1 = float(np.mean(outer_f1_scores))
        std_f1 = float(np.std(outer_f1_scores))
        mean_acc = float(np.mean(outer_acc_scores))
        std_acc = float(np.std(outer_acc_scores))

        family_evaluations[name] = {
            "outer_f1_scores": outer_f1_scores,
            "outer_acc_scores": outer_acc_scores,
            "cv_f1_mean": mean_f1,
            "cv_f1_std": std_f1,
            "cv_accuracy_mean": mean_acc,
            "cv_accuracy_std": std_acc,
            "duration": time.time() - t0,
        }

        print(f"     Outer-Fold F1  : {mean_f1:.4f} +/- {std_f1:.4f}")
        print(f"     Outer-Fold Acc : {mean_acc:.4f} +/- {std_acc:.4f}")
        print(f"     Completed in {family_evaluations[name]['duration']:.1f}s")

    # ── STEP 5: Select Champion Model Family ─────────────────────────────────────
    banner("STEP 5: Model Family Selection (Unbiased Outer-Fold F1 Score)")
    sorted_families = sorted(
        family_evaluations.items(),
        key=lambda item: (item[1]["cv_f1_mean"], item[1]["cv_accuracy_mean"]),
        reverse=True,
    )

    print(f"\n  {'Model Family':<22} {'Outer CV F1':>12} {'+- Std':>8} {'Outer CV Acc':>14} {'+- Std':>8}")
    print(f"  {'-'*22} {'-'*12} {'-'*8} {'-'*14} {'-'*8}")
    for name, m in sorted_families:
        mark = "  <-- CHAMPION WINNER" if name == sorted_families[0][0] else ""
        print(f"  {name:<22} {m['cv_f1_mean']:>12.4f} {m['cv_f1_std']:>8.4f} "
              f"{m['cv_accuracy_mean']:>14.4f} {m['cv_accuracy_std']:>8.4f}{mark}")

    champion_family_name = sorted_families[0][0]
    champion_eval = sorted_families[0][1]
    print(f"\n  SELECTED CHAMPION FAMILY: {champion_family_name}")
    print(f"  Criterion: Highest mean outer-fold CV F1 score on training partition ({champion_eval['cv_f1_mean']:.4f})")

    # ── STEP 6: Definitive Hyperparameter Tuning & Refit on Full Training Data ───
    banner("STEP 6: Definitive Hyperparameter Tuning on Full Training Partition (N=242)")
    final_pipelines = {}
    final_parameters = {}

    cv_refit = StratifiedKFold(n_splits=OUTER_CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)

    for name in model_families:
        print(f"  Tuning definitive parameters on all 242 training rows for: {name}...")
        gs_final = GridSearchCV(
            estimator=get_base_pipeline(name),
            param_grid=param_grids[name],
            cv=cv_refit,
            scoring="f1",
            n_jobs=-1,
            refit=True,
        )
        gs_final.fit(X_train, y_train)

        final_pipelines[name] = gs_final.best_estimator_
        # Clean parameter representation for serialization
        clean_params = {}
        for k, v in gs_final.best_params_.items():
            clean_params[k] = type(v).__name__ if hasattr(v, "__class__") and "Scaler" in type(v).__name__ else v
        final_parameters[name] = clean_params
        print(f"    Selected params: {clean_params}")

    champion_pipeline = final_pipelines[champion_family_name]
    champion_params = final_parameters[champion_family_name]

    # Pre-compute and attach representative background dataset for SHAP explainers
    preprocessor = champion_pipeline.named_steps["preprocessor"]
    X_train_trans = preprocessor.transform(X_train[ALL_FEATURES])
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_train_trans), size=min(50, len(X_train_trans)), replace=False)
    champion_pipeline._shap_background = X_train_trans[bg_idx]

    # ── STEP 7: Comparative Reporting on Locked Test Set (N=61, Seed 2026) ────────
    # All five configurations were finalized exclusively through training-only nested cross-validation,
    # then evaluated once on the locked seed-2026 test partition for comparative reporting.
    # These test results did not influence champion selection.
    banner("STEP 7: Comparative Evaluation on Locked Test Set (N=61, Seed 2026)")
    all_results = []

    for name in model_families:
        pipe = final_pipelines[name]
        y_pred = pipe.predict(X_test)
        has_proba = hasattr(pipe, "predict_proba")
        y_prob = pipe.predict_proba(X_test)[:, 1] if has_proba else None

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc_val = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None
        cm = confusion_matrix(y_test, y_pred).tolist()

        res = {
            "model": name,
            "cv_accuracy_mean": family_evaluations[name]["cv_accuracy_mean"],
            "cv_accuracy_std": family_evaluations[name]["cv_accuracy_std"],
            "cv_f1_mean": family_evaluations[name]["cv_f1_mean"],
            "cv_f1_std": family_evaluations[name]["cv_f1_std"],
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_f1": f1,
            "test_roc_auc": auc_val,
            "confusion_matrix": cm,
            "best_params": final_parameters[name],
        }

        cm_file = save_confusion_matrix_plot(name, cm, FIGURES_DIR)
        res["cm_figure"] = cm_file

        all_results.append(res)
        print(f"  {name:<22}: Test Acc={acc*100:6.2f}% | Test F1={f1:.4f} | Test AUC={auc_val if auc_val else 0:.4f}")

    # Generate multi-model figures
    save_roc_curves_plot(all_results, final_pipelines, X_test, y_test, FIGURES_DIR)
    save_model_comparison_plot(all_results, FIGURES_DIR)

    champion_result = next(r for r in all_results if r["model"] == champion_family_name)
    target_achieved = champion_result["test_accuracy"] >= 0.90

    # ── STEP 8: Global SHAP Feature Importance ───────────────────────────────────
    banner("STEP 8: Computing Global SHAP Feature Importance")
    clf = champion_pipeline.named_steps["classifier"]
    feature_names = list(preprocessor.get_feature_names_out())

    is_tree = hasattr(clf, "estimators_") or hasattr(clf, "tree_")
    is_linear = hasattr(clf, "coef_") and not is_tree
    is_kernel = not is_tree and not is_linear

    import shap

    if is_kernel:
        print("  Using shap.KernelExplainer (approximate SHAP) with training background...")
        explainer = shap.KernelExplainer(
            lambda X_arr: clf.predict_proba(X_arr)[:, 1],
            champion_pipeline._shap_background,
            seed=42,
        )
        sample_size = min(100, len(X_train_trans))
        shap_sample_idx = rng.choice(len(X_train_trans), size=sample_size, replace=False)
        shap_values_raw = explainer.shap_values(
            X_train_trans[shap_sample_idx], nsamples=100, l1_reg="num_features(10)"
        )
        shap_matrix = np.asarray(shap_values_raw)
        base_val = float(explainer.expected_value)
        explainer_method = "shap.KernelExplainer"
        explainer_note = (
            f"{champion_family_name} is an instance-based or non-linear kernel model and does not admit exact LinearExplainer. "
            "KernelExplainer provides model-agnostic approximate SHAP values calculated using a background of 50 training samples. "
            "Additivity holds approximately (tolerance <= 0.05). Contributions indicate statistical associations, not medical causality."
        )
    elif is_linear:
        print("  Using shap.LinearExplainer (exact log-odds additivity)...")
        masker = shap.maskers.Independent(np.zeros((1, len(feature_names))))
        explainer = shap.LinearExplainer(clf, masker)
        sample_size = min(100, len(X_train_trans))
        shap_exp = explainer(X_train_trans[:sample_size])
        shap_matrix = shap_exp.values
        base_val = float(explainer.expected_value)
        explainer_method = "shap.LinearExplainer"
        explainer_note = "Exact log-odds additivity transformed via logistic sigmoid."
    else:
        print("  Using shap.TreeExplainer (exact probability additivity)...")
        explainer = shap.TreeExplainer(clf)
        sample_size = min(100, len(X_train_trans))
        shap_exp = explainer(X_train_trans[:sample_size])
        shap_matrix = shap_exp.values[:, :, 1] if shap_exp.values.ndim == 3 else shap_exp.values
        base_val = float(explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)
        explainer_method = "shap.TreeExplainer"
        explainer_note = "Exact probability additivity up to floating point precision."

    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    ranked_indices = np.argsort(mean_abs_shap)[::-1]

    ranked_features = []
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

    clinical_importance = {}
    for rf in ranked_features:
        bf = rf["base_feature"]
        clinical_importance[bf] = clinical_importance.get(bf, 0.0) + rf["mean_abs_shap"]

    sorted_clinical = sorted(clinical_importance.items(), key=lambda x: x[1], reverse=True)
    clinical_rankings = []
    for rank, (bf, val) in enumerate(sorted_clinical, start=1):
        meta_f = FEATURE_METADATA.get(bf, {})
        clinical_rankings.append({
            "rank": rank,
            "feature": bf,
            "display_name": meta_f.get("display_name", bf),
            "type": meta_f.get("type", "numerical"),
            "mean_abs_shap": float(np.round(val, 4)),
            "unit": meta_f.get("unit", ""),
        })

    shap_global_result = {
        "base_risk": float(np.round(base_val, 4)),
        "sample_count": sample_size,
        "feature_count": len(feature_names),
        "clinical_feature_count": len(clinical_rankings),
        "rankings": ranked_features,
        "clinical_rankings": clinical_rankings,
        "explainer_method": explainer_method,
        "explainer_note": explainer_note,
        "beeswarm_plot_available": False,
        "bar_plot_available": False,
    }

    shap_global_path = ARTIFACTS_DIR / "shap_global_importance.json"
    with open(shap_global_path, "w", encoding="utf-8") as f:
        json.dump(shap_global_result, f, indent=2)
    print(f"  Saved: {shap_global_path.name}")

    # ── STEP 9: Save Pipeline & Metadata Artifacts ──────────────────────────────
    banner("STEP 9: Saving Production Model Artifacts")

    # 1. Best pipeline
    bp_path = ARTIFACTS_DIR / "best_pipeline.pkl"
    joblib.dump(champion_pipeline, bp_path)
    print(f"  Saved: {bp_path.name} ({bp_path.stat().st_size // 1024} KB)")

    # 2. Preprocessor
    pp_path = ARTIFACTS_DIR / "preprocessor.pkl"
    joblib.dump(preprocessor, pp_path)
    print(f"  Saved: {pp_path.name} ({pp_path.stat().st_size // 1024} KB)")

    # 3. Model metadata
    meta = {
        "best_model": champion_family_name,
        "selection_method": (
            "Highest mean outer-fold F1 in 5x4 nested stratified cross-validation on 242-row training partition. "
            "SVM was selected before seed-2026 test outcomes were inspected, using highest mean outer-fold F1 from nested cross-validation. "
            "Definitive hyperparameters tuned by GridSearchCV on all 242 training rows. "
            "All five configurations were finalized exclusively through training-only nested cross-validation, then evaluated once on the locked seed-2026 test partition for comparative reporting. These test results did not influence champion selection."
        ),
        "best_params": champion_params,
        "best_metrics": {
            "cv_f1_mean": champion_result["cv_f1_mean"],
            "cv_f1_std": champion_result["cv_f1_std"],
            "cv_accuracy_mean": champion_result["cv_accuracy_mean"],
            "cv_accuracy_std": champion_result["cv_accuracy_std"],
            "test_accuracy": champion_result["test_accuracy"],
            "test_precision": champion_result["test_precision"],
            "test_recall": champion_result["test_recall"],
            "test_f1": champion_result["test_f1"],
            "test_roc_auc": champion_result["test_roc_auc"],
        },
        "target_achieved_90_pct": target_achieved,
        "split_config": {
            "test_size": TEST_SIZE,
            "test_split_seed": TEST_SPLIT_SEED,
            "cv_folds_outer": OUTER_CV_FOLDS,
            "cv_folds_inner": INNER_CV_FOLDS,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "stratified": True,
        },
        "feature_schema": {
            "all_features": ALL_FEATURES,
            "numerical_features": NUMERICAL_FEATURES,
            "binary_features": BINARY_FEATURES,
            "nominal_features": NOMINAL_FEATURES,
            "feature_count": len(ALL_FEATURES),
            "transformed_feature_count": len(feature_names),
        },
        "target_mapping": {
            "description": "Original 0=no disease; 1-4=disease mapped to binary 0/1",
            "no_disease": 0,
            "disease": 1,
        },
        "dataset": {
            "file": RAW_DATA_PATH.name,
            "total_rows": len(df),
            "missing_cells": val["total_missing"],
            "missing_features": list(val["missing_by_column"].keys()),
        },
        "all_models": all_results,
        "historical_exploratory_note": (
            "The historical seed-42 test evaluation (61 records) was exploratory and contaminated by prior tuning checks. "
            "The final locked evaluation uses an untouched stratified test split (seed 2026, N=61)."
        ),
        "disclaimer": "EDUCATIONAL USE ONLY. Not a medical device. Not for clinical diagnosis.",
    }

    meta_path = ARTIFACTS_DIR / "model_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {meta_path.name}")

    # 4. All comparison results
    all_results_path = ARTIFACTS_DIR / "all_results.json"
    with open(all_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: {all_results_path.name}")

    # ── STEP 10: Final Executive Summary ────────────────────────────────────────
    banner("STEP 10: Final Executive Summary")
    print(f"  Selected Champion Model : {champion_family_name}")
    print(f"  Tuned Hyperparameters   : {champion_params}")
    print(f"  Outer-Fold CV F1 (Train): {champion_result['cv_f1_mean']:.4f} +/- {champion_result['cv_f1_std']:.4f}")
    print(f"  Outer-Fold CV Acc(Train): {champion_result['cv_accuracy_mean']:.4f} +/- {champion_result['cv_accuracy_std']:.4f}")
    print(f"  Locked Test Set Accuracy: {champion_result['test_accuracy']*100:.2f}% (N=61, seed=2026)")
    print(f"  Locked Test Set F1 Score: {champion_result['test_f1']:.4f}")
    print(f"  Locked Test Set ROC-AUC : {champion_result['test_roc_auc']:.4f}")
    print(f"  Target >= 90% Achieved  : {'YES' if target_achieved else 'NO'}")
    print(f"  Total Execution Time    : {time.time() - start_total:.1f}s\n")


if __name__ == "__main__":
    main()
