"""
================================================================================
@deprecated — HISTORICAL SCRIPT ONLY (CHUNK 10 ARTIFACT)
================================================================================
DO NOT RUN THIS SCRIPT FOR PRODUCTION OR BENCHMARKING.
This script was used in Chunk 10 for a manual champion assignment experiment.
The sole source of truth for model training, tuning, artifact generation, and
reproducible model selection is now ml/src/train_pipeline.py.
================================================================================

switch_to_knn.py — Retrain champion as KNN and update all artifacts (HISTORICAL).

Run from project root:
    python scripts/switch_to_knn.py

What this script does:
  1. Backs up current artifacts (best_pipeline.pkl, model_meta.json, shap_global_importance.json)
  2. Re-loads the same stratified split used in train_pipeline.py (same seed = same split)
  3. Trains the KNN pipeline (k=7, distance-weighted, euclidean)
  4. Saves as best_pipeline.pkl
  5. Updates model_meta.json: best_model -> "KNN"
  6. Recomputes global SHAP with KernelExplainer (background=50-sample median)
  7. Saves shap_global_importance.json
"""

import json, sys, shutil, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / "ml"
ARTIFACTS_DIR = ML_DIR / "artifacts"
sys.path.insert(0, str(ML_DIR / "src"))

import numpy as np
import pandas as pd
import joblib
import shap

from data_loader import load_raw, apply_binary_target
from preprocessing import ALL_FEATURES, build_preprocessor
from models import get_model_configs

# ── Step 1: Backup existing artifacts ────────────────────────────────────────
print("Step 1: Backing up current artifacts...")
files_to_backup = [
    ("best_pipeline.pkl",         "best_pipeline_lr_historical.pkl"),
    ("model_meta.json",           "model_meta_lr_historical.json"),
    ("shap_global_importance.json","shap_global_importance_lr_historical.json"),
]
for src_name, dst_name in files_to_backup:
    src = ARTIFACTS_DIR / src_name
    dst = ARTIFACTS_DIR / dst_name
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        print(f"  Backed up: {src_name} -> {dst_name}")
    elif src.exists() and dst.exists():
        print(f"  Backup already exists: {dst_name} (skipped)")
    else:
        print(f"  Source not found, skipping: {src_name}")

# ── Step 2: Recreate the same stratified split ─────────────────────────────
print("\nStep 2: Loading data and recreating identical 80/20 split (seed=42)...")
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

data_path = ML_DIR / "data" / "raw" / "processed.cleveland.data"
df = apply_binary_target(load_raw(data_path))
X = df[ALL_FEATURES]
y = df["target_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

# ── Step 3: Train KNN pipeline ────────────────────────────────────────────
print("\nStep 3: Training KNN pipeline (k=7, weights=distance, metric=euclidean)...")
configs = get_model_configs()
knn_pipeline = configs["KNN"]

# 5-fold CV on training data only (for reporting)
cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(knn_pipeline, X_train, y_train, cv=cv_splitter, scoring="accuracy", n_jobs=-1)
cv_f1  = cross_val_score(knn_pipeline, X_train, y_train, cv=cv_splitter, scoring="f1", n_jobs=-1)

print(f"  CV Accuracy : {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
print(f"  CV F1 Score : {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")

# Fit on full training partition
knn_pipeline.fit(X_train, y_train)

# Verify class order
clf = knn_pipeline.named_steps["classifier"]
print(f"  Classifier classes_: {clf.classes_}  (index 0 = no disease, index 1 = disease)")
assert list(clf.classes_) == [0, 1], "Class order unexpected! Verify class mapping."

# Test-set evaluation (for reporting only — not used for selection)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
y_pred = knn_pipeline.predict(X_test)
y_prob = knn_pipeline.predict_proba(X_test)[:, 1]

test_acc  = accuracy_score(y_test, y_pred)
test_prec = precision_score(y_test, y_pred, zero_division=0)
test_rec  = recall_score(y_test, y_pred, zero_division=0)
test_f1   = f1_score(y_test, y_pred, zero_division=0)
test_auc  = roc_auc_score(y_test, y_prob)

print(f"\n  Held-out test metrics (N=61, reported after selection — not used for selection):")
print(f"  Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"  Test F1       : {test_f1:.4f}")
print(f"  Test ROC-AUC  : {test_auc:.4f}")

# ── Step 4: Save best_pipeline.pkl ────────────────────────────────────────
print("\nStep 4: Saving KNN pipeline as best_pipeline.pkl...")
bp_path = ARTIFACTS_DIR / "best_pipeline.pkl"
joblib.dump(knn_pipeline, bp_path)
print(f"  Saved: {bp_path}  ({bp_path.stat().st_size // 1024} KB)")

# ── Step 5: Update model_meta.json ───────────────────────────────────────
print("\nStep 5: Updating model_meta.json...")
meta_path = ARTIFACTS_DIR / "model_meta.json"
with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

meta["best_model"] = "KNN"
meta["selection_method"] = (
    "MANUALLY REASSIGNED to KNN for Chunk 10. "
    "Original automatic selection was Logistic Regression (highest CV F1 = 0.8214). "
    "KNN is reassigned per project requirements; historical LR artifacts preserved."
)
meta["best_metrics"] = {
    "cv_accuracy_mean": float(cv_acc.mean()),
    "cv_accuracy_std":  float(cv_acc.std()),
    "cv_f1_mean":       float(cv_f1.mean()),
    "cv_f1_std":        float(cv_f1.std()),
    "test_accuracy":    float(test_acc),
    "test_precision":   float(test_prec),
    "test_recall":      float(test_rec),
    "test_f1":          float(test_f1),
    "test_roc_auc":     float(test_auc),
}
meta["knn_config"] = {
    "n_neighbors": 7,
    "weights": "distance",
    "metric": "euclidean",
    "note": (
        "KNN is a non-parametric, instance-based classifier. "
        "It cannot provide exact additive SHAP explanations; "
        "approximate KernelExplainer (SHAP) is used instead."
    ),
}

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print(f"  Updated: {meta_path.name}")
print(f"  best_model -> KNN")

# ── Step 6: Recompute global SHAP for KNN ─────────────────────────────────
print("\nStep 6: Computing global SHAP with KernelExplainer (background N=50, deterministic seed=42)...")
preprocessor = knn_pipeline.named_steps["preprocessor"]
X_train_trans = preprocessor.transform(X_train[ALL_FEATURES])
feature_names = list(preprocessor.get_feature_names_out())

# Use a small deterministic background set (median of training data)
rng = np.random.default_rng(42)
bg_idx = rng.choice(len(X_train_trans), size=50, replace=False)
X_background = shap.maskers.Independent(X_train_trans[bg_idx], max_samples=50)

print(f"  Background samples: 50  |  Features: {len(feature_names)}")
print("  Fitting KernelExplainer... (this may take 1-2 minutes)")

def knn_predict_class1(X):
    """Wrapper returning P(class=1) for SHAP KernelExplainer."""
    return knn_pipeline.predict_proba(
        pd.DataFrame(
            preprocessor.inverse_transform(X) if hasattr(preprocessor, "inverse_transform") else X,
            columns=feature_names
        )
    )[:, 1] if False else knn_pipeline.named_steps["classifier"].predict_proba(X)[:, 1]

# KernelExplainer operates in transformed space
explainer_global = shap.KernelExplainer(
    lambda X_arr: knn_pipeline.named_steps["classifier"].predict_proba(X_arr)[:, 1],
    X_train_trans[bg_idx],
    seed=42,
)

# Sample for global importance (use training set, not test set)
sample_idx = rng.choice(len(X_train_trans), size=min(100, len(X_train_trans)), replace=False)
X_global_sample = X_train_trans[sample_idx]

shap_values_global = explainer_global.shap_values(X_global_sample, nsamples=100, l1_reg="num_features(10)")
# shap_values_global shape: (n_samples, n_features) for single-output KernelExplainer

mean_abs_shap = np.mean(np.abs(shap_values_global), axis=0)
ranked_indices = np.argsort(mean_abs_shap)[::-1]

from explainability import parse_transformed_feature_name, FEATURE_METADATA

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

base_val = float(explainer_global.expected_value)

global_result = {
    "base_risk": float(np.round(base_val, 4)),
    "sample_count": len(X_global_sample),
    "feature_count": len(feature_names),
    "clinical_feature_count": len(clinical_rankings),
    "rankings": ranked_features,
    "clinical_rankings": clinical_rankings,
    "explainer_method": "shap.KernelExplainer",
    "explainer_note": (
        "KNN does not support exact LinearExplainer or TreeExplainer. "
        "KernelExplainer provides model-agnostic approximate SHAP values "
        "computed via a background dataset of 50 training samples. "
        "Additivity may not hold to machine precision — residuals up to ~0.02 are expected. "
        "SHAP contributions indicate statistical association only, not causal effects."
    ),
}

shap_global_path = ARTIFACTS_DIR / "shap_global_importance.json"
with open(shap_global_path, "w", encoding="utf-8") as f:
    json.dump(global_result, f, indent=2)
print(f"  Saved: {shap_global_path.name}")
print(f"  Base probability (E[f(X)]): {base_val:.4f}")

print("\n=== KNN Champion Switch Complete ===")
print(f"  best_pipeline.pkl  -> KNN (k=7, distance-weighted)")
print(f"  model_meta.json    -> best_model = KNN")
print(f"  shap_global_importance.json -> KernelExplainer (approx)")
print(f"  Historical LR artifacts preserved as *_lr_historical.*")

