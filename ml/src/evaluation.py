"""
evaluation.py — Compute, format and save all evaluation metrics.
=================================================================
Called by train_pipeline.py after each model is trained.
Never leaks test-set data into any training or selection step.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)


# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(
    name: str,
    pipeline,
    X_test,
    y_test,
    cv_scores: np.ndarray,
) -> dict:
    """
    Compute all metrics for a trained pipeline on the held-out test set.

    Parameters
    ----------
    name       : model display name
    pipeline   : fitted sklearn Pipeline
    X_test     : test feature DataFrame
    y_test     : test labels Series
    cv_scores  : array of 5 CV accuracy scores from cross_val_score

    Returns
    -------
    dict with all scalar metrics (JSON-serialisable).
    """
    y_pred = pipeline.predict(X_test)

    # ROC-AUC: use predict_proba if available, else predict_log_proba
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, y_proba))
    else:
        roc_auc = None

    results = {
        "model"           : name,
        "cv_accuracy_mean": float(np.mean(cv_scores)),
        "cv_accuracy_std" : float(np.std(cv_scores)),
        "test_accuracy"   : float(accuracy_score(y_test, y_pred)),
        "test_precision"  : float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall"     : float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1"         : float(f1_score(y_test, y_pred, zero_division=0)),
        "test_roc_auc"    : roc_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["No Disease", "Disease"], output_dict=True
        ),
    }

    return results


# ─────────────────────────────────────────────────────────────────────────────
def print_results(results: dict) -> None:
    """Pretty-print a single model's results to the console."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  {results['model']}")
    print(sep)
    print(f"  CV Accuracy (5-fold): {results['cv_accuracy_mean']:.4f}"
          f"  ±{results['cv_accuracy_std']:.4f}")
    print(f"  Test Accuracy       : {results['test_accuracy']:.4f}"
          f"  ({results['test_accuracy']*100:.2f}%)")
    print(f"  Test Precision      : {results['test_precision']:.4f}")
    print(f"  Test Recall         : {results['test_recall']:.4f}")
    print(f"  Test F1-Score       : {results['test_f1']:.4f}")
    if results["test_roc_auc"] is not None:
        print(f"  Test ROC-AUC        : {results['test_roc_auc']:.4f}")
    print(f"\n  Confusion Matrix:")
    cm = results["confusion_matrix"]
    print(f"    TN={cm[0][0]:3d}  FP={cm[0][1]:3d}")
    print(f"    FN={cm[1][0]:3d}  TP={cm[1][1]:3d}")


# ─────────────────────────────────────────────────────────────────────────────
def save_confusion_matrix_plot(
    name: str,
    cm: list[list[int]],
    figures_dir: Path,
) -> str:
    """Save a styled confusion matrix heatmap. Returns the filename."""
    fig, ax = plt.subplots(figsize=(5, 4))
    cm_arr = np.array(cm)
    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        linewidths=0.5, linecolor="#E2E8F0",
        ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    filename = f"cm_{name.lower().replace(' ', '_')}.png"
    fig.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
    plt.close("all")
    return filename


# ─────────────────────────────────────────────────────────────────────────────
def save_roc_curves_plot(
    all_results: list[dict],
    all_pipelines: dict,
    X_test,
    y_test,
    figures_dir: Path,
) -> str:
    """Save a combined ROC curve plot for all models that support predict_proba."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2DD4BF", "#F97316", "#7C3AED", "#0EA5E9", "#EF4444"]

    for (name, pipeline), color in zip(all_pipelines.items(), colors):
        if hasattr(pipeline, "predict_proba"):
            RocCurveDisplay.from_estimator(
                pipeline, X_test, y_test,
                name=name, ax=ax,
                curve_kwargs={"color": color, "alpha": 0.85, "linewidth": 2},
            )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.50)")
    ax.set_title("ROC Curves — All Models (Held-out Test Set)", fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(figures_dir / "roc_curves_all_models.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    return "roc_curves_all_models.png"


# ─────────────────────────────────────────────────────────────────────────────
def save_model_comparison_plot(all_results: list[dict], figures_dir: Path) -> str:
    """Save a grouped bar chart comparing all model metrics."""
    names   = [r["model"] for r in all_results]
    metrics = {
        "CV Accuracy"   : [r["cv_accuracy_mean"] for r in all_results],
        "Test Accuracy" : [r["test_accuracy"]    for r in all_results],
        "F1-Score"      : [r["test_f1"]          for r in all_results],
        "ROC-AUC"       : [r["test_roc_auc"] if r["test_roc_auc"] else 0
                           for r in all_results],
    }

    x     = np.arange(len(names))
    width = 0.2
    colors= ["#2DD4BF", "#F97316", "#7C3AED", "#0EA5E9"]

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (metric, values) in enumerate(metrics.items()):
        bars = ax.bar(x + i * width, values, width, label=metric,
                      color=colors[i], edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — CV Accuracy / Test Accuracy / F1 / ROC-AUC",
                 fontweight="bold")
    ax.axhline(0.90, color="#EF4444", linewidth=1.2, linestyle="--",
               label="90% target line")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(figures_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    return "model_comparison.png"
