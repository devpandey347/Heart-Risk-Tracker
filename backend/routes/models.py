"""
models.py — Model comparison and benchmarking evaluation endpoint.
==================================================================
"""

from fastapi import APIRouter
from backend.schemas import ModelComparisonResponse
from backend.service import ml_service

router = APIRouter(prefix="/api", tags=["Models"])


@router.get("/models/comparison", response_model=ModelComparisonResponse)
def get_model_comparison() -> ModelComparisonResponse:
    """
    Retrieve genuine benchmark evaluation metrics for all tested classifiers.
    Shows cross-validation performance and final held-out test evaluation.

    Data source precedence:
    - all_results (all_results.json): contains full per-model metrics including
      test_precision, test_recall, confusion_matrix — used for full comparison.
    - model_meta (model_meta.json): fallback if all_results not available.
    """
    meta = ml_service.model_meta
    all_results = ml_service.all_results  # List[dict] from all_results.json

    # Prefer all_results (has precision, recall) over meta["all_models"] (leaner)
    if all_results:
        models_data = [
            {
                "model": r["model"],
                "cv_accuracy_mean": r["cv_accuracy_mean"],
                "cv_accuracy_std": r["cv_accuracy_std"],
                "cv_f1_mean": r.get("cv_f1_mean"),
                "cv_f1_std": r.get("cv_f1_std"),
                "test_accuracy": r["test_accuracy"],
                "test_precision": r.get("test_precision", 0.0),
                "test_recall": r.get("test_recall", 0.0),
                "test_f1": r["test_f1"],
                "test_roc_auc": r.get("test_roc_auc"),
            }
            for r in all_results
        ]
    else:
        # Fallback to model_meta["all_models"] for backward compatibility
        models_data = meta.get("all_models", [])

    return ModelComparisonResponse(
        champion_model=meta.get("best_model", "Unknown"),
        selection_method=meta.get("selection_method", "N/A"),
        split_protocol=meta.get("split_config", {}),
        models=models_data,
    )
