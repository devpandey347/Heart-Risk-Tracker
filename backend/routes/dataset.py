"""
dataset.py — Dataset statistics and distribution metadata endpoint.
===================================================================
"""

from fastapi import APIRouter
from backend.schemas import DatasetStatsResponse
from backend.service import ml_service

router = APIRouter(prefix="/api", tags=["Dataset"])


@router.get("/dataset/stats", response_model=DatasetStatsResponse)
def get_dataset_stats() -> DatasetStatsResponse:
    """
    Retrieve verified exploratory summary statistics from the Cleveland dataset.
    Includes feature distributions, class counts, and clinical category definitions.
    """
    stats = ml_service.dataset_stats
    return DatasetStatsResponse(
        total_samples=stats.get("total_samples", 303),
        total_features=stats.get("total_features", 13),
        class_distribution=stats.get("class_distribution", {}),
        missing_values=stats.get("missing_values", {}),
        numerical_features_summary=stats.get("numerical_features_summary", {}),
        categorical_categories=stats.get("categorical_categories", {}),
    )
