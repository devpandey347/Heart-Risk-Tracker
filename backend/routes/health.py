"""
health.py — System health and readiness endpoint.
=================================================
"""

from fastapi import APIRouter
from backend.config import API_VERSION
from backend.schemas import HealthResponse
from backend.service import ml_service

router = APIRouter(prefix="/api", tags=["System"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Check API status and ML model operational readiness."""
    meta = ml_service.model_meta
    best_metrics = meta.get("best_metrics", {})
    champion = meta.get("best_model", "Unknown")
    selection = meta.get("selection_method", "N/A")

    return HealthResponse(
        status="healthy" if ml_service.is_loaded else "degraded",
        api_version=API_VERSION,
        model_ready=ml_service.is_loaded,
        champion_model=champion,
        selection_method=selection,
        metrics=best_metrics,
        features_expected=13,
    )
