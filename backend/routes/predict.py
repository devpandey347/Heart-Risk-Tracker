"""
predict.py — Patient cardiovascular risk inference endpoint.
============================================================
"""

from fastapi import APIRouter, HTTPException, status
from backend.schemas import PatientInput, PredictionResponse
from backend.service import ml_service

router = APIRouter(prefix="/api", tags=["Inference"])


@router.post("/predict", response_model=PredictionResponse)
def predict_heart_disease_risk(patient: PatientInput) -> PredictionResponse:
    """
    Generate cardiovascular risk assessment from 13 patient health inputs.
    Returns estimated probability, risk classification tier, and confidence score.
    """
    if not ml_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is currently unavailable. Please check system health.",
        )

    try:
        patient_data = patient.model_dump()
        result = ml_service.predict(patient_data)
        return PredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while computing model prediction.",
        )
