"""
explain.py — SHAP model interpretability and feature attribution endpoints.
===========================================================================
"""

from typing import Union
from fastapi import APIRouter, HTTPException, status
from backend.schemas import (
    ExplainRequest,
    GlobalExplanationResponse,
    PatientExplanationResponse,
)
from backend.service import ml_service

router = APIRouter(prefix="/api", tags=["Explainability"])


@router.post(
    "/explain",
    response_model=Union[PatientExplanationResponse, GlobalExplanationResponse],
)
def explain_model_prediction(request: ExplainRequest):
    """
    Generate model interpretability attributions using SHAP.
    - If mode='patient', computes local waterfall contributions, top risk drivers/reducers,
      and mathematical additivity verification for the supplied patient features.
    - If mode='global', returns dataset-wide mean absolute SHAP feature ranking.
    """
    if not ml_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is currently unavailable. Please check system health.",
        )

    try:
        if request.mode == "global":
            global_data = ml_service.get_global_explanation()
            return GlobalExplanationResponse(**global_data)

        # Mode == 'patient'
        if request.patient_data is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="patient_data must be provided when requesting patient-level explanation.",
            )

        patient_dict = request.patient_data.model_dump()
        patient_explanation = ml_service.explain_patient(patient_dict)
        return PatientExplanationResponse(**patient_explanation)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating model explanations.",
        )


@router.get("/explain/global", response_model=GlobalExplanationResponse)
def get_global_feature_importance():
    """Convenience GET endpoint to fetch global SHAP feature importance rankings."""
    if not ml_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is currently unavailable. Please check system health.",
        )

    try:
        global_data = ml_service.get_global_explanation()
        return GlobalExplanationResponse(**global_data)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving global feature importance.",
        )
