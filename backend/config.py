"""
config.py — Backend application configuration and file path resolution.
======================================================================
"""

from pathlib import Path
from typing import List

# Root directories
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
ML_DIR = PROJECT_ROOT / "ml"
ARTIFACTS_DIR = ML_DIR / "artifacts"
RAW_DATA_PATH = ML_DIR / "data" / "raw" / "processed.cleveland.data"

# Artifact file paths
BEST_PIPELINE_PATH = ARTIFACTS_DIR / "best_pipeline.pkl"
MODEL_META_PATH = ARTIFACTS_DIR / "model_meta.json"
ALL_RESULTS_PATH = ARTIFACTS_DIR / "all_results.json"
SHAP_GLOBAL_PATH = ARTIFACTS_DIR / "shap_global_importance.json"

# API Settings
API_TITLE = "Heart Risk Tracker API"
API_VERSION = "2.0.0"
API_DESCRIPTION = (
    "Production-style REST API for cardiovascular risk assessment and SHAP explainability. "
    "Designed for educational and analytical purposes only."
)

# CORS Origins (Development and Production Vite/React ports)
CORS_ORIGINS: List[str] = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

MEDICAL_DISCLAIMER = (
    "Educational Cardiovascular Risk Assessment Notice: "
    "This tool is designed for educational and informational purposes only. "
    "It is NOT a medical device, does NOT diagnose heart disease, and should never "
    "replace professional medical advice, clinical diagnosis, or treatment."
)
