"""
conftest.py — Pytest fixtures for FastAPI backend tests.
=======================================================
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "ml" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ml" / "src"))

from backend.main import app
from backend.service import ml_service


@pytest.fixture(scope="session")
def client():
    """Create a TestClient with lifespan context active."""
    with TestClient(app) as test_client:
        yield test_client
