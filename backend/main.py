"""
main.py — FastAPI application entry point.
=========================================
Configures middleware, lifecycle events, global error handling, and API routing.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    CORS_ORIGINS,
)
from backend.routes import dataset, explain, health, models, predict
from backend.service import ml_service

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backend.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle manager."""
    logger.info("Initializing Heart Risk Tracker Backend...")
    try:
        ml_service.load_artifacts()
        logger.info("Champion pipeline and dataset metadata successfully loaded.")
    except Exception as exc:
        logger.error(f"Failed to load ML artifacts at startup: {exc}")
    yield
    logger.info("Shutting down Heart Risk Tracker Backend.")


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global sanitizing exception handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Ensure internal stack traces and server filepaths are never exposed to clients."""
    logger.exception(f"Unhandled error on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please try again later."},
    )


# Include API Routers
app.include_router(health.router)
app.include_router(models.router)
app.include_router(dataset.router)
app.include_router(predict.router)
app.include_router(explain.router)


@app.get("/", tags=["Root"])
def root_status():
    """Root endpoint verifying API availability."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "status": "online",
        "documentation": "/docs",
    }
