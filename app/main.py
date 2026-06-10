"""
FastAPI application for Credit Risk Scoring Service.
Provides /health, /predict, and /batch_predict endpoints.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    Applicant,
    PredictionResult,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    RiskBand,
)
from app.inference import (
    load_artifacts,
    is_loaded,
    predict_single,
    predict_batch_endpoint,
)
from src.config import MODEL_VERSION, LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Credit Risk Scoring Service...")
    success = load_artifacts()
    if success:
        logger.info("Service started successfully")
    else:
        logger.warning("Service started without model. /predict will fail.")
    yield
    logger.info("Shutting down Credit Risk Scoring Service...")


app = FastAPI(
    title="Credit Risk Scoring Service",
    description="ML-powered loan default prediction service",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=is_loaded(),
        version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictionResult)
async def predict(applicant: Applicant):
    if not is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later.",
        )
    try:
        result = predict_single(applicant.model_dump())
        return PredictionResult(**result)
    except Exception as e:
        logger.error("Prediction error: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    if not is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later.",
        )
    try:
        applicants_data = [a.model_dump() for a in request.applicants]
        results = predict_batch_endpoint(applicants_data)
        return BatchPredictionResponse(
            predictions=[PredictionResult(**r) for r in results],
            count=len(results),
        )
    except Exception as e:
        logger.error("Batch prediction error: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}",
        )
