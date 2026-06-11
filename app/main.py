"""
FastAPI application for Credit Risk Scoring Service.
Provides /health, /predict, and /batch_predict endpoints with config, logging, rate limiting, and optional DB logging.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.logging import setup_logging, get_logger
from app.schemas import (
    Applicant,
    PredictionResult,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
)
from app.inference import load_artifacts, is_loaded, predict_single, predict_batch_endpoint
from src.config import MODEL_VERSION
from src.database import (
    close_db,
    create_tables,
    get_db,
    init_db,
    is_db_initialized,
    log_prediction,
)

logger = get_logger(__name__)

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(level=settings.LOG_LEVEL)
    logger.info("Starting Credit Risk Scoring Service...")

    if settings.DATABASE_URL:
        init_db(settings.DATABASE_URL)
        await create_tables()
        logger.info("Database initialized via %s", settings.DATABASE_URL.split("@")[-1] if "@" in settings.DATABASE_URL else "configured URL")
    else:
        logger.warning("No DATABASE_URL set — prediction logging disabled")

    success = load_artifacts()
    if success:
        logger.info("Model loaded successfully")
    else:
        logger.warning("Service started without model. /predict will fail.")

    yield

    await close_db()
    logger.info("Shutdown complete.")


app = FastAPI(
    title=settings.APP_NAME,
    description="ML-powered loan default prediction service. Accepts applicant data and returns risk assessment.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
@limiter.limit(settings.RATE_LIMIT)
async def health(request: Request):
    return HealthResponse(
        status="healthy",
        model_loaded=is_loaded(),
        version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictionResult)
@limiter.limit(settings.RATE_LIMIT)
async def predict(applicant: Applicant, request: Request):
    if not is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    try:
        applicant_data = applicant.model_dump()
        result = predict_single(applicant_data)
        await _log_prediction_if_enabled(applicant_data, result)
        return PredictionResult(**result)
    except Exception as e:
        logger.error("Prediction error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse)
@limiter.limit(settings.RATE_LIMIT)
async def batch_predict(req: BatchPredictionRequest, request: Request):
    if not is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    try:
        applicants_data = [a.model_dump() for a in req.applicants]
        results = predict_batch_endpoint(applicants_data)
        for applicant_data, result in zip(applicants_data, results):
            await _log_prediction_if_enabled(applicant_data, result)
        return BatchPredictionResponse(
            predictions=[PredictionResult(**r) for r in results],
            count=len(results),
        )
    except Exception as e:
        logger.error("Batch prediction error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


async def _log_prediction_if_enabled(applicant_data: dict, result: dict) -> None:
    if not is_db_initialized():
        return

    try:
        async for session in get_db():
            await log_prediction(
                session=session,
                input_features=applicant_data,
                probability=result["probability"],
                risk_band=result["risk_band"],
                model_version=result["model_version"],
            )
    except Exception as e:
        logger.warning("Prediction DB logging failed: %s", str(e))
