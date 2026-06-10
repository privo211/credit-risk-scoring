"""
Inference module for the FastAPI application.
Sets up model loading, logging, and prediction at startup.
"""

import logging
import time
import sys
from src.predict import load_model, load_preprocessor, predict_proba, predict_batch
from src.config import BEST_MODEL_PATH, PREPROCESSOR_PATH, MODEL_VERSION, LOGGER_NAME
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
if not logger.handlers:
    logger.addHandler(handler)
_model = None
_preprocessor = None


def load_artifacts():
    """Load model and preprocessor from disk."""
    global _model, _preprocessor
    try:
        logger.info("Loading model artifacts...")
        _model = load_model(BEST_MODEL_PATH)
        _preprocessor = load_preprocessor(PREPROCESSOR_PATH)
        logger.info("Model and preprocessor loaded successfully")
        return True
    except Exception as e:
        logger.error("Failed to load model: %s", str(e))
        _model = None
        _preprocessor = None
        return False


def is_loaded() -> bool:
    return _model is not None and _preprocessor is not None


def predict_single(applicant_data: dict) -> dict:
    """Predict risk for a single applicant."""
    if not is_loaded():
        raise RuntimeError("Model not loaded. Call load_artifacts() first.")

    proba = predict_proba(applicant_data, _model, _preprocessor)
    from src.predict import get_risk_band

    risk_band = get_risk_band(proba)
    return {
        "probability": proba,
        "risk_band": risk_band,
        "model_version": MODEL_VERSION,
    }


def predict_batch_endpoint(applicants_data: list[dict]) -> list[dict]:
    """Predict risk for multiple applicants."""
    if not is_loaded():
        raise RuntimeError("Model not loaded. Call load_artifacts() first.")

    results = predict_batch(applicants_data, _model, _preprocessor)
    return results
