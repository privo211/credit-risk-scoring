"""
Prediction module for Credit Risk Scoring.
Handles model loading, preprocessing, and risk prediction.
"""

import logging
import time
import joblib
import pandas as pd
import numpy as np
from src.config import (
    MODELS_DIR,
    PREPROCESSOR_PATH,
    BEST_MODEL_PATH,
    RISK_THRESHOLDS,
    MODEL_VERSION,
    RANDOM_STATE,
)
from src.preprocess import engineer_features

logger = logging.getLogger(__name__)


def get_risk_band(probability: float) -> str:
    """
    Map a probability to a risk band.

    Args:
        probability: Predicted probability of default (0-1).

    Returns:
        "LOW", "MEDIUM", or "HIGH".
    """
    if probability < 0.30:
        return "LOW"
    elif probability <= 0.60:
        return "MEDIUM"
    else:
        return "HIGH"


def load_model(model_path: str = BEST_MODEL_PATH):
    """
    Load a trained model from disk.

    Args:
        model_path: Path to the serialized model.

    Returns:
        Loaded model object.
    """
    model_path_str = str(model_path)
    if model_path_str.startswith(str(MODELS_DIR)):
        path = model_path
    else:
        path = MODELS_DIR / model_path
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model


def load_preprocessor(preprocessor_path: str = PREPROCESSOR_PATH):
    """
    Load the fitted preprocessor from disk.

    Args:
        preprocessor_path: Path to the serialized preprocessor.

    Returns:
        Loaded preprocessor object.
    """
    pp_path_str = str(preprocessor_path)
    if pp_path_str.startswith(str(MODELS_DIR)):
        path = preprocessor_path
    else:
        path = MODELS_DIR / preprocessor_path
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {path}")
    preprocessor = joblib.load(path)
    logger.info(f"Preprocessor loaded from {path}")
    return preprocessor


def predict_proba(
    applicant: dict,
    model,
    preprocessor,
) -> float:
    """
    Predict the probability of default for a single applicant.

    Args:
        applicant: Dictionary of applicant features.
        model: Trained model.
        preprocessor: Fitted preprocessor.

    Returns:
        Probability of default (class 1) as a float in [0, 1].
    """
    start_time = time.time()
    df = pd.DataFrame([applicant])

    # Ensure all required columns exist, fill missing with reasonable defaults
    required_cols = [
        "checking_status", "duration", "credit_history", "purpose",
        "credit_amount", "savings", "employment", "installment_rate",
        "personal_status", "guarantors", "residence_since", "property",
        "age", "other_plans", "housing", "num_credits", "job",
        "people_maintenance", "telephone", "foreign_worker",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # Feature engineering
    df_fe = engineer_features(df)

    # Preprocess
    X_processed = preprocessor.transform(df_fe)

    # Predict
    proba = model.predict_proba(X_processed)[0, 1]
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(
        "Prediction completed in %.0f ms | probability=%.4f | risk=%s",
        elapsed_ms,
        proba,
        get_risk_band(proba),
    )
    return float(proba)


def predict_batch(
    applicants: list[dict],
    model,
    preprocessor,
) -> list[dict]:
    """
    Predict default probability for multiple applicants.

    Args:
        applicants: List of applicant dictionaries.
        model: Trained model.
        preprocessor: Fitted preprocessor.

    Returns:
        List of dicts with probability, risk_band, model_version.
    """
    start_time = time.time()
    df = pd.DataFrame(applicants)

    required_cols = [
        "checking_status", "duration", "credit_history", "purpose",
        "credit_amount", "savings", "employment", "installment_rate",
        "personal_status", "guarantors", "residence_since", "property",
        "age", "other_plans", "housing", "num_credits", "job",
        "people_maintenance", "telephone", "foreign_worker",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    df_fe = engineer_features(df)
    X_processed = preprocessor.transform(df_fe)
    probas = model.predict_proba(X_processed)[:, 1]

    results = []
    for prob in probas:
        results.append(
            {
                "probability": float(prob),
                "risk_band": get_risk_band(float(prob)),
                "model_version": MODEL_VERSION,
            }
        )

    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(
        "Batch prediction completed for %d applicants in %.0f ms",
        len(applicants),
        elapsed_ms,
    )
    return results
