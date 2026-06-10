"""Tests for model loading and prediction."""

import joblib
import numpy as np
import pandas as pd
from src.predict import load_model, load_preprocessor, predict_proba, get_risk_band
from src.config import MODELS_DIR, BEST_MODEL_PATH, PREPROCESSOR_PATH
from src.data_loader import load_and_split_data


def test_model_loads():
    model = load_model()
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_preprocessor_loads():
    preprocessor = load_preprocessor()
    assert preprocessor is not None
    assert hasattr(preprocessor, "transform")


def test_prediction_probability_range():
    model = load_model()
    preprocessor = load_preprocessor()
    _, _, X_test, _, _, y_test = load_and_split_data()
    sample = X_test.iloc[0:1]
    prob = model.predict_proba(preprocessor.transform(sample))[0, 1]
    assert 0.0 <= prob <= 1.0, f"Probability {prob} outside [0,1]"


def test_risk_band_boundaries():
    assert get_risk_band(0.00) == "LOW"
    assert get_risk_band(0.29) == "LOW"
    assert get_risk_band(0.30) == "MEDIUM"
    assert get_risk_band(0.45) == "MEDIUM"
    assert get_risk_band(0.59) == "MEDIUM"
    assert get_risk_band(0.60) == "MEDIUM"
    assert get_risk_band(0.61) == "HIGH"
    assert get_risk_band(1.00) == "HIGH"


def test_batch_predictions_return_correct_length():
    model = load_model()
    preprocessor = load_preprocessor()
    _, _, X_test, _, _, _ = load_and_split_data()
    samples = X_test.head(5).to_dict("records")
    from src.predict import predict_batch
    results = predict_batch(samples, model, preprocessor)
    assert len(results) == 5, f"Expected 5, got {len(results)}"
    for r in results:
        assert "probability" in r
        assert "risk_band" in r
        assert "model_version" in r
        assert 0.0 <= r["probability"] <= 1.0
