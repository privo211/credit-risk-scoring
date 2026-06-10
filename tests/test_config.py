"""Tests for configuration modules (src/config.py, app/config.py)."""

import os
import importlib

import pytest

from src.config import ENGINEERED_FEATURES, MODEL_VERSION
from app.config import settings


class TestSrcConfig:
    def test_engineered_features_count(self):
        """ENGINEERED_FEATURES has exactly 10 items."""
        assert len(ENGINEERED_FEATURES) == 10

    def test_engineered_features_contains_all(self):
        """All expected engineered features are present."""
        expected = [
            "credit_amount_per_duration",
            "age_band",
            "installment_burden",
            "debt_to_income_proxy",
            "employment_stability_score",
            "savings_adequacy",
            "checking_balance_score",
            "high_risk_purpose_flag",
            "guarantor_buffer",
            "credit_utilization",
        ]
        for feat in expected:
            assert feat in ENGINEERED_FEATURES, f"Missing: {feat}"

    def test_model_version(self):
        """MODEL_VERSION is 2.0.0."""
        assert MODEL_VERSION == "2.0.0"

    def test_env_override_train_size(self):
        """Setting CREDIT_TRAIN_SIZE overrides the default."""
        os.environ["CREDIT_TRAIN_SIZE"] = "0.7"
        import src.config
        importlib.reload(src.config)
        assert src.config.TRAIN_SIZE == 0.7
        del os.environ["CREDIT_TRAIN_SIZE"]
        importlib.reload(src.config)

    def test_env_override_random_state(self):
        """Setting CREDIT_RANDOM_STATE overrides the default."""
        os.environ["CREDIT_RANDOM_STATE"] = "99"
        import src.config
        importlib.reload(src.config)
        assert src.config.RANDOM_STATE == 99
        del os.environ["CREDIT_RANDOM_STATE"]
        importlib.reload(src.config)


class TestAppConfig:
    def test_settings_defaults(self):
        """Settings singleton has correct default values."""
        assert settings.APP_NAME == "Credit Risk Scoring API"
        assert settings.HOST == "0.0.0.0"
        assert settings.PORT == 8000
        assert settings.LOG_LEVEL == "INFO"
        assert settings.DATABASE_URL == ""
        assert settings.API_PREFIX == "/api/v1"
        assert settings.CORS_ORIGINS == ["*"]
        assert settings.RATE_LIMIT == "100/minute"
        assert settings.API_KEY_ENABLED is False

    def test_settings_env_override(self):
        """CREDIT_DEBUG env var overrides the default."""
        os.environ["CREDIT_DEBUG"] = "true"
        import app.config
        importlib.reload(app.config)
        from app.config import settings
        assert settings.DEBUG is True
        del os.environ["CREDIT_DEBUG"]
        importlib.reload(app.config)