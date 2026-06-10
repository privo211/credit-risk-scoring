"""Test fixtures and configuration for the Credit Risk Scoring test suite."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def raw_data_path(project_root: Path) -> Path:
    """Path to the raw German Credit dataset CSV."""
    return project_root / "data" / "raw" / "german_credit_data.csv"


@pytest.fixture(scope="session")
def sample_data() -> pd.DataFrame:
    """Return a small sample DataFrame representing German Credit data structure.
    Uses real distributions: A11-A14 for checking_status, A30-A34 for credit_history, etc.
    """
    np.random.seed(42)
    n = 20
    return pd.DataFrame({
        "checking_status": np.random.choice(["A11", "A12", "A13", "A14"], n),
        "duration": np.random.randint(6, 72, n),
        "credit_history": np.random.choice(["A30", "A31", "A32", "A33", "A34"], n),
        "purpose": np.random.choice(["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A48", "A49", "A410"], n),
        "credit_amount": np.random.randint(500, 20000, n).astype(float),
        "savings": np.random.choice(["A61", "A62", "A63", "A64", "A65"], n),
        "employment": np.random.choice(["A71", "A72", "A73", "A74", "A75"], n),
        "installment_rate": np.random.randint(1, 5, n),
        "personal_status": np.random.choice(["A91", "A92", "A93", "A94", "A95"], n),
        "guarantors": np.random.choice(["A101", "A102", "A103"], n),
        "residence_since": np.random.randint(1, 5, n),
        "property": np.random.choice(["A121", "A122", "A123", "A124"], n),
        "age": np.random.randint(18, 75, n),
        "other_plans": np.random.choice(["A141", "A142", "A143"], n),
        "housing": np.random.choice(["A151", "A152", "A153"], n),
        "num_credits": np.random.randint(1, 5, n),
        "job": np.random.choice(["A171", "A172", "A173", "A174"], n),
        "people_maintenance": np.random.randint(1, 3, n),
        "telephone": np.random.choice(["A191", "A192"], n),
        "foreign_worker": np.random.choice(["A201", "A202"], n),
    })


@pytest.fixture(scope="session")
def sample_applicant(sample_data: pd.DataFrame) -> dict:
    """Return a single applicant dict (first row of sample_data)."""
    return sample_data.iloc[0].to_dict()


@pytest.fixture(scope="session")
def sample_applicants(sample_data: pd.DataFrame) -> list[dict]:
    """Return a list of applicant dicts (first 5 rows)."""
    return sample_data.head(5).to_dict(orient="records")


@pytest.fixture
def trained_preprocessor(sample_data):
    """Build and fit a preprocessor on the sample data."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.preprocess import engineer_features, build_preprocessor

    df_fe = engineer_features(sample_data)
    preprocessor = build_preprocessor()
    preprocessor.fit(df_fe)
    return preprocessor


@pytest.fixture
def trained_model(trained_preprocessor, sample_data):
    """Train a quick model for testing inference."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import joblib
    import tempfile
    from sklearn.ensemble import RandomForestClassifier
    from src.preprocess import engineer_features

    df_fe = engineer_features(sample_data)
    X = trained_preprocessor.transform(df_fe)
    y = np.random.randint(0, 2, len(sample_data))

    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X, y)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        joblib.dump(model, f.name)
        model_path = f.name

    yield model, model_path

    import os
    os.unlink(model_path)


@pytest.fixture
def test_client():
    """Create a FastAPI TestClient for API integration tests."""
    from fastapi.testclient import TestClient
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.main import app
    with TestClient(app) as client:
        yield client


def pytest_configure(config):
    """Configure pytest markers and settings."""
    config.addinivalue_line(
        "markers",
        "anyio: mark test to run with anyio backend (default: asyncio)",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (deselect with '-m \"not slow\"')",
    )
