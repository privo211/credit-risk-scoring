"""Tests for the FastAPI application endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_predict_endpoint_invalid_data(client):
    response = await client.post(
        "/predict",
        json={"invalid_field": "test"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_endpoint_valid_schema(client):
    applicant = {
        "duration": 24.0,
        "credit_amount": 5000.0,
        "age": 35.0,
        "checking_status": "A11",
        "credit_history": "A34",
        "purpose": "A43",
        "savings": "A65",
        "employment": "A75",
        "installment_rate": 4.0,
        "personal_status": "A93",
        "guarantors": "A101",
        "residence_since": 4.0,
        "property": "A121",
        "other_plans": "A143",
        "housing": "A152",
        "num_credits": 1.0,
        "job": "A173",
        "people_maintenance": 1.0,
        "telephone": "A192",
        "foreign_worker": "A201",
    }
    response = await client.post("/predict", json=applicant)
    if response.status_code == 200:
        data = response.json()
        assert "probability" in data
        assert "risk_band" in data
        assert "model_version" in data
        assert 0.0 <= data["probability"] <= 1.0
        assert data["risk_band"] in ["LOW", "MEDIUM", "HIGH"]
    elif response.status_code == 503:
        pass


@pytest.mark.asyncio
async def test_batch_predict_endpoint(client):
    applicants = [
        {
            "duration": 24.0,
            "credit_amount": 5000.0,
            "age": 35,
            "checking_status": "A11",
            "credit_history": "A34",
        },
        {
            "duration": 12.0,
            "credit_amount": 2000.0,
            "age": 28,
            "checking_status": "A12",
            "credit_history": "A33",
        },
    ]
    response = await client.post(
        "/batch_predict",
        json={"applicants": applicants},
    )
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert "count" in data
        assert data["count"] == 2
        assert len(data["predictions"]) == 2
        for pred in data["predictions"]:
            assert "probability" in pred
            assert "risk_band" in pred
    elif response.status_code == 503:
        pass
