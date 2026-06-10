# Credit Risk Scoring Service

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-35%2B%20passing-brightgreen)]()

> End-to-end machine learning pipeline for loan default prediction with FastAPI, Docker, model evaluation, and production-style deployment. Built as a portfolio project demonstrating production ML engineering skills.

**What this project shows:**
- 🧹 **Production ML pipeline** — data cleaning, preprocessing, feature engineering, model training, evaluation
- 🚀 **Deployable API** — FastAPI service with /predict, /batch_predict, /health endpoints
- 📦 **Containerized** — Docker-ready multi-stage build with PostgreSQL via docker-compose
- ✅ **Tested** — 35+ unit tests across all layers (80%+ coverage)
- 📊 **Evaluated** — 3 models compared (LR, RF, XGBoost) with GridSearchCV + SMOTE + Platt calibration
- 🎯 **Business-ready** — Risk bands (LOW/MEDIUM/HIGH) with cost-matrix threshold optimization

---

## Problem Statement

Predict whether a loan applicant is likely to default using structured financial and demographic features, then serve predictions through a production-style API.

**Task:** Binary classification  
**Target:** `0` = No default (good credit), `1` = Default (bad credit)  
**Output:** Probability of default + risk band (LOW / MEDIUM / HIGH)

---

## Dataset

**Source:** [Statlog (German Credit Data) - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)

| Property | Value |
|----------|-------|
| Instances | 1,000 |
| Features | 20 (7 numeric, 13 categorical) |
| Target | Binary (70% good, 30% bad) |
| Missing values | None |

The dataset contains financial and demographic attributes including checking account status, credit history, loan purpose, employment length, savings, and age.

---

## Architecture

```
Raw Dataset (German Credit)
         │
         ▼
    Feature Engineering
  (10 engineered features: debt_to_income,
   employment_stability, savings_adequacy, ...)
         │
         ▼
    SMOTE Oversampling
  (resample to balance classes)
         │
         ▼
    Preprocessing Pipeline
  (StandardScaler + OneHotEncoder)
         │
         ▼
    Model Training + GridSearchCV
  (Logistic Regression / Random Forest / XGBoost)
         │
         ▼
    Model Selection (best ROC-AUC)
         │
         ▼
    Platt Calibration
  (5-fold CV sigmoid calibration)
         │
         ▼
    Evaluation + Metrics
  (Gini, KS, ROC-AUC, cost-threshold opt)
         │
         ▼
    Saved Artifacts
  (best_model.pkl, preprocessor.pkl)
         │
         ▼
    FastAPI Service (structured logging, rate limiting)
  (/health, /predict, /batch_predict)
         │
         ▼
    Real-Time Predictions
  (probability + risk_band + model_version)
```

**Project Structure:**

```
credit-risk-scoring/
├── app/                    # FastAPI application
│   ├── config.py           # pydantic-settings (CREDIT_ env prefix)
│   ├── logging.py          # structlog configuration
│   ├── main.py             # API endpoints (rate-limited, CORS)
│   ├── schemas.py          # Pydantic request/response models
│   └── inference.py        # Model inference wrapper
├── src/                    # ML pipeline
│   ├── config.py           # Configuration constants + env-var overrides
│   ├── credit_metrics.py   # Gini coefficient, KS statistic, PSI
│   ├── data_loader.py      # Data loading and splitting
│   ├── database.py         # Async SQLAlchemy models (PostgreSQL)
│   ├── evaluate.py         # Evaluation + cost-matrix threshold opt
│   ├── explain.py          # SHAP feature importance (global + local)
│   ├── preprocess.py       # Feature engineering (10 features) + preprocessing
│   ├── train.py            # Model training (SMOTE + GridSearchCV)
│   └── predict.py          # Prediction and risk band mapping
├── models/                 # Trained model artifacts
├── data/                   # Dataset files
│   ├── raw/                # Raw dataset
│   └── processed/          # Processed train/val/test splits
├── tests/                  # Unit tests
│   ├── conftest.py         # Shared fixtures (8 fixtures)
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_api.py
│   ├── test_metrics.py
│   ├── test_config.py
│   └── test_explain.py
├── alembic/                # Database migrations
│   ├── versions/
│   └── env.py              # Async SQLAlchemy (asyncpg)
├── docs/
│   └── model_card.md       # ML model card
├── scripts/                # Utility scripts
│   ├── train_pipeline.py   # Full training pipeline entrypoint
│   ├── sample_payload.json
│   └── test_api.sh
├── docker-compose.yml      # PostgreSQL 16 + API service
├── Makefile                # Build/train/test/run commands
├── requirements.txt
├── Dockerfile              # Multi-stage, non-root user
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- pip
- Docker (optional, for containerized deployment)

### Setup & Run (3 commands)

```bash
# 1. Install dependencies
make setup

# 2. Train models (LR, RF, XGBoost) and save artifacts
make train

# 3. Start the API server
make run
```

### Test the API

Use the included test script to verify all endpoints:

```bash
# Run all API tests (requires server running)
./scripts/test_api.sh

# Or test individually:
# Health check
curl "http://localhost:8000/health"

# Single prediction (using sample payload)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @scripts/sample_payload.json

# Batch prediction
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '{"applicants": [{"duration": 24, "credit_amount": 5000, "age": 35}, {"duration": 12, "credit_amount": 2000, "age": 28}]}'
```

### Run Tests

```bash
make test
```

---

## How to Run with Docker

```bash
# Build the image
make docker-build

# Run the container
make docker-run

# Test it
curl "http://localhost:8000/health"

# Or manually:
docker build -t credit-risk-scoring .
docker run -p 8000:8000 credit-risk-scoring
```

---

## API Endpoints

### `GET /health`

Returns service health status and model state.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "2.0.0"
}
```

### `POST /predict`

Predict default risk for a single applicant.

**Request:** Applicant features (20 fields, all optional with default null handling)

**Response:**
```json
{
  "probability": 0.82,
  "risk_band": "HIGH",
  "model_version": "2.0.0"
}
```

### `POST /batch_predict`

Predict default risk for multiple applicants.

**Request:** `{"applicants": [{...}, {...}]}`

**Response:**
```json
{
  "predictions": [
    {"probability": 0.82, "risk_band": "HIGH", "model_version": "2.0.0"},
    {"probability": 0.15, "risk_band": "LOW", "model_version": "2.0.0"}
  ],
  "count": 2
}
```

---

## Risk Bands

| Probability Range | Risk Band |
|---|---|
| < 0.30 | LOW |
| 0.30 - 0.60 | MEDIUM |
| > 0.60 | HIGH |

---

## Model Performance

**Best Model:** Random Forest (selected by validation ROC-AUC, SMOTE + Platt calibration)

### Test Set Results (v2.0.0)

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.7827** |
| Gini Coefficient | 0.5655 |
| KS Statistic | 0.4571 |
| Accuracy | 0.7750 |
| Precision | 0.6829 |
| Recall | 0.4667 |
| F1 Score | 0.5545 |

### Model Comparison (Validation Set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 0.6450 | 0.4396 | 0.6667 | 0.5298 | 0.7239 |
| Random Forest | 0.7150 | 0.5349 | 0.3833 | 0.4466 | **0.7407** |
| XGBoost | 0.7050 | 0.5106 | 0.4000 | 0.4486 | 0.7236 |

---

## Risk Band Threshold Tuning

The risk bands (LOW < 0.30, MEDIUM 0.30-0.60, HIGH > 0.60) are configurable defaults. For production, use cost-matrix threshold optimization (find_optimal_threshold in src/evaluate.py) where false negatives are weighted 5x more than false positives — reflecting that approving a bad applicant is more costly than rejecting a good one.

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_preprocessing.py -v
pytest tests/test_model.py -v
pytest tests/test_api.py -v
```

```bash
# Run all 15 unit tests
make test

# Or directly:
pytest tests/ -v
```

All **35+ tests** pass with **80%+ coverage**:
- Preprocessing (feature engineering, pipeline shape, split proportions, stratification)
- Model (model loading, prediction range, risk band boundaries, batch predictions)
- API (health, predict valid/invalid, batch predict, edge cases)
- Metrics (Gini coefficient, KS statistic, threshold optimization)
- Config (env-var overrides, pydantic-settings defaults)
- SHAP explanations (global importance, local values, graceful degradation)

You can also run the integration test script against a running server:

```bash
./scripts/test_api.sh
```

---

## Features

- **Preprocessing Pipeline:** `ColumnTransformer` with `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`
- **Feature Engineering:** 10 engineered features (credit_amount_per_duration, age_band, installment_burden, debt_to_income_proxy, employment_stability_score, savings_adequacy, checking_balance_score, high_risk_purpose_flag, guarantor_buffer, credit_utilization)
- **Class Imbalance:** SMOTE oversampling before GridSearchCV
- **Models:** Logistic Regression, Random Forest, XGBoost with GridSearchCV hyperparameter tuning
- **Calibration:** Platt scaling (sigmoid, 5-fold CV) for calibrated probabilities
- **Threshold Optimization:** Cost-matrix threshold search (FN cost 5x FP cost)
- **Explainability:** SHAP global feature importance + local per-prediction values
- **Evaluation:** Accuracy, precision, recall, F1, ROC-AUC, Gini coefficient, KS statistic, confusion matrix
- **Inference:** Model loading, probability prediction, risk band mapping
- **API:** FastAPI with Pydantic validation, structlog, slowapi rate limiting, CORS middleware
- **Database:** Async SQLAlchemy + asyncpg (PostgreSQL via docker-compose), Alembic migrations
- **Testing:** 35+ tests with 80%+ coverage across all layers
- **Deployment:** Docker multi-stage build (non-root user, HEALTHCHECK), docker-compose with PostgreSQL

---

## Tradeoffs and Future Work

### Tradeoffs
- **Dataset size:** 1,000 rows is small for production; larger datasets would improve generalization
- **Feature count:** 20 raw features (10 engineered) is manageable but could benefit from additional external data
- **Model complexity:** Random Forest outperformed slightly, but XGBoost could surpass with deeper tuning
- **Imbalanced classes:** SMOTE mitigated the 70/30 split; recall remains at 0.47 — ADASYN or GAN-based augmentation could improve further

### Future Improvements
- API key authentication for secure deployment
- Drift monitoring on incoming predictions versus training distribution
- A/B testing framework for model rollout
- Model versioning and rollback support
- Automated hyperparameter optimization with Optuna
- Prediction caching for repeated requests
- Feature store for reproducible feature computation
- Monitoring dashboard with prediction drift alerts
- Data validation with Great Expectations

---



## License

MIT
