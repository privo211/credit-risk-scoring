# Credit Risk Scoring Service

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-16%20passing-brightgreen)]()

> End-to-end machine learning pipeline for loan default prediction with FastAPI, Docker, model evaluation, and production-style deployment. Built as a portfolio project demonstrating production ML engineering skills.

**What this project shows:**
- 🧹 **Production ML pipeline** — data cleaning, 10 engineered features, SMOTE, model training, calibration, evaluation
- 🚀 **Deployable API** — FastAPI service with /predict, /batch_predict, /health endpoints
- 🗄️ **Persistence-ready** — optional PostgreSQL prediction logging with async SQLAlchemy and Alembic migrations
- 📦 **Containerized** — Docker image plus docker-compose for API + PostgreSQL
- ✅ **Tested** — 16 unit tests across preprocessing, model, and API layers
- 📊 **Evaluated** — LR, RF, and XGBoost compared with GridSearchCV, ROC-AUC, recall, and cost-sensitive thresholding
- 🔎 **Explainable** — SHAP summary and local explanation artifacts generated during training

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
  (10 credit-domain features)
         │
         ▼
    SMOTE + GridSearchCV
  (LR / RF / XGBoost)
         │
         ▼
    Preprocessing Pipeline
  (StandardScaler + OneHotEncoder)
         │
         ▼
    Model Selection + Calibration
  (validation ROC-AUC + sigmoid calibration)
         │
         ▼
    Model Evaluation
  (ROC-AUC, recall, threshold cost)
         │
         ▼
    Saved Artifacts
  (model, preprocessor, metadata, SHAP plots)
         │
         ▼
    FastAPI Service
  (/health, /predict, /batch_predict, optional PostgreSQL logs)
         │
         ▼
    Real-Time Predictions
  (probability + risk_band + model_version)
```

**Project Structure:**

```
credit-risk-scoring/
├── app/                    # FastAPI application
│   ├── config.py           # pydantic-settings runtime configuration
│   ├── logging.py          # structlog setup
│   ├── main.py             # API endpoints, rate limiting, optional DB logging
│   ├── schemas.py          # Pydantic request/response models
│   └── inference.py        # Model inference wrapper
├── src/                    # ML pipeline
│   ├── config.py           # Configuration constants
│   ├── credit_metrics.py   # Gini, KS, PSI metrics
│   ├── data_loader.py      # Data loading and splitting
│   ├── database.py         # Async SQLAlchemy prediction log model
│   ├── explain.py          # SHAP summary/local explanation generation
│   ├── preprocess.py       # Feature engineering + preprocessing pipeline
│   ├── train.py            # Model training (LR, RF, XGBoost)
│   ├── evaluate.py         # Model evaluation and comparison
│   └── predict.py          # Prediction and risk band mapping
├── models/                 # Trained model artifacts
├── data/                   # Dataset files
│   ├── raw/                # Raw dataset
│   └── processed/          # Processed train/val/test splits
├── tests/                  # Unit tests
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_api.py
├── notebooks/              # EDA
├── alembic/                # PostgreSQL schema migrations
├── docs/                   # Model card and project documentation
├── scripts/                # Utility scripts
│   ├── train_pipeline.py   # Full train/evaluate/save pipeline
│   ├── sample_payload.json # Sample request for testing
│   └── test_api.sh         # API endpoint test script
├── docker-compose.yml      # API + PostgreSQL services
├── Makefile                # Build/train/test/run commands
├── requirements.txt
├── Dockerfile
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

For the PostgreSQL-backed path, run:

```bash
docker compose up --build
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
  "version": "1.1.0"
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
  "model_version": "1.1.0"
}
```

### `POST /batch_predict`

Predict default risk for multiple applicants.

**Request:** `{"applicants": [{...}, {...}]}`

**Response:**
```json
{
  "predictions": [
    {"probability": 0.82, "risk_band": "HIGH", "model_version": "1.1.0"},
    {"probability": 0.15, "risk_band": "LOW", "model_version": "1.1.0"}
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

**Best Model:** Random Forest pipeline (SMOTE + GridSearchCV, selected by validation ROC-AUC, sigmoid-calibrated)

### Test Set Results (v1.1.0)

| Metric | Value |
|--------|-------|
| Accuracy | 0.5200 |
| Precision | 0.3800 |
| Recall | **0.9500** |
| F1 Score | 0.5429 |
| ROC-AUC | **0.7955** |
| Optimal binary threshold | 0.1400 |

### Model Comparison (Validation Set)

| Model | Validation ROC-AUC |
|-------|--------------------|
| Logistic Regression | 0.7307 |
| Random Forest | **0.7453** |
| XGBoost | 0.7379 |

---

## Risk Band Threshold Tuning

The API risk bands remain interpretable probability bands (LOW < 0.30, MEDIUM 0.30-0.60, HIGH > 0.60). The training pipeline also computes a separate binary decision threshold from a validation-set cost matrix. In v1.1.0, false negatives are weighted 10x false positives, producing a threshold of 0.14 and intentionally favoring recall over precision.

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
# Run all 16 unit tests
make test

# Or directly:
pytest tests/ -v
```

All **16 tests** pass:
- 6 preprocessing tests (feature engineering, pipeline shape, split proportions, stratification)
- 5 model tests (model loading, prediction range, risk band boundaries, batch predictions)
- 4 API tests (health, predict valid/invalid, batch predict)
- 1 guard test ensuring engineered features are included in the fitted preprocessor

You can also run the integration test script against a running server:

```bash
./scripts/test_api.sh
```

---

## Features

- **Preprocessing Pipeline:** `ColumnTransformer` with `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`
- **Feature Engineering:** 10 engineered features including debt-to-income proxy, credit utilization, checking balance score, savings adequacy, and guarantor buffer
- **Models:** Logistic Regression, Random Forest, XGBoost with SMOTE inside GridSearchCV pipelines
- **Calibration:** Sigmoid probability calibration on the validation set
- **Evaluation:** Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and cost-sensitive threshold search
- **Explainability:** SHAP summary and local waterfall artifacts generated during training
- **Inference:** Model loading, probability prediction, risk band mapping
- **API:** FastAPI with Pydantic validation, CORS middleware, structured logging, rate limiting, optional PostgreSQL prediction logging
- **Testing:** 16 unit tests across preprocessing, model, and API layers
- **Deployment:** Docker containerization with slim Python image; docker-compose includes PostgreSQL

---

## Tradeoffs and Future Work

### Tradeoffs
- **Dataset size:** 1,000 rows is small for production; larger datasets would improve generalization
- **Feature count:** 20 raw features plus 10 engineered features is manageable but could benefit from additional external data
- **Model complexity:** Random Forest outperformed slightly, but XGBoost could surpass with deeper tuning
- **Threshold tuning:** The optimized binary threshold improves recall but creates many false positives; a real lender would choose this threshold from portfolio economics

### Future Improvements
- API key authentication for secure deployment
- Drift monitoring on incoming predictions versus training distribution
- A/B testing framework for model rollout
- Model versioning and rollback support
- Expanded test coverage for edge cases
- RAG-style documentation assistant over the model card, feature definitions, and operational runbook

---

## What I Would Do Differently with More Time

1. Use a larger dataset (e.g., Home Credit Default Risk with 300K+ rows)
2. Implement feature selection to reduce dimensionality
3. Add automated hyperparameter optimization with Optuna
4. Build a feature store for reproducible feature computation
5. Add prediction caching for repeated requests
6. Implement proper model registry with versioned artifacts
7. Add monitoring dashboard with prediction drift alerts
8. Include data validation with Great Expectations

---

## License

MIT
