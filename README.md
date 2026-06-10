# Credit Risk Scoring Service

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-15%20passing-brightgreen)]()

> End-to-end machine learning pipeline for loan default prediction with FastAPI, Docker, model evaluation, and production-style deployment. Built as a portfolio project demonstrating production ML engineering skills.

**What this project shows:**
- 🧹 **Production ML pipeline** — data cleaning, preprocessing, feature engineering, model training, evaluation
- 🚀 **Deployable API** — FastAPI service with /predict, /batch_predict, /health endpoints
- 📦 **Containerized** — Docker-ready with reproducible builds
- ✅ **Tested** — 15 unit tests across preprocessing, model, and API layers
- 📊 **Evaluated** — 3 models compared (LR, RF, XGBoost) with GridSearchCV hyperparameter tuning
- 🎯 **Business-ready** — Risk bands (LOW/MEDIUM/HIGH) with threshold tuning

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
  (credit_amount_per_duration,
   age_band, installment_burden)
         │
         ▼
    Preprocessing Pipeline
  (StandardScaler + OneHotEncoder)
         │
         ▼
    Model Training
  (Logistic Regression / Random Forest / XGBoost)
         │
         ▼
    Model Evaluation
  (CV + validation metrics comparison)
         │
         ▼
    Saved Artifacts
  (best_model.pkl, preprocessor.pkl)
         │
         ▼
    FastAPI Service
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
│   ├── main.py             # API endpoints
│   ├── schemas.py          # Pydantic request/response models
│   └── inference.py        # Model inference wrapper
├── src/                    # ML pipeline
│   ├── config.py           # Configuration constants
│   ├── data_loader.py      # Data loading and splitting
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
├── scripts/                # Utility scripts
│   ├── sample_payload.json # Sample request for testing
│   └── test_api.sh         # API endpoint test script
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
  "version": "1.0.0"
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
  "model_version": "1.0.0"
}
```

### `POST /batch_predict`

Predict default risk for multiple applicants.

**Request:** `{"applicants": [{...}, {...}]}`

**Response:**
```json
{
  "predictions": [
    {"probability": 0.82, "risk_band": "HIGH", "model_version": "1.0.0"},
    {"probability": 0.15, "risk_band": "LOW", "model_version": "1.0.0"}
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

**Best Model:** Random Forest (selected by validation ROC-AUC)

### Test Set Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.7650 |
| Precision | 0.6444 |
| Recall | 0.4833 |
| F1 Score | 0.5524 |
| ROC-AUC | **0.7805** |

### Model Comparison (Validation Set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 0.6450 | 0.4396 | 0.6667 | 0.5298 | 0.7233 |
| Random Forest | 0.7150 | 0.5349 | 0.3833 | 0.4466 | **0.7298** |
| XGBoost | 0.7050 | 0.5106 | 0.4000 | 0.4486 | 0.7196 |

---

## Risk Band Threshold Tuning

The risk bands were tuned based on the validation set performance to balance precision and recall for the default class. For a production deployment, these thresholds should be adjusted based on the business cost matrix — for credit risk, false negatives (approving a bad applicant) are typically 5x more costly than false positives (rejecting a good applicant).

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

All **15 tests** pass:
- 6 preprocessing tests (feature engineering, pipeline shape, split proportions, stratification)
- 5 model tests (model loading, prediction range, risk band boundaries, batch predictions)
- 4 API tests (health, predict valid/invalid, batch predict)

You can also run the integration test script against a running server:

```bash
./scripts/test_api.sh
```

---

## Features

- **Preprocessing Pipeline:** `ColumnTransformer` with `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`
- **Feature Engineering:** `credit_amount_per_duration`, `age_band`, `installment_burden`
- **Models:** Logistic Regression, Random Forest, XGBoost with GridSearchCV hyperparameter tuning
- **Evaluation:** Accuracy, precision, recall, F1, ROC-AUC, confusion matrix
- **Inference:** Model loading, probability prediction, risk band mapping
- **API:** FastAPI with Pydantic validation, CORS middleware, structured logging
- **Testing:** 15 unit tests across preprocessing, model, and API layers
- **Deployment:** Docker containerization with slim Python image

---

## Tradeoffs and Future Work

### Tradeoffs
- **Dataset size:** 1,000 rows is small for production; larger datasets would improve generalization
- **Feature count:** 20 features is manageable but could benefit from additional external data
- **Model complexity:** Random Forest outperformed slightly, but XGBoost could surpass with deeper tuning
- **Imbalanced classes:** The 70/30 split is moderately imbalanced; class weights helped but SMOTE or ADASYN could improve recall
- **Threshold tuning:** Current thresholds (0.30, 0.60) are heuristics; cost-sensitive optimization would be better

### Future Improvements
- Model calibration (Platt scaling or isotonic regression) for better probability estimates
- SHAP or LIME explanations for individual predictions
- Drift monitoring on incoming predictions versus training distribution
- A/B testing framework for model rollout
- CI/CD pipeline with GitHub Actions
- Model versioning and rollback support
- Expanded test coverage for edge cases

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

## Resume Bullets

These bullet points describe the project in language suitable for a resume:

- Built an end-to-end **credit risk scoring system** in Python using Pandas, scikit-learn, and XGBoost, covering preprocessing, feature engineering, training, and evaluation across 3 model types.
- Deployed a **FastAPI inference service** with Docker for reproducible real-time prediction, serving both single and batch prediction endpoints.
- Compared multiple classification models using **precision, recall, F1-score, and ROC-AUC** to support model selection in an imbalanced financial dataset (70/30 class split).
- Implemented **production-quality practices**: GridSearchCV hyperparameter tuning, ColumnTransformer preprocessing pipelines, Pydantic input validation, structured logging, and 15 unit tests.

---

## License

MIT
