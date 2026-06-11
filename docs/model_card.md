# Model Card: Credit Risk Scoring v2.0.0

## Model Overview
- **Model Name**: Credit Risk Scoring Service
- **Version**: 2.0.0
- **Model Type**: Ensemble (Random Forest with SMOTE inside GridSearchCV + Platt Calibration)
- **Date**: June 2026
- **Intended Use**: Predict probability of loan default for consumer credit applications

## Dataset
- **Name**: German Credit (Statlog) Dataset
- **Source**: UCI Machine Learning Repository
- **Samples**: 1,000
- **Features**: 20 raw (10 after feature engineering = 30 total input columns)
- **Target**: Binary (0 = no default, 1 = default)
- **Class Distribution**: 70% non-default, 30% default
- **Missing Values**: None

## Feature Engineering
The pipeline creates 10 engineered features from the 20 raw columns:
- **credit_amount_per_duration**: Average credit amount per month of loan
- **age_band**: Categorized age group (young, adult, middle, senior)
- **installment_burden**: Installment rate applied to credit amount
- **debt_to_income_proxy**: Credit amount relative to repayment capacity
- **employment_stability_score**: Ordinal encoding of employment category (A71=0 to A75=4)
- **savings_adequacy**: Ordinal encoding of savings level (A61=0 to A65=4)
- **checking_balance_score**: Ordinal encoding of checking account status (A11=0 to A14=3)
- **high_risk_purpose_flag**: Binary flag for high-risk loan purposes (A41, A43, A46, A410)
- **guarantor_buffer**: Binary flag for presence of guarantors
- **credit_utilization**: Average credit amount per existing credit

## Training Pipeline
1. Stratified 60/20/20 train/val/test split
2. SMOTE oversampling inside each GridSearchCV training fold
3. StandardScaler + OneHotEncoder via ColumnTransformer
4. GridSearchCV (5-fold) for Logistic Regression, Random Forest, XGBoost
5. Best model selected by validation ROC-AUC
6. Platt calibration (sigmoid, 5-fold CV) on training data

## Performance Metrics (Test Set)
| Metric | Value |
|---|---|
| ROC-AUC | 0.7861 |
| Gini Coefficient | 0.5721 |
| KS Statistic | 0.4405 |
| Accuracy | 0.7700 |
| Precision | 0.6522 |
| Recall | 0.5000 |
| F1 Score | 0.5660 |

## Confusion Matrix (Test Set)
| | Predicted 0 | Predicted 1 |
|---|---|---|
| Actual 0 | TN=124 | FP=16 |
| Actual 1 | FN=30 | TP=30 |

## Model Comparison (Validation ROC-AUC)
| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.7307 |
| Random Forest | 0.7453 |
| XGBoost | 0.7379 |

## Fairness Considerations
- The German Credit dataset contains demographic attributes (age, personal status, foreign worker status) that may correlate with protected groups
- These features are included in the model; no explicit debiasing has been applied
- Users should evaluate model fairness for their specific deployment context
- Future work: apply fairness metrics and disparate impact analysis

## Limitations
- Small dataset (1,000 samples) limits generalization
- Recall of 0.5000 means the model misses approximately 50% of defaults
- Dataset is from 1990s Germany — may not generalize to other populations or time periods
- No continuous monitoring or drift detection implemented

## Usage
- **API**: FastAPI service on port 8000 with /predict and /batch_predict endpoints
- **Rate Limiting**: 100 requests/minute per IP (configurable)
- **Database**: PostgreSQL 16 via docker-compose with async SQLAlchemy
- **Dependencies**: See requirements.txt
- **Deployment**: Docker multi-stage (python:3.12-slim, non-root user)
