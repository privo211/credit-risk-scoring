# Model Card: Credit Risk Scoring v1.1.0

## Model Overview

- Model name: Credit Risk Scoring Service
- Version: 1.1.0
- Model type: Random Forest pipeline with SMOTE and sigmoid calibration
- Dataset: German Credit (Statlog)
- Intended use: estimate consumer loan default probability and assign an operational risk band

## Training Data

- Samples: 1,000
- Raw features: 20
- Engineered features: 10
- Target: binary default label, where 0 means no default and 1 means default
- Split: stratified train/validation/test split of 60/20/20

## Pipeline

1. Load and map the German Credit target labels.
2. Create 10 domain-oriented engineered features.
3. Fit a ColumnTransformer with numeric scaling and categorical one-hot encoding.
4. Train Logistic Regression, Random Forest, and XGBoost candidates with SMOTE inside each GridSearchCV pipeline.
5. Select the best model by validation ROC-AUC.
6. Calibrate the selected model with sigmoid calibration on the validation set.
7. Tune a binary decision threshold with false negatives weighted 10x false positives.
8. Save model artifacts, metadata, and SHAP explanation plots.

## Test Performance

| Metric | Value |
|---|---:|
| ROC-AUC | 0.7955 |
| Accuracy | 0.5200 |
| Precision | 0.3800 |
| Recall | 0.9500 |
| F1 score | 0.5429 |
| Optimal binary threshold | 0.1400 |

## Confusion Matrix

| | Predicted no default | Predicted default |
|---|---:|---:|
| Actual no default | 47 | 93 |
| Actual default | 3 | 57 |

## Model Comparison

| Model | Validation ROC-AUC |
|---|---:|
| Logistic Regression | 0.7307 |
| Random Forest | 0.7453 |
| XGBoost | 0.7379 |

## Operational Notes

- The API returns calibrated default probability and LOW/MEDIUM/HIGH risk bands.
- PostgreSQL prediction logging is optional and enabled by setting CREDIT_DATABASE_URL.
- Docker Compose runs the API with PostgreSQL 16.
- SHAP plots are generated during training and saved under models/.

## Limitations

- German Credit has only 1,000 rows, so results should be treated as educational rather than production-ready.
- The optimized threshold is recall-heavy and creates many false positives.
- No fairness audit has been implemented, despite demographic attributes being present.
- No monitoring job currently computes drift metrics on live prediction logs.

## Appropriate Use

This model is suitable as a portfolio demonstration of ML engineering practices. It should not be used for real lending decisions without larger, current data, fairness review, compliance review, and ongoing monitoring.
