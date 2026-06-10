"""
Model training module for Credit Risk Scoring.
Trains and saves Logistic Regression, Random Forest, and XGBoost models.
"""

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.config import (
    MODELS_DIR,
    PREPROCESSOR_PATH,
    BEST_MODEL_PATH,
    MODEL_METADATA_PATH,
    RANDOM_STATE,
    CV_FOLDS,
    LOGISTIC_PARAMS,
    RF_PARAMS,
    XGB_PARAMS,
    POSITIVE_CLASS,
    MODEL_VERSION,
)


def train_logistic_regression(X_train, y_train, cv=CV_FOLDS):
    """
    Train Logistic Regression with hyperparameter tuning.
    Returns the best estimator.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    lr = LogisticRegression(random_state=RANDOM_STATE)
    grid = GridSearchCV(
        lr,
        LOGISTIC_PARAMS,
        cv=skf,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV ROC-AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_


def train_random_forest(X_train, y_train, cv=CV_FOLDS):
    """
    Train Random Forest with hyperparameter tuning.
    Returns the best estimator.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    grid = GridSearchCV(
        rf,
        RF_PARAMS,
        cv=skf,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV ROC-AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_


def train_xgboost(X_train, y_train, cv=CV_FOLDS):
    """
    Train XGBoost with hyperparameter tuning.
    Returns the best estimator.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  XGBoost not installed. Skipping.")
        return None

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    xgb = XGBClassifier(
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
    )
    grid = GridSearchCV(
        xgb,
        XGB_PARAMS,
        cv=skf,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV ROC-AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_


def train_all_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Train all three models and return them in a dictionary.
    """
    models = {}
    print("Training Logistic Regression...")
    models["Logistic Regression"] = train_logistic_regression(X_train, y_train)
    print("Training Random Forest...")
    models["Random Forest"] = train_random_forest(X_train, y_train)
    print("Training XGBoost...")
    models["XGBoost"] = train_xgboost(X_train, y_train)
    return models


def save_model(model, path: str):
    """Save a trained model to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"  Model saved to {path}")


def save_preprocessor(preprocessor, path: str = PREPROCESSOR_PATH):
    """Save the fitted preprocessor to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"  Preprocessor saved to {path}")


def save_metadata(model_name: str, metrics: dict):
    """Save training metadata as JSON."""
    metadata = {
        "model_name": model_name,
        "model_version": MODEL_VERSION,
        "training_date": datetime.now().isoformat(),
        "dataset": "German Credit (Statlog)",
        "num_samples": 1000,
        "metrics": metrics,
    }
    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {MODEL_METADATA_PATH}")


def select_best_model(
    models: dict, X_val: pd.DataFrame, y_val: pd.Series
) -> tuple[str, object]:
    """
    Select the best model based on ROC-AUC on validation set.
    Returns (model_name, model).
    """
    from sklearn.metrics import roc_auc_score

    best_name = None
    best_model = None
    best_score = -1

    for name, model in models.items():
        if model is None:
            continue
        y_prob = model.predict_proba(X_val)[:, POSITIVE_CLASS]
        score = roc_auc_score(y_val, y_prob)
        print(f"  {name}: Validation ROC-AUC = {score:.4f}")
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model

    print(f"\n  Best model: {best_name} (ROC-AUC: {best_score:.4f})")
    return best_name, best_model
