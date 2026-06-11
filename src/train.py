import json
import joblib
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

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
    GRID_N_JOBS,
)


def train_logistic_regression(X_train, y_train, cv=CV_FOLDS):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    lr = LogisticRegression(random_state=RANDOM_STATE)
    smote = SMOTE(random_state=RANDOM_STATE)
    pipeline = ImbPipeline([("smote", smote), ("model", lr)])

    grid_params = {f"model__{k}": v for k, v in LOGISTIC_PARAMS.items()}

    grid = GridSearchCV(
        pipeline,
        grid_params,
        cv=skf,
        scoring="roc_auc",
        n_jobs=GRID_N_JOBS,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV ROC-AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_


def train_random_forest(X_train, y_train, cv=CV_FOLDS):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    smote = SMOTE(random_state=RANDOM_STATE)
    pipeline = ImbPipeline([("smote", smote), ("model", rf)])

    grid_params = {f"model__{k}": v for k, v in RF_PARAMS.items()}

    grid = GridSearchCV(
        pipeline,
        grid_params,
        cv=skf,
        scoring="roc_auc",
        n_jobs=GRID_N_JOBS,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV ROC-AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_


def train_xgboost(X_train, y_train, cv=CV_FOLDS):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  XGBoost not installed. Skipping.")
        return None

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    xgb = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    )
    smote = SMOTE(random_state=RANDOM_STATE)
    pipeline = ImbPipeline([("smote", smote), ("model", xgb)])

    grid_params = {f"model__{k}": v for k, v in XGB_PARAMS.items()}

    grid = GridSearchCV(
        pipeline,
        grid_params,
        cv=skf,
        scoring="roc_auc",
        n_jobs=GRID_N_JOBS,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV ROC-AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_


def train_all_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    models = {}
    print("Training Logistic Regression...")
    models["Logistic Regression"] = train_logistic_regression(X_train, y_train)
    print("Training Random Forest...")
    models["Random Forest"] = train_random_forest(X_train, y_train)
    print("Training XGBoost...")
    models["XGBoost"] = train_xgboost(X_train, y_train)
    return models


def save_model(model, path: str):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"  Model saved to {path}")


def save_preprocessor(preprocessor, path: str = PREPROCESSOR_PATH):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"  Preprocessor saved to {path}")


def save_metadata(model_name: str, metrics: dict):
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
