"""
Model evaluation module for Credit Risk Scoring.
Computes metrics, confusion matrices, and model comparison.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.config import POSITIVE_CLASS


def evaluate_model(
    model, X: pd.DataFrame, y_true: pd.Series
) -> dict:
    """
    Compute comprehensive evaluation metrics for a model.

    Args:
        model: Trained classifier with predict_proba.
        X: Feature DataFrame.
        y_true: True labels.

    Returns:
        Dictionary with accuracy, precision, recall, f1, roc_auc,
        and confusion matrix.
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, POSITIVE_CLASS]

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def print_metrics(metrics: dict, label: str = ""):
    """Pretty-print evaluation metrics."""
    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}Evaluation Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")


def compare_models(
    models: dict, X_val: pd.DataFrame, y_val: pd.Series
) -> pd.DataFrame:
    """
    Evaluate all models and produce a comparison DataFrame.

    Args:
        models: Dictionary of {name: model}.
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        DataFrame with model comparison metrics.
    """
    results = []
    for name, model in models.items():
        if model is None:
            continue
        metrics = evaluate_model(model, X_val, y_val)
        results.append(
            {
                "Model": name,
                "Accuracy": f"{metrics['accuracy']:.4f}",
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "F1": f"{metrics['f1']:.4f}",
                "ROC-AUC": f"{metrics['roc_auc']:.4f}",
            }
        )

    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    # Find best model by ROC-AUC
    numeric_results = [
        (r["Model"], float(r["ROC-AUC"])) for r in results
    ]
    best_model = max(numeric_results, key=lambda x: x[1])
    print(f"\nBest model: {best_model[0]} (ROC-AUC: {best_model[1]:.4f})")

    return df


def print_classification_report(model, X_val, y_val):
    """Print scikit-learn classification report."""
    y_pred = model.predict(X_val)
    target_names = ["No Default (0)", "Default (1)"]
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=target_names))
