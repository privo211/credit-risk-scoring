"""
Model evaluation module for Credit Risk Scoring.
Computes metrics, confusion matrices, model comparison, and optimal threshold search.
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
    """Compute comprehensive evaluation metrics for a model."""
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
    """Evaluate all models and produce a comparison DataFrame."""
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


def find_optimal_threshold(
    model,
    X_val,
    y_val,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
    pos_class: int = 1,
):
    """Find probability threshold minimizing total cost.

    Cost matrix: FN costs cost_fn (default 5x), FP costs cost_fp (default 1x).
    Sweeps thresholds 0.10-0.90 in 0.05 steps.

    Returns:
        Tuple of (best_threshold, best_cost).
    """
    y_prob = model.predict_proba(X_val)[:, pos_class]
    thresholds = np.arange(0.10, 0.95, 0.05)
    best_threshold = 0.5
    best_cost = float("inf")

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        total_cost = fn * cost_fn + fp * cost_fp
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = t

    print(f"  Optimal threshold: {best_threshold:.2f} (total cost: {best_cost:.2f})")
    print(f"    Cost params: FN={cost_fn}x, FP={cost_fp}x")
    return best_threshold, best_cost
