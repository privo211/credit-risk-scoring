"""
SHAP-based model explanation module for Credit Risk Scoring.

Provides global feature importance and local per-prediction explanations.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import shap
except ImportError:
    shap = None
    logger.warning("shap not installed. explain module unavailable.")


def explain_global(
    model,
    X,
    feature_names=None,
    n_samples: int = 100,
) -> pd.DataFrame:
    """Compute global SHAP feature importance.

    Uses TreeExplainer for tree-based models (RF, XGB),
    falls back to KernelExplainer for non-tree models.

    Args:
        model: Fitted model with predict_proba.
        X: Feature matrix (array or DataFrame).
        feature_names: Column names. If None and X is a DataFrame, uses X.columns.
        n_samples: Background samples for KernelExplainer (ignored for TreeExplainer).

    Returns:
        DataFrame with columns ['feature', 'importance'] sorted descending.
    """
    if shap is None:
        logger.error("shap not installed — cannot compute explanations.")
        return pd.DataFrame(columns=["feature", "importance"])

    if feature_names is None and isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
    elif feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
    model_type = type(model).__name__

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_arr[:n_samples])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    except Exception:
        background = X_arr[: min(50, len(X_arr))]

        def model_predict(x):
            proba = model.predict_proba(x)
            return proba[:, 1] if proba.ndim == 2 else proba

        explainer = shap.KernelExplainer(model_predict, background)
        shap_values = explainer.shap_values(X_arr[: min(n_samples, len(X_arr))])

    importances = np.abs(shap_values).mean(axis=0)
    result = (
        pd.DataFrame(
            {"feature": feature_names[: len(importances)], "importance": importances}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    logger.info("Global SHAP importance computed for %s (%d features)", model_type, len(result))
    return result


def explain_local(
    model,
    X_sample,
    feature_names=None,
) -> dict:
    """Compute SHAP values for a single prediction or small batch.

    Args:
        model: Fitted model with predict_proba.
        X_sample: Single sample (1D) or batch (2D array/DataFrame).
        feature_names: Feature names for the returned dict.

    Returns:
        Dict with keys: 'shap_values', 'base_value', 'features'.
    """
    if shap is None:
        return {"error": "shap not installed"}

    if X_sample.ndim == 1:
        X_sample = X_sample.reshape(1, -1)
    if isinstance(X_sample, pd.DataFrame):
        if feature_names is None:
            feature_names = list(X_sample.columns)
        X_arr = X_sample.values
    else:
        X_arr = np.array(X_sample)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_arr)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        ev = explainer.expected_value
        base_value = float(ev[1] if isinstance(ev, (list, np.ndarray)) and np.ndim(ev) > 0 else ev)
    except Exception:
        return {"error": f"TreeExplainer failed for {type(model).__name__}"}

    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    per_sample = []
    for i in range(shap_values.shape[0]):
        per_sample.append(
            {name: float(shap_values[i, j]) for j, name in enumerate(feature_names[: shap_values.shape[1]])}
        )

    return {
        "shap_values": per_sample[0] if len(per_sample) == 1 else per_sample,
        "base_value": base_value,
        "features": feature_names,
    }