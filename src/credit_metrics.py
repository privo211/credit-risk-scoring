"""Financial-domain metrics for credit risk scoring."""

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def gini_coefficient(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute the Gini coefficient for a credit risk model.

    Gini = 2 * ROC-AUC - 1

    The Gini coefficient is a standard measure of discriminatory power
    in credit scoring. It ranges from -1 to 1, where:
        - 0 indicates a random model (ROC-AUC = 0.5)
        - 1 indicates a perfect model (ROC-AUC = 1.0)
        - -1 indicates a perfectly inverse model (ROC-AUC = 0.0)

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0 or 1). Shape (n_samples,).
    y_score : np.ndarray
        Predicted probabilities or scores. Shape (n_samples,).

    Returns
    -------
    float
        Gini coefficient in [-1, 1].

    Raises
    ------
    ValueError
        If input arrays have different lengths, fewer than 2 samples,
        or fewer than 2 unique classes in y_true.

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_score = np.array([0.1, 0.2, 0.8, 0.9])
    >>> gini_coefficient(y_true, y_score)
    0.8
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(
            f"Shape mismatch: y_true ({y_true.shape[0]}) vs y_score "
            f"({y_score.shape[0]}) must have same number of samples."
        )
    if y_true.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 samples, got {y_true.shape[0]}."
        )
    if np.unique(y_true).shape[0] < 2:
        raise ValueError(
            "y_true must contain at least two unique classes."
        )

    auc = roc_auc_score(y_true, y_score)
    return float(2.0 * auc - 1.0)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute the Kolmogorov-Smirnov statistic for credit scoring.

    KS = max(TPR - FPR) across all possible thresholds, where:
        - TPR = true positive rate (sensitivity)
        - FPR = false positive rate (1 - specificity)

    The KS statistic measures the maximum separation between the
    cumulative distributions of good (class 0) and bad (class 1)
    borrowers. Higher values indicate better discriminatory power.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0 or 1). Shape (n_samples,).
    y_score : np.ndarray
        Predicted probabilities or scores. Shape (n_samples,).

    Returns
    -------
    float
        KS statistic in [0, 1].

    Raises
    ------
    ValueError
        If input arrays have different lengths, fewer than 2 samples,
        or fewer than 2 unique classes in y_true.

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_score = np.array([0.1, 0.4, 0.6, 0.9])
    >>> ks_statistic(y_true, y_score)
    0.5
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(
            f"Shape mismatch: y_true ({y_true.shape[0]}) vs y_score "
            f"({y_score.shape[0]}) must have same number of samples."
        )
    if y_true.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 samples, got {y_true.shape[0]}."
        )
    if np.unique(y_true).shape[0] < 2:
        raise ValueError(
            "y_true must contain at least two unique classes."
        )

    scores_class1 = y_score[y_true == 1]
    scores_class0 = y_score[y_true == 0]

    ks_val = stats.ks_2samp(scores_class1, scores_class0)

    return float(ks_val.statistic)


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, n_bins: int = 10
) -> float:
    """Compute the Population Stability Index (PSI).

    PSI = sum_i ((actual_i - expected_i) * ln(actual_i / expected_i))

    PSI quantifies the shift between an expected (reference) score
    distribution and an actual (current) score distribution. It is
    widely used in credit risk monitoring to detect score drift.

    Interpretation:
        - PSI < 0.1: no significant change (stable)
        - 0.1 <= PSI <= 0.2: slight shift (review recommended)
        - PSI > 0.2: significant drift (investigate)

    Bins are created from the expected distribution using equal-sized
    intervals. The same bin edges are applied to the actual distribution.
    If a bin contains zero actual observations, a small epsilon (1e-6)
    is substituted to avoid division by zero.

    Parameters
    ----------
    expected : np.ndarray
        Reference predicted probabilities. Shape (n_samples,).
    actual : np.ndarray
        Current predicted probabilities. Shape (n_samples,).
    n_bins : int, default=10
        Number of bins for discretisation.

    Returns
    -------
    float
        PSI value (non-negative).

    Raises
    ------
    ValueError
        If input arrays have mismatched sizes, fewer than n_bins
        samples, or n_bins < 2.

    Examples
    --------
    >>> expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    >>> actual = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92, 0.96])
    >>> psi = population_stability_index(expected, actual, n_bins=5)
    """
    expected = np.asarray(expected).ravel()
    actual = np.asarray(actual).ravel()

    if expected.shape[0] != actual.shape[0]:
        raise ValueError(
            f"Shape mismatch: expected ({expected.shape[0]}) vs actual "
            f"({actual.shape[0]}) must have same number of samples."
        )
    if n_bins < 2:
        raise ValueError(
            f"n_bins must be at least 2, got {n_bins}."
        )
    if expected.shape[0] < n_bins:
        raise ValueError(
            f"Number of samples ({expected.shape[0]}) must be at least "
            f"n_bins ({n_bins})."
        )

    lo = min(expected.min(), actual.min())
    hi = max(expected.max(), actual.max())
    eps_range = 1e-6
    lo -= eps_range
    hi += eps_range

    bin_edges = np.linspace(lo, hi, n_bins + 1)

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_counts / expected.shape[0]
    actual_pct = actual_counts / actual.shape[0]

    epsilon = 1e-6
    actual_pct = np.clip(actual_pct, epsilon, None)
    expected_pct = np.clip(expected_pct, epsilon, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)
