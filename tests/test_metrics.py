"""Tests for evaluation metrics and credit risk financial metrics."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix

from src.credit_metrics import gini_coefficient, ks_statistic, population_stability_index
from src.evaluate import find_optimal_threshold


class TestGiniCoefficient:
    def test_gini_perfect(self):
        """Gini = 1.0 when predictions perfectly rank order."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        gini = gini_coefficient(y_true, y_prob)
        assert gini == pytest.approx(1.0, abs=0.01)

    def test_gini_random(self):
        """Gini close to 0 when predictions are random."""
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 50 + [1] * 50)
        y_prob = rng.uniform(0, 1, 100)
        gini = gini_coefficient(y_true, y_prob)
        assert gini == pytest.approx(0.0, abs=0.15)

    def test_gini_in_range(self):
        """Gini always in [-1, 1]."""
        rng = np.random.RandomState(99)
        for _ in range(5):
            y_true = rng.randint(0, 2, 200)
            y_prob = rng.uniform(0, 1, 200)
            gini = gini_coefficient(y_true, y_prob)
            assert -1.0 <= gini <= 1.0


class TestKSStatistic:
    def test_ks_perfect_separation(self):
        """KS = 1.0 when classes are perfectly separated."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.15, 0.05, 0.08, 0.9, 0.85, 0.95, 0.8, 0.88])
        ks = ks_statistic(y_true, y_prob)
        assert ks == pytest.approx(1.0, abs=0.05)

    def test_ks_overlap(self):
        """KS between 0 and 1 for overlapping distributions."""
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 30 + [1] * 30)
        y_prob = np.concatenate([rng.normal(0.3, 0.15, 30), rng.normal(0.7, 0.15, 30)])
        y_prob = np.clip(y_prob, 0, 1)
        ks = ks_statistic(y_true, y_prob)
        assert 0 < ks < 1.0

    def test_ks_no_discrimination(self):
        """KS near 0 when classes are indistinguishable."""
        y_true = np.array([0] * 50 + [1] * 50)
        y_prob = np.array([0.5] * 100)
        ks = ks_statistic(y_true, y_prob)
        assert ks < 0.05


class TestPopulationStabilityIndex:
    def test_psi_identical(self):
        """PSI = 0 when distributions are identical."""
        rng = np.random.RandomState(42)
        expected = rng.normal(0.5, 0.1, 100)
        actual = expected.copy()
        psi = population_stability_index(expected, actual, n_bins=10)
        assert psi == pytest.approx(0.0, abs=1e-10)

    def test_psi_different(self):
        """PSI > 0 when distributions differ."""
        rng = np.random.RandomState(42)
        expected = rng.normal(0.3, 0.1, 100)
        actual = rng.normal(0.7, 0.1, 100)
        psi = population_stability_index(expected, actual, n_bins=10)
        assert psi > 0

    def test_psi_epsilon_handling(self):
        """PSI handles edge bins gracefully."""
        rng = np.random.RandomState(42)
        expected = rng.normal(0.1, 0.05, 100)
        actual = rng.normal(0.9, 0.05, 100)
        psi = population_stability_index(expected, actual, n_bins=10)
        assert np.isfinite(psi)


class TestFindOptimalThreshold:
    def test_returns_tuple(self):
        """Returns (threshold, cost) tuple."""
        rng = np.random.RandomState(42)
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        X = rng.rand(50, 5)
        y = rng.randint(0, 2, 50)
        model.fit(X, y)
        result = find_optimal_threshold(model, X, y)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_threshold_in_range(self):
        """Threshold between 0.1 and 0.9."""
        rng = np.random.RandomState(42)
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        X = rng.rand(50, 5)
        y = rng.randint(0, 2, 50)
        model.fit(X, y)
        threshold, _ = find_optimal_threshold(model, X, y)
        assert 0.1 <= threshold <= 0.9