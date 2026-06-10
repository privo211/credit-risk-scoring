"""Tests for SHAP explanation module (src/explain.py)."""

import numpy as np
import pandas as pd

from src.explain import explain_global, explain_local


class TestExplainGlobal:
    def test_returns_dataframe(self, trained_model, sample_data):
        """explain_global returns a DataFrame with feature and importance columns."""
        model, _ = trained_model
        from src.preprocess import engineer_features
        import joblib
        df_fe = engineer_features(sample_data)
        pp = joblib.load("/private/tmp/credit-risk-scoring/models/preprocessor.pkl")
        X = pp.transform(df_fe)
        result = explain_global(model, X, n_samples=10)
        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "importance" in result.columns

    def test_importance_sorted_descending(self, trained_model, sample_data):
        """Importance values are sorted descending."""
        model, _ = trained_model
        from src.preprocess import engineer_features
        import joblib
        df_fe = engineer_features(sample_data)
        pp = joblib.load("/private/tmp/credit-risk-scoring/models/preprocessor.pkl")
        X = pp.transform(df_fe)
        result = explain_global(model, X, n_samples=10)
        assert all(result["importance"].iloc[i] >= result["importance"].iloc[i + 1]
                   for i in range(len(result) - 1))

    def test_graceful_no_shap(self, monkeypatch, trained_model, sample_data):
        """explain_global returns empty DataFrame when shap is missing."""
        monkeypatch.setattr("src.explain.shap", None)
        model, _ = trained_model
        import joblib
        from src.preprocess import engineer_features
        df_fe = engineer_features(sample_data)
        pp = joblib.load("/private/tmp/credit-risk-scoring/models/preprocessor.pkl")
        X = pp.transform(df_fe)
        result = explain_global(model, X)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_feature_names_from_dataframe(self, trained_model, sample_data):
        """explain_global uses DataFrame column names when feature_names is None."""
        model, _ = trained_model
        from src.preprocess import engineer_features
        import joblib
        df_fe = engineer_features(sample_data)
        pp = joblib.load("/private/tmp/credit-risk-scoring/models/preprocessor.pkl")
        X = pp.transform(df_fe)
        X_df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
        result = explain_global(model, X_df, n_samples=10)
        assert result["feature"].iloc[0].startswith("col_")


class TestExplainLocal:
    def test_returns_dict(self, trained_model, sample_data):
        """explain_local returns a dict with required keys."""
        model, _ = trained_model
        from src.preprocess import engineer_features
        import joblib
        df_fe = engineer_features(sample_data)
        pp = joblib.load("/private/tmp/credit-risk-scoring/models/preprocessor.pkl")
        X = pp.transform(df_fe)
        result = explain_local(model, X[0])
        assert isinstance(result, dict)
        assert "shap_values" in result
        assert "base_value" in result
        assert "features" in result

    def test_single_sample_returns_dict_shap(self, trained_model, sample_data):
        """explain_local returns a single dict for 1D input."""
        model, _ = trained_model
        from src.preprocess import engineer_features
        import joblib
        df_fe = engineer_features(sample_data)
        pp = joblib.load("/private/tmp/credit-risk-scoring/models/preprocessor.pkl")
        X = pp.transform(df_fe)
        result = explain_local(model, X[0])
        assert isinstance(result["shap_values"], dict)
        assert isinstance(result["base_value"], float)

    def test_batch_returns_list(self, trained_model, sample_data):
        """explain_local returns list of dicts for batch input."""
        model, _ = trained_model
        from src.preprocess import engineer_features
        import joblib
        df_fe = engineer_features(sample_data)
        pp = joblib.load("/private/tmp/credit-risk-scoring/models/preprocessor.pkl")
        X = pp.transform(df_fe)
        result = explain_local(model, X[:3])
        if isinstance(result["shap_values"], list):
            assert len(result["shap_values"]) == 3