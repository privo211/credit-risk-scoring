"""Tests for preprocessing pipeline and feature engineering."""

import pandas as pd
import numpy as np
from src.data_loader import load_and_split_data
from src.preprocess import engineer_features, build_preprocessor
from src.config import ENGINEERED_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def test_feature_engineering_adds_columns():
    X_train, _, _, _, _, _ = load_and_split_data()
    original_cols = X_train.columns.tolist()
    X_train_fe = engineer_features(X_train)
    new_cols = set(X_train_fe.columns) - set(original_cols)
    assert len(new_cols) == 3, f"Expected 3 new cols, got {len(new_cols)}"
    for feat in ENGINEERED_FEATURES:
        assert feat in X_train_fe.columns, f"Missing engineered feature: {feat}"


def test_feature_engineering_no_missing_values():
    X_train, _, _, _, _, _ = load_and_split_data()
    X_train_fe = engineer_features(X_train)
    for col in ENGINEERED_FEATURES:
        assert (
            X_train_fe[col].isna().sum() == 0
        ), f"NaN values in {col}"


def test_pipeline_output_shape():
    X_train, _, _, _, _, _ = load_and_split_data()
    X_train_fe = engineer_features(X_train)
    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X_train_fe)
    assert X_processed.shape[0] == X_train.shape[0], "Row count mismatch"
    assert X_processed.shape[1] > 0, "No features produced"


def test_split_proportions():
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()
    total = len(y_train) + len(y_val) + len(y_test)
    assert abs(len(y_train) / total - 0.6) < 0.01, "Train proportion off"
    assert abs(len(y_val) / total - 0.2) < 0.01, "Val proportion off"
    assert abs(len(y_test) / total - 0.2) < 0.01, "Test proportion off"


def test_stratification_preserves_class_ratio():
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()
    overall_ratio = pd.concat([y_train, y_val, y_test]).mean()
    train_ratio = y_train.mean()
    val_ratio = y_val.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - overall_ratio) < 0.05
    assert abs(val_ratio - overall_ratio) < 0.05
    assert abs(test_ratio - overall_ratio) < 0.05


def test_preprocessor_handles_unknown_categories():
    X_train, X_val, _, _, _, _ = load_and_split_data()
    X_train_fe = engineer_features(X_train)
    X_val_fe = engineer_features(X_val)
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train_fe)
    X_val_transformed = preprocessor.transform(X_val_fe)
    assert X_val_transformed.shape[0] == X_val.shape[0]
    assert not np.any(np.isnan(X_val_transformed.toarray() if hasattr(X_val_transformed, 'toarray') else X_val_transformed))
