"""
Preprocessing pipeline and feature engineering for the German Credit dataset.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    RANDOM_STATE,
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features: credit_amount_per_duration, age_band, installment_burden.
    Returns a copy of the DataFrame with new columns.
    """
    df = df.copy()

    # Credit amount per duration month
    df["credit_amount_per_duration"] = np.where(
        df["duration"] > 0, df["credit_amount"] / df["duration"], 0
    )

    # Age bands
    conditions = [
        df["age"] < 25,
        df["age"] < 40,
        df["age"] < 55,
    ]
    choices = ["young", "adult", "middle"]
    df["age_band"] = np.select(conditions, choices, default="senior")

    # Installment burden: installment_rate * credit_amount / 100
    df["installment_burden"] = (df["installment_rate"] / 100.0) * df["credit_amount"]

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer with separate pipelines for numeric and categorical features.
    Returns:
        Fitted ColumnTransformer ready for transforming.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    # Compute transformed column names for categorical features
    # (without fitting data - just for informational purposes)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Extract feature names from the fitted ColumnTransformer.
    """
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            names = transformer.get_feature_names_out(columns)
            feature_names.extend(names)
        else:
            feature_names.extend(columns)
    return feature_names


def create_pipeline() -> Pipeline:
    """
    Create a full pipeline combining feature engineering and preprocessing.
    Note: Feature engineering must be applied separately (outside pipeline)
    because it operates on raw DataFrames, not arrays.
    This function returns just the preprocessing half.

    Use in training:
        df = engineer_features(df)
        preprocessor = build_preprocessor()
        X_processed = preprocessor.fit_transform(df)
    """
    return build_preprocessor()


def get_all_feature_columns() -> list[str]:
    """Return all feature columns (original + engineered)."""
    return (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + ["credit_amount_per_duration", "age_band", "installment_burden"]
    )
