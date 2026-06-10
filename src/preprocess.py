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
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features: credit_amount_per_duration, age_band, installment_burden."""
    df = df.copy()

    df["credit_amount_per_duration"] = np.where(
        df["duration"] > 0, df["credit_amount"] / df["duration"], 0
    )

    conditions = [
        df["age"] < 25,
        df["age"] < 40,
        df["age"] < 55,
    ]
    choices = ["young", "adult", "middle"]
    df["age_band"] = np.select(conditions, choices, default="senior")

    df["installment_burden"] = (df["installment_rate"] / 100.0) * df["credit_amount"]

    return df


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer with separate pipelines for numeric and categorical features."""
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract feature names from the fitted ColumnTransformer."""
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
