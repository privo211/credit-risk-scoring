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
    ENGINEERED_NUMERIC_FEATURES,
    ENGINEERED_CATEGORICAL_FEATURES,
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 10 engineered features: credit_amount_per_duration, age_band, installment_burden,
    debt_to_income_proxy, employment_stability_score, savings_adequacy,
    checking_balance_score, high_risk_purpose_flag, guarantor_buffer, credit_utilization."""
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

    df["debt_to_income_proxy"] = df["credit_amount"] / (
        df["duration"] * df["installment_rate"] / 100 + 1
    )

    employment_map = {"A71": 0, "A72": 1, "A73": 2, "A74": 3, "A75": 4}
    df["employment_stability_score"] = df["employment"].map(employment_map).fillna(0)

    savings_map = {"A61": 0, "A62": 1, "A63": 2, "A64": 3, "A65": 4}
    df["savings_adequacy"] = df["savings"].map(savings_map).fillna(0)

    checking_map = {"A11": 0, "A12": 1, "A13": 2, "A14": 3}
    df["checking_balance_score"] = df["checking_status"].map(checking_map).fillna(0)

    df["high_risk_purpose_flag"] = np.where(
        df["purpose"].isin(["A41", "A43", "A46", "A410"]), 1, 0
    )

    df["guarantor_buffer"] = np.where(df["guarantors"] != "A101", 1, 0)

    df["credit_utilization"] = df["credit_amount"] / np.maximum(df["num_credits"], 1)

    return df


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer with separate pipelines for numeric and categorical features."""
    numeric_features = NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
    categorical_features = CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES

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
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
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
