"""
Configuration constants for the Credit Risk Scoring system.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Dataset files
RAW_DATA_FILE = DATA_RAW / "german_credit_data.csv"

# Artifact paths
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Random seed for reproducibility
RANDOM_STATE = 42

# Train/Val/Test split ratios
TRAIN_SIZE = 0.6
VALIDATION_SIZE = 0.2  # relative to full dataset
TEST_SIZE = 0.2  # relative to full dataset

# Target encoding
TARGET_MAP = {1: 0, 2: 1}  # Map original: 1=good→0, 2=bad→1
TARGET_COLUMN = "target"
POSITIVE_CLASS = 1  # default = bad

# Risk band thresholds
RISK_THRESHOLDS = {
    "LOW": (0.0, 0.30),
    "MEDIUM": (0.30, 0.60),
    "HIGH": (0.60, 1.0),
}

# Column names (raw dataset)
COLUMNS = [
    "checking_status",  # Status of existing checking account (categorical)
    "duration",  # Duration in months (numeric)
    "credit_history",  # Credit history (categorical)
    "purpose",  # Purpose (categorical)
    "credit_amount",  # Credit amount (numeric)
    "savings",  # Savings account/bonds (categorical)
    "employment",  # Present employment since (categorical)
    "installment_rate",  # Installment rate (numeric)
    "personal_status",  # Personal status and sex (categorical)
    "guarantors",  # Other debtors / guarantors (categorical)
    "residence_since",  # Present residence since (numeric)
    "property",  # Property (categorical)
    "age",  # Age in years (numeric)
    "other_plans",  # Other installment plans (categorical)
    "housing",  # Housing (categorical)
    "num_credits",  # Number of existing credits (numeric)
    "job",  # Job (categorical)
    "people_maintenance",  # Number of people liable for maintenance (numeric)
    "telephone",  # Telephone (categorical)
    "foreign_worker",  # Foreign worker (categorical)
]

# Numeric features (will be scaled)
NUMERIC_FEATURES = [
    "duration",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age",
    "num_credits",
    "people_maintenance",
]

# Categorical features (will be one-hot encoded)
CATEGORICAL_FEATURES = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings",
    "employment",
    "personal_status",
    "guarantors",
    "property",
    "other_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]

# Feature engineering columns
ENGINEERED_FEATURES = [
    "credit_amount_per_duration",
    "age_band",
    "installment_burden",
]

# Age bands
AGE_BANDS = {
    "young": (0, 25),
    "adult": (25, 40),
    "middle": (40, 55),
    "senior": (55, 200),
}

MODEL_VERSION = "1.0.0"

LOGISTIC_PARAMS = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "penalty": ["l2"],
    "solver": ["lbfgs"],
    "max_iter": [1000],
    "class_weight": ["balanced"],
}

RF_PARAMS = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "class_weight": ["balanced", "balanced_subsample"],
}

XGB_PARAMS = {
    "n_estimators": [100, 200],
    "max_depth": [3, 6],
    "learning_rate": [0.01, 0.1],
    "scale_pos_weight": [1, 2, 3],
}

CV_FOLDS = 5

LOGGER_NAME = "credit_risk"
