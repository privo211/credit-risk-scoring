"""
Data loading module for the Credit Risk Scoring system.
Loads the German Credit dataset and returns train/val/test splits.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import (
    RAW_DATA_FILE,
    TARGET_COLUMN,
    TARGET_MAP,
    RANDOM_STATE,
    TRAIN_SIZE,
    VALIDATION_SIZE,
    DATA_PROCESSED,
)


def load_raw_data() -> pd.DataFrame:
    """Load the raw German Credit dataset from CSV."""
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {RAW_DATA_FILE}. "
            "Run the dataset download script first."
        )
    df = pd.read_csv(RAW_DATA_FILE)
    return df


def map_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map target: original (1=good, 2=bad) -> binary (0=no default, 1=default)."""
    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(TARGET_MAP)
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """Split data into train/validation/test sets with stratification."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Second split: separate validation from train
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_and_split_data() -> tuple:
    """Convenience: load, map target, and split in one call."""
    df = load_raw_data()
    df = map_target(df)
    return split_data(df)


def save_processed_data(
    X_train, X_val, X_test, y_train, y_val, y_test
) -> None:
    """Save processed data splits to disk for reproducibility."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    pd.concat([X_train, y_train], axis=1).to_csv(
        DATA_PROCESSED / "train.csv", index=False
    )
    pd.concat([X_val, y_val], axis=1).to_csv(
        DATA_PROCESSED / "val.csv", index=False
    )
    pd.concat([X_test, y_test], axis=1).to_csv(
        DATA_PROCESSED / "test.csv", index=False
    )
    print(f"Processed data saved to {DATA_PROCESSED}")


def load_processed_data() -> tuple:
    """Load pre-saved processed data splits."""
    train = pd.read_csv(DATA_PROCESSED / "train.csv")
    val = pd.read_csv(DATA_PROCESSED / "val.csv")
    test = pd.read_csv(DATA_PROCESSED / "test.csv")

    y_train = train[TARGET_COLUMN]
    y_val = val[TARGET_COLUMN]
    y_test = test[TARGET_COLUMN]

    X_train = train.drop(columns=[TARGET_COLUMN])
    X_val = val.drop(columns=[TARGET_COLUMN])
    X_test = test.drop(columns=[TARGET_COLUMN])

    return X_train, X_val, X_test, y_train, y_val, y_test
