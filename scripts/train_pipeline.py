#!/usr/bin/env python3
import argparse
import logging
import sys
import warnings

from src.data_loader import load_and_split_data, save_processed_data
from src.preprocess import engineer_features, build_preprocessor
from src.train import train_all_models, select_best_model, save_model, save_preprocessor, save_metadata
from src.evaluate import evaluate_model, print_metrics
from src.config import MODEL_VERSION

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train credit risk scoring models.",
    )
    parser.add_argument(
        "--save-splits",
        action="store_true",
        help="Save processed train/val/test splits to disk after splitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("CREDIT RISK SCORING - MODEL TRAINING")
    logger.info("=" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()

    if args.save_splits:
        save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)

    X_train_fe = engineer_features(X_train)
    X_val_fe = engineer_features(X_val)
    X_test_fe = engineer_features(X_test)

    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train_fe)
    X_val_p = preprocessor.transform(X_val_fe)
    X_test_p = preprocessor.transform(X_test_fe)
    save_preprocessor(preprocessor)

    models = train_all_models(X_train_p, y_train)
    best_name, best_model = select_best_model(models, X_val_p, y_val)

    save_model(best_model, "models/best_model.pkl")
    save_model(models.get("Logistic Regression"), "models/logistic_regression.pkl")
    save_model(models.get("Random Forest"), "models/random_forest.pkl")
    save_model(models.get("XGBoost"), "models/xgboost.pkl")

    test_metrics = evaluate_model(best_model, X_test_p, y_test)
    print_metrics(test_metrics, "TEST SET")

    save_metadata(best_name, test_metrics)

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user. Exiting.")
        sys.exit(1)
