#!/usr/bin/env python3
import argparse
import logging
import sys
import warnings

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from src.data_loader import load_and_split_data, save_processed_data
from src.preprocess import engineer_features, build_preprocessor
from src.train import train_all_models, select_best_model, save_model, save_preprocessor, save_metadata
from src.evaluate import evaluate_model, print_metrics, find_optimal_threshold
from src.explain import SHAPExplainer
from src.config import MODEL_VERSION, MODELS_DIR

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

    logger.info("\nCalibrating probabilities for the best model...")
    calibrated_model = CalibratedClassifierCV(
        FrozenEstimator(best_model),
        method="sigmoid",
    )
    calibrated_model.fit(X_val_p, y_val)

    save_model(models.get("Logistic Regression"), "models/logistic_regression.pkl")
    save_model(models.get("Random Forest"), "models/random_forest.pkl")
    save_model(models.get("XGBoost"), "models/xgboost.pkl")

    y_prob_val = calibrated_model.predict_proba(X_val_p)[:, 1]
    optimal_threshold = find_optimal_threshold(y_val, y_prob_val, cost_fn=10.0, cost_fp=1.0)
    logger.info(f"Optimal threshold found: {optimal_threshold:.4f}")

    test_metrics = evaluate_model(calibrated_model, X_test_p, y_test, threshold=optimal_threshold)
    print_metrics(test_metrics, "TEST SET")

    test_metrics["optimal_threshold"] = optimal_threshold
    save_metadata(best_name, test_metrics)

    save_model(calibrated_model, "models/best_model.pkl")

    logger.info("\nGenerating SHAP explanations...")
    try:
        background = X_train_p[:100]
        explainer = SHAPExplainer(calibrated_model, background)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        feature_names = preprocessor.get_feature_names_out()

        X_test_df = pd.DataFrame(X_test_p[:200], columns=feature_names)
        explainer.explain_global(X_test_df, save_path=str(MODELS_DIR / "shap_summary.png"))

        x_local = pd.DataFrame(X_test_p[:1], columns=feature_names)
        explainer.explain_local(x_local, save_path=str(MODELS_DIR / "shap_local.png"))
    except Exception as e:
        logger.warning(f"Could not generate SHAP plots: {e}")

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user. Exiting.")
        sys.exit(1)
