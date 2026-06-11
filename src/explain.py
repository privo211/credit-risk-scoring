"""
SHAP explanation module for Credit Risk Scoring.
Generates local and global explanations for model predictions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from typing import Union, List

from src.config import POSITIVE_CLASS, MODELS_DIR

# Ensure SHAP doesn't open plots interactively by default during batch processes
plt.switch_backend("Agg")


class SHAPExplainer:
    """Wrapper around SHAP to generate model explanations."""

    def __init__(self, model, background_data: pd.DataFrame):
        """
        Initialize the explainer.

        Args:
            model: The trained scikit-learn compatible model pipeline or estimator.
            background_data: A representative sample of training data to use as background.
        """
        self.model = model
        self.background_data = background_data

        # Try TreeExplainer for tree-based models, fallback to KernelExplainer
        # or PermutationExplainer if needed. Since our final models could be
        # RF or XGB, TreeExplainer is preferred.
        try:
            # We must pass the raw estimator, not a pipeline, if using TreeExplainer.
            # But the 'model' here might be the actual estimator (e.g., best_estimator_ is
            # ImbPipeline or just model). Let's extract the actual model if it's a pipeline.
            if hasattr(model, 'named_steps'):
                self.estimator = model.named_steps['model']
            else:
                self.estimator = model

            # If the estimator is linear, use LinearExplainer
            if type(self.estimator).__name__ == "LogisticRegression":
                self.explainer = shap.LinearExplainer(self.estimator, background_data)
            else:
                self.explainer = shap.TreeExplainer(self.estimator)
        except Exception as e:
            print(f"  Warning: Falling back to Explainer (Permutation) due to: {e}")
            self.estimator = model
            self.explainer = shap.Explainer(self.estimator.predict_proba, background_data)

    def explain_global(self, X: pd.DataFrame, save_path: str = None) -> np.ndarray:
        """
        Generate global feature importance explanations (summary plot).

        Args:
            X: The dataset to explain.
            save_path: Optional path to save the plot.

        Returns:
            The SHAP values array.
        """
        shap_values = self.explainer.shap_values(X)

        # For classification, TreeExplainer returns a list of arrays (one per class).
        # We want the values for the positive class.
        if isinstance(shap_values, list):
            vals_to_plot = shap_values[POSITIVE_CLASS]
        else:
            vals_to_plot = shap_values

        plt.figure(figsize=(10, 8))
        shap.summary_plot(vals_to_plot, X, show=False)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"  Global SHAP summary plot saved to {save_path}")

        plt.close()
        return vals_to_plot

    def explain_local(self, x: pd.Series, save_path: str = None) -> Union[dict, None]:
        """
        Generate local explanation for a single prediction.

        Args:
            x: Single instance to explain (Series or single-row DataFrame).
            save_path: Optional path to save the waterfall or force plot.

        Returns:
            Dictionary containing base value and SHAP values for the instance.
        """
        if isinstance(x, pd.Series):
            x = x.to_frame().T

        # We need an Explainer object that returns an Explanation for waterfall plot
        try:
            # Re-create a generic explainer to get Explanation objects easily
            # if we didn't use the generic one above.
            if hasattr(self.model, "predict_proba"):
                local_explainer = shap.Explainer(lambda data: self.model.predict_proba(data)[:, POSITIVE_CLASS], self.background_data)
                shap_obj = local_explainer(x)

                if save_path:
                    plt.figure(figsize=(10, 5))
                    shap.waterfall_plot(shap_obj[0], show=False)
                    plt.savefig(save_path, bbox_inches="tight")
                    plt.close()
                    print(f"  Local SHAP waterfall plot saved to {save_path}")

                return {
                    "base_value": float(shap_obj[0].base_values),
                    "shap_values": shap_obj[0].values.tolist(),
                    "features": x.columns.tolist()
                }
            else:
                return None
        except Exception as e:
            print(f"  Failed to generate local explanation: {e}")
            return None
