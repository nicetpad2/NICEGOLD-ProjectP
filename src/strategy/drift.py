"""DriftObserver class and drift-related functions."""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class DriftObserver:
    """
    Observes and analyzes feature drift between training and testing folds
    using Wasserstein distance and T-tests.
    """
    def __init__(self, features_to_observe):
        """
        Initializes the DriftObserver.

        Args:
            features_to_observe (list): A list of feature names (strings) to monitor for drift.
        """
        if not isinstance(features_to_observe, list) or not all(isinstance(f, str) for f in features_to_observe):
            raise ValueError("features_to_observe must be a list of strings.")
        self.features = features_to_observe
        self.results = {} # Dictionary to store results per fold {fold_num: {feature: {metric: value}}}
        logger.info(f"   (DriftObserver) Initialized with {len(self.features)} features to observe.")

    def analyze_fold(self, train_df_pd, test_df_pd, fold_num):
        """
        Analyzes feature drift between the training and testing data for a specific fold.

        Args:
            train_df_pd (pd.DataFrame): Training data for the fold.
            test_df_pd (pd.DataFrame): Testing data for the fold.
            fold_num (int): The index of the current fold (starting from 0).
        """
        logger.info(f"    (DriftObserver) Analyzing Drift for Fold {fold_num + 1} (M1 Features)...")
        
        # Placeholder implementation
        fold_results = {}
        for feature in self.features:
            fold_results[feature] = {
                "wasserstein": 0.1,  # dummy value
                "ttest_stat": 0.0,
                "ttest_p": 0.5
            }
        
        self.results[fold_num] = fold_results
        logger.info(f"    (DriftObserver) Completed drift analysis for Fold {fold_num + 1}")

    def get_fold_drift_summary(self, fold_num):
        """
        Calculates the mean Wasserstein distance for a given fold based on analyzed features.

        Args:
            fold_num (int): The index of the fold.

        Returns:
            float: The mean Wasserstein distance for the fold, or np.nan if no valid results.
        """
        if fold_num not in self.results:
            return np.nan
        
        fold_data = self.results[fold_num]
        if not fold_data or not isinstance(fold_data, dict):
            return np.nan

        w_dists = [res["wasserstein"] for res in fold_data.values() 
                  if isinstance(res, dict) and pd.notna(res.get("wasserstein"))]
        mean_w_dist = np.mean(w_dists) if w_dists else np.nan
        
        return mean_w_dist

    def needs_retrain(self, fold_num, threshold=0.1):
        """Determine if the given fold requires re-training based on drift."""
        fold_data = self.results.get(fold_num)
        if not fold_data or not isinstance(fold_data, dict):
            return False
        
        for metrics in fold_data.values():
            if isinstance(metrics, dict):
                w_dist = metrics.get("wasserstein")
                if isinstance(w_dist, (int, float, np.number)) and pd.notna(w_dist) and w_dist > threshold:
                    logger.info(
                        f"    (DriftObserver) Retrain triggered: Wasserstein {w_dist:.4f} > {threshold:.2f} (Fold {fold_num + 1})"
                    )
                    return True
        return False

    def export_fold_summary(self, output_dir, fold_num):
        """
        Exports the detailed drift metrics for a specific fold to a CSV file.
        """
        logger.debug(f"          (Success) Exported Drift Summary for Fold {fold_num+1} (stub)")

    def summarize_and_save(self, output_dir, wasserstein_threshold=None, ttest_alpha=None):
        """
        Summarizes drift results across all analyzed folds and saves a summary CSV report.
        """
        logger.info("  (Success) Saved M1 drift summary (stub)")

# TODO: implement real drift observation functionality later
