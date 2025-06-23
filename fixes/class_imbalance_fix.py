
# Class Imbalance Fix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def handle_class_imbalance(X, y, method="combined"):
    """
    แก้ไขปัญหา class imbalance หลายวิธี
    """
    
    if method == "resampling":
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.pipeline import Pipeline as ImbPipeline
        
        # Combined over and under sampling
        over = SMOTE(sampling_strategy=0.3, random_state=42)
        under = RandomUnderSampler(sampling_strategy=0.7, random_state=42)
        
        pipeline = ImbPipeline([("over", over), ("under", under)])
        X_resampled, y_resampled = pipeline.fit_resample(X, y)
        
        return X_resampled, y_resampled
        
    elif method == "class_weights":
        # Calculate class weights
        classes = np.unique(y)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y
        )
        
        class_weight_dict = dict(zip(classes, class_weights))
        return class_weight_dict
        
    elif method == "threshold_optimization":
        # Optimize threshold for imbalanced data
        from sklearn.metrics import precision_recall_curve
        
        def find_optimal_threshold(y_true, y_proba):
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            return thresholds[optimal_idx]
            
        return find_optimal_threshold
        
    elif method == "cost_sensitive":
        # Cost-sensitive learning parameters
        return {
            "catboost": {"class_weights": [1, 5]},  # Higher cost for minority class
            "xgboost": {"scale_pos_weight": 5},
            "lightgbm": {"class_weight": {0: 1, 1: 5}}
        }

# Usage:
# X_balanced, y_balanced = handle_class_imbalance(X, y, method="resampling")
# class_weights = handle_class_imbalance(X, y, method="class_weights")
