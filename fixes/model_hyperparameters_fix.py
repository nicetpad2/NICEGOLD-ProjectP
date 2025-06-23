
# Fixed Model Hyperparameters
OPTIMIZED_CATBOOST_PARAMS = {
    # Core parameters
    "iterations": 500,
    "learning_rate": 0.05,  # ลดลงจาก 0.1
    "depth": 8,  # เพิ่มจาก 6
    
    # Regularization (สำคัญมาก!)
    "l2_leaf_reg": 5,
    "random_strength": 0.5,
    "bagging_temperature": 0.2,
    
    # Class handling
    "auto_class_weights": "Balanced",  # แก้ class imbalance
    "class_weights": [1, 3],  # ถ้า minority class คือ 1
    
    # Performance
    "eval_metric": "AUC",
    "custom_loss": "Logloss",
    "early_stopping_rounds": 50,
    
    # Randomness
    "random_seed": 42,
    "bootstrap_type": "Bayesian",
    
    # Advanced
    "grow_policy": "SymmetricTree",
    "score_function": "Cosine",
    
    # Overfitting prevention
    "od_type": "Iter",
    "od_wait": 20,
    
    "verbose": False
}

# Alternative ensemble approach
ENSEMBLE_CONFIG = {
    "catboost": OPTIMIZED_CATBOOST_PARAMS,
    "xgboost": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": 3,  # For imbalanced data
        "random_state": 42
    },
    "lightgbm": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "class_weight": "balanced",
        "random_state": 42
    }
}

# Cross-validation strategy for time series
TIME_SERIES_CV_CONFIG = {
    "method": "TimeSeriesSplit",
    "n_splits": 5,
    "test_size": 0.2,
    "gap": 10  # Gap between train and test to prevent leakage
}
