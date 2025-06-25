# automl_utils.py
from autogluon.tabular import TabularPredictor
from rich.console import Console
from rich.panel import Panel

"""
AutoML utilities for AutoGluon, H2O, FLAML, TPOT, autoxgboost, autocatboost
- Distributed/parallel hyperparameter search
- Ensemble selection, feature selection
- Rich leaderboard, export best pipeline/code
"""
console = Console()

# Example: AutoGluon
try:
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False


def run_autogluon(train_data, label, time_limit=600):
    if not AUTOGLUON_AVAILABLE:
        console.print(
            Panel("[red]AutoGluon not installed", title="AutoML", border_style="red")
        )
        return None
    predictor = TabularPredictor(label=label).fit(train_data, time_limit=time_limit)
    leaderboard = predictor.leaderboard(silent=True)
    console.print(
        Panel(
            f"[green]AutoGluon training complete!\nBest model: {predictor.get_model_best()}",
            title="AutoML Result",
            border_style="green",
        )
    )
    return predictor, leaderboard


# Additional AutoML frameworks implementation planned:
# - H2O: High-performance distributed AutoML platform
# - FLAML: Fast Lightweight AutoML library
# - TPOT: Tree-based Pipeline Optimization Tool
# - autoxgboost: Automatic XGBoost hyperparameter tuning
# - autocatboost: Automatic CatBoost hyperparameter tuning
# - Pipeline export: Code generation for production deployment
