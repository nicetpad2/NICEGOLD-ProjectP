from src.utils.resource_auto import get_optimal_resource_fraction
import os

def build_xgb_model(params: dict = None, use_gpu: bool = True, ram_fraction: float = 0.8, gpu_ram_fraction: float = 0.8, early_stopping_rounds: int = 20, eval_set=None, eval_metric='auc', callbacks=None):
    import xgboost as xgb
    ram_gb, gpu_gb = get_optimal_resource_fraction(ram_fraction, gpu_ram_fraction)
    params = dict(params) if params else {}
    params.setdefault('n_jobs', -1)
    params.setdefault('learning_rate', 0.05)
    params.setdefault('max_depth', 6)
    params.setdefault('n_estimators', 200)
    if use_gpu:
        params['tree_method'] = 'gpu_hist'
        params['gpu_id'] = 0
    model = xgb.XGBClassifier(**params)
    # Attach early stopping if eval_set is provided
    if eval_set is not None:
        fit_args = {'eval_set': eval_set, 'early_stopping_rounds': early_stopping_rounds, 'eval_metric': eval_metric}
        if callbacks:
            fit_args['callbacks'] = callbacks
        model._fit_args = fit_args
    return model

def build_lgbm_model(params: dict = None, use_gpu: bool = True, ram_fraction: float = 0.8, gpu_ram_fraction: float = 0.8, early_stopping_rounds: int = 20, eval_set=None, eval_metric='auc', callbacks=None, random_state=None):
    import lightgbm as lgb
    ram_gb, gpu_gb = get_optimal_resource_fraction(ram_fraction, gpu_ram_fraction)
    params = dict(params) if params else {}
    params.setdefault('n_jobs', -1)
    params.setdefault('learning_rate', 0.05)
    params.setdefault('max_depth', 6)
    params.setdefault('n_estimators', 200)
    
    # Handle random_state parameter
    if random_state is not None:
        params['random_state'] = random_state
        
    if use_gpu:
        params['device_type'] = 'gpu'
        params['gpu_device_id'] = 0
    model = lgb.LGBMClassifier(**params)
    if eval_set is not None:
        fit_args = {'eval_set': eval_set, 'early_stopping_rounds': early_stopping_rounds, 'eval_metric': eval_metric}
        if callbacks:
            fit_args['callbacks'] = callbacks
        model._fit_args = fit_args
    return model
