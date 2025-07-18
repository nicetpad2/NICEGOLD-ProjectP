

# [Patch v5.5.14] Simple dataclass - based pipeline config loader
from dataclasses import dataclass
from src.utils.errors import PipelineError
import os
import yaml
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
DEFAULT_CONFIG_FILE = os.path.join(CONFIG_DIR, 'pipeline.yaml')


@dataclass
class PipelineConfig:
    log_level: str = 'INFO'
    model_dir: str = 'models'
    threshold_file: str = 'threshold_wfv_optuna_results.csv'
    output_dir: str = 'output_default'
    features_filename: str = 'features_main.json'
    trade_log_pattern: str = 'trade_log_*.csv*'
    trade_log_file: str | None = None
    raw_m1_filename: str = 'XAUUSD_M1.csv'
    cleaning_fill_method: str = 'drop'
    parquet_dir: str | None = None


def load_config(path: str = DEFAULT_CONFIG_FILE) -> 'PipelineConfig':
    """Load configuration from YAML file if available."""
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding = 'utf - 8') as fh:
                data = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            mark = getattr(exc, 'problem_mark', None)
            if mark is not None:
                detail = f"(line {mark.line + 1}, column {mark.column + 1})"
            else:
                detail = ""
            msg = f"Invalid YAML in {path} {detail}: {exc}"
            raise PipelineError(msg) from exc
        if 'data' in data and isinstance(data['data'], dict):
            data = {**data, **data['data']}
            del data['data']
        if 'cleaning' in data and isinstance(data['cleaning'], dict):
            data['cleaning_fill_method'] = data['cleaning'].get('fill_method', PipelineConfig.cleaning_fill_method)
            del data['cleaning']
        return PipelineConfig(**{**PipelineConfig().__dict__, **data})
    return PipelineConfig()