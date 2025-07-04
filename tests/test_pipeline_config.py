

from src.utils import pipeline_config
import pytest
def test_load_config_defaults(tmp_path):
    cfg = pipeline_config.load_config(str(tmp_path / 'missing.yaml'))
    assert isinstance(cfg, pipeline_config.PipelineConfig)
    assert cfg.model_dir == 'models'


def test_load_config_override(tmp_path):
    conf_path = tmp_path / 'cfg.yaml'
    conf_path.write_text('log_level: DEBUG\nmodel_dir: demo\nthreshold_file: t.csv')
    cfg = pipeline_config.load_config(str(conf_path))
    assert cfg.log_level == 'DEBUG'
    assert cfg.model_dir == 'demo'
    assert cfg.threshold_file == 't.csv'


def test_load_config_data_section(tmp_path):
    conf_path = tmp_path / 'cfg.yaml'
    conf_path.write_text(
        'data:\n  output_dir: odir\n  features_filename: f.json\n  trade_log_pattern: tl.csv\n  trade_log_file: real.csv\n  raw_m1_filename: raw.csv\n'
    )
    cfg = pipeline_config.load_config(str(conf_path))
    assert cfg.output_dir == 'odir'
    assert cfg.features_filename == 'f.json'
    assert cfg.trade_log_pattern == 'tl.csv'
    assert cfg.raw_m1_filename == 'raw.csv'
    assert cfg.trade_log_file == 'real.csv'


def test_load_config_trade_log_file(tmp_path):
    conf_path = tmp_path / 'cfg.yaml'
    conf_path.write_text('data:\n  trade_log_file: log.csv\n')
    cfg = pipeline_config.load_config(str(conf_path))
    assert cfg.trade_log_file == 'log.csv'


def test_load_config_cleaning_section(tmp_path):
    conf_path = tmp_path / 'cfg.yaml'
    conf_path.write_text('cleaning:\n  fill_method: mean\n')
    cfg = pipeline_config.load_config(str(conf_path))
    assert cfg.cleaning_fill_method == 'mean'


def test_load_config_parquet_dir(tmp_path):
    conf_path = tmp_path / 'cfg.yaml'
    conf_path.write_text('data:\n  parquet_dir: cache\n')
    cfg = pipeline_config.load_config(str(conf_path))
    assert cfg.parquet_dir == 'cache'


def test_load_config_invalid_yaml(tmp_path):
    conf_path = tmp_path / 'bad.yaml'
    conf_path.write_text('log_level: [INFO')
    with pytest.raises(pipeline_config.PipelineError) as exc:
        pipeline_config.load_config(str(conf_path))
    assert 'line' in str(exc.value)