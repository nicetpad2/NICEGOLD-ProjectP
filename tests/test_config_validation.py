#
#
#         validate_config_yaml(str(config_path))
#         validate_config_yaml(str(config_path))
#     config_path = tmp_path / "bad_config.yaml"
#     config_path = tmp_path / "bad_config2.yaml"
#     config_path.write_text("""
#     config_path.write_text("model_class: RF\nmodel_params: {}\n")
#     with pytest.raises(ValueError):
#     with pytest.raises(ValueError):
# """)
# def test_config_invalid_types(tmp_path):
# def test_config_missing_fields(tmp_path):
# export: []
# from backtest_engine import validate_config_yaml
# import pytest
# metrics: {}
# model_class: RF
# model_params: []
# parallel: true
# visualization: true
# walk_forward: true