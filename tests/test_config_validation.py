# import pytest
# from backtest_engine import validate_config_yaml
#
# def test_config_missing_fields(tmp_path):
#     config_path = tmp_path / "bad_config.yaml"
#     config_path.write_text("model_class: RF\nmodel_params: {}\n")
#     with pytest.raises(ValueError):
#         validate_config_yaml(str(config_path))
#
# def test_config_invalid_types(tmp_path):
#     config_path = tmp_path / "bad_config2.yaml"
#     config_path.write_text("""
# model_class: RF
# model_params: []
# walk_forward: true
# metrics: {}
# export: []
# parallel: true
# visualization: true
# """)
#     with pytest.raises(ValueError):
#         validate_config_yaml(str(config_path))
