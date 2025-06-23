@echo off
echo Running test_strategy_imports_shap_helpers test...
python -m pytest tests/test_strategy_import_safe_load.py::test_strategy_imports_shap_helpers -v
echo Test complete.
pause
