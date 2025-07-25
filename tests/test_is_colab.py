
import importlib
import os
import sys
import types
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(1, os.path.join(ROOT_DIR, 'src'))


def _import_config(monkeypatch):
    # Provide dummy seaborn to satisfy import
    monkeypatch.setitem(sys.modules, 'seaborn', types.ModuleType('seaborn'))
    monkeypatch.setitem(sys.modules, 'requests', types.ModuleType('requests'))
    monkeypatch.setitem(sys.modules, 'shap', types.ModuleType('shap'))
    if 'src.config' in sys.modules:
        monkeypatch.delitem(sys.modules, 'src.config', raising = False)
    return importlib.import_module('src.config')


def test_is_colab_false(monkeypatch):
    if 'google.colab' in sys.modules:
        monkeypatch.delitem(sys.modules, 'google.colab', raising = False)  # pragma: no cover - optional cleanup
    monkeypatch.delenv('COLAB_RELEASE_TAG', raising = False)
    monkeypatch.delenv('COLAB_GPU', raising = False)
    ip_module = types.ModuleType('IPython')
    ip_module.get_ipython = lambda: None
    monkeypatch.setitem(sys.modules, 'IPython', ip_module)
    config = _import_config(monkeypatch)
    assert config.is_colab() is False


def test_is_colab_true(monkeypatch):
    dummy = types.ModuleType('google.colab')
    dummy.drive = types.SimpleNamespace(mount = lambda *a, **k: None)
    parent = types.ModuleType('google')
    parent.colab = dummy
    monkeypatch.setitem(sys.modules, 'google', parent)
    monkeypatch.setitem(sys.modules, 'google.colab', dummy)
    monkeypatch.setenv('COLAB_RELEASE_TAG', '1')
    ip_module = types.ModuleType('IPython')
    ip_module.get_ipython = lambda: types.SimpleNamespace(kernel = object())
    monkeypatch.setitem(sys.modules, 'IPython', ip_module)
    config = _import_config(monkeypatch)
    assert config.is_colab() is True