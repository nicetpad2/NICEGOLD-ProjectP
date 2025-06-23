import importlib
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)


# Disabled: This test expected DriftObserver in src.main, which does not exist.
# Remove or update this file if you implement the required class.
def test_drift_observer_is_imported():
    if 'src.main' in sys.modules:
        del sys.modules['src.main']
    main = importlib.import_module('src.main')
    assert hasattr(main, 'DriftObserver')
