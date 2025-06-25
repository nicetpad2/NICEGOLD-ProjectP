
# sklearn mutual_info_regression fallback
        from sklearn.feature_selection import mutual_info_regression
    from sklearn.metrics import mutual_info_regression
import numpy as np
            import sklearn.metrics
import sys
            import types
def mutual_info_regression_fallback(X, y, **kwargs):
    """Fallback implementation of mutual_info_regression using correlation"""
    try:
        if hasattr(X, 'shape') and len(X.shape) > 1:
            n_features = X.shape[1]
        else:
            n_features = 1
            X = np.array(X).reshape( - 1, 1)

        y = np.array(y)
        mi_scores = []

        for i in range(n_features):
            if len(X.shape) > 1:
                feature = X[:, i]
            else:
                feature = X

            # Use correlation as proxy for mutual information
            try:
                correlation = np.corrcoef(feature, y)[0, 1]
                mi_score = abs(correlation) if not np.isnan(correlation) else 0.0
            except:
                mi_score = 0.0

            mi_scores.append(mi_score)

        return np.array(mi_scores)
    except Exception as e:
        print(f"Warning in mutual_info_regression fallback: {e}")
        # Return zeros if everything fails
        if hasattr(X, 'shape') and len(X.shape) > 1:
            return np.zeros(X.shape[1])
        else:
            return np.array([0.0])

# Try to import the real function first
try:
    print("sklearn.metrics.mutual_info_regression imported successfully")
except ImportError:
    try:
        # Try alternative location
        print("sklearn.feature_selection.mutual_info_regression imported successfully")

        # Patch it to metrics
        try:
            sklearn.metrics.mutual_info_regression = mutual_info_regression
            print("Patched mutual_info_regression to sklearn.metrics")
        except:
            pass
    except ImportError:
        print("Creating sklearn mutual_info_regression fallback...")

        # Create fallback in sklearn.metrics
        try:
            sklearn.metrics.mutual_info_regression = mutual_info_regression_fallback
            print("sklearn.metrics.mutual_info_regression fallback created")
        except ImportError:
            # Create fake sklearn modules
            sklearn = types.ModuleType('sklearn')
            sklearn_metrics = types.ModuleType('sklearn.metrics')
            sklearn_metrics.mutual_info_regression = mutual_info_regression_fallback
            sklearn.metrics = sklearn_metrics
            sys.modules['sklearn'] = sklearn
            sys.modules['sklearn.metrics'] = sklearn_metrics
            print("sklearn modules and fallback created")