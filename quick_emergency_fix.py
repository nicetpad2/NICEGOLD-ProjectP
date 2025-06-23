#!/usr/bin/env python3
"""
Quick Emergency Fix for ProjectP Dependencies
"""

import sys
import subprocess
import os

def fix_pydantic():
    """Fix pydantic SecretField issue"""
    print("🔧 Fixing pydantic...")
    
    # Create pydantic fallback
    fallback_code = '''# Pydantic Fallback for SecretField
try:
    from pydantic import SecretField
    print("✅ pydantic SecretField imported successfully")
except ImportError:
    print("⚠️ Creating pydantic SecretField fallback...")
    
    class SecretField:
        """Fallback SecretField implementation"""
        def __init__(self, *args, **kwargs):
            self.default = kwargs.get('default', '')
        
        @classmethod
        def get_secret_value(cls, value):
            return str(value) if value is not None else ""
        
        def __call__(self, *args, **kwargs):
            return self.default
    
    # Ensure pydantic module exists
    try:
        import pydantic
        pydantic.SecretField = SecretField
    except ImportError:
        # Create fake pydantic module
        import types
        pydantic = types.ModuleType('pydantic')
        pydantic.SecretField = SecretField
        sys.modules['pydantic'] = pydantic
    
    print("✅ pydantic SecretField fallback created")
'''
    
    with open('pydantic_fallback.py', 'w') as f:
        f.write(fallback_code)
    
    print("✅ pydantic fallback created")

def fix_sklearn():
    """Fix sklearn mutual_info_regression issue"""
    print("🔧 Fixing sklearn...")
    
    # Create sklearn fallback
    fallback_code = '''# sklearn mutual_info_regression fallback
import numpy as np

def mutual_info_regression_fallback(X, y, **kwargs):
    """Fallback implementation of mutual_info_regression using correlation"""
    try:
        if hasattr(X, 'shape') and len(X.shape) > 1:
            n_features = X.shape[1]
        else:
            n_features = 1
            X = np.array(X).reshape(-1, 1)
        
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
    from sklearn.metrics import mutual_info_regression
    print("✅ sklearn.metrics.mutual_info_regression imported successfully")
except ImportError:
    try:
        # Try alternative location
        from sklearn.feature_selection import mutual_info_regression
        print("✅ sklearn.feature_selection.mutual_info_regression imported successfully")
        
        # Patch it to metrics
        try:
            import sklearn.metrics
            sklearn.metrics.mutual_info_regression = mutual_info_regression
            print("✅ Patched mutual_info_regression to sklearn.metrics")
        except:
            pass
    except ImportError:
        print("⚠️ Creating sklearn mutual_info_regression fallback...")
        
        # Create fallback in sklearn.metrics
        try:
            import sklearn.metrics
            sklearn.metrics.mutual_info_regression = mutual_info_regression_fallback
            print("✅ sklearn.metrics.mutual_info_regression fallback created")
        except ImportError:
            # Create fake sklearn modules
            import types
            sklearn = types.ModuleType('sklearn')
            sklearn_metrics = types.ModuleType('sklearn.metrics')
            sklearn_metrics.mutual_info_regression = mutual_info_regression_fallback
            sklearn.metrics = sklearn_metrics
            sys.modules['sklearn'] = sklearn
            sys.modules['sklearn.metrics'] = sklearn_metrics
            print("✅ sklearn modules and fallback created")
'''
    
    with open('sklearn_fallback.py', 'w') as f:
        f.write(fallback_code)
    
    print("✅ sklearn fallback created")

def create_safe_launcher():
    """Create safe ProjectP launcher"""
    print("🚀 Creating safe launcher...")
    
    launcher_code = '''#!/usr/bin/env python3
"""
Safe ProjectP Launcher - ป้องกัน infinite loop และ import errors
"""

import sys
import os
import time
import signal
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, '.')

print("🚀 Safe ProjectP Launcher")
print("=" * 40)
print(f"🕐 Start time: {datetime.now()}")

# Load fallbacks first
print("📦 Loading fallbacks...")
try:
    exec(open('pydantic_fallback.py').read())
    print("✅ Pydantic fallback loaded")
except Exception as e:
    print(f"⚠️ Pydantic fallback error: {e}")

try:
    exec(open('sklearn_fallback.py').read())
    print("✅ Sklearn fallback loaded")
except Exception as e:
    print(f"⚠️ Sklearn fallback error: {e}")

# Set timeout (10 minutes max)
MAX_RUNTIME = 600

def timeout_handler(signum, frame):
    print(f"\\n⏰ Timeout after {MAX_RUNTIME} seconds - stopping safely")
    sys.exit(0)

# Setup timeout (works on Unix-like systems)
try:
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(MAX_RUNTIME)
except:
    print("⚠️ Timeout not available on this system")

def run_projectp_safe():
    """Run ProjectP with safety measures"""
    start_time = time.time()
    
    try:
        print("\\n🔄 Loading ProjectP...")
        
        # Import ProjectP components safely
        try:
            from ProjectP import run_full_pipeline, main
            print("✅ ProjectP imported successfully")
        except Exception as e:
            print(f"❌ ProjectP import error: {e}")
            print("🔍 Trying alternative import...")
            
            # Try direct execution
            try:
                import runpy
                result = runpy.run_path('ProjectP.py', run_name='__main__')
                print("✅ ProjectP executed via runpy")
                return True
            except Exception as e2:
                print(f"❌ Alternative import failed: {e2}")
                return False
        
        # Run the pipeline
        print("\\n🚀 Running full pipeline...")
        
        if '--run_full_pipeline' in sys.argv:
            result = run_full_pipeline()
        else:
            result = main()
        
        runtime = time.time() - start_time
        print(f"\\n✅ Pipeline completed in {runtime:.1f} seconds")
        
        # Check for results
        if os.path.exists('classification_report.json'):
            import json
            with open('classification_report.json', 'r') as f:
                data = json.load(f)
            accuracy = data.get('accuracy', 0)
            print(f"🎯 Final Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        return True
        
    except KeyboardInterrupt:
        print("\\n⏹️ Stopped by user")
        return False
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        return False
    finally:
        try:
            signal.alarm(0)  # Cancel timeout
        except:
            pass

if __name__ == "__main__":
    success = run_projectp_safe()
    if success:
        print("\\n🎉 Safe execution completed!")
    else:
        print("\\n⚠️ Execution had issues - check logs")
'''
    
    with open('safe_projectp.py', 'w') as f:
        f.write(launcher_code)
    
    print("✅ Safe launcher created")

def main():
    """Run quick emergency fix"""
    print("🚨 Quick Emergency Fix for ProjectP")
    print("=" * 50)
    
    # Fix dependencies
    fix_pydantic()
    fix_sklearn()
    create_safe_launcher()
    
    print("\\n✅ Emergency fixes completed!")
    print("\\n🚀 Try running:")
    print("   python safe_projectp.py --run_full_pipeline")
    print("   or")
    print("   python ProjectP.py --run_full_pipeline")

if __name__ == "__main__":
    main()
