from auc_improvement_pipeline import AUCImprovementPipeline
import numpy as np
import pandas as pd
import sys
"""
Quick test for AUC Emergency Fix
"""

sys.path.append('.')


def test_emergency_fix():
    """Test the emergency AUC fix with synthetic imbalanced data"""
    print("🧪 Testing Emergency AUC Fix...")

    # Create severely imbalanced synthetic data (like what's causing the issue)
    np.random.seed(42)
    n_samples = 10000
    n_features = 20

    # Create features
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns = [f'feature_{i}' for i in range(n_features)])

    # Create severe class imbalance (200:1 ratio like in the error)
    y = np.zeros(n_samples)
    y[:50] = 1  # Only 50 positive samples out of 10000

    print(f"Created test data: {X.shape}, Imbalance ratio: {(y =  = 0).sum()}/{(y =  = 1).sum()}")

    # Test the pipeline
    pipeline = AUCImprovementPipeline()

    # Test the diagnosis function
    try:
        problems, baseline_aucs = pipeline.diagnose_auc_problems(X, y)
        print(f"✅ Diagnosis completed: {len(problems)} problems found")
        print(f"Baseline AUCs: {baseline_aucs}")

        if any(isinstance(auc, float) and not np.isnan(auc) and auc > 0.5 for auc in baseline_aucs.values()):
            print("🎉 SUCCESS: Emergency fix working - got valid AUC scores!")
            return True
        else:
            print("⚠️ WARNING: Still getting low/invalid AUC scores")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_emergency_fix()
    if success:
        print("\n🚀 Emergency fix validated - pipeline should work now!")
    else:
        print("\n🔧 Need additional fixes...")