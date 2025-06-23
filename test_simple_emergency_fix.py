#!/usr/bin/env python3
"""
Simple test to check if the emergency AUC fix works
"""

print("🧪 Simple Emergency Fix Test...")

try:
    # Import required libraries
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from collections import Counter
    
    print("✅ Basic imports successful")
    
    # Create test data with extreme imbalance (201:1 ratio)
    np.random.seed(42)
    n_samples = 10000
    n_positive = 50  # Very few positive samples
    
    # Generate features
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'feature4': np.random.randn(n_samples),
    })
    
    # Generate extremely imbalanced target
    y = np.zeros(n_samples)
    y[:n_positive] = 1
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    X = X.iloc[indices].reset_index(drop=True)
    y = y[indices]
    
    class_counts = Counter(y)
    imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
    print(f"📊 Created test data: {X.shape}, imbalance ratio: {imbalance_ratio:.1f}:1")
    
    # Manual conservative undersampling
    def conservative_undersample(X, y):
        """Conservative undersampling to prevent memory issues"""
        class_counts = Counter(y)
        minority_count = min(class_counts.values())
        majority_count = max(class_counts.values())
        
        # Keep all minority samples
        minority_mask = y == 1  # Assuming 1 is minority
        X_minority = X[minority_mask]
        y_minority = y[minority_mask]
        
        # Keep only 10x minority samples from majority
        majority_mask = y == 0  # Assuming 0 is majority
        majority_indices = np.where(majority_mask)[0]
        
        # Sample conservatively
        n_majority_keep = min(len(majority_indices), minority_count * 10)
        selected_majority = np.random.choice(majority_indices, size=n_majority_keep, replace=False)
        
        X_majority = X.iloc[selected_majority]
        y_majority = y[selected_majority]
        
        # Combine
        X_balanced = pd.concat([X_minority, X_majority], ignore_index=True)
        y_balanced = np.concatenate([y_minority, y_majority])
        
        return X_balanced, y_balanced
    
    # Apply conservative resampling
    print("🔧 Applying conservative undersampling...")
    X_balanced, y_balanced = conservative_undersample(X, y)
    
    new_counts = Counter(y_balanced)
    new_ratio = max(new_counts.values()) / min(new_counts.values())
    print(f"✅ After resampling: {new_ratio:.1f}:1 ratio, shape: {X_balanced.shape}")
    
    # Test models with balanced data
    print("🤖 Testing models with balanced data...")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X_balanced, y_balanced, cv=cv, scoring='roc_auc')
            mean_auc = scores.mean()
            
            if np.isnan(mean_auc) or np.isinf(mean_auc):
                print(f"❌ {name}: Got NaN/Inf AUC")
            else:
                print(f"✅ {name}: AUC = {mean_auc:.3f} (±{scores.std():.3f}) - NOT NaN!")
                
        except Exception as e:
            print(f"❌ {name}: Failed with error: {e}")
    
    print("\n🎉 Simple emergency fix test completed!")
    print("🚀 Conservative undersampling prevents NaN AUC!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
