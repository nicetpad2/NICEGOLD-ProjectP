"""
Fix for 'Unknown class label: "-1"' error in the Train step
This script demonstrates and tests the solution.
"""

import os
import pandas as pd
import sys
import warnings
from fix_target_values import fix_target_values, prepare_data_for_training
from projectp.pro_log import pro_log

def verify_fix():
    """Run verification that our fix works"""
    # Check if the file exists
    preprocessed_path = 'output_default/preprocessed.csv'
    if not os.path.exists(preprocessed_path):
        print(f"❌ Error: {preprocessed_path} not found. Please run preprocess step first.")
        return False
    
    # Load the data
    print(f"Loading {preprocessed_path}...")
    df = pd.read_csv(preprocessed_path)
    
    # Check if target column exists
    if 'target' not in df.columns:
        print(f"❌ Error: 'target' column not found in {preprocessed_path}")
        return False
    
    # Check if there are -1 values in target
    neg_one_count = (df['target'] == -1).sum() 
    if neg_one_count == 0:
        print("ℹ️ No '-1' values found in target column. No fix needed.")
        return True
    
    print(f"Found {neg_one_count} '-1' values in target column")
    
    # Apply fix
    print("Applying fix to convert -1 to 2...")
    df_fixed = fix_target_values(df)
    
    # Verify fix worked
    remaining_neg_one = (df_fixed['target'] == -1).sum()
    if remaining_neg_one > 0:
        print(f"❌ Fix failed: {remaining_neg_one} '-1' values still remain")
        return False
    
    # Save to a new file
    fixed_path = 'output_default/preprocessed_fixed.csv'
    df_fixed.to_csv(fixed_path, index=False)
    print(f"✅ Fix successful! Saved fixed data to {fixed_path}")
    
    # Show unique values before and after
    print("Original target values:", df['target'].unique())
    print("Fixed target values:", df_fixed['target'].unique())
    
    # Run a quick test with sklearn to confirm the fix works
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Filter numeric columns only
        numeric_cols = df_fixed.select_dtypes(include=['number']).columns
        feature_cols = [c for c in numeric_cols if c != 'target']
        
        X = df_fixed[feature_cols]
        y = df_fixed['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        print(f"✅ ML model training successful with fixed target values!")
        score = model.score(X_test, y_test)
        print(f"Model accuracy: {score:.4f}")
        
    except Exception as e:
        print(f"❌ Error during model testing: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("========== TARGET FIX TEST ==========")
    success = verify_fix()
    if success:
        print("\n✅ VERIFICATION PASSED: The fix for 'Unknown class label: -1' is working!")
        print("\nYou can now run the full pipeline again with:")
        print("python ProjectP.py --run_full_pipeline --maximize_ram")
    else:
        print("\n❌ VERIFICATION FAILED: The fix did not work completely")
    print("====================================")
