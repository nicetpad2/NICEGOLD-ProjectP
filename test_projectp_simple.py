"""
Simple ProjectP Test with Real Data
"""

print("🧪 Testing ProjectP with Real Data...")

try:
    # Import real data patch
    exec(open("real_data_patch.py").read())
    
    # Test data loading
    from pipeline_data_loader import load_pipeline_data
    features_df, targets_df, stats = load_pipeline_data()
    
    print(f"✅ Loaded data: {features_df.shape} features, {targets_df.shape} targets")
    print(f"📊 Stats: {stats.get('data_source', 'Unknown')}")
    
    print("🎯 ProjectP ready with real data!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
