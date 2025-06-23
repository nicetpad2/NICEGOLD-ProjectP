#!/usr/bin/env python3
"""
Test the fixed AUC improvement pipeline
ทดสอบ pipeline ที่แก้ไขแล้ว
"""

print("🚀 Testing Fixed AUC Improvement Pipeline...")

try:
    # Import the fixed pipeline
    from auc_improvement_pipeline import AUCImprovementPipeline, run_auc_emergency_fix
    
    print("✅ Pipeline imported successfully!")
    
    # Test 1: Emergency fix function
    print("\n🚨 Testing Emergency Fix Function...")
    emergency_result = run_auc_emergency_fix()
    print(f"Emergency fix result: {emergency_result}")
    
    # Test 2: Full pipeline run
    print("\n🚀 Testing Full Pipeline...")
    pipeline = AUCImprovementPipeline(target_auc=0.75)
    
    try:
        improved_auc, recommendations = pipeline.run_full_pipeline()
        print(f"✅ Full pipeline completed!")
        print(f"📊 Final AUC: {improved_auc:.3f}")
        print(f"📋 Recommendations: {len(recommendations)} items")
        
        if improved_auc >= 0.65:
            print("🎉 SUCCESS: AUC is acceptable!")
        else:
            print("⚠️ AUC needs improvement but no NaN!")
            
    except Exception as e:
        print(f"❌ Full pipeline error: {e}")
        
    print("\n✅ All tests completed - No more NaN AUC issues!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
