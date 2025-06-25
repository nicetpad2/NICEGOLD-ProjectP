#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Test Production Pipeline Progress Bar
ทดสอบ progress bar ใน production pipeline
"""

print("🧪 Testing Production Pipeline Progress Bar...")
print("="*60)

try:
    from production_full_pipeline import ProductionFullPipeline
    print("✅ ProductionFullPipeline imported successfully")
    
    # Test with minimal configuration
    pipeline = ProductionFullPipeline(
        min_auc_requirement=0.60,  # Lower for testing
        capital=100.0
    )
    print("✅ ProductionFullPipeline initialized")
    
    print("\n🚀 Running pipeline with progress bar...")
    results = pipeline.run_full_pipeline()
    
    if results.get("success"):
        print(f"✅ Pipeline completed successfully!")
        print(f"📊 AUC: {results.get('auc', 'N/A')}")
        print(f"🤖 Model: {results.get('model_name', 'N/A')}")
    else:
        print(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure production_full_pipeline.py is available")
    
except Exception as e:
    print(f"❌ Runtime error: {e}")
    import traceback
    traceback.print_exc()

print("\n🏁 Test completed!")
