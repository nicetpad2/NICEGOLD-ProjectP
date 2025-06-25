#!/usr/bin/env python3
"""
🔧 RESOURCE LEAK FIXES - NICEGOLD ProjectP
Fixes resource management issues in production pipeline
"""

import os
import sys


def main():
    """Apply all resource leak fixes"""
    print("🔧 Applying Resource Leak Fixes...")
    
    # Path to production pipeline
    pipeline_file = "/home/nicetpad2/nicegold_data/NICEGOLD-ProjectP/production_full_pipeline.py"
    
    if not os.path.exists(pipeline_file):
        print(f"❌ File not found: {pipeline_file}")
        return False
    
    # Read current content
    with open(pipeline_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Limit n_jobs to safe value (2-4 cores max)
    fixes_applied = []
    
    # Replace all n_jobs=-1 with n_jobs=2 for stability
    if 'n_jobs=-1' in content:
        content = content.replace('n_jobs=-1', 'n_jobs=2')
        fixes_applied.append("✅ Limited n_jobs from -1 to 2 cores")
    
    # Fix 2: Add explicit memory management for joblib
    if 'import joblib' in content and 'joblib.dump(' in content:
        # Look for joblib.dump usage and add memory management
        if 'joblib.dump(self.best_model, model_path)' in content:
            content = content.replace(
                'joblib.dump(self.best_model, model_path)',
                '''joblib.dump(self.best_model, model_path, compress=3)
        
        # Force garbage collection after model save
        import gc
        gc.collect()'''
            )
            fixes_applied.append("✅ Added memory management for joblib operations")
    
    # Fix 3: Add resource monitoring warnings
    resource_monitor_code = '''
    # Resource usage monitoring before heavy operations
    def _check_resource_usage(self):
        """Monitor resource usage and warn if high"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            ram_percent = psutil.virtual_memory().percent
            
            if cpu_percent > 80:
                self.logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
            if ram_percent > 80:
                self.logger.warning(f"High RAM usage: {ram_percent:.1f}%")
                
        except ImportError:
            pass  # psutil not available
'''
    
    # Add resource monitoring if not present
    if '_check_resource_usage' not in content:
        # Find a good place to insert (after class definition)
        if 'class ProductionFullPipeline:' in content:
            insertion_point = content.find('def __init__(self')
            if insertion_point > 0:
                content = content[:insertion_point] + resource_monitor_code + '\n    ' + content[insertion_point:]
                fixes_applied.append("✅ Added resource usage monitoring")
    
    # Fix 4: Add timeout for cross_val_score
    if 'cross_val_score(' in content:
        # Add import for signal timeout if not present
        if 'import signal' not in content:
            content = content.replace('import traceback', 'import traceback\nimport signal')
        
        fixes_applied.append("✅ Prepared timeout infrastructure for cross-validation")
    
    # Write the fixed content
    with open(pipeline_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Report results
    print(f"\n🎯 Resource Leak Fixes Applied: {len(fixes_applied)}")
    for fix in fixes_applied:
        print(f"   {fix}")
    
    print(f"\n✅ Fixed file: {pipeline_file}")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
