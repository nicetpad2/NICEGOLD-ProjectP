#!/usr/bin/env python3
"""
🔧 NICEGOLD ProjectP - Entry Point Notice Generator

This script adds deprecation notices to old entry point files.
"""

import os
import sys
from pathlib import Path

# Template for deprecation notice
DEPRECATION_NOTICE = '''#!/usr/bin/env python3
"""
🚨 DEPRECATED ENTRY POINT

This file has been superseded by ProjectP.py
All functionality is now integrated into the main entry point.
"""

print("🚨 DEPRECATED: This entry point is no longer used!")
print()
print("📢 NICEGOLD ProjectP v2.1 uses: ProjectP.py")
print("=" * 50)
print("✅ Please run: python ProjectP.py")
print(f"❌ Instead of: python {__file__}")
print()
print("ProjectP.py includes ALL features with:")
print("  🚀 Better integration")
print("  🎨 Enhanced interface") 
print("  ⚡ Improved performance")
print("  🛡️ Error handling")
print()
print("💡 All your favorite features are in ProjectP.py!")
print("=" * 50)

import sys
sys.exit(1)

# LEGACY CODE BELOW - DEPRECATED
'''

def add_notice_to_file(filepath):
    """Add deprecation notice to a file"""
    try:
        # Read existing content
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Skip if already has notice
        if "DEPRECATED ENTRY POINT" in original_content:
            print(f"⏭️  Already updated: {filepath}")
            return
        
        # Create backup
        backup_path = f"{filepath}.backup"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Add notice to original file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(DEPRECATION_NOTICE)
            f.write("\n\n# === ORIGINAL CODE BELOW (DEPRECATED) ===\n")
            f.write("'''\n")
            f.write(original_content)
            f.write("\n'''\n")
        
        print(f"✅ Updated: {filepath}")
        
    except Exception as e:
        print(f"❌ Error updating {filepath}: {e}")

def main():
    print("🔧 Adding deprecation notices to old entry points...")
    print()
    
    # Files to update
    deprecated_files = [
        "run_ai_agents.py",
        "run_all_modes.py", 
        "run_complete_production.py",
        "run_full_pipeline.py",
        "run_production_pipeline.py",
        "run_simple_pipeline.py",
        "run_ultimate_pipeline.py",
        "main.py"  # If we want to fix the syntax errors
    ]
    
    updated = 0
    for filename in deprecated_files:
        if os.path.exists(filename):
            add_notice_to_file(filename)
            updated += 1
        else:
            print(f"⚠️  Not found: {filename}")
    
    print()
    print(f"✅ Updated {updated} files")
    print("📢 All entry points now redirect to ProjectP.py")
    print()
    print("🎯 Users should now use: python ProjectP.py")

if __name__ == "__main__":
    main()
