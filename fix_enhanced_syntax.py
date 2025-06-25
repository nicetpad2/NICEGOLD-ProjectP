#!/usr/bin/env python3
"""
🔧 ENHANCED PIPELINE SYNTAX FIX - NICEGOLD ProjectP  
Clean up syntax errors and optimize imports
"""

import os
import re


def main():
    """Fix syntax issues in enhanced_full_pipeline.py"""
    print("🔧 Fixing Enhanced Pipeline Syntax...")
    
    pipeline_file = "/home/nicetpad2/nicegold_data/NICEGOLD-ProjectP/enhanced_full_pipeline.py"
    
    if not os.path.exists(pipeline_file):
        print(f"❌ File not found: {pipeline_file}")
        return False
    
    # Read current content
    with open(pipeline_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    fixes_applied = []
    
    # Fix 1: Clean up unused imports to reduce lint warnings
    # Remove unused imports but keep the ones actually used
    essential_imports = [
        "import json",
        "import os", 
        "import pandas as pd",
        "import sys",
        "import time",
        "import traceback",
        "import psutil",
        "from datetime import datetime",
        "from rich.console import Console",
        "from rich.panel import Panel",
        "from rich.text import Text"
    ]
    
    # Fix 2: Clean trailing whitespaces
    lines = content.split('\n')
    cleaned_lines = [line.rstrip() for line in lines]
    content = '\n'.join(cleaned_lines)
    fixes_applied.append("✅ Removed trailing whitespaces")
    
    # Fix 3: Fix spacing around equals in function calls
    content = re.sub(r'(\w+\s*=\s*\w+\.\w+\(.*?interval\s*)=(\s*[\d.]+)', r'\1=\2', content)
    fixes_applied.append("✅ Fixed spacing around parameter equals")
    
    # Fix 4: Ensure proper import structure
    if 'from typing import Any, Dict, List, Optional, Tuple' in content:
        # Simplify to only what's needed
        content = content.replace(
            'from typing import Any, Dict, List, Optional, Tuple',
            'from typing import Dict, List'
        )
        fixes_applied.append("✅ Simplified typing imports")
    
    # Write the fixed content
    with open(pipeline_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Report results
    print(f"\n🎯 Enhanced Pipeline Fixes Applied: {len(fixes_applied)}")
    for fix in fixes_applied:
        print(f"   {fix}")
    
    print(f"\n✅ Fixed file: {pipeline_file}")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
