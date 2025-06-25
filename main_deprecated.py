#!/usr/bin/env python3
"""
🚨 NOTICE: DEPRECATED ENTRY POINT

This script redirects to the official ProjectP.py entry point.
"""

import os
import subprocess
import sys
import time

print("🚨 NOTICE: You are trying to run a DEPRECATED entry point!")
print()
print("📢 NICEGOLD ProjectP v2.1 Official Entry Point: ProjectP.py")
print("=" * 60)
print("✅ Please run: python ProjectP.py")
print("❌ Do NOT run this file directly")
print()
print("ProjectP.py provides:")
print("  🚀 Complete feature integration")
print("  🎨 Enhanced user interface")
print("  ⚡ Optimized performance")
print("  🛡️ Better error handling")
print("  📊 Advanced progress tracking")
print("  ⚠️ Risk management system")
print("  📱 Interactive dashboard")
print()
print("🎯 Redirecting to ProjectP.py in 3 seconds...")
print("   (Press Ctrl+C to cancel)")

try:
    time.sleep(3)
    
    # Check if ProjectP.py exists
    if os.path.exists("ProjectP.py"):
        print("🔄 Starting ProjectP.py...")
        # Run ProjectP.py
        subprocess.run([sys.executable, "ProjectP.py"])
    else:
        print("❌ ProjectP.py not found in current directory!")
        print("📂 Please ensure you're in the correct project directory")
        sys.exit(1)
        
except KeyboardInterrupt:
    print("\n⏹️ Redirect cancelled by user")
    print("💡 Remember to use: python ProjectP.py")
    sys.exit(0)
except Exception as e:
    print(f"❌ Error: {e}")
    print("💡 Please run manually: python ProjectP.py")
    sys.exit(1)
