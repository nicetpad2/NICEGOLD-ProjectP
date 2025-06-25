#!/usr/bin/env python3
"""
TensorFlow Installation and Testing Script for NICEGOLD ProjectP
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

This script helps resolve TensorFlow import issues and provides installation options.
"""

import os
import subprocess
import sys


def check_tensorflow():
    """Check if TensorFlow is installed and working"""
    print("🔍 Checking TensorFlow installation...")

    try:
        # Suppress TensorFlow warnings
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf

        print(f"✅ TensorFlow is installed: {tf.__version__}")

        # Check GPU availability
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            print(f"🚀 GPU devices found: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")

            # Try to set memory growth
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ GPU memory growth configured")
            except Exception as e:
                print(f"⚠️ Could not configure GPU memory growth: {e}")
        else:
            print("📱 No GPU devices found - using CPU")

        # Test basic TensorFlow operation
        print("🧪 Testing TensorFlow basic operations...")
        x = tf.constant([1, 2, 3, 4])
        y = tf.constant([2, 3, 4, 5])
        result = tf.add(x, y)
        print(f"✅ Basic operation test passed: {result.numpy()}")

        return True

    except ImportError:
        print("❌ TensorFlow is not installed")
        return False
    except Exception as e:
        print(f"❌ TensorFlow error: {e}")
        return False


def install_tensorflow():
    """Install TensorFlow with appropriate options"""
    print("\n📦 TensorFlow Installation Options:")
    print("1. TensorFlow (standard) - includes GPU support if available")
    print("2. TensorFlow - CPU (CPU only) - smaller, CPU - optimized")
    print("3. Skip installation")

    choice = input("\nSelect option (1 - 3): ").strip()

    if choice == "1":
        print("📦 Installing TensorFlow (standard)...")
        return run_pip_install("tensorflow")
    elif choice == "2":
        print("📦 Installing TensorFlow - CPU...")
        return run_pip_install("tensorflow - cpu")
    else:
        print("⏭️ Skipping TensorFlow installation")
        return False


def run_pip_install(package):
    """Run pip install for a package"""
    try:
        subprocess.check_call([sys.executable, " - m", "pip", "install", package])
        print(f"✅ {package} installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False


def main():
    """Main function"""
    print("🚀 NICEGOLD ProjectP - TensorFlow Setup Helper")
    print(" = " * 50)

    # Check current status
    tf_available = check_tensorflow()

    if not tf_available:
        print(
            "\n💡 TensorFlow is not installed. This is optional for NICEGOLD ProjectP."
        )
        print(
            "   The system will work fine without TensorFlow using other ML libraries."
        )

        install_choice = (
            input("\nWould you like to install TensorFlow? (y/N): ").strip().lower()
        )
        if install_choice in ["y", "yes"]:
            success = install_tensorflow()
            if success:
                print("\n🔄 Testing installation...")
                check_tensorflow()
        else:
            print("\n✅ No problem! NICEGOLD ProjectP works great without TensorFlow.")
            print(
                "   The system uses scikit - learn, XGBoost, CatBoost, and LightGBM for ML."
            )
    else:
        print("\n✅ TensorFlow is working correctly!")

    print("\n" + " = " * 50)
    print("🎯 Summary:")
    print("• TensorFlow is OPTIONAL for NICEGOLD ProjectP")
    print("• The system works perfectly with sklearn, XGBoost, CatBoost, LightGBM")
    print("• TensorFlow adds deep learning capabilities (neural networks)")
    print("• You can install it later using: pip install tensorflow")
    print("\n💡 To suppress TensorFlow warnings in the future:")
    print("   export TF_CPP_MIN_LOG_LEVEL = 3")


if __name__ == "__main__":
    main()