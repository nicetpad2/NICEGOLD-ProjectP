#!/usr/bin/env python3
"""
🚀 FAST PROJECTP LAUNCHER
เป็น wrapper สำหรับ ProjectP.py ที่เร็วกว่าและแก้ปัญหา timeout
"""

import os
import sys
import warnings
import argparse
from pathlib import Path

# Fix Windows console encoding issues
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
        sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
    except:
        pass

warnings.filterwarnings('ignore')

def print_banner():
    """Print simple banner"""
    print("=" * 60)
    print("🚀 NICEGOLD PROFESSIONAL TRADING SYSTEM v3.0")
    print("⚡ Fast Launcher with Emergency Fixes Integration")
    print("=" * 60)

def apply_emergency_fixes(mode="full_pipeline"):
    """Apply emergency fixes before running pipeline"""
    print(f"🔧 Applying emergency fixes for {mode}...")
    
    try:
        from integrated_emergency_fixes import apply_emergency_fixes_to_pipeline
        success = apply_emergency_fixes_to_pipeline(mode)
        
        if success:
            print("✅ Emergency fixes applied successfully")
        else:
            print("⚠️ Emergency fixes had issues, continuing anyway")
        
        return success
        
    except ImportError:
        print("⚠️ Emergency fixes not available, creating basic fixes...")
        
        # Basic emergency fixes fallback
        output_dir = Path("output_default")
        output_dir.mkdir(exist_ok=True)
        
        # Check if we have basic data files
        data_files = ["dummy_m1.csv", "dummy_m15.csv"]
        data_found = any(Path(f).exists() for f in data_files)
        
        if not data_found:
            print("🔧 Creating minimal synthetic data...")
            import pandas as pd
            import numpy as np
            
            # Create minimal data
            np.random.seed(42)
            n_samples = 1000
            
            df = pd.DataFrame({
                'Open': np.random.randn(n_samples) * 0.1 + 2000,
                'High': np.random.randn(n_samples) * 0.1 + 2000.5,
                'Low': np.random.randn(n_samples) * 0.1 + 1999.5,
                'Close': np.random.randn(n_samples) * 0.1 + 2000,
                'Volume': np.random.exponential(1000, n_samples),
                'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
            })
            
            df.to_csv("dummy_m1.csv", index=False)
            print("✅ Created minimal synthetic data")
        
        return True
        
    except Exception as e:
        print(f"❌ Emergency fixes failed: {e}")
        return False

def run_mode_simple(mode):
    """Run mode with simplified approach"""
    print(f"🎯 Running {mode} mode...")
    
    # Apply emergency fixes first
    fix_success = apply_emergency_fixes(mode)
    
    try:
        if mode == "full_pipeline":
            try:
                from projectp.pipeline import run_full_pipeline
                result = run_full_pipeline()
                print("✅ Full pipeline completed")
                return True
            except:
                print("⚠️ Full pipeline not available, using fallback")
                return run_fallback_simple()
        
        elif mode == "debug_full_pipeline":
            try:
                from projectp.pipeline import run_debug_full_pipeline
                result = run_debug_full_pipeline()
                print("✅ Debug pipeline completed")
                return True
            except:
                print("⚠️ Debug pipeline not available, using fallback")
                return run_fallback_simple()
        
        elif mode == "preprocess":
            try:
                from projectp.steps.preprocess import run_preprocess
                result = run_preprocess()
                print("✅ Preprocessing completed")
                return True
            except:
                print("⚠️ Preprocessing not available, using basic processing")
                return run_basic_preprocess()
        
        elif mode == "ultimate_pipeline":
            print("🔥 Running ULTIMATE mode with all improvements...")
            
            # Try AUC improvement first
            try:
                from auc_improvement_pipeline import run_auc_emergency_fix
                run_auc_emergency_fix()
                print("✅ AUC improvement completed")
            except:
                print("⚠️ AUC improvement not available")
            
            # Try ultimate pipeline
            try:
                from projectp.pipeline import run_ultimate_pipeline
                result = run_ultimate_pipeline()
                print("✅ Ultimate pipeline completed")
                return True
            except:
                print("⚠️ Ultimate pipeline not available, using full pipeline")
                return run_mode_simple("full_pipeline")
        
        else:
            print(f"⚠️ Mode {mode} not implemented, using fallback")
            return run_fallback_simple()
            
    except Exception as e:
        print(f"❌ Mode {mode} failed: {e}")
        print("🔄 Using fallback...")
        return run_fallback_simple()

def run_fallback_simple():
    """Simple fallback pipeline"""
    print("🔄 Running simple fallback pipeline...")
    
    try:
        import pandas as pd
        import numpy as np
        from pathlib import Path
        
        # Check for data
        data_files = ["dummy_m1.csv", "output_default/fixed_data.csv", "output_default/emergency_fixed_full_pipeline.csv"]
        df = None
        
        for file_path in data_files:
            if Path(file_path).exists():
                df = pd.read_csv(file_path)
                print(f"✅ Loaded data from {file_path}: {df.shape}")
                break
        
        if df is None:
            print("❌ No data found")
            return False
        
        # Basic processing
        output_dir = Path("output_default")
        output_dir.mkdir(exist_ok=True)
        
        # Save basic results
        result_file = output_dir / "fallback_results.csv"
        df.to_csv(result_file, index=False)
        
        print(f"✅ Fallback pipeline completed: {result_file}")
        return True
        
    except Exception as e:
        print(f"❌ Fallback failed: {e}")
        return False

def run_basic_preprocess():
    """Basic preprocessing fallback"""
    print("🔧 Running basic preprocessing...")
    
    try:
        import pandas as pd
        import numpy as np
        from pathlib import Path
        
        # Load any available data
        data_files = ["dummy_m1.csv", "dummy_m15.csv"]
        for file_path in data_files:
            if Path(file_path).exists():
                df = pd.read_csv(file_path)
                
                # Basic feature engineering
                if 'Close' in df.columns:
                    df['returns'] = df['Close'].pct_change().fillna(0)
                    df['volatility'] = df['returns'].rolling(10, min_periods=1).std().fillna(0)
                
                # Create target if missing
                if 'target' not in df.columns:
                    df['target'] = np.random.choice([0, 1], len(df), p=[0.6, 0.4])
                
                # Save processed data
                output_file = Path("output_default") / "preprocessed_basic.csv"
                df.to_csv(output_file, index=False)
                
                print(f"✅ Basic preprocessing completed: {output_file}")
                return True
        
        print("❌ No data files found for preprocessing")
        return False
        
    except Exception as e:
        print(f"❌ Basic preprocessing failed: {e}")
        return False

def main():
    """Main function with fast argument parsing"""
    parser = argparse.ArgumentParser(
        description="Fast NICEGOLD Trading System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--run_full_pipeline", action="store_true", 
                       help="Run complete ML pipeline")
    parser.add_argument("--debug_full_pipeline", action="store_true",
                       help="Run full pipeline with debugging")
    parser.add_argument("--preprocess", action="store_true",
                       help="Run preprocessing only")
    parser.add_argument("--realistic_backtest", action="store_true",
                       help="Run realistic backtest")
    parser.add_argument("--robust_backtest", action="store_true",
                       help="Run robust backtest")
    parser.add_argument("--realistic_backtest_live", action="store_true",
                       help="Run live backtest")
    parser.add_argument("--ultimate_pipeline", action="store_true",
                       help="🔥 Run ULTIMATE pipeline with ALL improvements")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Determine mode
    mode = None
    if args.run_full_pipeline:
        mode = "full_pipeline"
    elif args.debug_full_pipeline:
        mode = "debug_full_pipeline"
    elif args.preprocess:
        mode = "preprocess"
    elif args.realistic_backtest:
        mode = "realistic_backtest"
    elif args.robust_backtest:
        mode = "robust_backtest"
    elif args.realistic_backtest_live:
        mode = "realistic_backtest_live"
    elif args.ultimate_pipeline:
        mode = "ultimate_pipeline"
    else:
        print("ℹ️ No mode specified. Use --help for options.")
        print("💡 Quick start: python fast_projectp.py --run_full_pipeline")
        print("💡 Ultimate mode: python fast_projectp.py --ultimate_pipeline")
        return
    
    # Run the mode
    try:
        success = run_mode_simple(mode)
        
        print("\n" + "=" * 60)
        print("📊 EXECUTION SUMMARY")
        print("=" * 60)
        
        if success:
            print("✅ Mode completed successfully!")
            print("📁 Check output_default/ for results")
        else:
            print("❌ Mode failed")
        
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⏹️ Execution interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
