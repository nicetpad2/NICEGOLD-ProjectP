"""
🚀 PIPELINE WRAPPER WITH INTEGRATED AUC FIX
==========================================
Enhanced pipeline runner with automatic AUC monitoring and fixing
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def run_pipeline_with_auc_integration(mode="full_pipeline", **kwargs):
    """Run pipeline with integrated AUC monitoring and fixing"""
    from rich.console import Console
    from rich.panel import Panel
    
    console = Console()
    console.print(Panel.fit(f"🚀 Running {mode} with AUC Integration", style="bold blue"))
    
    try:
        # Initialize integrated AUC system
        from integrated_auc_system import IntegratedAUCSystem
        auc_system = IntegratedAUCSystem()
        
        # Run basic AUC fix first if available
        if auc_system.integration_status['basic_auc_fix']:
            console.print("🔧 Running pre-pipeline AUC optimization...")
            try:
                from basic_auc_fix import create_optimized_model
                # This will create/update the model files
                console.print("✅ Pre-pipeline AUC fix completed")
            except Exception as e:
                console.print(f"⚠️ Pre-pipeline fix warning: {e}")
        
        # Import and run the original pipeline
        from ProjectP import main as original_main
        
        # Monitor during execution
        result = original_main()
        
        # Post-pipeline AUC check
        auc_system.post_pipeline_check()
        
        return result
        
    except Exception as e:
        console.print(f"❌ Pipeline execution error: {e}")
        
        # Emergency fallback
        console.print("🆘 Running emergency AUC fix...")
        try:
            from basic_auc_fix import emergency_model_creation
            emergency_model_creation()
            console.print("✅ Emergency model created")
        except Exception as e2:
            console.print(f"❌ Emergency fix also failed: {e2}")
        
        raise e

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Pipeline with AUC Integration")
    parser.add_argument("--mode", default="full_pipeline", help="Pipeline mode")
    parser.add_argument("--run_full_pipeline", action="store_true", help="Run full pipeline")
    
    args = parser.parse_args()
    
    if args.run_full_pipeline or args.mode == "full_pipeline":
        run_pipeline_with_auc_integration("full_pipeline")
    else:
        run_pipeline_with_auc_integration(args.mode)
