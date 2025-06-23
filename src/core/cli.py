"""
🎯 CLI Handler
=============

จัดการ command line interface และ argument parsing
แยกออกจาก main logic เพื่อความชัดเจน
"""

import argparse
import sys
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import os

from src.core.config import config_manager
from src.core.display import banner_manager
from src.core.resource_monitor import resource_monitor
from src.core.pipeline_modes import pipeline_manager


class CLIHandler:
    """จัดการ CLI arguments และ execution flow"""
    
    def __init__(self):
        self.parser = self._create_parser()
        self.execution_results = {}
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """สร้าง argument parser"""
        parser = argparse.ArgumentParser(
            description="🚀 NICEGOLD PROFESSIONAL TRADING SYSTEM v3.0 - ULTIMATE EDITION",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
EXAMPLES:
  python ProjectP.py --run_full_pipeline          # Run complete pipeline
  python ProjectP.py --class_balance_fix          # Fix class imbalance
  python ProjectP.py --ultimate_pipeline          # Ultimate mode with all improvements
  python ProjectP.py --run_all_modes              # Run all modes sequentially
  python ProjectP.py --check_resources            # Check system resources only
"""
        )
        
        # Main pipeline modes
        parser.add_argument(
            "--run_full_pipeline",
            action="store_true",
            help="🚀 Run complete ML pipeline (preprocess → train → validate → export)"
        )
        
        parser.add_argument(
            "--debug_full_pipeline",
            action="store_true",
            help="🐛 Run full pipeline with detailed debugging"
        )
        
        parser.add_argument(
            "--ultimate_pipeline",
            action="store_true",
            help="🔥 Run ULTIMATE pipeline with ALL improvements (Emergency Fixes + Class Balance + Full Pipeline)"
        )
        
        # Specialized modes
        parser.add_argument(
            "--class_balance_fix",
            action="store_true",
            help="🎯 Run dedicated class balance fix mode (แก้ไขปัญหา Class Imbalance แบบสมบูรณ์)"
        )
        
        parser.add_argument(
            "--preprocess",
            action="store_true",
            help="📊 Run preprocessing and feature engineering only"
        )
        
        # Backtesting modes
        parser.add_argument(
            "--realistic_backtest",
            action="store_true",
            help="📈 Run realistic backtest simulation"
        )
        
        parser.add_argument(
            "--robust_backtest",
            action="store_true",
            help="🛡️ Run robust backtest with multiple scenarios"
        )
        
        parser.add_argument(
            "--realistic_backtest_live",
            action="store_true",
            help="🔴 Run live-style backtest simulation"
        )
        
        # Utility modes
        parser.add_argument(
            "--check_resources",
            action="store_true",
            help="🖥️ Check system resources only"
        )
        
        parser.add_argument(
            "--run_all_modes",
            action="store_true",
            help="🚀 Run ALL modes sequentially (for comprehensive testing)"
        )
        
        # Output options
        parser.add_argument(
            "--output_dir",
            type=str,
            default="output_default",
            help="Output directory for results (default: output_default)"
        )
        
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose output"
        )
        
        return parser
    
    def parse_and_execute(self, args: Optional[List[str]] = None) -> None:
        """Parse arguments และ execute ตาม mode ที่เลือก"""
        parsed_args = self.parser.parse_args(args)
        
        # Print banner
        banner_manager.print_main_banner()
        
        # Track execution
        result = None
        
        try:
            # Check resources mode
            if parsed_args.check_resources:
                self._handle_check_resources()
                return
            
            # Run all modes
            if parsed_args.run_all_modes:
                result = self._handle_run_all_modes(parsed_args.output_dir)
            
            # Individual modes
            elif parsed_args.run_full_pipeline:
                result = pipeline_manager.run_mode('full_pipeline')
                
            elif parsed_args.debug_full_pipeline:
                result = pipeline_manager.run_mode('debug_full_pipeline')
                
            elif parsed_args.ultimate_pipeline:
                result = pipeline_manager.run_mode('ultimate_pipeline')
                
            elif parsed_args.class_balance_fix:
                result = pipeline_manager.run_mode('class_balance_fix')
                
            elif parsed_args.preprocess:
                result = pipeline_manager.run_mode('preprocess')
                
            elif parsed_args.realistic_backtest:
                result = pipeline_manager.run_mode('realistic_backtest')
                
            elif parsed_args.robust_backtest:
                result = pipeline_manager.run_mode('robust_backtest')
                
            elif parsed_args.realistic_backtest_live:
                result = pipeline_manager.run_mode('realistic_backtest_live')
            
            # No mode specified
            else:
                self._show_help_message()
                return
            
        except KeyboardInterrupt:
            banner_manager.print_warning("Execution interrupted by user")
        except Exception as e:
            banner_manager.print_error(f"Unexpected error: {e}")
            if parsed_args.verbose:
                import traceback
                traceback.print_exc()
        
        finally:
            self._print_execution_summary(result)
    
    def _handle_check_resources(self) -> None:
        """Handle resource checking"""
        banner_manager.print_mode_banner("Resource Check", "Checking system resources")
        resource_monitor.print_status()
        banner_manager.print_success("Resource check completed")
    
    def _handle_run_all_modes(self, output_dir: str) -> str:
        """Handle running all modes sequentially"""
        self.execution_results = pipeline_manager.run_all_modes()
        
        # Save all results
        all_results_path = os.path.join(output_dir, "all_modes_results.json")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(all_results_path, 'w', encoding='utf-8') as f:
            json.dump(self.execution_results, f, indent=2, ensure_ascii=False)
        
        banner_manager.print_success(f"All modes results saved to {all_results_path}")
        return all_results_path
    
    def _show_help_message(self) -> None:
        """Show help message when no mode is specified"""
        banner_manager.print_info("No mode specified. Available options:")
        print(f"💡 Quick start: python ProjectP.py --run_full_pipeline")
        print(f"🎯 Class balance: python ProjectP.py --class_balance_fix")
        print(f"🔥 Ultimate mode: python ProjectP.py --ultimate_pipeline")
        print(f"🚀 All modes: python ProjectP.py --run_all_modes")
        print(f"📖 Help: python ProjectP.py --help")
    
    def _print_execution_summary(self, result: Optional[str]) -> None:
        """Print execution summary"""
        print(f"\n{'='*80}")
        print(f"📊 EXECUTION SUMMARY")
        print(f"{'='*80}")
        
        if result:
            banner_manager.print_success("Execution completed successfully")
            print(f"📁 Results location: {result}")
        else:
            banner_manager.print_error("Execution failed or no result")
        
        # Show execution results if available
        if self.execution_results:
            print(f"\n📈 MODE EXECUTION SUMMARY:")
            for mode, details in self.execution_results.items():
                status = "✅ SUCCESS" if details.get('success') else "❌ FAILED"
                time_taken = details.get('execution_time', 0)
                print(f"   {mode}: {status} ({time_taken:.2f}s)")
        
        print(f"{'='*80}")
        banner_manager.print_success("Thank you for using NICEGOLD Professional Trading System v3.0!")


# Singleton instance
cli_handler = CLIHandler()
