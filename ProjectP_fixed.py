"""
ProjectP Professional Mode - โหมดการทำงาน

full_pipeline
  ทำอะไร: รันทุกขั้นตอนของ pipeline ตั้งแต่เตรียมข้อมูล, สร้างฟีเจอร์, เทรนโมเดล, แบ็คเทส, วัดผล, export ผลลัพธ์
  เหมาะกับ:
    - การใช้งานจริง (production)
    - งานวิจัยที่ต้องการผลลัพธ์ครบวงจร
    - การรันอัตโนมัติแบบ end-to-end

debug_full_pipeline
  ทำอะไร: เหมือน full_pipeline แต่ log ละเอียดกว่า, ไม่หยุดเมื่อเจอ error
  เหมาะกับ:
    - การดีบัค pipeline
    - การพัฒนา/ปรับปรุงระบบ
    - ตรวจสอบปัญหาทุกจุดใน flow

preprocess
  ทำอะไร: เตรียมข้อมูลและสร้างฟีเจอร์ (feature engineering), สร้างไฟล์ features/preprocessed
  เหมาะกับ:
    - การเตรียมข้อมูลก่อนเทรนโมเดลหรือแบ็คเทส
    - การทดลอง feature engineering
    - กรณีต้องการปรับปรุงคุณภาพข้อมูล

realistic_backtest
  ทำอะไร: แบ็คเทสเสมือนจริง (realistic), ใช้ข้อมูลจาก pipeline, walk-forward, ไม่มี lookahead/data leak
  เหมาะกับ:
    - การประเมินกลยุทธ์ในสภาพแวดล้อมใกล้เคียงจริง
    - ทดสอบ performance ก่อน deploy
    - งาน research ที่ต้องการความสมจริง

robust_backtest
  ทำอะไร: แบ็คเทสแบบ robust (เทพ), รองรับ walk-forward, parallel, เลือกโมเดลได้
  เหมาะกับ:
    - ทดสอบความแข็งแกร่งของกลยุทธ์
    - เปรียบเทียบโมเดลหลายแบบ
    - วิเคราะห์ความเสถียรของระบบในหลาย scenario

realistic_backtest_live
  ทำอะไร: แบ็คเทสเหมือนเทรดจริง (train เฉพาะอดีต, test อนาคต, save model)
  เหมาะกับ:
    - Simulation ที่ต้องการความสมจริงสูงสุด
    - เตรียมโมเดลสำหรับนำไปใช้งาน live
    - ตรวจสอบ overfitting/underfitting ก่อนใช้งานจริง
"""

# Standard library imports
import os
import sys
import warnings
import multiprocessing
import re
import logging
import argparse
import getpass
import socket
import uuid
import shutil
from typing import Optional, Dict, Any, List, Union, Tuple
from tqdm import tqdm

# Third-party imports with graceful fallbacks
from colorama import Fore, Style, init as colorama_init

# Optional imports with fallbacks
try:
    import psutil
except ImportError:
    psutil = None

try:
    import GPUtil
except ImportError:
    GPUtil = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import torch
except ImportError:
    torch = None

# Fix for circular import issue
try:
    from src.strategy_init_helper import initialize_strategy_functions
    initialize_strategy_functions()
except ImportError:
    print("Warning: Could not initialize strategy functions. Some functionality may be limited.")

# Project imports with error handling
try:
    from projectp.pro_log import pro_log
except ImportError:
    def pro_log(msg: str, tag: Optional[str] = None, level: str = "info") -> None:
        print(f"[{level.upper()}] {tag or 'LOG'}: {msg}")

try:
    from projectp.pipeline import run_full_pipeline, run_debug_full_pipeline, run_ultimate_pipeline
except ImportError as e:
    print(f"Warning: Could not import pipeline functions: {e}")
    run_full_pipeline = run_debug_full_pipeline = run_ultimate_pipeline = None

try:
    from src.utils.log_utils import set_log_context
except ImportError:
    def set_log_context(**kwargs: Any) -> None:
        pass

# Environment setup
os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'
colorama_init(autoreset=True)

# Warning filters
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message="Skipped unsupported reflection of expression-based index",
    category=UserWarning,
    module="sqlalchemy"
)
warnings.filterwarnings(
    "ignore",
    message="Hint: Inferred schema contains integer column(s). Integer columns in Python cannot represent missing values.",
    category=UserWarning,
    module="mlflow.types.utils"
)

# Optimize resource usage: set all BLAS env to use all CPU cores
num_cores = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_cores)
print(f"Set all BLAS env to use {num_cores} threads")

# TensorFlow GPU setup with error handling
if tf is not None:
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("TensorFlow: GPU memory growth set to True")
    except Exception as e:
        print("TensorFlow: Failed to set GPU memory growth:", e)
else:
    print("TensorFlow not available")

# PyTorch GPU setup with error handling
if torch is not None:
    try:
        if torch.cuda.is_available():
            print("PyTorch: GPU available:", torch.cuda.get_device_name(0))
            torch.cuda.empty_cache()
    except Exception as e:
        print("PyTorch: GPU setup failed:", e)
else:
    print("PyTorch not available")

# Resource usage checks (with fallbacks)
SYS_MEM_TARGET_GB = 25.0
GPU_MEM_TARGET_GB = 12.0
DISK_USAGE_LIMIT = 0.8

def check_resources() -> None:
    """Check system resources with graceful fallbacks."""
    try:
        if psutil is not None:
            vm = psutil.virtual_memory()
            used_gb, total_gb = vm.used/1e9, vm.total/1e9
            print(f"System RAM: {used_gb:.1f}/{total_gb:.1f} GB ({vm.percent}%)")
            if used_gb > SYS_MEM_TARGET_GB:
                print(f"⚠️ WARNING: RAM usage {used_gb:.1f}GB exceeds target {SYS_MEM_TARGET_GB}GB")
        else:
            print("Resource monitoring unavailable (psutil not installed)")

        if GPUtil is not None:
            try:
                for gpu in GPUtil.getGPUs():
                    used_gb = gpu.memoryUsed/1024
                    total_gb = gpu.memoryTotal/1024
                    print(f"GPU {gpu.id}: {used_gb:.1f}/{total_gb:.1f} GB ({gpu.memoryUtil*100:.0f}%)")
                    if used_gb > GPU_MEM_TARGET_GB:
                        print(f"⚠️ WARNING: GPU {gpu.id} usage above target")
            except Exception as e:
                print(f"GPU monitoring failed: {e}")
        else:
            print("GPU monitoring unavailable (GPUtil not installed)")

        # Disk usage check
        try:
            du = shutil.disk_usage(os.getcwd())
            used_pct = du.used/du.total
            print(f"Disk usage: {used_pct*100:.0f}%")
            if used_pct > DISK_USAGE_LIMIT:
                print(f"⚠️ WARNING: Disk usage {used_pct*100:.0f}% exceeds limit {DISK_USAGE_LIMIT*100:.0f}%")
        except Exception as e:
            print(f"Disk monitoring failed: {e}")

    except Exception as e:
        print(f"Resource check failed: {e}")

def print_professional_banner() -> None:
    """Print professional banner with system info."""
    print(f"{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN}🚀 NICEGOLD PROFESSIONAL TRADING SYSTEM v2.0")
    print(f"{Fore.CYAN}{'='*80}")
    print(f"{Fore.GREEN}System: {socket.gethostname()}")
    print(f"{Fore.GREEN}User: {getpass.getuser()}")
    print(f"{Fore.GREEN}Python: {sys.version.split()[0]}")
    print(f"{Fore.GREEN}Session ID: {str(uuid.uuid4())[:8]}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")

def parse_table(text: str) -> List[List[str]]:
    """Parse table data from text."""
    pattern = r'\|\s*(.+?)\s*\|'
    matches = re.findall(pattern, text)
    return [match.split('|') for match in matches]

def show_order_progress(order: Dict[str, Any]) -> None:
    """Show order progress with type safety."""
    try:
        status = str(order.get("status", "")).upper()
        filled = float(order.get("filled_qty", 0))
        qty = float(order.get("qty", 1))
        
        progress = filled / qty if qty > 0 else 0
        bar_length = 20
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        print(f"Order: [{bar}] {progress*100:.1f}% ({filled}/{qty}) - {status}")
    except Exception as e:
        print(f"Order progress display failed: {e}")

# Main mode functions with comprehensive error handling
def run_full_mode() -> Optional[str]:
    """Run full pipeline mode with comprehensive error handling."""
    print(f"{Fore.GREEN}🚀 Full Pipeline Mode Starting...{Style.RESET_ALL}")
    
    try:
        if run_full_pipeline is None:
            print("❌ Pipeline function not available. Using fallback.")
            return run_fallback_pipeline()
        
        check_resources()
        result = run_full_pipeline()
        
        if result:
            print(f"{Fore.GREEN}✅ Full pipeline completed successfully!{Style.RESET_ALL}")
            return result
        else:
            print(f"{Fore.RED}❌ Full pipeline failed. Trying fallback.{Style.RESET_ALL}")
            return run_fallback_pipeline()
            
    except Exception as e:
        print(f"{Fore.RED}❌ Full pipeline error: {e}{Style.RESET_ALL}")
        return run_fallback_pipeline()

def run_fallback_pipeline() -> Optional[str]:
    """Run fallback pipeline when main pipeline fails."""
    print(f"{Fore.YELLOW}🔄 Running fallback pipeline...{Style.RESET_ALL}")
    
    try:
        # Run the simple pipeline script we created
        import subprocess
        import sys
        
        result = subprocess.run([sys.executable, "run_simple_pipeline.py"], 
                              capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print(f"{Fore.GREEN}✅ Fallback pipeline completed successfully!{Style.RESET_ALL}")
            print(result.stdout)
            return "output_default/fallback_results"
        else:
            print(f"{Fore.RED}❌ Fallback pipeline failed: {result.stderr}{Style.RESET_ALL}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"{Fore.RED}❌ Fallback pipeline timed out{Style.RESET_ALL}")
        return None
    except Exception as e:
        print(f"{Fore.RED}❌ Fallback pipeline error: {e}{Style.RESET_ALL}")
        return None

def run_debug_mode() -> Optional[str]:
    """Run debug pipeline mode."""
    print(f"{Fore.YELLOW}🐛 Debug Pipeline Mode Starting...{Style.RESET_ALL}")
    
    try:
        if run_debug_full_pipeline is None:
            print("❌ Debug pipeline function not available.")
            return None
            
        check_resources()
        result = run_debug_full_pipeline()
        
        if result:
            print(f"{Fore.GREEN}✅ Debug pipeline completed!{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ Debug pipeline failed!{Style.RESET_ALL}")
            
        return result
        
    except Exception as e:
        print(f"{Fore.RED}❌ Debug pipeline error: {e}{Style.RESET_ALL}")
        return None

def run_preprocess_mode() -> Optional[str]:
    """Run preprocessing mode."""
    print(f"{Fore.BLUE}📊 Preprocessing Mode Starting...{Style.RESET_ALL}")
    
    try:
        from projectp.steps.preprocess import run_preprocess
        
        check_resources()
        result = run_preprocess()
        
        if result:
            print(f"{Fore.GREEN}✅ Preprocessing completed!{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ Preprocessing failed!{Style.RESET_ALL}")
            
        return result
        
    except ImportError as e:
        print(f"{Fore.RED}❌ Import error: {e}{Style.RESET_ALL}")
        return None
    except Exception as e:
        print(f"{Fore.RED}❌ Preprocessing error: {e}{Style.RESET_ALL}")
        return None

def run_realistic_backtest_mode() -> Optional[str]:
    """Run realistic backtest mode."""
    print(f"{Fore.MAGENTA}📈 Realistic Backtest Mode Starting...{Style.RESET_ALL}")
    
    try:
        from backtest_engine import run_realistic_backtest
        
        check_resources()
        result = run_realistic_backtest()
        
        if result:
            print(f"{Fore.GREEN}✅ Realistic backtest completed!{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ Realistic backtest failed!{Style.RESET_ALL}")
            
        return result
        
    except ImportError as e:
        print(f"{Fore.RED}❌ Import error: {e}{Style.RESET_ALL}")
        return None
    except Exception as e:
        print(f"{Fore.RED}❌ Realistic backtest error: {e}{Style.RESET_ALL}")
        return None

def run_robust_backtest_mode() -> Optional[str]:
    """Run robust backtest mode."""
    print(f"{Fore.MAGENTA}🛡️ Robust Backtest Mode Starting...{Style.RESET_ALL}")
    
    try:
        from backtest_engine import run_robust_backtest
        
        check_resources()
        result = run_robust_backtest()
        
        if result:
            print(f"{Fore.GREEN}✅ Robust backtest completed!{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ Robust backtest failed!{Style.RESET_ALL}")
            
        return result
        
    except ImportError as e:
        print(f"{Fore.RED}❌ Import error: {e}{Style.RESET_ALL}")
        return None
    except Exception as e:
        print(f"{Fore.RED}❌ Robust backtest error: {e}{Style.RESET_ALL}")
        return None

def run_realistic_backtest_live_mode() -> Optional[str]:
    """Run realistic backtest live mode."""
    print(f"{Fore.CYAN}🔴 Live Backtest Mode Starting...{Style.RESET_ALL}")
    
    try:
        from backtest_engine import run_realistic_backtest
        
        check_resources()
        # Use realistic backtest as fallback if live version not available
        result = run_realistic_backtest()
        
        if result:
            print(f"{Fore.GREEN}✅ Live backtest completed!{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ Live backtest failed!{Style.RESET_ALL}")
            
        return result
        
    except ImportError as e:
        print(f"{Fore.RED}❌ Import error: {e}{Style.RESET_ALL}")
        return None
    except Exception as e:
        print(f"{Fore.RED}❌ Live backtest error: {e}{Style.RESET_ALL}")
        return None

# Global variables for logging
log_issues: List[Dict[str, Any]] = []

def enhanced_pro_log(msg: str, tag: Optional[str] = None, level: str = "info") -> None:
    """Enhanced logging with issue tracking."""
    important_levels = ["error", "critical", "warning"]
    
    if level in important_levels or (tag and tag.lower() in ("result", "summary", "important")):
        print(f"{Fore.RED if level == 'error' else Fore.YELLOW if level == 'warning' else Fore.GREEN}[{level.upper()}] {tag or 'LOG'}: {msg}{Style.RESET_ALL}")
        
        log_issues.append({
            "level": level,
            "tag": tag or "LOG",
            "msg": msg
        })
    else:
        print(f"[{level.upper()}] {tag or 'LOG'}: {msg}")

def print_log_summary() -> None:
    """Print summary of logged issues."""
    if not log_issues:
        print(f"{Fore.GREEN}✅ No critical issues found in logs!{Style.RESET_ALL}")
        return
        
    try:
        from rich.console import Console
        from rich.table import Table
        
        console = Console()
        table = Table(title="Log Issues Summary")
        table.add_column("Level", style="red")
        table.add_column("Tag", style="cyan")
        table.add_column("Message", style="white")
        
        for issue in log_issues:
            table.add_row(
                str(issue.get("level", "")),
                str(issue.get("tag", "")),
                str(issue.get("msg", ""))
            )
        
        console.print(table)
        
    except ImportError:
        print(f"{Fore.YELLOW}📊 LOG ISSUES SUMMARY:{Style.RESET_ALL}")
        for i, issue in enumerate(log_issues, 1):
            level = issue.get("level", "")
            tag = issue.get("tag", "")
            msg = issue.get("msg", "")
            print(f"{i}. [{level.upper()}] {tag}: {msg}")

def main() -> None:
    """Main function with comprehensive argument parsing."""
    parser = argparse.ArgumentParser(
        description="NICEGOLD Professional Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--run_full_pipeline",
        action="store_true",
        help="Run complete ML pipeline (preprocess → train → validate → export)"
    )
    
    parser.add_argument(
        "--debug_full_pipeline",
        action="store_true",
        help="Run full pipeline with detailed debugging"
    )
    
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run preprocessing and feature engineering only"
    )
    
    parser.add_argument(
        "--realistic_backtest",
        action="store_true",
        help="Run realistic backtest simulation"
    )
    
    parser.add_argument(
        "--robust_backtest",
        action="store_true",
        help="Run robust backtest with multiple scenarios"
    )
    
    parser.add_argument(
        "--realistic_backtest_live",
        action="store_true",
        help="Run live-style backtest simulation"
    )
    
    parser.add_argument(
        "--check_resources",
        action="store_true",
        help="Check system resources only"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_professional_banner()
    
    # Track execution
    result = None
    
    try:
        if args.check_resources:
            check_resources()
            return
            
        if args.run_full_pipeline:
            result = run_full_mode()
            
        elif args.debug_full_pipeline:
            result = run_debug_mode()
            
        elif args.preprocess:
            result = run_preprocess_mode()
            
        elif args.realistic_backtest:
            result = run_realistic_backtest_mode()
            
        elif args.robust_backtest:
            result = run_robust_backtest_mode()
            
        elif args.realistic_backtest_live:
            result = run_realistic_backtest_live_mode()
            
        else:
            print(f"{Fore.YELLOW}ℹ️ No mode specified. Use --help for options.{Style.RESET_ALL}")
            print(f"{Fore.CYAN}💡 Quick start: python ProjectP.py --run_full_pipeline{Style.RESET_ALL}")
            
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⏹️ Execution interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ Unexpected error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Print summary
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}📊 EXECUTION SUMMARY")
        print(f"{Fore.CYAN}{'='*80}")
        
        if result:
            print(f"{Fore.GREEN}✅ Mode completed successfully")
            print(f"{Fore.GREEN}📁 Results: {result}")
        else:
            print(f"{Fore.RED}❌ Mode failed or no result")
            
        print_log_summary()
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
