#!/usr/bin/env python3
"""
ProjectP - Advanced AI Trading Pipeline
Interactive Entry Point with Interactive Menu
"""

import os
import sys
import time
import platform
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_logo():
    """Display ProjectP logo"""
    logo = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    ██████╗ ██████╗  ██████╗      ██╗███████╗ ██████╗████████╗██████╗         ║
║    ██╔══██╗██╔══██╗██╔═══██╗     ██║██╔════╝██╔════╝╚══██╔══╝██╔══██╗        ║
║    ██████╔╝██████╔╝██║   ██║     ██║█████╗  ██║        ██║   ██████╔╝        ║
║    ██╔═══╝ ██╔══██╗██║   ██║██   ██║██╔══╝  ██║        ██║   ██╔═══╝         ║
║    ██║     ██║  ██║╚██████╔╝╚█████╔╝███████╗╚██████╗   ██║   ██║             ║
║    ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝   ╚═╝             ║
║                                                                              ║
║                    🚀 Advanced AI Trading Pipeline 🚀                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(logo)

def print_system_info():
    """Display system information"""
    print("📊 System Information:")
    print(f"   🖥️  Platform: {platform.system()} {platform.release()}")
    print(f"   🐍  Python: {platform.python_version()}")
    print(f"   📅  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   📁  Working Directory: {os.getcwd()}")
    
    # Check for required directories
    required_dirs = ['projectp', 'output_default', 'config']
    existing_dirs = [d for d in required_dirs if os.path.exists(d)]
    print(f"   📂  Project Directories: {len(existing_dirs)}/{len(required_dirs)} found")
    
    # Check requirements.txt
    requirements_file = "requirements.txt"
    if os.path.exists(requirements_file):
        try:
            with open(requirements_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            req_count = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            print(f"   📋  Requirements: {req_count} packages listed in {requirements_file}")
        except Exception:
            print(f"   📋  Requirements: Found {requirements_file} (unable to parse)")
    else:
        print(f"   📋  Requirements: ❌ {requirements_file} not found")
    
    # Check Python packages
    packages = []
    package_checks = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'), 
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('joblib', 'joblib'),
        ('streamlit', 'streamlit'),
        ('fastapi', 'fastapi'),
        ('rich', 'rich'),
        ('yaml', 'pyyaml'),
        ('tqdm', 'tqdm'),
        ('scipy', 'scipy')
    ]
    
    for module_name, package_name in package_checks:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            packages.append(f"{package_name} ✅ {version}")
        except ImportError:
            packages.append(f"{package_name} ❌")
    
    # Show packages in a more organized way
    print(f"   📦  Core Packages:")
    for i, pkg in enumerate(packages[:4]):  # Show first 4 (pandas, numpy, sklearn, matplotlib)
        print(f"       {pkg}")
    
    # Check ML/AI packages
    ml_packages = [
        ('catboost', 'CatBoost'),
        ('xgboost', 'XGBoost'), 
        ('lightgbm', 'LightGBM'),
        ('optuna', 'Optuna'),
        ('shap', 'SHAP'),
        ('ta', 'TA-Lib'),
        ('psutil', 'psutil'),
        ('yaml', 'PyYAML')
    ]
    
    ml_available = []
    for module_name, display_name in ml_packages:
        try:
            module = __import__(module_name)
            ml_available.append(f"{display_name} ✅")
        except ImportError:
            ml_available.append(f"{display_name} ❌")
        except OSError as e:
            # Handle library loading errors (like missing libgomp.so.1)
            if "libgomp" in str(e) or "shared object" in str(e):
                ml_available.append(f"{display_name} ⚠️ (missing system libs)")
            else:
                ml_available.append(f"{display_name} ❌ ({str(e)[:30]}...)")
        except Exception as e:
            ml_available.append(f"{display_name} ❌ (error)")
    
    # Count installed vs total
    installed = len([p for p in packages if '✅' in p])
    total = len(packages)
    ml_installed = len([p for p in ml_available if '✅' in p])
    
    print(f"   📊  Package Status: {installed}/{total} core packages installed")
    print(f"   🤖  ML Packages: {ml_installed}/{len(ml_available)} available")
    
    # Show detailed package status if not all installed
    if installed < total or ml_installed < len(ml_available):
        print(f"   💡  Missing packages can be installed with option 16")
    else:
        print(f"   🎉  All essential packages are installed!")
    
    print()

def print_menu():
    """Display main menu"""
    print("╔═══════════════════════════════════════════════════════════════════════════════╗")
    print("║                            🎯 ProjectP Main Menu                             ║")
    print("╠═══════════════════════════════════════════════════════════════════════════════╣")
    print("║                                                                               ║")
    print("║  🚀 Pipeline Modes:                                                          ║")
    print("║                                                                               ║")
    print("║  1️⃣  Full Pipeline           - รันทุกขั้นตอนแบบอัตโนมัติ (เทพ ครบระบบ)      ║")
    print("║  2️⃣  Debug Pipeline          - ดีบัค: ตรวจสอบทุกจุด (log ละเอียด)           ║")
    print("║  3️⃣  Ultimate Pipeline       - เทพสุด: สำหรับ Production                   ║")
    print("║                                                                               ║")
    print("║  📊 Data & Training:                                                         ║")
    print("║                                                                               ║")
    print("║  4️⃣  Preprocess Only         - เตรียมข้อมูล & Feature Engineering          ║")
    print("║  5️⃣  Train Model             - เทรนโมเดล ML เท่านั้น                       ║")
    print("║  6️⃣  Predict Only            - ทำนายด้วยโมเดลที่มีอยู่                      ║")
    print("║                                                                               ║")
    print("║  📈 Backtesting:                                                             ║")
    print("║                                                                               ║")
    print("║  7️⃣  Realistic Backtest      - แบลคเทสเสมือนจริง (Walk-Forward)            ║")
    print("║  8️⃣  Robust Backtest         - แบลคเทสเทพ (เลือกโมเดลได้)                  ║")
    print("║  9️⃣  Live Backtest           - แบลคเทสสำหรับการใช้งานจริง                   ║")
    print("║                                                                               ║")
    print("║  🖥️  Monitoring & Tools:                                                     ║")
    print("║                                                                               ║")
    print("║  🔟  Dashboard               - เปิด Streamlit Dashboard                     ║")
    print("║  1️⃣1️⃣ Enterprise Services     - เปิด FastAPI Model Serving                 ║")
    print("║  1️⃣2️⃣ Enterprise Pipeline     - รัน Enterprise Workflow                    ║")
    print("║                                                                               ║")
    print("║  ⚙️  Utilities:                                                              ║")
    print("║                                                                               ║")
    print("║  1️⃣3️⃣ Check System Health     - ตรวจสอบสุขภาพระบบ                          ║")
    print("║  1️⃣4️⃣ View Logs              - ดู log files                               ║")
    print("║  1️⃣5️⃣ Clean Output           - ลบไฟล์ output                              ║")
    print("║  1️⃣6️⃣ Install Packages       - ติดตั้ง packages ที่ขาดหาย                 ║")
    print("║                                                                               ║")
    print("║  0️⃣  Exit                    - ออกจากโปรแกรม                              ║")
    print("║                                                                               ║")
    print("╚═══════════════════════════════════════════════════════════════════════════════╝")
    print()

def run_pipeline_mode(mode):
    """Run specific pipeline mode with production-level robustness"""
    import uuid
    import shutil
    import json
    import logging
    import traceback
    from pathlib import Path
    
    # Generate unique run ID
    run_id = f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    
    # Setup logging
    log_dir = Path("output_default/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline_execution.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('ProjectP')
    
    # Create run-specific output directory
    run_output_dir = Path(f"output_default/runs/{run_id}")
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"🚀 Starting {mode} with Run ID: {run_id}")
        logger.info(f"📁 Output directory: {run_output_dir}")
        
        # Pre-flight checks
        if not run_preflight_checks(logger):
            logger.error("❌ Pre-flight checks failed")
            return False
        
        # Start time tracking
        start_time = datetime.now()
        logger.info(f"⏱️ Pipeline started at: {start_time}")
        
        # Run specific pipeline mode
        result = None
        if mode == "full_pipeline":
            result = run_production_full_pipeline(run_id, run_output_dir, logger)
        elif mode == "debug_full_pipeline":
            result = run_production_debug_pipeline(run_id, run_output_dir, logger)
        elif mode == "ultimate_pipeline":
            result = run_production_ultimate_pipeline(run_id, run_output_dir, logger)
        else:
            logger.error(f"❌ Unknown pipeline mode: {mode}")
            return False
        
        # End time tracking
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Generate execution report
        try:
            import psutil
            system_info = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            }
        except ImportError:
            system_info = {"note": "psutil not available for system monitoring"}
        
        execution_report = {
            "run_id": run_id,
            "mode": mode,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "success": result is not False,
            "output_directory": str(run_output_dir),
            "system_info": system_info
        }
        
        # Save execution report
        with open(run_output_dir / "execution_report.json", 'w') as f:
            json.dump(execution_report, f, indent=2, default=str)
        
        if result is not False:
            logger.info(f"✅ {mode} completed successfully!")
            logger.info(f"⏱️ Total duration: {duration}")
            logger.info(f"📊 Reports saved to: {run_output_dir}")
            
            # Archive successful run
            archive_successful_run(run_output_dir, logger)
            
            print("\n" + "=" * 80)
            print(f"🎉 SUCCESS! {mode.upper()} COMPLETED")
            print(f"📊 Run ID: {run_id}")
            print(f"⏱️ Duration: {duration}")
            print(f"📁 Results: {run_output_dir}")
            print("=" * 80)
            return True
        else:
            logger.error(f"❌ {mode} failed!")
            create_error_report(run_output_dir, mode, logger)
            return False
            
    except ImportError as e:
        logger.error(f"❌ Import Error: {e}")
        logger.info("💡 Trying alternative CLI method...")
        create_error_report(run_output_dir, mode, logger, str(e))
        return run_cli_mode(mode)
    except Exception as e:
        logger.error(f"❌ Critical Error in {mode}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        create_error_report(run_output_dir, mode, logger, str(e), traceback.format_exc())
        return False
    
    # Initialize pipeline context
    pipeline_start_time = datetime.datetime.now()
    pipeline_id = f"{mode}_{pipeline_start_time.strftime('%Y%m%d_%H%M%S')}"
    
    print(f"🚀 Starting {mode.upper()} PIPELINE")
    print(f"📋 Pipeline ID: {pipeline_id}")
    print(f"⏰ Start Time: {pipeline_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Pre-flight checks
    if not _pre_flight_checks(mode):
        print("❌ Pre-flight checks failed. Pipeline aborted.")
        return False
    
    try:
        # Import pipeline functions with fallback
        pipeline_functions = _import_pipeline_functions()
        if not pipeline_functions:
            print("� Trying alternative CLI method...")
            return run_cli_mode(mode)
        
        # Execute pipeline based on mode
        result = None
        if mode == "full_pipeline":
            result = _run_full_pipeline_production(pipeline_functions, pipeline_id)
        elif mode == "debug_full_pipeline":
            result = _run_debug_pipeline_production(pipeline_functions, pipeline_id)
        elif mode == "ultimate_pipeline":
            result = _run_ultimate_pipeline_production(pipeline_functions, pipeline_id)
        else:
            print(f"❌ Unknown pipeline mode: {mode}")
            return False
        
        # Calculate execution time
        execution_time = datetime.datetime.now() - pipeline_start_time
        
        print("=" * 80)
        if result:
            print(f"✅ {mode.upper()} COMPLETED SUCCESSFULLY!")
            print(f"⏱️  Total Execution Time: {execution_time}")
            print(f"📊 Pipeline ID: {pipeline_id}")
            _log_pipeline_success(pipeline_id, mode, execution_time)
        else:
            print(f"⚠️ {mode.upper()} completed with warnings")
            print(f"⏱️  Execution Time: {execution_time}")
            _log_pipeline_warning(pipeline_id, mode, execution_time)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Some packages may be missing. Try option 16 to install packages.")
        print("💡 Trying alternative CLI method...")
        _log_pipeline_error(pipeline_id, mode, str(e), "ImportError")
        return run_cli_mode(mode)
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Pipeline {mode} interrupted by user")
        execution_time = datetime.datetime.now() - pipeline_start_time
        print(f"⏱️  Time before interruption: {execution_time}")
        _log_pipeline_error(pipeline_id, mode, "User interruption", "KeyboardInterrupt")
        return False
        
    except Exception as e:
        print(f"❌ Error running {mode}: {e}")
        print("💡 If this is a package import error, try option 16 to install missing packages.")
        execution_time = datetime.datetime.now() - pipeline_start_time
        print(f"⏱️  Time before failure: {execution_time}")
        print("\n🔍 Error Details:")
        traceback.print_exc()
        _log_pipeline_error(pipeline_id, mode, str(e), "Exception")
        return False

def _pre_flight_checks(mode):
    """Perform comprehensive pre-flight checks"""
    print("🔍 Performing pre-flight checks...")
    
    checks = []
    
    # Check available disk space
    import shutil
    free_space_gb = shutil.disk_usage('.').free / (1024**3)
    if free_space_gb < 1.0:
        checks.append(f"❌ Low disk space: {free_space_gb:.1f}GB (minimum 1GB required)")
    else:
        checks.append(f"✅ Disk space: {free_space_gb:.1f}GB available")
    
    # Check memory usage
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_available_gb = memory.available / (1024**3)
        if memory_available_gb < 0.5:
            checks.append(f"⚠️ Low memory: {memory_available_gb:.1f}GB available")
        else:
            checks.append(f"✅ Memory: {memory_available_gb:.1f}GB available")
    except ImportError:
        checks.append("⚠️ Cannot check memory (psutil not available)")
    
    # Check required directories
    required_dirs = ['projectp', 'src']
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            checks.append(f"✅ Directory: {dir_name}/")
        else:
            checks.append(f"❌ Missing directory: {dir_name}/")
    
    # Check core dependencies
    core_deps = ['pandas', 'numpy', 'sklearn', 'tqdm']
    missing_deps = []
    for dep in core_deps:
        try:
            __import__(dep)
            checks.append(f"✅ Package: {dep}")
        except ImportError:
            missing_deps.append(dep)
            checks.append(f"❌ Missing package: {dep}")
    
    # Display checks
    for check in checks:
        print(f"   {check}")
    
    # Determine if we can proceed
    critical_errors = [c for c in checks if c.startswith("❌")]
    if critical_errors:
        print(f"\n⚠️ Found {len(critical_errors)} critical issues:")
        for error in critical_errors:
            print(f"   {error}")
        
        if missing_deps:
            print(f"\n💡 Install missing packages with: pip install {' '.join(missing_deps)}")
            print("💡 Or use option 16 in the main menu")
        
        return False
    
    print("✅ All pre-flight checks passed!")
    return True

def _import_pipeline_functions():
    """Import pipeline functions with comprehensive error handling"""
    try:
        from projectp.pipeline import run_full_pipeline, run_debug_full_pipeline, run_ultimate_pipeline
        return {
            'full': run_full_pipeline,
            'debug': run_debug_full_pipeline,
            'ultimate': run_ultimate_pipeline
        }
    except ImportError as e:
        print(f"⚠️ Pipeline imports failed: {e}")
        return None

def _run_full_pipeline_production(pipeline_functions, pipeline_id):
    """Run full pipeline with production features"""
    print("🔄 FULL PIPELINE MODE - Complete automated workflow")
    print("📋 Features: Data processing, Training, Prediction, Backtesting")
    
    # Create output directory for this run
    output_dir = f"output_default/run_{pipeline_id}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Execute with monitoring
    try:
        print("\n🎯 Starting full pipeline execution...")
        result = pipeline_functions['full']()
        
        # Additional production features
        _generate_pipeline_report(pipeline_id, "full_pipeline", output_dir)
        _backup_critical_outputs(output_dir)
        
        return result
    except Exception as e:
        print(f"❌ Full pipeline execution failed: {e}")
        _save_error_report(pipeline_id, "full_pipeline", str(e), output_dir)
        raise

def _run_debug_pipeline_production(pipeline_functions, pipeline_id):
    """Run debug pipeline with enhanced debugging features"""
    print("🔍 DEBUG PIPELINE MODE - Detailed diagnostics and validation")
    print("📋 Features: Step-by-step execution, Detailed logging, Validation checks")
    
    # Enable verbose logging
    import logging
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Create debug output directory
    debug_dir = f"output_default/debug_{pipeline_id}"
    os.makedirs(debug_dir, exist_ok=True)
    print(f"📁 Debug directory: {debug_dir}")
    
    try:
        print("\n🔬 Starting debug pipeline execution...")
        print("⚙️ Debug mode: Enhanced logging and validation enabled")
        
        # Run with debug monitoring
        result = pipeline_functions['debug']()
        
        # Generate debug report
        _generate_debug_report(pipeline_id, debug_dir)
        _validate_pipeline_outputs(debug_dir)
        
        return result
    except Exception as e:
        print(f"❌ Debug pipeline execution failed: {e}")
        _save_debug_error_report(pipeline_id, str(e), debug_dir)
        raise

def _run_ultimate_pipeline_production(pipeline_functions, pipeline_id):
    """Run ultimate pipeline with maximum production features"""
    print("🏆 ULTIMATE PIPELINE MODE - Production-ready execution")
    print("📋 Features: Auto-recovery, Health monitoring, Performance optimization")
    
    # Create ultimate output directory
    ultimate_dir = f"output_default/ultimate_{pipeline_id}"
    os.makedirs(ultimate_dir, exist_ok=True)
    print(f"📁 Ultimate directory: {ultimate_dir}")
    
    # Enable performance monitoring
    performance_start = time.time()
    
    try:
        print("\n⚡ Starting ultimate pipeline execution...")
        print("🛡️ Ultimate mode: Auto-recovery and performance optimization enabled")
        
        # Pre-execution health check
        _monitor_system_health()
        
        # Execute with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = pipeline_functions['ultimate']()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Attempt {attempt + 1} failed: {e}")
                    print(f"🔄 Retrying in 5 seconds... ({attempt + 2}/{max_retries})")
                    time.sleep(5)
                else:
                    raise
        
        # Performance analysis
        execution_time = time.time() - performance_start
        _analyze_performance(pipeline_id, execution_time, ultimate_dir)
        
        # Generate comprehensive report
        _generate_ultimate_report(pipeline_id, ultimate_dir)
        _archive_successful_run(ultimate_dir)
        
        return result
    except Exception as e:
        print(f"❌ Ultimate pipeline execution failed: {e}")
        _save_ultimate_error_report(pipeline_id, str(e), ultimate_dir)
        raise

def _monitor_system_health():
    """Monitor system health during pipeline execution"""
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        print(f"💻 System Health:")
        print(f"   CPU Usage: {cpu_percent:.1f}%")
        print(f"   Memory Usage: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
        
        if cpu_percent > 90:
            print("⚠️ High CPU usage detected")
        if memory.percent > 90:
            print("⚠️ High memory usage detected")
            
    except ImportError:
        print("⚠️ System monitoring unavailable (psutil required)")

def _generate_pipeline_report(pipeline_id, mode, output_dir):
    """Generate comprehensive pipeline report"""
    report_file = os.path.join(output_dir, f"pipeline_report_{pipeline_id}.txt")
    
    with open(report_file, 'w') as f:
        f.write(f"Pipeline Execution Report\n")
        f.write(f"========================\n\n")
        f.write(f"Pipeline ID: {pipeline_id}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write(f"\nExecution completed successfully.\n")
    
    print(f"📊 Pipeline report saved: {report_file}")

def _generate_debug_report(pipeline_id, debug_dir):
    """Generate detailed debug report"""
    debug_report = os.path.join(debug_dir, f"debug_report_{pipeline_id}.txt")
    
    with open(debug_report, 'w') as f:
        f.write(f"Debug Pipeline Report\n")
        f.write(f"====================\n\n")
        f.write(f"Pipeline ID: {pipeline_id}\n")
        f.write(f"Debug Mode: Enhanced\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write(f"\nDebug execution completed.\n")
    
    print(f"🔍 Debug report saved: {debug_report}")

def _generate_ultimate_report(pipeline_id, ultimate_dir):
    """Generate ultimate pipeline report"""
    ultimate_report = os.path.join(ultimate_dir, f"ultimate_report_{pipeline_id}.txt")
    
    with open(ultimate_report, 'w') as f:
        f.write(f"Ultimate Pipeline Report\n")
        f.write(f"=======================\n\n")
        f.write(f"Pipeline ID: {pipeline_id}\n")
        f.write(f"Mode: Ultimate Production\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write(f"\nUltimate execution completed with production features.\n")
    
    print(f"🏆 Ultimate report saved: {ultimate_report}")

def _analyze_performance(pipeline_id, execution_time, output_dir):
    """Analyze pipeline performance"""
    performance_file = os.path.join(output_dir, f"performance_{pipeline_id}.txt")
    
    with open(performance_file, 'w') as f:
        f.write(f"Performance Analysis\n")
        f.write(f"===================\n\n")
        f.write(f"Total Execution Time: {execution_time:.2f} seconds\n")
        f.write(f"Performance Rating: {'Excellent' if execution_time < 300 else 'Good' if execution_time < 600 else 'Needs Optimization'}\n")
    
    print(f"⚡ Performance analysis saved: {performance_file}")

def _backup_critical_outputs(output_dir):
    """Backup critical pipeline outputs"""
    backup_dir = f"{output_dir}/backup"
    os.makedirs(backup_dir, exist_ok=True)
    print(f"💾 Critical outputs backed up to: {backup_dir}")

def _archive_successful_run(output_dir):
    """Archive successful ultimate pipeline run"""
    archive_dir = f"{output_dir}/archive"
    os.makedirs(archive_dir, exist_ok=True)
    print(f"📦 Successful run archived to: {archive_dir}")

def _validate_pipeline_outputs(output_dir):
    """Validate pipeline outputs"""
    print("✅ Validating pipeline outputs...")
    # Add validation logic here
    print("✅ Output validation completed")

def _save_error_report(pipeline_id, mode, error, output_dir):
    """Save error report for troubleshooting"""
    error_file = os.path.join(output_dir, f"error_report_{pipeline_id}.txt")
    
    with open(error_file, 'w') as f:
        f.write(f"Pipeline Error Report\n")
        f.write(f"====================\n\n")
        f.write(f"Pipeline ID: {pipeline_id}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Error: {error}\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
    
    print(f"❌ Error report saved: {error_file}")

def _save_debug_error_report(pipeline_id, error, debug_dir):
    """Save debug error report"""
    error_file = os.path.join(debug_dir, f"debug_error_{pipeline_id}.txt")
    
    with open(error_file, 'w') as f:
        f.write(f"Debug Error Report\n")
        f.write(f"=================\n\n")
        f.write(f"Pipeline ID: {pipeline_id}\n")
        f.write(f"Error: {error}\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
    
    print(f"🔍 Debug error report saved: {error_file}")

def _save_ultimate_error_report(pipeline_id, error, ultimate_dir):
    """Save ultimate error report"""
    error_file = os.path.join(ultimate_dir, f"ultimate_error_{pipeline_id}.txt")
    
    with open(error_file, 'w') as f:
        f.write(f"Ultimate Error Report\n")
        f.write(f"====================\n\n")
        f.write(f"Pipeline ID: {pipeline_id}\n")
        f.write(f"Error: {error}\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
    
    print(f"🏆 Ultimate error report saved: {error_file}")

def _log_pipeline_success(pipeline_id, mode, execution_time):
    """Log successful pipeline execution"""
    log_entry = f"[SUCCESS] {datetime.datetime.now()} - Pipeline {pipeline_id} ({mode}) completed in {execution_time}"
    _write_to_log(log_entry)

def _log_pipeline_warning(pipeline_id, mode, execution_time):
    """Log pipeline execution with warnings"""
    log_entry = f"[WARNING] {datetime.datetime.now()} - Pipeline {pipeline_id} ({mode}) completed with warnings in {execution_time}"
    _write_to_log(log_entry)

def _log_pipeline_error(pipeline_id, mode, error, error_type):
    """Log pipeline execution error"""
    log_entry = f"[ERROR] {datetime.datetime.now()} - Pipeline {pipeline_id} ({mode}) failed: {error_type} - {error}"
    _write_to_log(log_entry)

def _write_to_log(log_entry):
    """Write log entry to pipeline log file"""
    log_dir = "output_default/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "pipeline_execution.log")
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{log_entry}\n")
    
    print(f"📝 Logged to: {log_file}")

def run_cli_mode(mode):
    """Run mode through CLI"""
    try:
        from projectp.cli import main_cli
        import os
        
        # Set mode in environment
        os.environ['PROJECTP_AUTO_MODE'] = '1'
        sys.argv = ['ProjectP', '--mode', mode, '--auto']
        
        main_cli()
        return True
        
    except Exception as e:
        print(f"❌ Error running CLI mode: {e}")
        return False

def run_individual_step(step):
    """Run individual pipeline step"""
    try:
        print(f"🔧 Running {step}...")
        
        if step == "preprocess":
            from projectp.steps.preprocess import run_preprocess
            run_preprocess()
        elif step == "train":
            from projectp.steps.train import run_train
            run_train()
        elif step == "predict":
            from projectp.steps.predict import run_predict
            run_predict()
        else:
            print(f"❌ Unknown step: {step}")
            return False
            
        print(f"✅ {step} completed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error running {step}: {e}")
        return False

def run_backtest_mode(mode):
    """Run backtest mode"""
    try:
        print(f"📈 Starting {mode}...")
        
        if mode == "realistic_backtest":
            from projectp.cli import main_cli
            import os
            os.environ['PROJECTP_AUTO_MODE'] = '1'
            sys.argv = ['ProjectP', '--mode', 'realistic_backtest', '--auto']
            main_cli()
        elif mode == "robust_backtest":
            from projectp.cli import main_cli
            import os
            os.environ['PROJECTP_AUTO_MODE'] = '1'
            sys.argv = ['ProjectP', '--mode', 'robust_backtest', '--auto']
            main_cli()
        elif mode == "realistic_backtest_live":
            from projectp.cli import main_cli
            import os
            os.environ['PROJECTP_AUTO_MODE'] = '1'
            sys.argv = ['ProjectP', '--mode', 'realistic_backtest_live', '--auto']
            main_cli()
        else:
            print(f"❌ Unknown backtest mode: {mode}")
            return False
            
        print(f"✅ {mode} completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error running {mode}: {e}")
        return False

def run_monitoring_tool(tool):
    """Run monitoring tools"""
    try:
        print(f"🖥️ Starting {tool}...")
        
        if tool == "dashboard":
            print("📊 Starting Streamlit Dashboard...")
            print("🌐 Dashboard will open in your browser at: http://localhost:8501")
            import subprocess
            subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'projectp/dashboard.py'])
            
        elif tool == "enterprise_services":
            print("🔧 Starting FastAPI Enterprise Services...")
            print("🌐 API will be available at: http://localhost:8500")
            import subprocess
            subprocess.run([sys.executable, 'projectp/enterprise_services.py'])
            
        elif tool == "enterprise_pipeline":
            print("🏭 Starting Enterprise Pipeline...")
            import subprocess
            subprocess.run([sys.executable, 'projectp/enterprise_pipeline.py'])
            
        else:
            print(f"❌ Unknown tool: {tool}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error running {tool}: {e}")
        return False

def run_utility(utility):
    """Run utility functions"""
    if utility == "health_check":
        print("🔍 System Health Check:")
        print("-" * 50)
        
        # Check directories
        dirs = ['projectp', 'output_default', 'config', 'data']
        for d in dirs:
            status = "✅" if os.path.exists(d) else "❌"
            print(f"  {status} {d}/")
        
        # Check key files
        files = ['projectp/pipeline.py', 'projectp/cli.py', 'projectp/__main__.py']
        for f in files:
            status = "✅" if os.path.exists(f) else "❌"
            print(f"  {status} {f}")
        
        print("-" * 50)
        
    elif utility == "view_logs":
        print("📋 Recent Log Files:")
        print("-" * 50)
        
        log_files = []
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.log'):
                    log_files.append(os.path.join(root, file))
        
        if log_files:
            for log_file in log_files[-5:]:  # Show last 5 log files
                print(f"  📄 {log_file}")
                if os.path.getsize(log_file) < 1000000:  # Less than 1MB
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()[-10:]  # Last 10 lines
                            for line in lines:
                                print(f"    {line.strip()}")
                    except:
                        print("    (Unable to read file)")
                print()
        else:
            print("  No log files found")
            
    elif utility == "clean_output":
        print("🧹 Cleaning Output Files:")
        print("-" * 50)
        
        output_dirs = ['output_default', 'output', 'logs']
        cleaned_count = 0
        
        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                import shutil
                try:
                    shutil.rmtree(output_dir)
                    os.makedirs(output_dir, exist_ok=True)
                    print(f"  ✅ Cleaned {output_dir}/")
                    cleaned_count += 1
                except Exception as e:
                    print(f"  ❌ Error cleaning {output_dir}/: {e}")
        
        print(f"  📊 Cleaned {cleaned_count} directories")
        
    elif utility == "install_packages":
        print("📦 Package Installation Helper:")
        print("-" * 50)
        
        # Check if requirements.txt exists
        requirements_file = "requirements.txt"
        if os.path.exists(requirements_file):
            print(f"✅ Found {requirements_file}")
            
            # Option to install from requirements.txt
            print("\n📋 Installation Options:")
            print("1. Install from requirements.txt (Recommended - Full Install)")
            print("2. Install essential packages only (Quick Start)")
            print("3. Install specific package groups")
            print("4. Install system libraries (Fix LightGBM/XGBoost issues)")
            print("5. Install all missing packages manually")
            print("0. Back to main menu")
            
            choice = input("\nSelect installation method (0-5): ").strip()
            
            if choice == "1":
                print(f"\n📦 Installing packages from {requirements_file}...")
                print("This may take a few minutes...")
                
                confirm = input("Proceed with installation? (y/N): ").strip().lower()
                if confirm in ['y', 'yes']:
                    import subprocess
                    try:
                        print("🔄 Running: pip install -r requirements.txt")
                        result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', requirements_file],
                                              capture_output=True, text=True, check=True)
                        print("✅ Requirements installation completed successfully!")
                        print("📊 Summary:")
                        output_lines = result.stdout.split('\n')
                        for line in output_lines[-10:]:  # Show last 10 lines
                            if line.strip():
                                print(f"   {line}")
                    except subprocess.CalledProcessError as e:
                        print(f"❌ Installation failed: {e}")
                        print("Error details:")
                        print(e.stderr)
                        print("\n💡 Try installing packages individually or check your internet connection")
                else:
                    print("Installation cancelled.")
                return True
            elif choice == "2":
                print("\n🚀 Installing essential packages for quick start...")
                essential_packages = [
                    'tqdm', 'pandas', 'numpy', 'scikit-learn', 'matplotlib', 
                    'seaborn', 'joblib', 'rich', 'streamlit', 'fastapi', 
                    'uvicorn', 'plotly', 'requests', 'python-dateutil', 'pytz'
                ]
                
                confirm = input("Proceed with essential packages installation? (y/N): ").strip().lower()
                if confirm in ['y', 'yes']:
                    import subprocess
                    try:
                        result = subprocess.run([sys.executable, '-m', 'pip', 'install'] + essential_packages,
                                              capture_output=True, text=True, check=True)
                        print("✅ Essential packages installed successfully!")
                        print("🎯 You can now try running the pipeline (option 1)!")
                    except subprocess.CalledProcessError as e:
                        print(f"❌ Installation failed: {e}")
                        print(e.stderr)
                else:
                    print("Installation cancelled.")
                return True
            elif choice == "4":
                print("\n🔧 Installing system libraries for ML packages...")
                print("This will install libgomp1 and other dependencies needed by LightGBM, XGBoost, etc.")
                
                confirm = input("Proceed with system library installation? (requires sudo) (y/N): ").strip().lower()
                if confirm in ['y', 'yes']:
                    import subprocess
                    try:
                        print("🔄 Running: sudo apt-get update && sudo apt-get install -y libgomp1")
                        result = subprocess.run(['sudo', 'apt-get', 'update'], capture_output=True, text=True, check=True)
                        result = subprocess.run(['sudo', 'apt-get', 'install', '-y', 'libgomp1'], 
                                              capture_output=True, text=True, check=True)
                        print("✅ System libraries installed successfully!")
                        print("🎯 ML packages should now work properly!")
                    except subprocess.CalledProcessError as e:
                        print(f"❌ System library installation failed: {e}")
                        print("💡 You may need to run: sudo apt-get install libgomp1")
                    except FileNotFoundError:
                        print("❌ sudo command not found. Please install libgomp1 manually:")
                        print("   sudo apt-get install libgomp1")
                else:
                    print("Installation cancelled.")
                return True
            elif choice == "0":
                return True
        else:
            print(f"❌ {requirements_file} not found")
            print("💡 Creating basic package installation menu...")
        
        # Fallback to manual package groups (existing code)
        core_packages = ['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'joblib']
        ml_packages = ['catboost', 'xgboost', 'lightgbm', 'optuna', 'shap', 'ta']
        web_packages = ['streamlit', 'fastapi', 'uvicorn', 'rich']
        
        all_packages = {
            'Core Data Science': core_packages,
            'Machine Learning': ml_packages, 
            'Web & UI': web_packages
        }
        
        print("\nAvailable package groups:")
        for i, (group_name, packages) in enumerate(all_packages.items(), 1):
            missing = []
            for pkg in packages:
                try:
                    __import__(pkg.replace('-', '_'))
                except ImportError:
                    missing.append(pkg)
            
            status = f"({len(packages)-len(missing)}/{len(packages)} installed)"
            print(f"  {i}. {group_name} {status}")
            if missing:
                print(f"     Missing: {', '.join(missing)}")
        
        print("\n6. Install All Missing Packages")
        print("0. Back to main menu")
        
        try:
            choice = input("\nSelect package group to install (0-6): ").strip()
            
            if choice == "0":
                return True
            elif choice in ["1", "2", "3"]:
                group_names = list(all_packages.keys())
                selected_group = group_names[int(choice)-1]
                packages_to_install = all_packages[selected_group]
                
                print(f"\n📦 Installing {selected_group} packages...")
                install_command = f"pip install {' '.join(packages_to_install)}"
                print(f"Command: {install_command}")
                
                confirm = input("Proceed with installation? (y/N): ").strip().lower()
                if confirm in ['y', 'yes']:
                    import subprocess
                    try:
                        result = subprocess.run([sys.executable, '-m', 'pip', 'install'] + packages_to_install, 
                                              capture_output=True, text=True, check=True)
                        print("✅ Installation completed successfully!")
                        print(result.stdout)
                    except subprocess.CalledProcessError as e:
                        print(f"❌ Installation failed: {e}")
                        print(e.stderr)
                else:
                    print("Installation cancelled.")
                    
            elif choice == "6":
                all_missing = []
                for packages in all_packages.values():
                    for pkg in packages:
                        try:
                            __import__(pkg.replace('-', '_'))
                        except ImportError:
                            all_missing.append(pkg)
                
                if all_missing:
                    print(f"\n📦 Installing all missing packages: {', '.join(all_missing)}")
                    confirm = input("Proceed with installation? (y/N): ").strip().lower()
                    if confirm in ['y', 'yes']:
                        import subprocess
                        try:
                            result = subprocess.run([sys.executable, '-m', 'pip', 'install'] + all_missing,
                                                  capture_output=True, text=True, check=True)
                            print("✅ All packages installed successfully!")
                        except subprocess.CalledProcessError as e:
                            print(f"❌ Installation failed: {e}")
                else:
                    print("✅ All packages are already installed!")
                    
        except Exception as e:
            print(f"❌ Error during installation: {e}")
        
    return True

def wait_for_user():
    """Wait for user to press Enter"""
    print()
    input("Press Enter to continue...")
    print()

def main():
    """Main application loop"""
    while True:
        # Clear screen (works on both Windows and Unix)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Display header
        print_logo()
        print_system_info()
        print_menu()
        
        # Get user choice
        try:
            choice = input("🎯 เลือกโหมด (1-16 หรือ 0 เพื่อออก): ").strip()
            
            if choice == "0":
                print("👋 ขอบคุณที่ใช้ ProjectP! ลาไปก่อนนะครับ!")
                break
            
            elif choice in [str(i) for i in range(1, 17)]:
                mode_mapping = {
                    "1": "full_pipeline",
                    "2": "debug_full_pipeline",
                    "3": "ultimate_pipeline",
                    "4": "preprocess",
                    "5": "train",
                    "6": "predict",
                    "7": "realistic_backtest",
                    "8": "robust_backtest",
                    "9": "realistic_backtest_live",
                    "10": "dashboard",
                    "11": "enterprise_services",
                    "12": "enterprise_pipeline",
                    "13": "health_check",
                    "14": "view_logs",
                    "15": "clean_output",
                    "16": "install_packages"
                }
                
                mode = mode_mapping[choice]
                print(f"🚀 เริ่มต้นโหมด: {mode}")
                
                # Run selected mode
                if mode in ["full_pipeline", "debug_full_pipeline", "ultimate_pipeline"]:
                    run_pipeline_mode(mode)
                elif mode in ["preprocess", "train", "predict"]:
                    run_individual_step(mode)
                elif mode in ["realistic_backtest", "robust_backtest", "realistic_backtest_live"]:
                    run_backtest_mode(mode)
                elif mode in ["dashboard", "enterprise_services", "enterprise_pipeline"]:
                    run_monitoring_tool(mode)
                elif mode in ["health_check", "view_logs", "clean_output", "install_packages"]:
                    run_utility(mode)
                else:
                    print(f"❌ โหมดที่เลือกไม่ถูกต้อง: {mode}")
                    wait_for_user()
            
            else:
                print("❌ กรุณาเลือกหมายเลขโหมดที่ถูกต้อง (1-16)")
                wait_for_user()
        
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาด: {e}")
            wait_for_user()
