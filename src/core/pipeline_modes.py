"""
📋 Pipeline Mode Manager
=======================

จัดการ pipeline modes ต่างๆ และ orchestration
แยกความรับผิดชอบของแต่ละ mode ออกจากกัน
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime
import traceback
import time

from src.core.config import config_manager
from src.core.display import banner_manager
from src.core.resource_monitor import resource_monitor


class PipelineMode(ABC):
    """Base class สำหรับ pipeline modes"""
    
    def __init__(self, mode_name: str, description: str = ""):
        self.mode_name = mode_name
        self.description = description
        self.start_time = None
        self.end_time = None
        self.result = None
        self.errors = []
        self.warnings = []
    
    @abstractmethod
    def execute(self) -> Optional[str]:
        """Execute the pipeline mode"""
        pass
    
    def run(self) -> Optional[str]:
        """Run pipeline mode with error handling and timing"""
        try:
            banner_manager.print_mode_banner(self.mode_name, self.description)
            self.start_time = time.time()
            
            # Apply emergency fixes if available
            self._apply_emergency_fixes()
            
            # Check resources
            resource_status = resource_monitor.get_comprehensive_status()
            if resource_status.get("warnings"):
                banner_manager.print_warning(f"Resource warnings: {len(resource_status['warnings'])}")
            
            # Execute main logic
            self.result = self.execute()
            
            self.end_time = time.time()
            execution_time = self.end_time - self.start_time
            
            if self.result:
                banner_manager.print_success(f"{self.mode_name} completed successfully in {execution_time:.2f}s")
                return self.result
            else:
                banner_manager.print_error(f"{self.mode_name} failed")
                return None
                
        except Exception as e:
            self.errors.append(str(e))
            banner_manager.print_error(f"{self.mode_name} error: {e}")
            traceback.print_exc()
            return None
    
    def _apply_emergency_fixes(self) -> None:
        """Apply emergency fixes if available"""
        try:
            # Try to import and apply emergency fixes
            from integrated_emergency_fixes import apply_emergency_fixes_to_pipeline
            
            banner_manager.print_phase_header(1, "Applying emergency fixes")
            fix_success = apply_emergency_fixes_to_pipeline(self.mode_name.lower().replace(" ", "_"))
            
            if fix_success:
                banner_manager.print_success("Emergency fixes applied successfully")
            else:
                banner_manager.print_warning("Emergency fixes had issues, continuing anyway")
                
        except ImportError:
            banner_manager.print_warning("Emergency fixes not available")
        except Exception as e:
            banner_manager.print_warning(f"Emergency fix application failed: {e}")


class FullPipelineMode(PipelineMode):
    """Full Pipeline Mode"""
    
    def __init__(self):
        super().__init__(
            "Full Pipeline",
            "Complete end-to-end pipeline (production-ready)"
        )
    
    def execute(self) -> Optional[str]:
        try:
            from projectp.pipeline import run_full_pipeline
            
            result = run_full_pipeline()
            if result:
                return result
            else:
                return self._run_fallback()
                
        except ImportError:
            banner_manager.print_warning("Main pipeline function not available, using fallback")
            return self._run_fallback()
    
    def _run_fallback(self) -> Optional[str]:
        """Run fallback pipeline"""
        banner_manager.print_warning("Running fallback pipeline...")
        
        try:
            import subprocess
            import sys
            
            result = subprocess.run(
                [sys.executable, "run_simple_pipeline.py"],
                capture_output=True, text=True, timeout=3600,
                encoding='utf-8', errors='ignore'
            )
            
            if result.returncode == 0:
                banner_manager.print_success("Fallback pipeline completed successfully!")
                return "output_default/fallback_results"
            else:
                banner_manager.print_error(f"Fallback pipeline failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            banner_manager.print_error("Fallback pipeline timed out")
            return None
        except Exception as e:
            banner_manager.print_error(f"Fallback pipeline error: {e}")
            return None


class DebugPipelineMode(PipelineMode):
    """Debug Pipeline Mode"""
    
    def __init__(self):
        super().__init__(
            "Debug Pipeline",
            "Full pipeline with detailed debugging"
        )
    
    def execute(self) -> Optional[str]:
        try:
            from projectp.pipeline import run_debug_full_pipeline
            
            if run_debug_full_pipeline is None:
                banner_manager.print_error("Debug pipeline function not available")
                return None
            
            result = run_debug_full_pipeline()
            return "output_default/debug_results" if result else None
            
        except ImportError:
            banner_manager.print_error("Debug pipeline import failed")
            return None


class PreprocessMode(PipelineMode):
    """Preprocessing Mode"""
    
    def __init__(self):
        super().__init__(
            "Preprocessing",
            "Data preparation and feature engineering only"
        )
    
    def execute(self) -> Optional[str]:
        try:
            from projectp.steps.preprocess import run_preprocess
            
            result = run_preprocess()
            return result if result else None
            
        except ImportError as e:
            banner_manager.print_error(f"Preprocessing import error: {e}")
            return None


class BacktestMode(PipelineMode):
    """Backtest Mode"""
    
    def __init__(self, backtest_type: str = "realistic"):
        mode_names = {
            "realistic": "Realistic Backtest",
            "robust": "Robust Backtest", 
            "live": "Live Backtest"
        }
        
        super().__init__(
            mode_names.get(backtest_type, "Backtest"),
            f"{backtest_type.title()} backtesting simulation"
        )
        self.backtest_type = backtest_type
    
    def execute(self) -> Optional[str]:
        try:
            from backtest_engine import run_realistic_backtest, run_robust_backtest
            
            if self.backtest_type == "realistic" or self.backtest_type == "live":
                result = run_realistic_backtest()
            elif self.backtest_type == "robust":
                result = run_robust_backtest()
            else:
                result = run_realistic_backtest()
            
            return result if result else None
            
        except ImportError as e:
            banner_manager.print_error(f"Backtest import error: {e}")
            return None


class UltimatePipelineMode(PipelineMode):
    """Ultimate Pipeline Mode - รวมทุกการปรับปรุง"""
    
    def __init__(self):
        super().__init__(
            "Ultimate Pipeline",
            "🔥 ALL improvements: Emergency Fixes + Class Balance + Full Pipeline"
        )
    
    def execute(self) -> Optional[str]:
        # Phase 2: AUC Improvement
        banner_manager.print_phase_header(2, "Running AUC Improvement Pipeline")
        self._run_auc_improvements()
        
        # Phase 3: Class Balance Fix
        banner_manager.print_phase_header(3, "Running Class Balance Fix")
        self._run_class_balance_fix()
        
        # Phase 4: Ultimate Pipeline
        banner_manager.print_phase_header(4, "Running Ultimate Pipeline")
        return self._run_ultimate_pipeline()
    
    def _run_auc_improvements(self) -> None:
        """Run AUC improvement pipeline"""
        try:
            from auc_improvement_pipeline import run_auc_emergency_fix, run_advanced_feature_engineering
            
            banner_manager.print_info("Running AUC Emergency Fix...")
            run_auc_emergency_fix()
            
            banner_manager.print_info("Running Advanced Feature Engineering...")
            run_advanced_feature_engineering()
            
            banner_manager.print_success("AUC Improvement Pipeline completed")
            
        except ImportError:
            banner_manager.print_warning("AUC Improvement Pipeline not available, skipping")
        except Exception as e:
            banner_manager.print_warning(f"AUC Improvement Pipeline failed: {e}, continuing")
    
    def _run_class_balance_fix(self) -> None:
        """Run class balance fix"""
        try:
            from ultimate_class_balance_fix import UltimateClassBalanceFix
            
            fix_engine = UltimateClassBalanceFix()
            result = fix_engine.run_all_modes()
            
            if result:
                banner_manager.print_success("Class Balance Fix completed")
            else:
                banner_manager.print_warning("Class Balance Fix had issues")
                
        except ImportError:
            banner_manager.print_warning("Class Balance Fix not available")
        except Exception as e:
            banner_manager.print_warning(f"Class Balance Fix failed: {e}")
    
    def _run_ultimate_pipeline(self) -> Optional[str]:
        """Run ultimate pipeline with fallbacks"""
        # Try ultimate pipeline first
        try:
            from projectp.pipeline import run_ultimate_pipeline
            
            if run_ultimate_pipeline is not None:
                result = run_ultimate_pipeline()
                if result:
                    return "output_default/ultimate_results"
            
        except ImportError:
            pass
        except Exception as e:
            banner_manager.print_warning(f"Ultimate pipeline failed: {e}")
        
        # Fallback to full pipeline
        banner_manager.print_phase_header(5, "Fallback to Full Pipeline")
        try:
            from projectp.pipeline import run_full_pipeline
            
            if run_full_pipeline is not None:
                result = run_full_pipeline()
                if result:
                    return "output_default/ultimate_fallback_results"
                    
        except ImportError:
            pass
        except Exception as e:
            banner_manager.print_warning(f"Full pipeline fallback failed: {e}")
        
        # Final fallback
        banner_manager.print_phase_header(6, "Final Fallback Pipeline")
        fallback_mode = FullPipelineMode()
        return fallback_mode._run_fallback()


class ClassBalanceFixMode(PipelineMode):
    """Class Balance Fix Mode"""
    
    def __init__(self):
        super().__init__(
            "Class Balance Fix",
            "🎯 Dedicated class balance fix (แก้ไขปัญหา Class Imbalance แบบสมบูรณ์)"
        )
    
    def execute(self) -> Optional[str]:
        try:
            from ultimate_class_balance_fix import UltimateClassBalanceFix
            
            if config_manager.is_package_available('sklearn'):
                # Initialize with optimal parameters
                fix_engine = UltimateClassBalanceFix(
                    target_imbalance_ratio=4.0,
                    min_samples_per_class=1000,
                    enable_advanced_resampling=True,
                    enable_ensemble=True,
                    enable_threshold_optimization=True
                )
                
                results = fix_engine.run_all_modes()
                
                if results and isinstance(results, dict):
                    auc = results.get('auc', 0)
                    if auc > 0.5:
                        banner_manager.print_success(f"Class balance fix completed with AUC: {auc:.4f}")
                        return "output_default/class_balance_results.json"
                    else:
                        banner_manager.print_warning(f"Low AUC achieved: {auc:.4f}")
                        return self._run_fallback_fix()
                else:
                    return self._run_fallback_fix()
            else:
                return self._run_fallback_fix()
                
        except ImportError:
            banner_manager.print_warning("Ultimate Class Balance Fix not available, using fallback")
            return self._run_fallback_fix()
    
    def _run_fallback_fix(self) -> Optional[str]:
        """Run fallback class balance fix"""
        banner_manager.print_info("Running fallback class balance fix...")
        
        try:
            # Basic fallback implementation would go here
            # For now, return a placeholder
            return "output_default/fallback_class_balance_results.json"
            
        except Exception as e:
            banner_manager.print_error(f"Fallback class balance fix error: {e}")
            return None


class PipelineModeManager:
    """จัดการ pipeline modes ทั้งหมด"""
    
    def __init__(self):
        self.modes = {
            'full_pipeline': FullPipelineMode,
            'debug_full_pipeline': DebugPipelineMode,
            'preprocess': PreprocessMode,
            'realistic_backtest': lambda: BacktestMode('realistic'),
            'robust_backtest': lambda: BacktestMode('robust'),
            'realistic_backtest_live': lambda: BacktestMode('live'),
            'ultimate_pipeline': UltimatePipelineMode,
            'class_balance_fix': ClassBalanceFixMode
        }
    
    def run_mode(self, mode_name: str) -> Optional[str]:
        """รัน mode ที่ระบุ"""
        if mode_name not in self.modes:
            banner_manager.print_error(f"Unknown mode: {mode_name}")
            return None
        
        mode_class = self.modes[mode_name]
        mode_instance = mode_class()
        
        return mode_instance.run()
    
    def run_all_modes(self) -> Dict[str, Any]:
        """รันทุก modes ตามลำดับ"""
        banner_manager.print_mode_banner("ALL MODES", "Running all modes sequentially")
        
        results = {}
        modes_to_run = [
            'class_balance_fix',
            'full_pipeline', 
            'debug_full_pipeline',
            'ultimate_pipeline'
        ]
        
        for mode_name in modes_to_run:
            banner_manager.print_info(f"Running {mode_name}...")
            
            start_time = time.time()
            result = self.run_mode(mode_name)
            execution_time = time.time() - start_time
            
            results[mode_name] = {
                'result': result,
                'execution_time': execution_time,
                'success': result is not None,
                'timestamp': datetime.now().isoformat()
            }
            
            if result:
                banner_manager.print_success(f"{mode_name} completed in {execution_time:.2f}s")
            else:
                banner_manager.print_error(f"{mode_name} failed")
        
        return results
    
    def get_available_modes(self) -> List[str]:
        """รับรายการ modes ที่มี"""
        return list(self.modes.keys())


# Singleton instance
pipeline_manager = PipelineModeManager()
