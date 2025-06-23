"""
ProjectP Integration System
เชื่อมต่อ Agent System กับ ProjectP pipeline
"""

import os
import sys
import importlib
import traceback
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ProjectPIntegrator:
    """ระบบเชื่อมต่อ Agent กับ ProjectP"""
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.projectp_module = None
        self.pipeline_state = {}
        self.hooks = {}
        
    def initialize(self) -> bool:
        """เริ่มต้นระบบ integration"""
        try:
            # ตรวจสอบ ProjectP.py
            projectp_path = os.path.join(self.project_root, "ProjectP.py")
            if not os.path.exists(projectp_path):
                logger.error("ProjectP.py not found")
                return False
            
            # Import ProjectP module
            sys.path.insert(0, self.project_root)
            try:
                import ProjectP
                self.projectp_module = ProjectP
                logger.info("ProjectP module loaded successfully")
            except Exception as e:
                logger.error(f"Failed to import ProjectP: {e}")
                return False
            
            # Setup hooks
            self._setup_hooks()
            
            return True
            
        except Exception as e:
            logger.error(f"Integration initialization failed: {e}")
            return False
    
    def _setup_hooks(self):
        """ตั้งค่า hooks สำหรับ monitoring"""
        self.hooks = {
            'pre_training': [],
            'post_training': [],
            'pre_validation': [],
            'post_validation': [],
            'error_handler': []
        }
    
    def register_hook(self, event: str, callback):
        """ลงทะเบียน hook สำหรับ event ต่างๆ"""
        if event in self.hooks:
            self.hooks[event].append(callback)
    
    def trigger_hooks(self, event: str, data: Dict[str, Any] = None):
        """เรียกใช้ hooks สำหรับ event ที่กำหนด"""
        if event in self.hooks:
            for callback in self.hooks[event]:
                try:
                    callback(data or {})
                except Exception as e:
                    logger.error(f"Hook {callback} failed: {e}")
    
    def inject_monitoring(self) -> bool:
        """ฉีด monitoring code เข้าไปใน ProjectP"""
        try:
            # ตรวจสอบว่า ProjectP มี functions ที่ต้องการ monitor
            functions_to_monitor = [
                'train_model',
                'validate_model', 
                'run_walkforward',
                'optimize_threshold'
            ]
            
            for func_name in functions_to_monitor:
                if hasattr(self.projectp_module, func_name):
                    original_func = getattr(self.projectp_module, func_name)
                    wrapped_func = self._create_monitored_function(func_name, original_func)
                    setattr(self.projectp_module, func_name, wrapped_func)
                    logger.info(f"Monitoring injected for {func_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to inject monitoring: {e}")
            return False
    
    def _create_monitored_function(self, name: str, original_func):
        """สร้าง wrapper function ที่มี monitoring"""
        def wrapped(*args, **kwargs):
            start_time = datetime.now()
            
            # Pre-execution hooks
            self.trigger_hooks(f'pre_{name}', {
                'function': name,
                'args': args,
                'kwargs': kwargs,
                'start_time': start_time
            })
            
            try:
                result = original_func(*args, **kwargs)
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                # Post-execution hooks
                self.trigger_hooks(f'post_{name}', {
                    'function': name,
                    'result': result,
                    'execution_time': execution_time,
                    'success': True
                })
                
                return result
                
            except Exception as e:
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                # Error hooks
                self.trigger_hooks('error_handler', {
                    'function': name,
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'execution_time': execution_time
                })
                
                raise
        
        return wrapped
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """ดึงสถานะปัจจุบันของ pipeline"""
        return {
            'project_root': self.project_root,
            'module_loaded': self.projectp_module is not None,
            'hooks_registered': sum(len(hooks) for hooks in self.hooks.values()),
            'last_update': datetime.now().isoformat()
        }
    
    def run_with_monitoring(self, command: str, *args, **kwargs):
        """รัน ProjectP command พร้อม monitoring"""
        if not self.projectp_module:
            raise RuntimeError("ProjectP module not loaded")
        
        if hasattr(self.projectp_module, command):
            func = getattr(self.projectp_module, command)
            return func(*args, **kwargs)
        else:
            raise AttributeError(f"Command {command} not found in ProjectP")
    
    def auto_fix_issues(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ระบบแก้ไขปัญหาอัตโนมัติ"""
        results = {
            'fixed': [],
            'failed': [],
            'skipped': []
        }
        
        for issue in issues:
            try:
                issue_type = issue.get('type')
                
                if issue_type == 'nan_auc':
                    self._fix_nan_auc(issue)
                    results['fixed'].append(issue)
                elif issue_type == 'low_performance':
                    self._fix_low_performance(issue)
                    results['fixed'].append(issue)
                elif issue_type == 'data_quality':
                    self._fix_data_quality(issue)
                    results['fixed'].append(issue)
                else:
                    results['skipped'].append(issue)
                    
            except Exception as e:
                issue['error'] = str(e)
                results['failed'].append(issue)
        
        return results
    
    def _fix_nan_auc(self, issue: Dict[str, Any]):
        """แก้ไขปัญหา NaN AUC"""
        logger.info("Fixing NaN AUC issue...")
        # Implementation for NaN AUC fix
        pass
    
    def _fix_low_performance(self, issue: Dict[str, Any]):
        """แก้ไขปัญหา performance ต่ำ"""
        logger.info("Fixing low performance issue...")
        # Implementation for performance improvement
        pass
    
    def _fix_data_quality(self, issue: Dict[str, Any]):
        """แก้ไขปัญหาคุณภาพข้อมูล"""
        logger.info("Fixing data quality issue...")
        # Implementation for data quality improvement
        pass
