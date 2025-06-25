from pathlib import Path
from typing import Dict, Any, Optional
            import colorama
            import GPUtil
            import imblearn
import multiprocessing
import os
            import psutil
            import sklearn
import sys
            import tensorflow as tf
            import torch
import warnings
"""
🔧 Configuration Manager
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

จัดการการตั้งค่าระบบทั้งหมด รวมถึง:
- Environment variables
- Package availability checks
- Resource limits
- Warning filters
"""


class ConfigManager:
    """จัดการการตั้งค่าระบบทั้งหมด"""

    def __init__(self):
        self.config = {}
        self.package_availability = {}
        self.resource_limits = {}
        self._setup_environment()
        self._check_packages()
        self._setup_warnings()
        self._setup_performance()

    def _setup_environment(self) -> None:
        """ตั้งค่า environment variables"""
        # Console encoding
        if sys.platform.startswith('win'):
            os.environ['PYTHONIOENCODING'] = 'utf - 8'
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding = 'utf - 8', errors = 'ignore')
                sys.stderr.reconfigure(encoding = 'utf - 8', errors = 'ignore')
        else:
            os.environ['PYTHONIOENCODING'] = 'utf - 8'

        # Framework settings
        os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['PYTHONHASHSEED'] = '42'

    def _check_packages(self) -> None:
        """ตรวจสอบ package availability"""
        packages_to_check = {
            'colorama': self._check_colorama, 
            'psutil': self._check_psutil, 
            'GPUtil': self._check_gputil, 
            'sklearn': self._check_sklearn, 
            'tensorflow': self._check_tensorflow, 
            'torch': self._check_torch, 
            'imbalanced': self._check_imbalanced
        }

        for name, checker in packages_to_check.items():
            self.package_availability[name] = checker()

    def _check_colorama(self) -> bool:
        try:
            colorama.init(autoreset = True)
            return True
        except ImportError:
            return False

    def _check_psutil(self) -> bool:
        try:
            return True
        except ImportError:
            return False

    def _check_gputil(self) -> bool:
        try:
            return True
        except ImportError:
            return False

    def _check_sklearn(self) -> bool:
        try:
            return True
        except ImportError:
            return False

    def _check_tensorflow(self) -> bool:
        try:
            # Setup GPU if available
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def _check_torch(self) -> bool:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def _check_imbalanced(self) -> bool:
        try:
            return True
        except ImportError:
            return False

    def _setup_warnings(self) -> None:
        """ตั้งค่า warning filters"""
        warnings.filterwarnings("ignore", category = UserWarning)
        warnings.filterwarnings("ignore", category = FutureWarning)
        warnings.filterwarnings("ignore", category = DeprecationWarning)
        warnings.filterwarnings("ignore", message = "Skipped unsupported reflection of expression - based index")
        warnings.filterwarnings("ignore", message = "Hint: Inferred schema contains integer column")
        warnings.filterwarnings("ignore", message = ".*does not have many workers.*")
        warnings.filterwarnings("ignore", message = ".*Trying to estimate the number of CPU cores.*")

    def _setup_performance(self) -> None:
        """ตั้งค่าประสิทธิภาพ"""
        num_cores = multiprocessing.cpu_count()
        blas_env_vars = [
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", 
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"
        ]

        for env_var in blas_env_vars:
            os.environ[env_var] = str(num_cores)

        self.config['num_cores'] = num_cores

    def get_resource_limits(self) -> Dict[str, float]:
        """รับค่า resource limits"""
        return {
            'sys_mem_target_gb': 25.0, 
            'gpu_mem_target_gb': 12.0, 
            'disk_usage_limit': 0.8
        }

    def is_package_available(self, package_name: str) -> bool:
        """ตรวจสอบว่า package พร้อมใช้งานหรือไม่"""
        return self.package_availability.get(package_name, False)

    def get_config(self, key: str, default: Any = None) -> Any:
        """รับค่า config"""
        return self.config.get(key, default)


# Singleton instance
config_manager = ConfigManager()