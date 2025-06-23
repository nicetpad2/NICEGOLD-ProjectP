"""
📊 Resource Monitor
==================

ตรวจสอบและติดตาม system resources:
- Memory usage
- GPU memory
- Disk usage
- Performance metrics
"""

import shutil
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.core.config import config_manager


class ResourceMonitor:
    """ตรวจสอบและติดตาม system resources"""
    
    def __init__(self):
        self.limits = config_manager.get_resource_limits()
        self.warnings: List[str] = []
    
    def check_system_memory(self) -> Dict[str, Any]:
        """ตรวจสอบ system memory"""
        if not config_manager.is_package_available('psutil'):
            return {
                "available": False,
                "message": "psutil not available"
            }
        
        try:
            import psutil
            vm = psutil.virtual_memory()
            used_gb = vm.used / 1e9
            total_gb = vm.total / 1e9
            
            result = {
                "available": True,
                "used_gb": round(used_gb, 2),
                "total_gb": round(total_gb, 2),
                "percent": round(vm.percent, 1)
            }
            
            if used_gb > self.limits['sys_mem_target_gb']:
                warning = f"RAM usage {used_gb:.1f}GB exceeds target {self.limits['sys_mem_target_gb']}GB"
                self.warnings.append(warning)
                result["warning"] = warning
            
            return result
            
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }
    
    def check_gpu_memory(self) -> Dict[str, Any]:
        """ตรวจสอบ GPU memory"""
        if not config_manager.is_package_available('GPUtil'):
            return {
                "available": False,
                "message": "GPUtil not available"
            }
        
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            gpu_info = []
            
            for gpu in gpus:
                used_gb = gpu.memoryUsed / 1024
                total_gb = gpu.memoryTotal / 1024
                
                gpu_data = {
                    "id": gpu.id,
                    "name": gpu.name,
                    "used_gb": round(used_gb, 2),
                    "total_gb": round(total_gb, 2),
                    "percent": round(gpu.memoryUtil * 100, 1)
                }
                
                if used_gb > self.limits['gpu_mem_target_gb']:
                    warning = f"GPU {gpu.id} usage {used_gb:.1f}GB exceeds target {self.limits['gpu_mem_target_gb']}GB"
                    self.warnings.append(warning)
                    gpu_data["warning"] = warning
                
                gpu_info.append(gpu_data)
            
            return {
                "available": True,
                "gpus": gpu_info
            }
            
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }
    
    def check_disk_usage(self) -> Dict[str, Any]:
        """ตรวจสอบ disk usage"""
        try:
            du = shutil.disk_usage(os.getcwd())
            used_pct = du.used / du.total
            
            result = {
                "available": True,
                "used_gb": round(du.used / 1e9, 2),
                "total_gb": round(du.total / 1e9, 2),
                "percent": round(used_pct * 100, 1)
            }
            
            if used_pct > self.limits['disk_usage_limit']:
                warning = f"Disk usage {used_pct*100:.1f}% exceeds limit {self.limits['disk_usage_limit']*100:.1f}%"
                self.warnings.append(warning)
                result["warning"] = warning
            
            return result
            
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """รับสถานะทรัพยากรทั้งหมด"""
        self.warnings.clear()  # Clear previous warnings
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "system_memory": self.check_system_memory(),
            "gpu_memory": self.check_gpu_memory(),
            "disk_usage": self.check_disk_usage(),
            "warnings": self.warnings.copy()
        }
        
        return status
    
    def print_status(self) -> None:
        """แสดงสถานะทรัพยากร"""
        status = self.get_comprehensive_status()
        
        print("🖥️ SYSTEM RESOURCES STATUS")
        print("=" * 40)
        
        # System Memory
        mem = status["system_memory"]
        if mem["available"]:
            print(f"💾 RAM: {mem['used_gb']:.1f}/{mem['total_gb']:.1f} GB ({mem['percent']:.1f}%)")
            if "warning" in mem:
                print(f"   ⚠️ {mem['warning']}")
        else:
            print(f"💾 RAM: Not available - {mem.get('message', mem.get('error', 'Unknown'))}")
        
        # GPU Memory
        gpu = status["gpu_memory"]
        if gpu["available"]:
            for gpu_info in gpu["gpus"]:
                print(f"🎮 GPU {gpu_info['id']}: {gpu_info['used_gb']:.1f}/{gpu_info['total_gb']:.1f} GB ({gpu_info['percent']:.1f}%)")
                if "warning" in gpu_info:
                    print(f"   ⚠️ {gpu_info['warning']}")
        else:
            print(f"🎮 GPU: Not available - {gpu.get('message', gpu.get('error', 'Unknown'))}")
        
        # Disk Usage
        disk = status["disk_usage"]
        if disk["available"]:
            print(f"💿 Disk: {disk['used_gb']:.1f}/{disk['total_gb']:.1f} GB ({disk['percent']:.1f}%)")
            if "warning" in disk:
                print(f"   ⚠️ {disk['warning']}")
        else:
            print(f"💿 Disk: Not available - {disk.get('error', 'Unknown')}")
        
        if status["warnings"]:
            print(f"\n⚠️  Total warnings: {len(status['warnings'])}")


# Singleton instance
resource_monitor = ResourceMonitor()
