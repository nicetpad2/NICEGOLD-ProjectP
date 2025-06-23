"""
🖨️ Banner และ Display Utils
===========================

จัดการการแสดงผล banner, progress และ UI elements
"""

import socket
import getpass
import sys
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.core.config import config_manager


class BannerManager:
    """จัดการการแสดง banner และ system info"""
    
    def __init__(self):
        self.has_colorama = config_manager.is_package_available('colorama')
        if self.has_colorama:
            from colorama import Fore, Style
            self.Fore = Fore
            self.Style = Style
        else:
            # Fallback classes
            class MockFore:
                RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = ""
            class MockStyle:
                RESET_ALL = ""
            self.Fore = MockFore()
            self.Style = MockStyle()
    
    def print_main_banner(self) -> None:
        """แสดง banner หลัก"""
        print(f"{self.Fore.CYAN}{'='*80}")
        print(f"{self.Fore.CYAN}🚀 NICEGOLD PROFESSIONAL TRADING SYSTEM v3.0 - ULTIMATE EDITION")
        print(f"{self.Fore.CYAN}🎯 COMPLETE SOLUTION FOR EXTREME CLASS IMBALANCE (201.7:1)")
        print(f"{self.Fore.CYAN}{'='*80}")
        print(f"{self.Fore.GREEN}🖥️  System: {socket.gethostname()}")
        print(f"{self.Fore.GREEN}👤 User: {getpass.getuser()}")
        print(f"{self.Fore.GREEN}🐍 Python: {sys.version.split()[0]}")
        print(f"{self.Fore.GREEN}🆔 Session ID: {str(uuid.uuid4())[:8]}")
        print(f"{self.Fore.GREEN}📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show package status
        self._show_package_status()
        
        print(f"{self.Fore.CYAN}{'='*80}{self.Style.RESET_ALL}")
    
    def _show_package_status(self) -> None:
        """แสดงสถานะ packages"""
        packages_status = []
        package_map = {
            'sklearn': 'Sklearn',
            'tensorflow': 'TensorFlow', 
            'torch': 'PyTorch',
            'imbalanced': 'Imbalanced-learn',
            'psutil': 'Psutil',
            'GPUtil': 'GPUtil'
        }
        
        for key, name in package_map.items():
            available = config_manager.is_package_available(key)
            status = '✅' if available else '❌'
            packages_status.append(f"{name}: {status}")
        
        print(f"{self.Fore.YELLOW}📦 Packages: {' | '.join(packages_status)}")
    
    def print_mode_banner(self, mode_name: str, description: str = "") -> None:
        """แสดง banner สำหรับ mode เฉพาะ"""
        print(f"\n{self.Fore.CYAN}{'='*60}")
        print(f"{self.Fore.CYAN}🚀 {mode_name.upper()} MODE")
        if description:
            print(f"{self.Fore.CYAN}{description}")
        print(f"{self.Fore.CYAN}{'='*60}{self.Style.RESET_ALL}")
    
    def print_phase_header(self, phase_num: int, phase_name: str) -> None:
        """แสดง header ของ phase"""
        print(f"\n{self.Fore.CYAN}🔧 Phase {phase_num}: {phase_name}...{self.Style.RESET_ALL}")
    
    def print_success(self, message: str) -> None:
        """แสดงข้อความสำเร็จ"""
        print(f"{self.Fore.GREEN}✅ {message}{self.Style.RESET_ALL}")
    
    def print_warning(self, message: str) -> None:
        """แสดงข้อความเตือน"""
        print(f"{self.Fore.YELLOW}⚠️ {message}{self.Style.RESET_ALL}")
    
    def print_error(self, message: str) -> None:
        """แสดงข้อความข้อผิดพลาด"""
        print(f"{self.Fore.RED}❌ {message}{self.Style.RESET_ALL}")
    
    def print_info(self, message: str) -> None:
        """แสดงข้อความสารสนเทศ"""
        print(f"{self.Fore.BLUE}ℹ️ {message}{self.Style.RESET_ALL}")
    
    def print_professional_banner(self) -> None:
        """แสดง banner แบบมืออาชีพ (alias สำหรับ print_main_banner)"""
        self.print_main_banner()
    
    def print_section_header(self, title: str, icon: str = "📋") -> None:
        """แสดง section header"""
        print(f"\n{self.Fore.CYAN}{icon} {title}")
        print(f"{self.Fore.CYAN}{'-' * (len(title) + 3)}{self.Style.RESET_ALL}")


class ProgressDisplay:
    """จัดการการแสดง progress และ order status"""
    
    def __init__(self):
        self.has_colorama = config_manager.is_package_available('colorama')
        if self.has_colorama:
            from colorama import Fore, Style
            self.Fore = Fore
            self.Style = Style
        else:
            class MockFore:
                RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = ""
            class MockStyle:
                RESET_ALL = ""
            self.Fore = MockFore()
            self.Style = MockStyle()
    
    def show_order_progress(self, order: Dict[str, Any]) -> None:
        """แสดง order progress"""
        try:
            status = str(order.get("status", "UNKNOWN")).upper()
            filled = float(order.get("filled_qty", 0))
            qty = float(order.get("qty", 1))
            
            if qty <= 0:
                qty = 1  # Avoid division by zero
                
            progress = min(filled / qty, 1.0)  # Cap at 100%
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            color = self.Fore.GREEN if progress >= 1.0 else self.Fore.YELLOW if progress >= 0.5 else self.Fore.RED
            print(f"{color}Order Progress: [{bar}] {progress*100:.1f}% ({filled:.2f}/{qty:.2f}) - {status}{self.Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{self.Fore.RED}❌ Order progress display failed: {e}{self.Style.RESET_ALL}")
    
    def show_pipeline_progress(self, stage: str, progress: float, total_stages: int, current_stage: int) -> None:
        """แสดง pipeline progress"""
        try:
            bar_length = 40
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            print(f"\n📊 Pipeline Progress:")
            print(f"   Stage: {stage} ({current_stage}/{total_stages})")
            print(f"   [{bar}] {progress*100:.1f}%")
            
        except Exception as e:
            print(f"❌ Pipeline progress display failed: {e}")


# Singleton instances
banner_manager = BannerManager()
progress_display = ProgressDisplay()
