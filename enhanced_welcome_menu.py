#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 ENHANCED WELCOME MENU FOR NICEGOLD ProjectP
เมนูต้อนรับที่สวยงามและทันสมัย
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Rich imports with fallback
RICH_AVAILABLE = False
try:
    from rich import box
    from rich.align import Align
    from rich.columns import Columns
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    pass


class EnhancedWelcomeMenu:
    """เมนูต้อนรับที่สวยงามสำหรับ NICEGOLD ProjectP"""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.project_info = {
            "name": "NICEGOLD ProjectP",
            "version": "v2.1",
            "subtitle": "Professional AI Trading System",
            "description": "Enterprise-Grade Gold Trading with Advanced ML",
            "author": "NICEGOLD Enterprise",
            "date": "June 25, 2025"
        }
    
    def show_welcome_screen(self):
        """แสดงหน้าจอต้อนรับแบบสวยงาม"""
        if RICH_AVAILABLE:
            self._show_rich_welcome()
        else:
            self._show_fallback_welcome()
    
    def _show_rich_welcome(self):
        """แสดงเมนูต้อนรับด้วย Rich UI"""
        console = self.console
        console.clear()
        
        # หัวข้อหลัก
        title_text = Text()
        title_text.append("🏆 ", style="gold1")
        title_text.append("NICEGOLD ProjectP ", style="bold bright_magenta")
        title_text.append("v2.1", style="bold bright_cyan")
        
        subtitle_text = Text()
        subtitle_text.append("✨ Professional AI Trading System ✨", style="bright_green")
        
        description_text = Text()
        description_text.append("🚀 Enterprise-Grade Gold Trading with Advanced ML", style="cyan")
        
        # สร้าง Header Panel
        header_content = Align.center(
            Text.assemble(
                title_text, "\n",
                subtitle_text, "\n",
                description_text
            )
        )
        
        header_panel = Panel(
            header_content,
            title="🎯 Welcome to NICEGOLD",
            border_style="bright_magenta",
            box=box.DOUBLE,
            padding=(1, 2)
        )
        
        console.print(header_panel)
        console.print()
        
        # ข้อมูลระบบ
        system_table = Table(title="📊 System Information", box=box.ROUNDED)
        system_table.add_column("Component", style="cyan", width=20)
        system_table.add_column("Status", style="white", width=30)
        system_table.add_column("Details", style="green", width=25)
        
        # ตรวจสอบระบบต่างๆ
        system_info = self._check_system_status()
        
        for component, status, details in system_info:
            status_icon = "✅" if "Available" in status else "❌"
            system_table.add_row(
                component,
                f"{status_icon} {status}",
                details
            )
        
        console.print(system_table)
        console.print()
        
        # เมนูหลัก
        menu_table = Table(title="🎯 Main Menu Options", box=box.HEAVY_EDGE)
        menu_table.add_column("Option", style="bold yellow", width=8)
        menu_table.add_column("Description", style="white", width=35)
        menu_table.add_column("Features", style="cyan", width=30)
        
        menu_options = [
            ("1", "🚀 Full Pipeline", "Complete ML Trading Pipeline with Progress"),
            ("2", "📊 Data Analysis", "Advanced Market Data Analysis & Insights"),
            ("3", "🔧 Quick Test", "System Health & Component Testing"),
            ("4", "🩺 Health Check", "Comprehensive System Diagnostics"),
            ("5", "📦 Dependencies", "Install & Manage Required Packages"),
            ("6", "🧹 Clean System", "Remove Cache & Temporary Files"),
            ("7", "⚡ Performance", "Monitor System Performance & Resources"),
            ("0", "👋 Exit", "Graceful Shutdown with Cleanup")
        ]
        
        for option, desc, features in menu_options:
            if option == "1":  # Highlight Full Pipeline
                menu_table.add_row(
                    f"[bold bright_green]{option}[/bold bright_green]",
                    f"[bold bright_green]{desc}[/bold bright_green]",
                    f"[bold bright_green]{features}[/bold bright_green]"
                )
            else:
                menu_table.add_row(option, desc, features)
        
        console.print(menu_table)
        console.print()
        
        # ข้อมูลเพิ่มเติม
        info_panels = [
            Panel(
                "[bold cyan]🎯 Full Pipeline Features[/bold cyan]\n"
                "• Production-Ready ML Models\n"
                "• Real-time Progress Tracking\n"
                "• Rich Visual Feedback\n"
                "• Thai Language Support\n"
                "• HTML Dashboard Generation",
                title="Pipeline Info",
                border_style="cyan"
            ),
            Panel(
                "[bold yellow]⚡ Performance Features[/bold yellow]\n"
                "• Resource Monitoring\n"
                "• Memory Management\n"
                "• CPU Usage Tracking\n"
                "• Graceful Shutdown\n"
                "• Auto-cleanup System",
                title="Performance",
                border_style="yellow"
            ),
            Panel(
                "[bold green]🔧 System Status[/bold green]\n"
                f"• Python: {sys.version.split()[0]}\n"
                f"• Platform: {sys.platform}\n"
                f"• Time: {datetime.now().strftime('%H:%M:%S')}\n"
                f"• Directory: {os.path.basename(os.getcwd())}\n"
                f"• Rich UI: {'✅' if RICH_AVAILABLE else '❌'}",
                title="Environment",
                border_style="green"
            )
        ]
        
        console.print(Columns(info_panels, equal=True, expand=True))
        console.print()
        
        # คำแนะนำ
        tip_panel = Panel(
            "[bold bright_yellow]💡 Quick Start Tips[/bold bright_yellow]\n"
            "🔸 Start with option [bold]1[/bold] for Full Pipeline experience\n"
            "🔸 Use option [bold]4[/bold] to check system health first\n"
            "🔸 Install dependencies with option [bold]5[/bold] if needed\n"
            "🔸 Monitor performance with option [bold]7[/bold] during execution",
            title="💡 Tips",
            border_style="bright_yellow"
        )
        
        console.print(tip_panel)
    
    def _show_fallback_welcome(self):
        """แสดงเมนูต้อนรับแบบ fallback (ไม่มี Rich)"""
        os.system('clear' if os.name != 'nt' else 'cls')
        
        print("="*80)
        print("🏆 NICEGOLD ProjectP v2.1 - Professional AI Trading System")
        print("="*80)
        print("✨ Enterprise-Grade Gold Trading with Advanced ML ✨")
        print("🚀 Author: NICEGOLD Enterprise | Date: June 25, 2025")
        print("="*80)
        print()
        
        print("📊 SYSTEM INFORMATION:")
        print("-"*40)
        system_info = self._check_system_status()
        for component, status, details in system_info:
            status_icon = "✅" if "Available" in status else "❌"
            print(f"  {status_icon} {component:15} | {status:20} | {details}")
        print()
        
        print("🎯 MAIN MENU OPTIONS:")
        print("-"*40)
        menu_options = [
            ("1", "🚀 Full Pipeline", "Complete ML Trading Pipeline"),
            ("2", "📊 Data Analysis", "Market Data Analysis"),
            ("3", "🔧 Quick Test", "System Testing"),
            ("4", "🩺 Health Check", "System Diagnostics"),
            ("5", "📦 Dependencies", "Package Management"),
            ("6", "🧹 Clean System", "System Cleanup"),
            ("7", "⚡ Performance", "Performance Monitor"),
            ("0", "👋 Exit", "Graceful Shutdown")
        ]
        
        for option, desc, features in menu_options:
            if option == "1":
                print(f"  >> {option}. {desc:20} - {features} << RECOMMENDED")
            else:
                print(f"     {option}. {desc:20} - {features}")
        print()
        
        print("💡 QUICK START TIPS:")
        print("-"*40)
        print("  🔸 Start with option 1 for Full Pipeline experience")
        print("  🔸 Check system health with option 4 first")
        print("  🔸 Install dependencies with option 5 if needed")
        print("="*80)
    
    def _check_system_status(self):
        """ตรวจสอบสถานะระบบต่างๆ"""
        system_info = []
        
        # ตรวจสอบ Rich
        rich_status = "Available" if RICH_AVAILABLE else "Not Available"
        rich_details = "Enhanced UI" if RICH_AVAILABLE else "Fallback UI"
        system_info.append(("Rich Library", rich_status, rich_details))
        
        # ตรวจสอบ Core modules
        try:
            from core.menu_operations import MenuOperations
            core_status = "Available"
            core_details = "Full Features"
        except ImportError:
            core_status = "Not Available"
            core_details = "Limited Features"
        system_info.append(("Core Modules", core_status, core_details))
        
        # ตรวจสอบ Enhanced Progress
        try:
            from utils.enhanced_progress import EnhancedProgressProcessor
            progress_status = "Available"
            progress_details = "Beautiful Progress"
        except ImportError:
            progress_status = "Not Available"
            progress_details = "Basic Progress"
        system_info.append(("Progress System", progress_status, progress_details))
        
        # ตรวจสอบ Production Pipeline
        try:
            from production_full_pipeline import ProductionFullPipeline
            production_status = "Available"
            production_details = "Production Ready"
        except ImportError:
            production_status = "Not Available"
            production_details = "Demo Mode"
        system_info.append(("Production Pipeline", production_status, production_details))
        
        # ตรวจสอบ ML Libraries
        ml_libs = []
        for lib in ['pandas', 'numpy', 'scikit-learn', 'xgboost', 'lightgbm']:
            try:
                __import__(lib)
                ml_libs.append(lib)
            except ImportError:
                pass
        
        ml_status = "Available" if len(ml_libs) > 3 else "Partial" if ml_libs else "Not Available"
        ml_details = f"{len(ml_libs)}/5 Libraries"
        system_info.append(("ML Libraries", ml_status, ml_details))
        
        return system_info
    
    def show_loading_animation(self, message="Loading", duration=2):
        """แสดง loading animation"""
        if RICH_AVAILABLE:
            console = self.console
            with Progress(
                SpinnerColumn(spinner_style="cyan"),
                TextColumn(f"[bold blue]{message}..."),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("loading", total=None)
                time.sleep(duration)
        else:
            # Fallback animation
            chars = "|/-\\"
            for i in range(duration * 4):
                print(f"\r{message}... {chars[i % len(chars)]}", end="", flush=True)
                time.sleep(0.25)
            print(f"\r{message}... ✓")


def main():
    """ฟังก์ชันหลักสำหรับทดสอบเมนู"""
    welcome_menu = EnhancedWelcomeMenu()
    welcome_menu.show_welcome_screen()
    
    # แสดง loading animation
    print()
    welcome_menu.show_loading_animation("Initializing ProjectP", 1)
    print()
    
    # แสดงข้อความสรุป
    if RICH_AVAILABLE:
        console = Console()
        console.print(Panel(
            "[bold green]🎉 Welcome menu displayed successfully![/bold green]\n"
            "[cyan]Ready to start your NICEGOLD ProjectP experience![/cyan]",
            title="Status",
            border_style="green"
        ))
    else:
        print("🎉 Welcome menu displayed successfully!")
        print("Ready to start your NICEGOLD ProjectP experience!")


if __name__ == "__main__":
    main()
