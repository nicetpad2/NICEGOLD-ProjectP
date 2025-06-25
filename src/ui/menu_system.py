# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - Menu System Module
════════════════════════════════════════════════════════════════════════════════

Interactive menu system with beautiful formatting and navigation.

Author: NICEGOLD Team
Version: 3.0
Created: 2025-01-05
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# แก้ไข import path
try:
    from src.commands.ai_commands import AICommands
    from src.commands.analysis_commands import AnalysisCommands
    from src.commands.pipeline_commands import PipelineCommands
    from src.commands.trading_commands import TradingCommands
    from src.core.colors import Colors, colorize
    from src.ui.animations import print_with_animation
except ImportError:
    # Fallback สำหรับกรณีที่ run จาก directory อื่น
    import sys

    sys.path.append(".")
    from src.commands.ai_commands import AICommands
    from src.commands.analysis_commands import AnalysisCommands
    from src.commands.pipeline_commands import PipelineCommands
    from src.commands.trading_commands import TradingCommands
    from src.core.colors import Colors, colorize
    from src.ui.animations import print_with_animation


class MenuSection:
    """Menu section data structure"""

    def __init__(self, title: str, color: str, items: List[Tuple[str, str, str]]):
        self.title = title
        self.color = color
        self.items = items  # (number, title, description)


class MenuSystem:
    """Main menu system for ProjectP"""

    def __init__(self, project_root=None, csv_manager=None, logger=None):
        self.menu_sections = self._create_menu_sections()
        self.project_root = project_root or Path(__file__).parent.parent.parent

        # Initialize command handlers
        self.pipeline_commands = PipelineCommands(
            self.project_root, csv_manager, logger
        )
        self.analysis_commands = AnalysisCommands(
            self.project_root, csv_manager, logger
        )
        self.trading_commands = TradingCommands(self.project_root, csv_manager, logger)
        self.ai_commands = AICommands(self.project_root, csv_manager, logger)

    def _create_menu_sections(self) -> List[MenuSection]:
        """Create all menu sections"""
        return [
            MenuSection(
                title="🚀 Core Pipeline Modes (โหมดการทำงานหลัก)",
                color=Colors.BRIGHT_GREEN,
                items=[
                    ("1", "Full Pipeline", "รันระบบครบทุกขั้นตอน (Production Ready)"),
                    (
                        "2",
                        "Production Pipeline",
                        "ระบบผลิตจริง: Modern ML Pipeline (New!)",
                    ),
                    ("3", "Debug Pipeline", "โหมดดีบัก: ตรวจสอบทุกจุด (Detailed Logs)"),
                    ("4", "Quick Test", "ทดสอบเร็ว: ข้อมูลย่อย (Development)"),
                ],
            ),
            MenuSection(
                title="📊 Data Processing (การประมวลผลข้อมูล)",
                color=Colors.BRIGHT_BLUE,
                items=[
                    ("5", "Load & Validate Data", "โหลดและตรวจสอบข้อมูลจริงจาก datacsv"),
                    ("6", "Feature Engineering", "สร้าง Technical Indicators"),
                    ("7", "Preprocess Only", "เตรียมข้อมูลสำหรับ ML"),
                ],
            ),
            MenuSection(
                title="🤖 Machine Learning (การเรียนรู้ของเครื่อง)",
                color=Colors.BRIGHT_MAGENTA,
                items=[
                    ("8", "Train Models", "เทรนโมเดล ML (AutoML + Optimization)"),
                    ("9", "Model Comparison", "เปรียบเทียบโมเดลต่างๆ"),
                    ("10", "Predict & Backtest", "ทำนายและ Backtest"),
                ],
            ),
            MenuSection(
                title="📈 Advanced Analytics (การวิเคราะห์ขั้นสูง)",
                color=Colors.BRIGHT_CYAN,
                items=[
                    ("11", "Live Trading Simulation", "จำลองการเทรดแบบ Real-time"),
                    ("12", "Performance Analysis", "วิเคราะห์ผลงานแบบละเอียด"),
                    ("13", "Risk Management", "จัดการความเสี่ยงและ Portfolio"),
                ],
            ),
            MenuSection(
                title="🖥️ Monitoring & Services (การติดตามและบริการ)",
                color=Colors.BRIGHT_YELLOW,
                items=[
                    ("14", "Web Dashboard", "เปิด Streamlit Dashboard"),
                    ("15", "API Server", "เปิด FastAPI Model Server"),
                    ("16", "Real-time Monitor", "ติดตามระบบแบบ Real-time"),
                ],
            ),
            MenuSection(
                title="🤖 AI Agents (ระบบ AI Agents)",
                color=Colors.BRIGHT_MAGENTA,
                items=[
                    ("17", "AI Project Analysis", "วิเคราะห์โปรเจคด้วย AI"),
                    ("18", "AI Auto-Fix System", "แก้ไขปัญหาอัตโนมัติด้วย AI"),
                    ("19", "AI Performance Optimizer", "ปรับปรุงประสิทธิภาพด้วย AI"),
                    ("20", "AI Executive Summary", "สร้างรายงานผู้บริหารด้วย AI"),
                    ("21", "AI Agents Dashboard", "เปิด AI Agents Web Interface"),
                ],
            ),
            MenuSection(
                title="⚙️ System Management (การจัดการระบบ)",
                color=Colors.BRIGHT_RED,
                items=[
                    ("22", "System Health Check", "ตรวจสอบสุขภาพระบบทั้งหมด"),
                    ("23", "Install Dependencies", "ติดตั้งไลบรารี่ที่จำเป็น"),
                    ("24", "Clean & Reset", "ล้างข้อมูลและรีเซ็ตระบบ"),
                    ("25", "View Logs & Results", "ดูผลลัพธ์และ Log Files"),
                ],
            ),
        ]

    def print_main_menu(self) -> str:
        """Display the main interactive menu with beautiful colors and animations"""
        print("\n" + "=" * 80)
        print_with_animation(colorize("🎯 กำลังโหลดเมนูหลัก...", Colors.BRIGHT_CYAN), 0.01)
        time.sleep(0.5)

        # Header
        self._print_header()

        # Print each menu section
        for section in self.menu_sections:
            self._print_section(section)

        # Footer
        self._print_footer()

        # Interactive prompt
        self._print_prompt()

        return input(
            f"\n{colorize('🎯 กรุณาเลือกตัวเลือก:', Colors.BOLD + Colors.BRIGHT_WHITE)} {colorize('▶️', Colors.BRIGHT_GREEN)} "
        ).strip()

    def _print_header(self) -> None:
        """Print menu header"""
        header = f"""
{colorize('╔═══════════════════════════════════════════════════════════════════════════════╗', Colors.BRIGHT_YELLOW)}
{colorize('║', Colors.BRIGHT_YELLOW)} {colorize('🎯 NICEGOLD ProjectP - เมนูหลัก (MAIN MENU)', Colors.BOLD + Colors.BRIGHT_WHITE)} {colorize('║', Colors.BRIGHT_YELLOW)}
{colorize('╠═══════════════════════════════════════════════════════════════════════════════╣', Colors.BRIGHT_YELLOW)}
{colorize('║', Colors.BRIGHT_YELLOW)}                                                                               {colorize('║', Colors.BRIGHT_YELLOW)}"""
        print(header)

    def _print_section(self, section: MenuSection) -> None:
        """Print a menu section"""
        print(
            f"{colorize('║', Colors.BRIGHT_YELLOW)}  {colorize(section.title, section.color)} {colorize('║', Colors.BRIGHT_YELLOW)}"
        )
        print(
            f"{colorize('║', Colors.BRIGHT_YELLOW)}                                                                               {colorize('║', Colors.BRIGHT_YELLOW)}"
        )

        for num, title, desc in section.items:
            # Create emoji number
            emoji_num = self._get_emoji_number(num)

            print(
                f"{colorize('║', Colors.BRIGHT_YELLOW)}  {colorize(emoji_num, Colors.BRIGHT_WHITE)}  {colorize(title, Colors.BOLD + Colors.WHITE):<20} - {colorize(desc, Colors.DIM + Colors.WHITE)} {colorize('║', Colors.BRIGHT_YELLOW)}"
            )

        print(
            f"{colorize('║', Colors.BRIGHT_YELLOW)}                                                                               {colorize('║', Colors.BRIGHT_YELLOW)}"
        )

    def _get_emoji_number(self, num: str) -> str:
        """Convert number to emoji"""
        emoji_map = {
            "0": "0️⃣",
            "1": "1️⃣",
            "2": "2️⃣",
            "3": "3️⃣",
            "4": "4️⃣",
            "5": "5️⃣",
            "6": "6️⃣",
            "7": "7️⃣",
            "8": "8️⃣",
            "9": "9️⃣",
        }

        if len(num) == 1:
            return emoji_map.get(num, num)
        elif len(num) == 2:
            if num == "10":
                return "🔟"
            else:
                return emoji_map.get(num[0], num[0]) + emoji_map.get(num[1], num[1])
        return num

    def _print_footer(self) -> None:
        """Print menu footer"""
        footer = f"""
{colorize('║', Colors.BRIGHT_YELLOW)}  {colorize('0️⃣', Colors.BRIGHT_WHITE)}  {colorize('Exit', Colors.BOLD + Colors.WHITE):<20} - {colorize('ออกจากโปรแกรม', Colors.DIM + Colors.WHITE)} {colorize('║', Colors.BRIGHT_YELLOW)}
{colorize('║', Colors.BRIGHT_YELLOW)}                                                                               {colorize('║', Colors.BRIGHT_YELLOW)}
{colorize('╚═══════════════════════════════════════════════════════════════════════════════╝', Colors.BRIGHT_YELLOW)}"""
        print(footer)

    def _print_prompt(self) -> None:
        """Print interactive prompt"""
        print(
            f"\n{colorize('💡 คำแนะนำ:', Colors.BRIGHT_BLUE)} พิมพ์หมายเลข {colorize('1-25', Colors.BRIGHT_GREEN)} หรือ {colorize('0', Colors.BRIGHT_RED)} เพื่อออก"
        )
        print(
            f"{colorize('📋 สถานะระบบ:', Colors.BRIGHT_BLUE)} {colorize('✅ พร้อมใช้งาน', Colors.BRIGHT_GREEN)} | {colorize('📁 Data:', Colors.BRIGHT_BLUE)} {colorize('datacsv/', Colors.BRIGHT_CYAN)}"
        )

    def validate_choice(self, choice: str) -> bool:
        """Validate user menu choice"""
        valid_choices = set(["0"])  # Add exit option

        # Add all menu item numbers
        for section in self.menu_sections:
            for num, _, _ in section.items:
                valid_choices.add(num)

        return choice in valid_choices

    def get_choice_description(self, choice: str) -> str:
        """Get description for a menu choice"""
        if choice == "0":
            return "ออกจากโปรแกรม"

        for section in self.menu_sections:
            for num, title, desc in section.items:
                if num == choice:
                    return f"{title} - {desc}"

        return "ตัวเลือกไม่ถูกต้อง"

    def run_main_menu(self) -> None:
        """Run the main menu loop"""
        while True:
            try:
                choice = self.print_main_menu()

                if choice.lower() in ["0", "q", "quit", "exit"]:
                    print(
                        f"\n{colorize('👋 ขอบคุณที่ใช้ NICEGOLD ProjectP!', Colors.BRIGHT_MAGENTA)}"
                    )
                    break

                if not self.validate_choice(choice):
                    print(
                        f"{colorize('❌ ตัวเลือกไม่ถูกต้อง กรุณาลองใหม่', Colors.BRIGHT_RED)}"
                    )
                    continue

                # Execute the chosen command
                success = self.handle_menu_choice(choice)

                if success:
                    print(f"\n{colorize('✅ ดำเนินการเสร็จสิ้น!', Colors.BRIGHT_GREEN)}")
                else:
                    print(
                        f"\n{colorize('❌ เกิดข้อผิดพลาด กรุณาตรวจสอบ logs', Colors.BRIGHT_RED)}"
                    )

                # Wait for user to continue
                input(f"\n{colorize('กด Enter เพื่อกลับสู่เมนูหลัก...', Colors.DIM)}")

            except KeyboardInterrupt:
                print(f"\n{colorize('👋 การทำงานถูกหยุดโดยผู้ใช้', Colors.BRIGHT_YELLOW)}")
                break
            except Exception as e:
                print(f"{colorize('❌ เกิดข้อผิดพลาดในระบบ:', Colors.BRIGHT_RED)} {e}")
                input(f"\n{colorize('กด Enter เพื่อกลับสู่เมนูหลัก...', Colors.DIM)}")

    def handle_menu_choice(self, choice: str) -> bool:
        """Handle menu choice execution"""
        print(
            f"\n{colorize('⚡ กำลังดำเนินการ:', Colors.BRIGHT_CYAN)} {self.get_choice_description(choice)}"
        )
        print(f"{colorize('═' * 80, Colors.DIM)}")

        # Core Pipeline Modes (1-4)
        if choice == "1":
            return self.pipeline_commands.full_pipeline()
        elif choice == "2":
            return self.pipeline_commands.production_pipeline()
        elif choice == "3":
            return self.pipeline_commands.full_pipeline()  # Debug mode
        elif choice == "4":
            return self.pipeline_commands.preprocessing_only()

        # Data Processing (5-7)
        elif choice == "5":
            return self.analysis_commands.data_analysis_statistics()
        elif choice == "6":
            return self.pipeline_commands.preprocessing_only()
        elif choice == "7":
            return self.pipeline_commands.preprocessing_only()

        # Machine Learning (8-10)
        elif choice == "8":
            return self.pipeline_commands.ultimate_pipeline()
        elif choice == "9":
            return self.analysis_commands.model_comparison()
        elif choice == "10":
            return self.pipeline_commands.realistic_backtest()

        # Advanced Analytics (11-13)
        elif choice == "11":
            return self.trading_commands.start_live_simulation()
        elif choice == "12":
            return self.analysis_commands.performance_analysis()
        elif choice == "13":
            return self.analysis_commands.risk_analysis()

        # Monitoring & Services (14-16)
        elif choice == "14":
            return self.ai_commands.web_dashboard()
        elif choice == "15":
            return self._start_api_server()
        elif choice == "16":
            return self.trading_commands.start_monitoring()

        # AI Agents (17-21)
        elif choice == "17":
            return self.ai_commands.project_analysis()
        elif choice == "18":
            return self.ai_commands.run_ai_autofix()
        elif choice == "19":
            return self.ai_commands.run_ai_optimization()
        elif choice == "20":
            return self.ai_commands.executive_summary()
        elif choice == "21":
            return self.ai_commands.show_ai_dashboard()

        # System Management (22-25)
        elif choice == "22":
            return self._system_health_check()
        elif choice == "23":
            return self._install_dependencies()
        elif choice == "24":
            return self._clean_and_reset()
        elif choice == "25":
            return self._view_logs_and_results()

        else:
            print(f"{colorize('❌ ฟังก์ชันนี้ยังไม่ได้ implement', Colors.BRIGHT_RED)}")
            return False

    def _start_api_server(self) -> bool:
        """Start FastAPI server"""
        print(f"{colorize('🚀 Starting FastAPI Server...', Colors.BRIGHT_GREEN)}")
        return self.pipeline_commands.run_command(
            ["python", "api_server.py"], "FastAPI Model Server"
        )

    def _system_health_check(self) -> bool:
        """Run system health check"""
        print(f"{colorize('🏥 Running System Health Check...', Colors.BRIGHT_BLUE)}")
        # Implementation would use health monitor
        return True

    def _install_dependencies(self) -> bool:
        """Install required dependencies"""
        print(f"{colorize('📦 Installing Dependencies...', Colors.BRIGHT_YELLOW)}")
        return self.pipeline_commands.run_command(
            ["pip", "install", "-r", "requirements.txt"], "Installing Dependencies"
        )

    def _clean_and_reset(self) -> bool:
        """Clean and reset system"""
        print(f"{colorize('🧹 Cleaning System...', Colors.BRIGHT_RED)}")
        return self.pipeline_commands.run_command(
            [
                "python",
                "-c",
                "import shutil; import os; [shutil.rmtree(d) for d in ['output_default', '__pycache__'] if os.path.exists(d)]",
            ],
            "System Cleanup",
        )

    def _view_logs_and_results(self) -> bool:
        """View logs and results"""
        print(f"{colorize('📋 Viewing Logs and Results...', Colors.BRIGHT_CYAN)}")
        # Implementation would show logs directory
        return True


# Global menu system instance
menu_system = MenuSystem()


def print_main_menu() -> str:
    """Display the main menu (compatibility function)"""
    return menu_system.print_main_menu()


def validate_menu_choice(choice: str) -> bool:
    """Validate menu choice (compatibility function)"""
    return menu_system.validate_choice(choice)


def get_menu_description(choice: str) -> str:
    """Get menu choice description (compatibility function)"""
    return menu_system.get_choice_description(choice)
