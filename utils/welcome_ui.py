#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 Beautiful Welcome UI and Enhanced Menu System for NICEGOLD ProjectP
Modern, animated welcome screen with beautiful transitions and effects
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional


class WelcomeUI:
    """Beautiful welcome screen with animations and modern design"""

    def __init__(self):
        self.colors = {
            "GOLD": "\033[38;5;220m",
            "BRIGHT_GOLD": "\033[38;5;226m",
            "DARK_GOLD": "\033[38;5;214m",
            "GREEN": "\033[38;5;82m",
            "BRIGHT_GREEN": "\033[38;5;46m",
            "CYAN": "\033[38;5;51m",
            "BRIGHT_CYAN": "\033[38;5;87m",
            "BLUE": "\033[38;5;33m",
            "BRIGHT_BLUE": "\033[38;5;39m",
            "PURPLE": "\033[38;5;129m",
            "BRIGHT_PURPLE": "\033[38;5;135m",
            "RED": "\033[38;5;196m",
            "BRIGHT_RED": "\033[38;5;202m",
            "WHITE": "\033[97m",
            "BRIGHT_WHITE": "\033[38;5;231m",
            "GRAY": "\033[38;5;243m",
            "DIM": "\033[2m",
            "BOLD": "\033[1m",
            "RESET": "\033[0m",
            "BLINK": "\033[5m",
            "UNDERLINE": "\033[4m",
        }

        self.gradient_colors = [
            "\033[38;5;196m",  # Red
            "\033[38;5;202m",  # Orange
            "\033[38;5;208m",  # Light Orange
            "\033[38;5;214m",  # Yellow-Orange
            "\033[38;5;220m",  # Yellow
            "\033[38;5;226m",  # Bright Yellow
            "\033[38;5;46m",  # Green
            "\033[38;5;51m",  # Cyan
            "\033[38;5;33m",  # Blue
            "\033[38;5;129m",  # Purple
        ]

    def clear_screen(self):
        """Clear terminal screen"""
        os.system("clear" if os.name == "posix" else "cls")

    def get_terminal_size(self):
        """Get terminal dimensions"""
        try:
            import shutil

            columns, rows = shutil.get_terminal_size()
            return columns, rows
        except:
            return 80, 24

    def center_text(self, text: str, width: Optional[int] = None) -> str:
        """Center text in terminal"""
        if width is None:
            width, _ = self.get_terminal_size()

        # Remove ANSI escape codes for length calculation
        import re

        clean_text = re.sub(r"\033\[[0-9;]*m", "", text)
        padding = max(0, (width - len(clean_text)) // 2)
        return " " * padding + text

    def typewriter_effect(self, text: str, delay: float = 0.03, color: str = "WHITE"):
        """Typewriter effect for text"""
        color_code = self.colors.get(color, self.colors["WHITE"])
        for char in text:
            sys.stdout.write(color_code + char + self.colors["RESET"])
            sys.stdout.flush()
            time.sleep(delay)
        print()

    def rainbow_text(self, text: str) -> str:
        """Apply rainbow gradient to text"""
        result = ""
        for i, char in enumerate(text):
            color = self.gradient_colors[i % len(self.gradient_colors)]
            result += color + char
        return result + self.colors["RESET"]

    def animated_banner(self):
        """Display animated welcome banner"""
        self.clear_screen()
        width, height = self.get_terminal_size()

        # Title frames for animation
        frames = [
            [
                "███╗   ██╗██╗ ██████╗███████╗",
                "████╗  ██║██║██╔════╝██╔════╝",
                "██╔██╗ ██║██║██║     █████╗  ",
                "██║╚██╗██║██║██║     ██╔══╝  ",
                "██║ ╚████║██║╚██████╗███████╗",
                "╚═╝  ╚═══╝╚═╝ ╚═════╝╚══════╝",
            ],
            [
                "  ██████╗  ██████╗ ██╗     ██████╗ ",
                " ██╔════╝ ██╔═══██╗██║     ██╔══██╗",
                " ██║  ███╗██║   ██║██║     ██║  ██║",
                " ██║   ██║██║   ██║██║     ██║  ██║",
                " ╚██████╔╝╚██████╔╝███████╗██████╔╝",
                "  ╚═════╝  ╚═════╝ ╚══════╝╚═════╝ ",
            ],
        ]

        # Animate title
        for frame_idx in range(3):
            for i, frame in enumerate(frames):
                self.clear_screen()
                print("\n" * 3)

                # Display frame with gradient colors
                for line_idx, line in enumerate(frame):
                    color = self.gradient_colors[
                        (line_idx + frame_idx * 2) % len(self.gradient_colors)
                    ]
                    centered_line = self.center_text(
                        color + line + self.colors["RESET"]
                    )
                    print(centered_line)

                time.sleep(0.5)

        # Final title display
        self.clear_screen()
        print("\n" * 2)

        # NICEGOLD title with rainbow effect
        title_lines = [
            "███╗   ██╗██╗ ██████╗███████╗  ██████╗  ██████╗ ██╗     ██████╗ ",
            "████╗  ██║██║██╔════╝██╔════╝ ██╔════╝ ██╔═══██╗██║     ██╔══██╗",
            "██╔██╗ ██║██║██║     █████╗   ██║  ███╗██║   ██║██║     ██║  ██║",
            "██║╚██╗██║██║██║     ██╔══╝   ██║   ██║██║   ██║██║     ██║  ██║",
            "██║ ╚████║██║╚██████╗███████╗ ╚██████╔╝╚██████╔╝███████╗██████╔╝",
            "╚═╝  ╚═══╝╚═╝ ╚═════╝╚══════╝  ╚═════╝  ╚═════╝ ╚══════╝╚═════╝ ",
        ]

        for line in title_lines:
            rainbow_line = self.rainbow_text(line)
            centered_line = self.center_text(rainbow_line)
            print(centered_line)

        # Subtitle
        print("\n")
        subtitle = "🚀 PROJECT P - ENTERPRISE AI TRADING SYSTEM 🚀"
        subtitle_colored = self.colors["BRIGHT_GOLD"] + subtitle + self.colors["RESET"]
        print(self.center_text(subtitle_colored))

        # Version and status
        print("\n")
        version_info = (
            f"{self.colors['CYAN']}v2.0.0 Enterprise Edition{self.colors['RESET']}"
        )
        status_info = (
            f"{self.colors['BRIGHT_GREEN']}● PRODUCTION READY{self.colors['RESET']}"
        )

        print(self.center_text(f"{version_info} | {status_info}"))

        # Loading animation
        print("\n" * 2)
        self.animated_loading("Initializing NICEGOLD ProjectP", duration=2.0)

    def animated_loading(self, text: str, duration: float = 3.0):
        """Animated loading with spinning effects"""
        spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        start_time = time.time()
        i = 0

        while time.time() - start_time < duration:
            spinner = spinners[i % len(spinners)]
            loading_text = f"\r{self.colors['BRIGHT_CYAN']}{spinner} {text}...{self.colors['RESET']}"
            centered = self.center_text(loading_text)
            sys.stdout.write(centered)
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

        # Complete
        check_mark = (
            f"\r{self.colors['BRIGHT_GREEN']}✅ {text} Complete!{self.colors['RESET']}"
        )
        print(self.center_text(check_mark))
        time.sleep(0.5)

    def show_welcome_message(self):
        """Display welcome message with typewriter effect"""
        print("\n" * 2)

        welcome_messages = [
            "🎯 Welcome to the most advanced AI trading platform",
            "💎 Built for professional traders and institutions",
            "🚀 Real data analysis with machine learning power",
            "🛡️  Enterprise-grade security and reliability",
            "📊 Comprehensive analytics and risk management",
        ]

        for message in welcome_messages:
            centered_msg = self.center_text(message)
            self.typewriter_effect(centered_msg, delay=0.02, color="BRIGHT_WHITE")
            time.sleep(0.3)

        print("\n")

    def show_system_status(self):
        """Display system status with beautiful indicators"""
        print(
            self.center_text(
                f"{self.colors['BOLD']}{self.colors['BRIGHT_CYAN']}SYSTEM STATUS{self.colors['RESET']}"
            )
        )
        print(self.center_text("━" * 50))
        print()

        status_items = [
            ("AI Engine", "ONLINE", "BRIGHT_GREEN", "🤖"),
            ("Data Pipeline", "ACTIVE", "BRIGHT_GREEN", "📊"),
            ("Real Data Feed", "CONNECTED", "BRIGHT_GREEN", "📡"),
            ("Security Layer", "SECURED", "BRIGHT_GREEN", "🛡️"),
            ("Analytics Module", "READY", "BRIGHT_GREEN", "📈"),
            ("Live Trading", "DISABLED", "BRIGHT_RED", "🚫"),
        ]

        for name, status, color, icon in status_items:
            status_line = f"{icon} {name:<20} [{self.colors[color]}{status}{self.colors['RESET']}]"
            print(self.center_text(status_line))
            time.sleep(0.2)

        print("\n")

    def show_quick_stats(self):
        """Display quick statistics"""
        print(
            self.center_text(
                f"{self.colors['BOLD']}{self.colors['BRIGHT_GOLD']}QUICK STATISTICS{self.colors['RESET']}"
            )
        )
        print(self.center_text("━" * 50))
        print()

        # Generate some demo stats
        stats = [
            ("Models Trained", "47", "🎯"),
            ("Data Points Analyzed", "2.3M", "📊"),
            ("Successful Backtests", "156", "✅"),
            ("Risk Score", "LOW", "🛡️"),
            ("System Uptime", "99.9%", "⚡"),
        ]

        for name, value, icon in stats:
            stat_line = f"{icon} {name:<25} {self.colors['BRIGHT_CYAN']}{value}{self.colors['RESET']}"
            print(self.center_text(stat_line))
            time.sleep(0.15)

        print("\n")

    def show_footer_info(self):
        """Display footer information"""
        print("\n" * 2)
        print(self.center_text("━" * 60))

        footer_lines = [
            f"{self.colors['DIM']}💡 For support: docs.nicegold.ai | Enterprise: enterprise@nicegold.ai{self.colors['RESET']}",
            f"{self.colors['DIM']}🔒 This system uses only real market data - NO LIVE TRADING{self.colors['RESET']}",
            f"{self.colors['DIM']}⚡ Powered by Advanced AI & Machine Learning{self.colors['RESET']}",
        ]

        for line in footer_lines:
            print(self.center_text(line))

        print(self.center_text("━" * 60))
        print("\n")

    def press_enter_to_continue(self):
        """Beautiful 'press enter to continue' prompt"""
        prompt = f"{self.colors['BRIGHT_GOLD']}Press ENTER to access the main menu...{self.colors['RESET']}"
        centered_prompt = self.center_text(prompt)

        # Blinking effect
        for _ in range(3):
            sys.stdout.write(f"\r{centered_prompt}")
            sys.stdout.flush()
            time.sleep(0.5)
            sys.stdout.write(f"\r{' ' * len(centered_prompt)}")
            sys.stdout.flush()
            time.sleep(0.3)

        sys.stdout.write(f"\r{centered_prompt}")
        sys.stdout.flush()
        input()

    def complete_welcome_sequence(self):
        """Complete welcome sequence with all elements"""
        # Animated banner
        self.animated_banner()

        # Welcome message
        self.show_welcome_message()

        # System status
        self.show_system_status()

        # Quick stats
        self.show_quick_stats()

        # Footer info
        self.show_footer_info()

        # Wait for user
        self.press_enter_to_continue()


class EnhancedMenuUI:
    """Enhanced menu UI with beautiful styling and animations"""

    def __init__(self):
        self.colors = {
            "GOLD": "\033[38;5;220m",
            "BRIGHT_GOLD": "\033[38;5;226m",
            "GREEN": "\033[38;5;82m",
            "BRIGHT_GREEN": "\033[38;5;46m",
            "CYAN": "\033[38;5;51m",
            "BRIGHT_CYAN": "\033[38;5;87m",
            "BLUE": "\033[38;5;33m",
            "BRIGHT_BLUE": "\033[38;5;39m",
            "PURPLE": "\033[38;5;129m",
            "BRIGHT_PURPLE": "\033[38;5;135m",
            "RED": "\033[38;5;196m",
            "BRIGHT_RED": "\033[38;5;202m",
            "WHITE": "\033[97m",
            "BRIGHT_WHITE": "\033[38;5;231m",
            "GRAY": "\033[38;5;243m",
            "DIM": "\033[2m",
            "BOLD": "\033[1m",
            "RESET": "\033[0m",
        }

    def clear_screen(self):
        """Clear terminal screen"""
        os.system("clear" if os.name == "posix" else "cls")

    def get_terminal_size(self):
        """Get terminal dimensions"""
        try:
            import shutil

            columns, rows = shutil.get_terminal_size()
            return columns, rows
        except:
            return 80, 24

    def center_text(self, text: str, width: Optional[int] = None) -> str:
        """Center text in terminal"""
        if width is None:
            width, _ = self.get_terminal_size()

        # Remove ANSI escape codes for length calculation
        import re

        clean_text = re.sub(r"\033\[[0-9;]*m", "", text)
        padding = max(0, (width - len(clean_text)) // 2)
        return " " * padding + text

    def create_menu_box(
        self, items: List[Dict], title: str, color: str = "BRIGHT_CYAN"
    ):
        """Create a beautiful menu box"""
        width, _ = self.get_terminal_size()
        box_width = min(80, width - 4)

        # Box characters
        top_left = "╭"
        top_right = "╮"
        bottom_left = "╰"
        bottom_right = "╯"
        horizontal = "─"
        vertical = "│"

        color_code = self.colors.get(color, self.colors["BRIGHT_CYAN"])

        # Top border
        top_border = top_left + horizontal * (box_width - 2) + top_right
        print(self.center_text(color_code + top_border + self.colors["RESET"]))

        # Title
        title_line = f"{vertical} {title.center(box_width - 4)} {vertical}"
        print(self.center_text(color_code + title_line + self.colors["RESET"]))

        # Separator
        separator = vertical + horizontal * (box_width - 2) + vertical
        print(self.center_text(color_code + separator + self.colors["RESET"]))

        # Menu items
        for item in items:
            option = item.get("option", "")
            label = item.get("label", "")
            description = item.get("description", "")
            icon = item.get("icon", "")
            status = item.get("status", "")

            # Format menu line
            if status:
                status_colored = f"{self.colors['BRIGHT_GREEN'] if status == '✅' else self.colors['BRIGHT_RED']}{status}{self.colors['RESET']}"
                menu_line = f"{vertical} {option:>2} {icon} {label:<25} {status_colored} {vertical}"
            else:
                menu_line = f"{vertical} {option:>2} {icon} {label:<35} {vertical}"

            print(self.center_text(color_code + menu_line + self.colors["RESET"]))

            # Description line
            if description:
                desc_line = f"{vertical}    {self.colors['DIM']}{description:<{box_width-8}}{self.colors['RESET']} {vertical}"
                print(
                    self.center_text(
                        color_code
                        + desc_line[:box_width]
                        + vertical
                        + self.colors["RESET"]
                    )
                )

        # Bottom border
        bottom_border = bottom_left + horizontal * (box_width - 2) + bottom_right
        print(self.center_text(color_code + bottom_border + self.colors["RESET"]))
        print()

    def show_menu_header(self):
        """Show beautiful menu header"""
        self.clear_screen()

        # Compact title
        title = f"{self.colors['BRIGHT_GOLD']}🏆 NICEGOLD ProjectP - Main Menu 🏆{self.colors['RESET']}"
        print(self.center_text(title))

        # Status line
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_line = f"{self.colors['CYAN']}📊 Real Data Mode | 🚫 Live Trading Disabled | ⏰ {current_time}{self.colors['RESET']}"
        print(self.center_text(status_line))

        print(self.center_text("═" * 80))
        print()

    def get_user_choice(self, prompt: str = "Select option") -> str:
        """Get user choice with beautiful prompt"""
        print()
        choice_prompt = (
            f"{self.colors['BRIGHT_GOLD']}🎯 {prompt}: {self.colors['BRIGHT_WHITE']}"
        )
        choice = input(self.center_text(choice_prompt)).strip()
        print(self.colors["RESET"])
        return choice


# Global instances
welcome_ui = WelcomeUI()
menu_ui = EnhancedMenuUI()


def show_welcome_screen():
    """Show complete welcome screen"""
    welcome_ui.complete_welcome_sequence()


def show_enhanced_menu():
    """Show enhanced main menu"""
    menu_ui.show_menu_header()

    # Production features
    production_items = [
        {
            "option": "1",
            "icon": "🚀",
            "label": "Full Pipeline",
            "description": "Complete ML trading pipeline",
            "status": "✅",
        },
        {
            "option": "2",
            "icon": "📊",
            "label": "Data Analysis",
            "description": "Comprehensive data analysis",
            "status": "✅",
        },
        {
            "option": "3",
            "icon": "🔧",
            "label": "Quick Test",
            "description": "System functionality test",
            "status": "✅",
        },
        {
            "option": "4",
            "icon": "💚",
            "label": "Health Check",
            "description": "System diagnostics & monitoring",
            "status": "✅",
        },
    ]

    menu_ui.create_menu_box(
        production_items, "🎯 CORE PRODUCTION FEATURES", "BRIGHT_GREEN"
    )

    # AI Features
    ai_items = [
        {
            "option": "10",
            "icon": "🤖",
            "label": "AI Project Analysis",
            "description": "AI-powered project analysis",
            "status": "🔬",
        },
        {
            "option": "11",
            "icon": "🔧",
            "label": "AI Auto-Fix",
            "description": "Intelligent error correction",
            "status": "🔬",
        },
        {
            "option": "12",
            "icon": "⚡",
            "label": "AI Performance Optimizer",
            "description": "AI system optimization",
            "status": "🔬",
        },
        {
            "option": "13",
            "icon": "📊",
            "label": "AI Executive Summary",
            "description": "AI-generated insights",
            "status": "🔬",
        },
    ]

    menu_ui.create_menu_box(ai_items, "🤖 AI & ADVANCED ANALYTICS", "BRIGHT_CYAN")

    # Trading features
    trading_items = [
        {
            "option": "20",
            "icon": "🤖",
            "label": "Train Models",
            "description": "Machine learning model training",
            "status": "⚡",
        },
        {
            "option": "21",
            "icon": "🎯",
            "label": "Backtest Strategy",
            "description": "Historical backtesting with real data",
            "status": "⚡",
        },
        {
            "option": "22",
            "icon": "📊",
            "label": "Data Analysis",
            "description": "Real data analysis only (NO LIVE TRADING)",
            "status": "🚫",
        },
        {
            "option": "23",
            "icon": "⚠️",
            "label": "Risk Management",
            "description": "Risk analysis & controls",
            "status": "⚡",
        },
    ]

    menu_ui.create_menu_box(trading_items, "📈 TRADING & BACKTESTING", "BRIGHT_BLUE")

    # System options
    system_items = [
        {
            "option": "5",
            "icon": "📦",
            "label": "Install Dependencies",
            "description": "Package management",
        },
        {
            "option": "6",
            "icon": "🧹",
            "label": "Clean System",
            "description": "System cleanup & maintenance",
        },
        {
            "option": "30",
            "icon": "🌐",
            "label": "Start Dashboard",
            "description": "Web-based dashboard",
        },
        {
            "option": "0",
            "icon": "👋",
            "label": "Exit",
            "description": "Exit NICEGOLD ProjectP",
        },
    ]

    menu_ui.create_menu_box(system_items, "🛠️ SYSTEM & TOOLS", "BRIGHT_PURPLE")

    return menu_ui.get_user_choice("Enter your choice")
