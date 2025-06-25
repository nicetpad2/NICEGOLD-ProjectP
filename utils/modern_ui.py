#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 Enhanced Modern UI Components for NICEGOLD ProjectP
Beautiful animations, progress bars, and modern interface elements
"""

import os
import random
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


class ModernProgressBar:
    """Modern animated progress bar with various styles"""

    def __init__(
        self, title: str = "", total: int = 100, width: int = 50, style: str = "modern"
    ):
        self.title = title
        self.total = total
        self.width = width
        self.current = 0
        self.style = style
        self.start_time = time.time()
        self.is_complete = False

        # Different progress bar styles
        self.styles = {
            "modern": {"empty": "░", "filled": "█", "head": "▓"},
            "classic": {"empty": "-", "filled": "=", "head": ">"},
            "dots": {"empty": "⚬", "filled": "⚫", "head": "●"},
            "blocks": {"empty": "□", "filled": "■", "head": "▣"},
            "circles": {"empty": "○", "filled": "●", "head": "◉"},
            "gradient": {"empty": "░", "filled": "▓", "head": "█"},
        }

    def update(self, value: int, message: str = ""):
        """Update progress bar"""
        self.current = min(value, self.total)
        percentage = (self.current / self.total) * 100

        # Calculate bar components
        filled_length = int(self.width * self.current / self.total)
        style_chars = self.styles.get(self.style, self.styles["modern"])

        # Create progress bar
        bar = ""
        for i in range(self.width):
            if i < filled_length - 1:
                bar += style_chars["filled"]
            elif i == filled_length - 1 and filled_length < self.width:
                bar += style_chars["head"]
            else:
                bar += style_chars["empty"]

        # Calculate timing
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
        else:
            eta = 0

        # Format time
        eta_str = self._format_time(eta)
        elapsed_str = self._format_time(elapsed)

        # Color coding based on percentage
        if percentage < 30:
            color_code = "\033[91m"  # Red
        elif percentage < 70:
            color_code = "\033[93m"  # Yellow
        else:
            color_code = "\033[92m"  # Green

        reset_code = "\033[0m"

        # Create display line
        display_line = (
            f"\r{self.title} |{color_code}{bar}{reset_code}| {percentage:6.1f}% "
        )
        display_line += f"({self.current}/{self.total}) "
        display_line += f"[{elapsed_str} < {eta_str}]"

        if message:
            display_line += f" - {message}"

        sys.stdout.write(display_line)
        sys.stdout.flush()

        if self.current >= self.total:
            self.is_complete = True
            print()  # New line when complete

    def _format_time(self, seconds: float) -> str:
        """Format time in human readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m {seconds%60:.0f}s"
        else:
            return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m"


class ModernSpinner:
    """Modern animated spinner with various styles"""

    def __init__(self, title: str = "Loading", style: str = "dots"):
        self.title = title
        self.style = style
        self.is_spinning = False
        self.thread = None

        # Different spinner styles
        self.spinners = {
            "dots": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
            "bars": ["▁", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃"],
            "arrows": ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"],
            "clock": [
                "🕐",
                "🕑",
                "🕒",
                "🕓",
                "🕔",
                "🕕",
                "🕖",
                "🕗",
                "🕘",
                "🕙",
                "🕚",
                "🕛",
            ],
            "moon": ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"],
            "earth": ["🌍", "🌎", "🌏"],
            "pulse": ["💙", "💚", "💛", "🧡", "❤️"],
        }

    def start(self):
        """Start spinning animation"""
        self.is_spinning = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.start()

    def stop(self, final_message: str = ""):
        """Stop spinning animation"""
        self.is_spinning = False
        if self.thread:
            self.thread.join()

        # Clear spinner line and show final message
        sys.stdout.write("\r" + " " * (len(self.title) + 20) + "\r")
        if final_message:
            print(f"✅ {final_message}")
        sys.stdout.flush()

    def _spin(self):
        """Internal spinning animation"""
        chars = self.spinners.get(self.style, self.spinners["dots"])
        i = 0

        while self.is_spinning:
            sys.stdout.write(f"\r{chars[i]} {self.title}...")
            sys.stdout.flush()
            time.sleep(0.1)
            i = (i + 1) % len(chars)


class ModernBanner:
    """Beautiful ASCII art banners and headers"""

    @staticmethod
    def create_gradient_text(text: str, colors: List[str]) -> str:
        """Create gradient colored text"""
        if len(colors) < 2:
            return text

        result = ""
        text_len = len(text)

        for i, char in enumerate(text):
            # Calculate color position
            pos = i / max(1, text_len - 1)

            if pos <= 0.5:
                # First half - transition from first to middle color
                color_idx = 0
            else:
                # Second half - transition from middle to last color
                color_idx = min(1, len(colors) - 2)

            result += f"{colors[color_idx]}{char}"

        result += "\033[0m"  # Reset color
        return result

    @staticmethod
    def create_box(text: str, style: str = "double", padding: int = 1) -> str:
        """Create beautiful text boxes"""
        lines = text.split("\n")
        max_width = max(len(line) for line in lines) if lines else 0

        box_styles = {
            "single": {
                "top_left": "┌",
                "top_right": "┐",
                "bottom_left": "└",
                "bottom_right": "┘",
                "horizontal": "─",
                "vertical": "│",
            },
            "double": {
                "top_left": "╔",
                "top_right": "╗",
                "bottom_left": "╚",
                "bottom_right": "╝",
                "horizontal": "═",
                "vertical": "║",
            },
            "rounded": {
                "top_left": "╭",
                "top_right": "╮",
                "bottom_left": "╰",
                "bottom_right": "╯",
                "horizontal": "─",
                "vertical": "│",
            },
            "thick": {
                "top_left": "┏",
                "top_right": "┓",
                "bottom_left": "┗",
                "bottom_right": "┛",
                "horizontal": "━",
                "vertical": "┃",
            },
        }

        chars = box_styles.get(style, box_styles["double"])
        width = max_width + (padding * 2)

        # Top border
        result = (
            chars["top_left"] + chars["horizontal"] * width + chars["top_right"] + "\n"
        )

        # Empty padding lines
        for _ in range(padding):
            result += chars["vertical"] + " " * width + chars["vertical"] + "\n"

        # Content lines
        for line in lines:
            padded_line = " " * padding + line.ljust(max_width) + " " * padding
            result += chars["vertical"] + padded_line + chars["vertical"] + "\n"

        # Empty padding lines
        for _ in range(padding):
            result += chars["vertical"] + " " * width + chars["vertical"] + "\n"

        # Bottom border
        result += (
            chars["bottom_left"] + chars["horizontal"] * width + chars["bottom_right"]
        )

        return result

    @staticmethod
    def create_welcome_banner() -> str:
        """Create beautiful welcome banner"""
        banner = """
    ███╗   ██╗██╗ ██████╗███████╗ ██████╗  ██████╗ ██╗     ██████╗ 
    ████╗  ██║██║██╔════╝██╔════╝██╔════╝ ██╔═══██╗██║     ██╔══██╗
    ██╔██╗ ██║██║██║     █████╗  ██║  ███╗██║   ██║██║     ██║  ██║
    ██║╚██╗██║██║██║     ██╔══╝  ██║   ██║██║   ██║██║     ██║  ██║
    ██║ ╚████║██║╚██████╗███████╗╚██████╔╝╚██████╔╝███████╗██████╔╝
    ╚═╝  ╚═══╝╚═╝ ╚═════╝╚══════╝ ╚═════╝  ╚═════╝ ╚══════╝╚═════╝ 
                                                                     
         ██████╗ ██████╗  ██████╗      ██╗███████╗ ██████╗████████╗
         ██╔══██╗██╔══██╗██╔═══██╗     ██║██╔════╝██╔════╝╚══██╔══╝
         ██████╔╝██████╔╝██║   ██║     ██║█████╗  ██║        ██║   
         ██╔═══╝ ██╔══██╗██║   ██║██   ██║██╔══╝  ██║        ██║   
         ██║     ██║  ██║╚██████╔╝╚█████╔╝███████╗╚██████╗   ██║   
         ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝   
        """
        return banner


class ModernMenu:
    """Enhanced modern menu with animations and beautiful UI"""

    def __init__(self):
        self.colors = {
            "primary": "\033[94m",  # Blue
            "secondary": "\033[96m",  # Cyan
            "success": "\033[92m",  # Green
            "warning": "\033[93m",  # Yellow
            "danger": "\033[91m",  # Red
            "info": "\033[95m",  # Magenta
            "muted": "\033[90m",  # Gray
            "reset": "\033[0m",  # Reset
            "bold": "\033[1m",  # Bold
            "dim": "\033[2m",  # Dim
        }

        self.icons = {
            "rocket": "🚀",
            "chart": "📊",
            "gear": "⚙️",
            "robot": "🤖",
            "target": "🎯",
            "globe": "🌐",
            "shield": "🛡️",
            "heart": "❤️",
            "star": "⭐",
            "fire": "🔥",
            "lightning": "⚡",
            "gem": "💎",
            "crown": "👑",
            "trophy": "🏆",
            "medal": "🥇",
            "diamond": "💠",
        }

    def clear_screen(self):
        """Clear terminal screen with animation"""
        os.system("clear" if os.name == "posix" else "cls")

    def animate_text(self, text: str, delay: float = 0.03, color: str = "primary"):
        """Animate text character by character"""
        color_code = self.colors.get(color, "")
        reset_code = self.colors["reset"]

        for char in text:
            sys.stdout.write(f"{color_code}{char}{reset_code}")
            sys.stdout.flush()
            time.sleep(delay)
        print()

    def create_header(
        self, title: str, subtitle: str = "", version: str = "v2.0"
    ) -> str:
        """Create beautiful header with animations"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        header = f"""
{self.colors['primary']}╔══════════════════════════════════════════════════════════════════════════════════╗{self.colors['reset']}
{self.colors['primary']}║{self.colors['reset']} {self.colors['bold']}{self.colors['success']}{title:^80}{self.colors['reset']} {self.colors['primary']}║{self.colors['reset']}
{self.colors['primary']}║{self.colors['reset']} {self.colors['secondary']}{subtitle:^80}{self.colors['reset']} {self.colors['primary']}║{self.colors['reset']}
{self.colors['primary']}╠══════════════════════════════════════════════════════════════════════════════════╣{self.colors['reset']}
{self.colors['primary']}║{self.colors['reset']} {self.colors['muted']}Version: {version:10} │ Timestamp: {timestamp:19} │ Status: {self.colors['success']}ONLINE{self.colors['muted']}{self.colors['reset']} {self.colors['primary']}║{self.colors['reset']}
{self.colors['primary']}╚══════════════════════════════════════════════════════════════════════════════════╝{self.colors['reset']}
        """
        return header

    def create_menu_item(
        self,
        number: int,
        title: str,
        description: str,
        icon: str = "star",
        status: str = "active",
    ) -> str:
        """Create beautiful menu item"""
        status_colors = {
            "active": self.colors["success"],
            "warning": self.colors["warning"],
            "disabled": self.colors["muted"],
            "new": self.colors["info"],
        }

        status_icons = {"active": "✅", "warning": "⚠️", "disabled": "❌", "new": "🆕"}

        color = status_colors.get(status, self.colors["primary"])
        status_icon = status_icons.get(status, "")
        menu_icon = self.icons.get(icon, "⭐")

        return f"{color}{number:2}. {menu_icon} {self.colors['bold']}{title:<25}{self.colors['reset']} {color}│{self.colors['reset']} {description} {status_icon}"

    def create_separator(self, style: str = "line") -> str:
        """Create beautiful separators"""
        separators = {
            "line": f"{self.colors['muted']}{'─' * 80}{self.colors['reset']}",
            "double": f"{self.colors['primary']}{'═' * 80}{self.colors['reset']}",
            "dotted": f"{self.colors['muted']}{'·' * 80}{self.colors['reset']}",
            "wave": f"{self.colors['secondary']}{'~' * 80}{self.colors['reset']}",
        }
        return separators.get(style, separators["line"])

    def show_loading_animation(self, text: str = "Loading", duration: float = 2.0):
        """Show beautiful loading animation"""
        spinner = ModernSpinner(text, "pulse")
        spinner.start()
        time.sleep(duration)
        spinner.stop(f"{text} completed!")

    def show_progress_demo(self, title: str = "Processing", steps: int = 10):
        """Show progress bar demo"""
        progress = ModernProgressBar(title, steps, 60, "gradient")

        for i in range(steps + 1):
            progress.update(i, f"Step {i}/{steps}")
            time.sleep(0.3)

        print(
            f"\n{self.colors['success']}✅ {title} completed successfully!{self.colors['reset']}"
        )

    def display_system_info(self) -> str:
        """Display beautiful system information"""
        info = f"""
{self.colors['info']}┌─ System Information ────────────────────────────────────────────────────────────┐{self.colors['reset']}
{self.colors['info']}│{self.colors['reset']} {self.colors['success']}🖥️  Platform:{self.colors['reset']} {sys.platform.upper():15} {self.colors['success']}🐍 Python:{self.colors['reset']} {sys.version.split()[0]:10} {self.colors['info']}│{self.colors['reset']}
{self.colors['info']}│{self.colors['reset']} {self.colors['warning']}⚡ Performance:{self.colors['reset']} {'OPTIMIZED':12} {self.colors['warning']}🔒 Security:{self.colors['reset']} {'ENABLED':8} {self.colors['info']}│{self.colors['reset']}
{self.colors['info']}│{self.colors['reset']} {self.colors['primary']}📊 Data Mode:{self.colors['reset']} {'REAL DATA ONLY':14} {self.colors['primary']}🛡️  Safety:{self.colors['reset']} {'MAXIMUM':8} {self.colors['info']}│{self.colors['reset']}
{self.colors['info']}└─────────────────────────────────────────────────────────────────────────────────┘{self.colors['reset']}
        """
        return info


# Initialize global modern UI instance
modern_ui = ModernMenu()
progress_bar = ModernProgressBar
spinner = ModernSpinner
banner = ModernBanner


def demo_modern_ui():
    """Demo function to showcase modern UI components"""
    ui = ModernMenu()

    # Clear screen and show banner
    ui.clear_screen()
    print(banner.create_welcome_banner())

    # Animated welcome
    ui.animate_text(
        "🎉 Welcome to NICEGOLD ProjectP - Enhanced Modern UI!", 0.05, "success"
    )
    time.sleep(1)

    # Show loading
    ui.show_loading_animation("Initializing system", 2)

    # Show progress
    ui.show_progress_demo("Loading components", 8)

    # Show menu
    print(
        ui.create_header(
            "🚀 NICEGOLD PROJECTP", "Professional AI Trading System", "v2.0"
        )
    )
    print(ui.display_system_info())

    print(
        f"\n{ui.colors['bold']}{ui.colors['primary']}📋 MAIN MENU{ui.colors['reset']}"
    )
    print(ui.create_separator("double"))

    menu_items = [
        (
            1,
            "Full Pipeline",
            "Complete ML trading analysis workflow",
            "rocket",
            "active",
        ),
        (
            2,
            "Data Analysis",
            "Comprehensive data exploration and insights",
            "chart",
            "active",
        ),
        (
            3,
            "Quick Test",
            "Fast system functionality verification",
            "lightning",
            "active",
        ),
        (
            4,
            "Train Models",
            "Advanced machine learning model training",
            "robot",
            "active",
        ),
        (5, "Health Check", "Complete system health monitoring", "heart", "active"),
    ]

    for item in menu_items:
        print(ui.create_menu_item(*item))

    print(ui.create_separator())

    input(f"\n{ui.colors['primary']}Press Enter to continue...{ui.colors['reset']}")


if __name__ == "__main__":
    demo_modern_ui()
