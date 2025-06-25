# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - UI Animations Module
════════════════════════════════════════════════════════════════════════════════

Beautiful animations and visual effects for terminal interface.

Author: NICEGOLD Team
Version: 3.0
Created: 2025-01-05
"""

import os
import time

# แก้ไข import path
try:
    from src.core.colors import Colors, colorize
except ImportError:
    # Fallback สำหรับกรณีที่ run จาก directory อื่น
    import sys

    sys.path.append(".")
    from src.core.colors import Colors, colorize


def clear_screen() -> None:
    """Clear the terminal screen"""
    os.system("cls" if os.name == "nt" else "clear")


def print_with_animation(text: str, delay: float = 0.03) -> None:
    """
    Print text with typewriter animation

    Args:
        text (str): Text to print with animation
        delay (float): Delay between characters in seconds
    """
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()


def show_loading_animation(message: str, duration: float = 2.0) -> None:
    """
    Show loading animation with spinning indicator

    Args:
        message (str): Loading message to display
        duration (float): Duration of animation in seconds
    """
    spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    end_time = time.time() + duration

    while time.time() < end_time:
        for char in spinner_chars:
            if time.time() >= end_time:
                break
            print(
                f"\r{colorize(char, Colors.BRIGHT_CYAN)} {message}", end="", flush=True
            )
            time.sleep(0.1)

    print(f"\r{colorize('✅', Colors.BRIGHT_GREEN)} {message} - เสร็จสิ้น!")


def show_progress_bar(
    current: int, total: int, width: int = 50, message: str = "กำลังประมวลผล"
) -> None:
    """
    Show progress bar animation

    Args:
        current (int): Current progress value
        total (int): Total progress value
        width (int): Width of progress bar
        message (str): Progress message
    """
    if total == 0:
        return

    percent = (current / total) * 100
    filled = int(width * current // total)
    bar = "█" * filled + "░" * (width - filled)

    print(
        f"\r{colorize(message, Colors.BRIGHT_BLUE)}: [{colorize(bar, Colors.BRIGHT_GREEN)}] {percent:.1f}%",
        end="",
        flush=True,
    )

    if current == total:
        print()  # New line when complete


def print_logo() -> None:
    """Display beautiful NICEGOLD ProjectP logo with colors and animation"""
    clear_screen()

    # Animated loading dots
    print(colorize("🔥 NICEGOLD ProjectP กำลังเริ่มต้น", Colors.BRIGHT_CYAN), end="")
    for i in range(3):
        time.sleep(0.5)
        print(colorize(".", Colors.BRIGHT_YELLOW), end="", flush=True)
    print()
    time.sleep(0.5)
    clear_screen()

    # Main logo with animation
    logo_lines = [
        "╔══════════════════════════════════════════════════════════════════════════════╗",
        "║                                                                              ║",
        "║    ██████╗ ██████╗  ██████╗      ██╗███████╗ ██████╗████████╗██████╗         ║",
        "║    ██╔══██╗██╔══██╗██╔═══██╗     ██║██╔════╝██╔════╝╚══██╔══╝██╔══██╗        ║",
        "║    ██████╔╝██████╔╝██║   ██║     ██║█████╗  ██║        ██║   ██████╔╝        ║",
        "║    ██╔═══╝ ██╔══██╗██║   ██║██   ██║██╔══╝  ██║        ██║   ██╔═══╝         ║",
        "║    ██║     ██║  ██║╚██████╔╝╚█████╔╝███████╗╚██████╗   ██║   ██║             ║",
        "║    ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝   ╚═╝             ║",
        "║                                                                              ║",
        "║                    🚀 NICEGOLD PROFESSIONAL TRADING SYSTEM 🚀               ║",
        "║                          Advanced AI Trading Pipeline                        ║",
        "║                                                                              ║",
        "╚══════════════════════════════════════════════════════════════════════════════╝",
    ]

    # Print logo with gradient colors
    colors = [
        Colors.BRIGHT_RED,
        Colors.BRIGHT_YELLOW,
        Colors.BRIGHT_GREEN,
        Colors.BRIGHT_CYAN,
        Colors.BRIGHT_BLUE,
        Colors.BRIGHT_MAGENTA,
    ]

    for i, line in enumerate(logo_lines):
        color = colors[i % len(colors)]
        print(colorize(line, color))
        time.sleep(0.1)  # Animation delay

    print()
    print(
        colorize(
            "                    ═══ ระบบเทรดดิ้งอัจฉริยะที่ทันสมัยที่สุด ═══",
            Colors.BRIGHT_WHITE + Colors.BOLD,
        )
    )
    print(
        colorize(
            "                           Version 3.0 | Production Ready",
            Colors.BRIGHT_GREEN,
        )
    )
    print()


def print_separator(
    width: int = 80, char: str = "═", color: str = Colors.BRIGHT_CYAN
) -> None:
    """
    Print a colored separator line

    Args:
        width (int): Width of separator
        char (str): Character to use for separator
        color (str): Color for separator
    """
    print(colorize(char * width, color))


def print_box(text: str, width: int = 80, padding: int = 2) -> None:
    """
    Print text in a decorative box

    Args:
        text (str): Text to display in box
        width (int): Width of box
        padding (int): Padding inside box
    """
    # Calculate text centering
    text_width = len(text)
    total_padding = width - text_width - 4  # 4 for border chars
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding

    # Top border
    print(colorize("╔" + "═" * (width - 2) + "╗", Colors.BRIGHT_CYAN))

    # Empty padding lines
    for _ in range(padding):
        print(colorize("║" + " " * (width - 2) + "║", Colors.BRIGHT_CYAN))

    # Text line
    content = "║" + " " * left_padding + text + " " * right_padding + "║"
    print(colorize(content, Colors.BRIGHT_CYAN))

    # Empty padding lines
    for _ in range(padding):
        print(colorize("║" + " " * (width - 2) + "║", Colors.BRIGHT_CYAN))

    # Bottom border
    print(colorize("╚" + "═" * (width - 2) + "╝", Colors.BRIGHT_CYAN))


def flash_text(text: str, color: str, times: int = 3, delay: float = 0.5) -> None:
    """
    Flash text on screen

    Args:
        text (str): Text to flash
        color (str): Color of text
        times (int): Number of flashes
        delay (float): Delay between flashes
    """
    for _ in range(times):
        print(f"\r{colorize(text, color)}", end="", flush=True)
        time.sleep(delay)
        print(f"\r{' ' * len(text)}", end="", flush=True)
        time.sleep(delay)
    print(f"\r{colorize(text, color)}")  # Final display


def countdown(seconds: int, message: str = "เริ่มต้นใน") -> None:
    """
    Display countdown timer

    Args:
        seconds (int): Number of seconds to count down
        message (str): Message to display with countdown
    """
    for i in range(seconds, 0, -1):
        print(
            f"\r{colorize(f'{message} {i} วินาที...', Colors.BRIGHT_YELLOW)}",
            end="",
            flush=True,
        )
        time.sleep(1)
    print(f"\r{colorize('🚀 เริ่มต้น!', Colors.BRIGHT_GREEN)}")


def typing_effect(text: str, delay: float = 0.05) -> None:
    """
    Simulate typing effect for text

    Args:
        text (str): Text to type
        delay (float): Delay between characters
    """
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()  # New line at end
