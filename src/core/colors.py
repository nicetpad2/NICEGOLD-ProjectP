# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - Terminal Colors Module
════════════════════════════════════════════════════════════════════════════════

Terminal color definitions and styling utilities for beautiful console output.

Author: NICEGOLD Team
Version: 3.0
Created: 2025-01-05
"""


class Colors:
    """ANSI Color codes for beautiful terminal output"""

    # Reset and formatting
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    # Basic colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


def colorize(text: str, color: str) -> str:
    """
    Apply color to text with automatic reset

    Args:
        text (str): Text to colorize
        color (str): Color code from Colors class

    Returns:
        str: Colorized text with reset
    """
    return f"{color}{text}{Colors.RESET}"


def print_colored(text: str, color: str) -> None:
    """
    Print colored text to console

    Args:
        text (str): Text to print
        color (str): Color code from Colors class
    """
    print(colorize(text, color))


# Predefined color combinations for common use cases
class ColorThemes:
    """Predefined color themes for consistent styling"""

    SUCCESS = Colors.BRIGHT_GREEN
    ERROR = Colors.BRIGHT_RED
    WARNING = Colors.BRIGHT_YELLOW
    INFO = Colors.BRIGHT_BLUE
    HEADER = Colors.BRIGHT_CYAN + Colors.BOLD
    HIGHLIGHT = Colors.BRIGHT_WHITE + Colors.BOLD
    DIM_TEXT = Colors.DIM + Colors.WHITE

    @classmethod
    def success(cls, text: str) -> str:
        """Format success message"""
        return colorize(f"✅ {text}", cls.SUCCESS)

    @classmethod
    def error(cls, text: str) -> str:
        """Format error message"""
        return colorize(f"❌ {text}", cls.ERROR)

    @classmethod
    def warning(cls, text: str) -> str:
        """Format warning message"""
        return colorize(f"⚠️ {text}", cls.WARNING)

    @classmethod
    def info(cls, text: str) -> str:
        """Format info message"""
        return colorize(f"ℹ️ {text}", cls.INFO)

    @classmethod
    def header(cls, text: str) -> str:
        """Format header text"""
        return colorize(text, cls.HEADER)
