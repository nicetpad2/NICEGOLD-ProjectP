# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - UI Module
════════════════════════════════════════════════════════════════════════════════

User interface components for terminal-based interaction.

Author: NICEGOLD Team
Version: 3.0
Created: 2025-01-05
"""

from .animations import (
    clear_screen,
    countdown,
    flash_text,
    print_box,
    print_logo,
    print_separator,
    print_with_animation,
    show_loading_animation,
    show_progress_bar,
    typing_effect,
)
from .menu_system import (
    MenuSection,
    MenuSystem,
    get_menu_description,
    print_main_menu,
    validate_menu_choice,
)

__all__ = [
    # Animations
    "clear_screen",
    "print_with_animation",
    "show_loading_animation",
    "show_progress_bar",
    "print_logo",
    "print_separator",
    "print_box",
    "flash_text",
    "countdown",
    "typing_effect",
    # Menu System
    "MenuSystem",
    "MenuSection",
    "print_main_menu",
    "validate_menu_choice",
    "get_menu_description",
]

__version__ = "3.0"
__author__ = "NICEGOLD Team"
