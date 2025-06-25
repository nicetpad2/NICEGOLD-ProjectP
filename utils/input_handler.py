#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safe input handler for NICEGOLD ProjectP
Handles input in both interactive and non-interactive environments
"""

import sys
from typing import Optional


def safe_input(
    prompt: str = "", default: str = "", timeout: Optional[float] = None
) -> str:
    """
    Safe input function that handles EOFError and non-interactive environments

    Args:
        prompt: Input prompt message
        default: Default value if input fails
        timeout: Optional timeout (not implemented yet)

    Returns:
        str: User input or default value
    """
    try:
        # Check if we're in an interactive environment
        if not sys.stdin.isatty():
            print(f"{prompt}[Non-interactive mode - using default: {default}]")
            return default

        # Try to get input
        return input(prompt)
    except EOFError:
        print(f"\n[EOFError - using default: {default}]")
        return default
    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
        raise
    except Exception as e:
        print(f"\n[Input error: {e} - using default: {default}]")
        return default


def safe_input_choice(prompt: str, valid_choices: list, default: str = "") -> str:
    """
    Safe input for menu choices with validation

    Args:
        prompt: Input prompt message
        valid_choices: List of valid choice values
        default: Default choice if input fails

    Returns:
        str: Valid choice or default
    """
    while True:
        choice = safe_input(prompt, default).strip().lower()

        if choice in valid_choices:
            return choice
        elif choice == default:
            return choice
        else:
            print(f"Invalid choice. Please select from: {', '.join(valid_choices)}")
            if not sys.stdin.isatty():
                return default


def confirm_action(message: str, default: bool = False) -> bool:
    """
    Ask for confirmation with y/n input

    Args:
        message: Confirmation message
        default: Default value if input fails

    Returns:
        bool: True for yes, False for no
    """
    default_str = "y" if default else "n"
    choice = safe_input(f"{message} (y/n): ", default_str).lower()
    return choice in ["y", "yes", "ใช่", "1", "true"]
