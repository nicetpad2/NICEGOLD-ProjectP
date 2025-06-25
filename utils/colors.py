#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colors utility module for NICEGOLD ProjectP
Provides color constants and formatting functions
"""


class Colors:
    """ANSI color codes for terminal output"""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    # Standard colors
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


def colorize(text, color):
    """Apply color to text"""
    return f"{color}{text}{Colors.RESET}"


def print_with_animation(text, delay=0.03):
    """Print text with typing animation"""
    import time

    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()


def clear_screen():
    """Clear terminal screen"""
    import os

    os.system("clear" if os.name == "posix" else "cls")


def show_loading_animation(message, duration=2):
    """Show loading animation with spinning characters"""
    import time

    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    end_time = time.time() + duration
    i = 0

    while time.time() < end_time:
        print(
            f"\r{colorize(chars[i % len(chars)], Colors.BRIGHT_CYAN)} {colorize(message, Colors.WHITE)}",
            end="",
            flush=True,
        )
        time.sleep(0.1)
        i += 1

    print(
        f"\r{colorize('✅', Colors.BRIGHT_GREEN)} {colorize(message, Colors.WHITE)} - เสร็จสิ้น!"
    )


class LoadingAnimation:
    """Context manager for loading animation"""

    def __init__(self, message):
        self.message = message
        self.chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.i = 0
        self.active = False
        import threading

        self.stop_event = threading.Event()
        self.thread = None

    def __enter__(self):
        """Start the loading animation"""
        import threading
        import time

        self.active = True
        self.stop_event.clear()

        def animate():
            while not self.stop_event.is_set():
                if self.active:
                    print(
                        f"\r{colorize(self.chars[self.i % len(self.chars)], Colors.BRIGHT_CYAN)} {colorize(self.message, Colors.WHITE)}",
                        end="",
                        flush=True,
                    )
                    self.i += 1
                    time.sleep(0.1)

        self.thread = threading.Thread(target=animate)
        self.thread.daemon = True
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the loading animation and show completion"""
        self.active = False
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=0.5)

        # Show completion message
        if exc_type is None:
            print(
                f"\r{colorize('✅', Colors.BRIGHT_GREEN)} {colorize(self.message, Colors.WHITE)} - เสร็จสิ้น!"
            )
        else:
            print(
                f"\r{colorize('❌', Colors.BRIGHT_RED)} {colorize(self.message, Colors.WHITE)} - ล้มเหลว!"
            )


def loading_animation(message):
    """Factory function to create a loading animation context manager"""
    return LoadingAnimation(message)
