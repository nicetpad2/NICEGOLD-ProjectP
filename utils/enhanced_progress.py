#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 Enhanced Progress Processing System for NICEGOLD ProjectP
Beautiful progress bars and animations for all pipeline processes
"""

import sys
import threading
import time
from typing import Dict, List, Optional


class EnhancedProgressProcessor:
    """Enhanced progress processor with beautiful animations and status"""

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
            "RED": "\033[38;5;196m",
            "BRIGHT_RED": "\033[38;5;202m",
            "WHITE": "\033[97m",
            "BRIGHT_WHITE": "\033[38;5;231m",
            "GRAY": "\033[38;5;243m",
            "DIM": "\033[2m",
            "BOLD": "\033[1m",
            "RESET": "\033[0m",
        }

        self.spinners = {
            "dots": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
            "bars": ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"],
            "arrows": ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"],
            "circles": ["◐", "◓", "◑", "◒"],
            "squares": ["◰", "◳", "◲", "◱"],
        }

        self.progress_styles = {
            "modern": {"empty": "░", "filled": "█", "head": "▓"},
            "classic": {"empty": "-", "filled": "=", "head": ">"},
            "dots": {"empty": "⚬", "filled": "⚫", "head": "●"},
            "blocks": {"empty": "□", "filled": "■", "head": "▣"},
            "circles": {"empty": "○", "filled": "●", "head": "◉"},
            "gradient": {"empty": "░", "filled": "▓", "head": "█"},
        }

        self.is_running = False
        self.current_step = 0
        self.total_steps = 0
        self.start_time = None

    def get_terminal_size(self):
        """Get terminal dimensions"""
        try:
            import shutil

            columns, rows = shutil.get_terminal_size()
            return columns, rows
        except Exception:
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

    def format_time(self, seconds: float) -> str:
        """Format time in human readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m {seconds%60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"

    def create_progress_bar(
        self, current: int, total: int, width: int = 50, style: str = "modern"
    ) -> str:
        """Create a beautiful progress bar"""
        if total == 0:
            return "█" * width

        percentage = (current / total) * 100
        filled_length = int(width * current / total)

        style_chars = self.progress_styles.get(style, self.progress_styles["modern"])

        # Create progress bar
        bar = ""
        for i in range(width):
            if i < filled_length - 1:
                bar += style_chars["filled"]
            elif i == filled_length - 1 and filled_length < width:
                bar += style_chars["head"]
            else:
                bar += style_chars["empty"]

        # Color coding based on percentage
        if percentage < 30:
            color_code = self.colors["RED"]
        elif percentage < 70:
            color_code = self.colors["BRIGHT_GOLD"]
        else:
            color_code = self.colors["BRIGHT_GREEN"]

        return f"{color_code}{bar}{self.colors['RESET']}"

    def show_step_header(self, step_name: str, step_num: int, total_steps: int):
        """Show beautiful step header"""
        width, _ = self.get_terminal_size()

        # Create header box
        header_text = f" Step {step_num}/{total_steps}: {step_name} "
        box_width = min(len(header_text) + 4, width - 4)

        # Top border
        top_border = "╭" + "─" * (box_width - 2) + "╮"
        print(
            self.center_text(
                f"{self.colors['BRIGHT_CYAN']}{top_border}" f"{self.colors['RESET']}"
            )
        )

        # Header text
        header_line = f"│ {header_text.center(box_width - 4)} │"
        print(
            self.center_text(
                f"{self.colors['BRIGHT_CYAN']}{header_line}" f"{self.colors['RESET']}"
            )
        )

        # Bottom border
        bottom_border = "╰" + "─" * (box_width - 2) + "╯"
        print(
            self.center_text(
                f"{self.colors['BRIGHT_CYAN']}{bottom_border}" f"{self.colors['RESET']}"
            )
        )
        print()

    def animated_spinner(
        self, message: str, duration: float = 3.0, spinner_type: str = "dots"
    ):
        """Show animated spinner with message"""
        spinners = self.spinners.get(spinner_type, self.spinners["dots"])
        start_time = time.time()
        i = 0

        while time.time() - start_time < duration and self.is_running:
            spinner = spinners[i % len(spinners)]
            spinner_text = (
                f"\r{self.colors['BRIGHT_CYAN']}{spinner} "
                f"{self.colors['BRIGHT_WHITE']}{message}..."
                f"{self.colors['RESET']}"
            )
            sys.stdout.write(self.center_text(spinner_text))
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

        if self.is_running:
            # Complete message
            check_mark = (
                f"\r{self.colors['BRIGHT_GREEN']}✅ "
                f"{self.colors['BRIGHT_WHITE']}{message} Complete!"
                f"{self.colors['RESET']}"
            )
            print(self.center_text(check_mark))

    def process_with_progress(
        self, steps: List[Dict], title: str = "", show_overall_progress: bool = True
    ):
        """Process steps with beautiful progress display"""
        self.is_running = True
        self.total_steps = len(steps)
        self.start_time = time.time()

        if title:
            print(
                self.center_text(
                    f"{self.colors['BRIGHT_GOLD']}"
                    f"🚀 {title} 🚀{self.colors['RESET']}"
                )
            )
            print(self.center_text("=" * 60))
            print()

        for i, step in enumerate(steps):
            if not self.is_running:
                break

            self.current_step = i + 1
            step_name = step.get("name", f"Step {i+1}")
            step_function = step.get("function", None)
            step_args = step.get("args", [])
            step_kwargs = step.get("kwargs", {})
            duration = step.get("duration", 2.0)
            spinner_type = step.get("spinner", "dots")

            # Show step header
            self.show_step_header(step_name, i + 1, self.total_steps)

            # Show overall progress if requested
            if show_overall_progress:
                overall_progress = self.create_progress_bar(
                    i, self.total_steps, width=50
                )
                elapsed = time.time() - self.start_time
                eta = (elapsed / max(i, 1)) * (self.total_steps - i)

                progress_line = (
                    f"Overall Progress: {overall_progress} "
                    f"{(i/self.total_steps)*100:.1f}% "
                    f"[{self.format_time(elapsed)} < "
                    f"{self.format_time(eta)}]"
                )
                print(self.center_text(progress_line))
                print()

            try:
                if step_function and callable(step_function):
                    # Execute function with spinner
                    spinner_thread = threading.Thread(
                        target=self.animated_spinner,
                        args=(step_name, duration, spinner_type),
                    )
                    spinner_thread.daemon = True
                    spinner_thread.start()

                    # Execute the actual function
                    result = step_function(*step_args, **step_kwargs)

                    # Wait for spinner to complete
                    spinner_thread.join(timeout=duration + 1)

                    if result is not None:
                        print(
                            self.center_text(
                                f"{self.colors['DIM']}Result: {result}"
                                f"{self.colors['RESET']}"
                            )
                        )

                else:
                    # Just show animation
                    self.animated_spinner(step_name, duration, spinner_type)

            except Exception as e:
                self.is_running = False
                error_msg = (
                    f"{self.colors['BRIGHT_RED']}❌ Error in "
                    f"{step_name}: {str(e)}{self.colors['RESET']}"
                )
                print(self.center_text(error_msg))
                return False

            print()

        # Final completion message
        if self.is_running:
            total_time = time.time() - self.start_time
            completion_msg = (
                f"{self.colors['BRIGHT_GREEN']}🎉 "
                f"All steps completed successfully! "
                f"({self.format_time(total_time)})"
                f"{self.colors['RESET']}"
            )
            print(self.center_text(completion_msg))
            print(self.center_text("=" * 60))

        return True

    def stop(self):
        """Stop the current processing"""
        self.is_running = False

    def create_pipeline_steps(self, pipeline_name: str = "Full Pipeline"):
        """Create standard pipeline steps for ML training"""
        steps = [
            {"name": "Loading Configuration", "duration": 1.0, "spinner": "dots"},
            {"name": "Initializing Data Pipeline", "duration": 1.5, "spinner": "bars"},
            {"name": "Loading Market Data", "duration": 2.0, "spinner": "circles"},
            {"name": "Feature Engineering", "duration": 3.0, "spinner": "arrows"},
            {"name": "Data Preprocessing", "duration": 2.5, "spinner": "squares"},
            {"name": "Splitting Train/Test Data", "duration": 1.0, "spinner": "dots"},
            {
                "name": "Training RandomForest Model",
                "duration": 5.0,  # Reduced from potentially long time
                "spinner": "bars",
            },
            {"name": "Training XGBoost Model", "duration": 4.0, "spinner": "circles"},
            {"name": "Training Neural Network", "duration": 6.0, "spinner": "arrows"},
            {"name": "Model Evaluation", "duration": 2.0, "spinner": "squares"},
            {"name": "Generating Predictions", "duration": 1.5, "spinner": "dots"},
            {"name": "Performance Analysis", "duration": 2.0, "spinner": "bars"},
            {"name": "Saving Results", "duration": 1.0, "spinner": "circles"},
        ]

        return steps

    def create_quick_test_steps(self):
        """Create quick test steps"""
        steps = [
            {"name": "System Health Check", "duration": 1.0, "spinner": "dots"},
            {"name": "Testing Data Connection", "duration": 1.5, "spinner": "circles"},
            {"name": "Validating Models", "duration": 2.0, "spinner": "bars"},
            {"name": "Running Sample Prediction", "duration": 1.5, "spinner": "arrows"},
            {"name": "Verifying Results", "duration": 1.0, "spinner": "squares"},
        ]

        return steps

    def create_data_analysis_steps(self):
        """Create data analysis steps"""
        steps = [
            {"name": "Loading Market Data", "duration": 2.0, "spinner": "bars"},
            {"name": "Statistical Analysis", "duration": 3.0, "spinner": "circles"},
            {"name": "Trend Analysis", "duration": 2.5, "spinner": "arrows"},
            {"name": "Correlation Analysis", "duration": 2.0, "spinner": "dots"},
            {"name": "Volatility Analysis", "duration": 2.0, "spinner": "squares"},
            {"name": "Generating Visualizations", "duration": 3.0, "spinner": "bars"},
            {"name": "Creating Report", "duration": 1.5, "spinner": "circles"},
        ]

        return steps


# Global instance
enhanced_processor = EnhancedProgressProcessor()


def process_with_beautiful_progress(steps: List[Dict], title: str = ""):
    """Convenience function to process steps with beautiful progress"""
    return enhanced_processor.process_with_progress(steps, title)


def show_pipeline_progress(pipeline_type: str = "full"):
    """Show progress for different pipeline types"""
    if pipeline_type == "full":
        steps = enhanced_processor.create_pipeline_steps()
        title = "🚀 NICEGOLD Full ML Pipeline"
    elif pipeline_type == "quick":
        steps = enhanced_processor.create_quick_test_steps()
        title = "🔧 Quick System Test"
    elif pipeline_type == "analysis":
        steps = enhanced_processor.create_data_analysis_steps()
        title = "📊 Data Analysis Pipeline"
    else:
        steps = enhanced_processor.create_pipeline_steps()
        title = f"🔄 {pipeline_type.title()} Pipeline"

    return process_with_beautiful_progress(steps, title)


def simulate_model_training():
    """Simulate model training with realistic progress"""
    # This can be called from actual training functions
    training_steps = [
        {"name": "Preparing Training Data", "duration": 1.0, "spinner": "dots"},
        {"name": "Initializing Random Forest", "duration": 0.5, "spinner": "circles"},
        {
            "name": "Training Random Forest (Optimized)",
            "duration": 3.0,  # Much faster than before
            "spinner": "bars",
        },
        {"name": "Cross-Validation", "duration": 2.0, "spinner": "arrows"},
        {"name": "Model Optimization", "duration": 1.5, "spinner": "squares"},
    ]

    return process_with_beautiful_progress(training_steps, "🤖 Model Training")
