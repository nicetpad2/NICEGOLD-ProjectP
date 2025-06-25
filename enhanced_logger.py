#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 Enhanced Logger Integration for NICEGOLD ProjectP v2.1
Beautiful scrolling display and error management system
"""

import os
import sys
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MessageType(Enum):
    """Message types with visual styling"""
    SUCCESS = ("✅", "\033[92m", "\033[102m", "SUCCESS")
    INFO = ("ℹ️", "\033[96m", "\033[106m", "INFO")
    WARNING = ("⚠️", "\033[93m", "\033[103m", "WARNING")
    ERROR = ("❌", "\033[91m", "\033[101m", "ERROR")
    CRITICAL = ("🚨", "\033[95m", "\033[105m", "CRITICAL")
    DEBUG = ("🐛", "\033[94m", "\033[104m", "DEBUG")
    PROGRESS = ("🔄", "\033[97m", "\033[107m", "PROGRESS")


class EnhancedDisplay:
    """Enhanced display system for beautiful console output"""
    
    def __init__(self, width: int = 80, max_history: int = 100):
        self.width = width
        self.max_history = max_history
        self.message_history = []
        self.show_timestamps = True
        self.auto_scroll = True
        
    def print_with_style(self, message: str, msg_type: MessageType = MessageType.INFO,
                        animate: bool = False, center: bool = False):
        """Print message with beautiful styling"""
        
        icon, text_color, bg_color, type_name = msg_type.value
        
        # Add timestamp if enabled
        timestamp = ""
        if self.show_timestamps:
            timestamp = f"[{datetime.now().strftime('%H:%M:%S')}] "
        
        # Format message
        if center:
            content = f"{icon} {message}"
            formatted_msg = f"{text_color}{content:^{self.width}}\033[0m"
        else:
            formatted_msg = f"{timestamp}{bg_color} {icon} {type_name} \033[0m {text_color}{message}\033[0m"
        
        # Store in history
        self.message_history.append({
            "timestamp": datetime.now(),
            "message": message,
            "type": msg_type,
            "formatted": formatted_msg
        })
        
        # Keep history size manageable
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history:]
        
        # Display message
        if animate:
            self._animate_message(formatted_msg)
        else:
            print(formatted_msg)
    
    def _animate_message(self, message: str):
        """Animate message appearance"""
        # Typewriter effect for important messages
        if "\033[91m" in message or "\033[95m" in message:  # Error or Critical
            for char in message:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(0.03)
            print()
        else:
            # Fade in effect
            print(message)
    
    def create_separator(self, char: str = "═", color: str = "\033[94m") -> str:
        """Create a beautiful separator line"""
        return f"{color}{char * self.width}\033[0m"
    
    def create_box(self, content: List[str], title: str = "", 
                   box_color: str = "\033[96m") -> None:
        """Create a beautiful box around content"""
        
        # Calculate box width
        max_content_width = max(len(line) for line in content) if content else 0
        box_width = max(max_content_width + 4, len(title) + 4, 40)
        
        # Top border
        if title:
            title_line = f"╔══ {title} {'═' * (box_width - len(title) - 6)}╗"
        else:
            title_line = f"╔{'═' * (box_width - 2)}╗"
        
        print(f"{box_color}{title_line}\033[0m")
        
        # Content lines
        for line in content:
            padding = box_width - len(line) - 4
            content_line = f"║ {line}{' ' * padding} ║"
            print(f"{box_color}{content_line}\033[0m")
        
        # Bottom border
        bottom_line = f"╚{'═' * (box_width - 2)}╝"
        print(f"{box_color}{bottom_line}\033[0m")
    
    def create_progress_bar(self, current: int, total: int, 
                           title: str = "", width: int = 40) -> str:
        """Create beautiful progress bar"""
        
        if total == 0:
            percentage = 0
        else:
            percentage = min(100, (current / total) * 100)
        
        filled_width = int((percentage / 100) * width)
        empty_width = width - filled_width
        
        # Create gradient colors based on progress
        if percentage < 30:
            bar_color = "\033[91m"  # Red
        elif percentage < 70:
            bar_color = "\033[93m"  # Yellow
        else:
            bar_color = "\033[92m"  # Green
        
        # Build progress bar
        filled_bar = bar_color + "█" * filled_width + "\033[0m"
        empty_bar = "\033[90m" + "░" * empty_width + "\033[0m"
        
        progress_text = f"{percentage:5.1f}%"
        counter_text = f"({current}/{total})"
        
        if title:
            return f"{title}: [{filled_bar}{empty_bar}] {progress_text} {counter_text}"
        else:
            return f"[{filled_bar}{empty_bar}] {progress_text} {counter_text}"
    
    def show_loading_animation(self, message: str, duration: float = 2.0):
        """Show loading animation"""
        
        spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        start_time = time.time()
        i = 0
        
        while (time.time() - start_time) < duration:
            spinner = f"\033[96m{spinner_chars[i % len(spinner_chars)]}\033[0m"
            print(f"\r{spinner} {message}", end="", flush=True)
            time.sleep(0.1)
            i += 1
        
        print(f"\r✅ {message} - Complete!")
    
    def clear_screen(self):
        """Clear screen with style"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self, title: str, subtitle: str = ""):
        """Print beautiful header"""
        
        header_color = "\033[96m"
        border_color = "\033[94m"
        reset = "\033[0m"
        
        # Create header box
        border = border_color + "═" * self.width + reset
        
        print(border)
        print(f"{header_color}{title:^{self.width}}{reset}")
        
        if subtitle:
            print(f"{border_color}{subtitle:^{self.width}}{reset}")
        
        print(border)
        print()


class PremiumLogger:
    """Premium logging system for NICEGOLD ProjectP"""
    
    def __init__(self, display: Optional[EnhancedDisplay] = None):
        self.display = display or EnhancedDisplay()
        self.log_count = {msg_type: 0 for msg_type in MessageType}
        self.session_start = datetime.now()
        
    def success(self, message: str, animate: bool = False):
        """Log success message"""
        self.log_count[MessageType.SUCCESS] += 1
        self.display.print_with_style(message, MessageType.SUCCESS, animate)
    
    def info(self, message: str, animate: bool = False):
        """Log info message"""
        self.log_count[MessageType.INFO] += 1
        self.display.print_with_style(message, MessageType.INFO, animate)
    
    def warning(self, message: str, animate: bool = False):
        """Log warning message"""
        self.log_count[MessageType.WARNING] += 1
        self.display.print_with_style(message, MessageType.WARNING, animate)
    
    def error(self, message: str, animate: bool = True):
        """Log error message with attention-grabbing animation"""
        self.log_count[MessageType.ERROR] += 1
        self.display.print_with_style(message, MessageType.ERROR, animate)
        
        # Flash effect for errors
        if animate:
            for _ in range(2):
                print("\033[41m" + " " * self.display.width + "\033[0m")
                time.sleep(0.1)
                print("\033[K", end="")  # Clear line
                time.sleep(0.1)
    
    def critical(self, message: str, animate: bool = True):
        """Log critical error with maximum attention"""
        self.log_count[MessageType.CRITICAL] += 1
        
        if animate:
            # Alert sound effect (visual)
            print("\n🚨" * 20)
            self.display.print_with_style("CRITICAL ERROR DETECTED", MessageType.CRITICAL, True)
            print("🚨" * 20 + "\n")
        
        self.display.print_with_style(message, MessageType.CRITICAL, animate)
    
    def debug(self, message: str):
        """Log debug message"""
        self.log_count[MessageType.DEBUG] += 1
        self.display.print_with_style(message, MessageType.DEBUG)
    
    def progress(self, message: str):
        """Log progress message"""
        self.log_count[MessageType.PROGRESS] += 1
        self.display.print_with_style(message, MessageType.PROGRESS)
    
    def show_progress_bar(self, current: int, total: int, title: str = ""):
        """Show progress bar"""
        progress_bar = self.display.create_progress_bar(current, total, title)
        print(f"\r{progress_bar}", end="", flush=True)
        
        if current >= total:
            print()  # New line when complete
    
    def show_summary(self):
        """Show session summary"""
        session_duration = datetime.now() - self.session_start
        
        summary_content = [
            f"Session Duration: {str(session_duration).split('.')[0]}",
            "",
            "Message Summary:",
            f"✅ Success: {self.log_count[MessageType.SUCCESS]}",
            f"ℹ️ Info: {self.log_count[MessageType.INFO]}",
            f"⚠️ Warning: {self.log_count[MessageType.WARNING]}",
            f"❌ Error: {self.log_count[MessageType.ERROR]}",
            f"🚨 Critical: {self.log_count[MessageType.CRITICAL]}",
            f"🐛 Debug: {self.log_count[MessageType.DEBUG]}"
        ]
        
        self.display.create_box(summary_content, "Session Summary")
    
    def create_menu(self, title: str, options: List[str], 
                   descriptions: Optional[List[str]] = None) -> None:
        """Create beautiful menu display"""
        
        self.display.print_header(title)
        
        for i, option in enumerate(options):
            option_num = f"{i + 1:2d}"
            option_text = f"  {option_num}. {option}"
            
            if descriptions and i < len(descriptions):
                option_text += f" - {descriptions[i]}"
            
            # Highlight even numbers for better readability
            if i % 2 == 0:
                print(f"\033[97m{option_text}\033[0m")
            else:
                print(f"\033[90m{option_text}\033[0m")
        
        print(self.display.create_separator("─"))
    
    def loading_animation(self, message: str, duration: float = 2.0):
        """Show loading animation"""
        self.display.show_loading_animation(message, duration)


def create_enhanced_logger() -> PremiumLogger:
    """Factory function to create enhanced logger"""
    display = EnhancedDisplay(width=80)
    logger = PremiumLogger(display)
    return logger


# Example usage and demo
def demo_enhanced_logger():
    """Demo of enhanced logger system"""
    
    logger = create_enhanced_logger()
    
    # Demo header
    logger.display.print_header(
        "🚀 NICEGOLD ProjectP v2.1 Enhanced Logger Demo",
        "Beautiful Console Display with Error Management"
    )
    
    # Demo different message types
    logger.success("System initialized successfully!")
    logger.info("Loading configuration files...")
    logger.warning("Some optional components are missing")
    logger.error("Failed to connect to data source")
    logger.debug("Memory usage: 45.2MB")
    
    # Demo progress bar
    print("\n📊 Progress Bar Demo:")
    for i in range(101):
        logger.show_progress_bar(i, 100, "Loading Market Data")
        time.sleep(0.02)
    
    # Demo loading animation
    logger.loading_animation("Processing market analysis", 3.0)
    
    # Demo critical error
    logger.critical("Database connection lost!", animate=True)
    
    # Demo menu
    logger.create_menu(
        "Main Menu",
        ["Full Pipeline", "Data Analysis", "Settings", "Exit"],
        ["Run complete analysis", "Advanced tools", "Configure system", "Quit application"]
    )
    
    # Show session summary
    logger.show_summary()


if __name__ == "__main__":
    demo_enhanced_logger()
