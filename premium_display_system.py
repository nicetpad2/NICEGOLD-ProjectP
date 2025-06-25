#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 NICEGOLD ProjectP v2.1 - Premium Display System
Advanced Scrolling & Error Management with Designer-Level UI

Features:
- Beautiful scrolling text display
- Error severity levels with color coding
- Professional animations and transitions
- Multi-language support (Thai/English)
- Designer-quality visual effects
"""

import json
import os
import sys
import threading
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ErrorSeverity(Enum):
    """Error severity levels with color coding"""
    SUCCESS = ("SUCCESS", "🎉", "\033[92m", "\033[102m")     # Bright Green
    INFO = ("INFO", "ℹ️", "\033[96m", "\033[106m")           # Bright Cyan  
    WARNING = ("WARNING", "⚠️", "\033[93m", "\033[103m")     # Bright Yellow
    ERROR = ("ERROR", "❌", "\033[91m", "\033[101m")         # Bright Red
    CRITICAL = ("CRITICAL", "🚨", "\033[95m", "\033[105m")   # Bright Magenta
    DEBUG = ("DEBUG", "🐛", "\033[94m", "\033[104m")         # Bright Blue


class DisplayEffects:
    """Professional display effects and animations"""
    
    @staticmethod
    def typewriter_effect(text: str, delay: float = 0.03, color: str = "\033[97m"):
        """Typewriter effect for important messages"""
        sys.stdout.write(color)
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
        sys.stdout.write("\033[0m")
        print()
    
    @staticmethod
    def wave_animation(text: str, waves: int = 3, delay: float = 0.1):
        """Wave animation for loading states"""
        for _ in range(waves):
            for i in range(len(text) + 10):
                display_text = " " * 10
                if i < len(text):
                    display_text = text[:i] + "🌊" + text[i:]
                elif i < len(text) + 5:
                    display_text = text + "🌊" * (i - len(text))
                else:
                    display_text = text
                
                print(f"\r{display_text[:80]}", end="", flush=True)
                time.sleep(delay)
        print()
    
    @staticmethod
    def progress_bar_beautiful(current: int, total: int, width: int = 50, 
                             prefix: str = "", suffix: str = "", 
                             fill: str = "█", empty: str = "░"):
        """Beautiful progress bar with gradient effect"""
        percent = current / total
        filled_length = int(width * percent)
        
        # Create gradient effect
        bar_colors = [
            "\033[91m",  # Red
            "\033[93m",  # Yellow  
            "\033[92m",  # Green
        ]
        
        bar = ""
        for i in range(width):
            if i < filled_length:
                # Gradient based on position
                color_index = min(int(i / width * len(bar_colors)), len(bar_colors) - 1)
                bar += bar_colors[color_index] + fill + "\033[0m"
            else:
                bar += "\033[90m" + empty + "\033[0m"
        
        percentage = f"{percent:.1%}"
        print(f"\r{prefix} [{bar}] {percentage} {suffix}", end="", flush=True)
        
        if current == total:
            print()


class ScrollingDisplay:
    """Advanced scrolling display system with professional UI"""
    
    def __init__(self, max_lines: int = 20, width: int = 80):
        self.max_lines = max_lines
        self.width = width
        self.lines = []
        self.scroll_position = 0
        self.auto_scroll = True
        self.show_timestamp = True
        self.show_line_numbers = True
        
    def add_line(self, content: str, severity: ErrorSeverity = ErrorSeverity.INFO, 
                 timestamp: bool = True, animate: bool = False):
        """Add a line to the scrolling display"""
        
        # Format timestamp
        ts = datetime.now().strftime("%H:%M:%S") if timestamp and self.show_timestamp else ""
        
        # Create formatted line
        line_data = {
            "content": content,
            "severity": severity,
            "timestamp": ts,
            "line_number": len(self.lines) + 1
        }
        
        self.lines.append(line_data)
        
        # Auto-scroll if enabled
        if self.auto_scroll and len(self.lines) > self.max_lines:
            self.scroll_position = len(self.lines) - self.max_lines
        
        # Display with animation if requested
        if animate:
            self._animate_new_line(line_data)
        else:
            self.refresh_display()
    
    def _animate_new_line(self, line_data: Dict):
        """Animate new line addition"""
        severity = line_data["severity"]
        content = line_data["content"]
        
        # Slide in effect
        for i in range(self.width + 1):
            display_content = content[:i] if i <= len(content) else content
            formatted_line = self._format_line(line_data, len(display_content))
            
            print(f"\r{formatted_line}", end="", flush=True)
            time.sleep(0.01)
        
        print()
    
    def _format_line(self, line_data: Dict, max_content_length: Optional[int] = None) -> str:
        """Format a line with colors and decorations"""
        severity = line_data["severity"]
        content = line_data["content"]
        timestamp = line_data["timestamp"]
        line_number = line_data["line_number"]
        
        # Apply content length limit if specified
        if max_content_length is not None:
            content = content[:max_content_length]
        
        # Get severity styling
        severity_name, icon, text_color, bg_color = severity.value
        
        # Build line components
        parts = []
        
        # Line number (if enabled)
        if self.show_line_numbers:
            line_num = f"{line_number:4d}"
            parts.append(f"\033[90m{line_num}\033[0m")
        
        # Timestamp (if enabled and available)
        if timestamp:
            parts.append(f"\033[90m[{timestamp}]\033[0m")
        
        # Severity indicator
        severity_indicator = f"{bg_color} {icon} {severity_name} \033[0m"
        parts.append(severity_indicator)
        
        # Content with color
        colored_content = f"{text_color}{content}\033[0m"
        parts.append(colored_content)
        
        # Join and ensure proper width
        line = " ".join(parts)
        
        # Add padding or truncate to fit width
        display_line = line[:self.width] if len(line) > self.width else line
        
        return display_line
    
    def refresh_display(self):
        """Refresh the entire display"""
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Display header
        self._display_header()
        
        # Calculate visible lines
        start_idx = max(0, self.scroll_position)
        end_idx = min(len(self.lines), start_idx + self.max_lines)
        
        # Display visible lines
        for i in range(start_idx, end_idx):
            line_data = self.lines[i]
            formatted_line = self._format_line(line_data)
            print(formatted_line)
        
        # Display footer
        self._display_footer()
    
    def _display_header(self):
        """Display beautiful header"""
        header_color = "\033[96m"
        border_color = "\033[94m"
        reset = "\033[0m"
        
        border = border_color + "═" * self.width + reset
        title = f"{header_color}🚀 NICEGOLD ProjectP v2.1 - Premium Display System 🚀{reset}"
        subtitle = f"{header_color}📊 Real-time Scrolling Display with Error Management{reset}"
        
        print(border)
        print(f"{title:^{self.width + 20}}")  # +20 for color codes
        print(f"{subtitle:^{self.width + 20}}")
        print(border)
        print()
    
    def _display_footer(self):
        """Display informative footer"""
        footer_color = "\033[90m"
        accent_color = "\033[93m"
        reset = "\033[0m"
        
        total_lines = len(self.lines)
        visible_start = self.scroll_position + 1
        visible_end = min(self.scroll_position + self.max_lines, total_lines)
        
        status_info = (
            f"{footer_color}Lines: {accent_color}{visible_start}-{visible_end}"
            f"{footer_color}/{accent_color}{total_lines}{footer_color} | "
            f"Auto-scroll: {accent_color}{'ON' if self.auto_scroll else 'OFF'}{footer_color} | "
            f"Time: {accent_color}{datetime.now().strftime('%H:%M:%S')}{reset}"
        )
        
        border = footer_color + "─" * self.width + reset
        
        print()
        print(border)
        print(status_info)
        print(border)
    
    def scroll_up(self, lines: int = 1):
        """Scroll up by specified lines"""
        self.scroll_position = max(0, self.scroll_position - lines)
        self.auto_scroll = False
        self.refresh_display()
    
    def scroll_down(self, lines: int = 1):
        """Scroll down by specified lines"""
        max_scroll = max(0, len(self.lines) - self.max_lines)
        self.scroll_position = min(max_scroll, self.scroll_position + lines)
        
        # Re-enable auto-scroll if at bottom
        if self.scroll_position >= max_scroll:
            self.auto_scroll = True
        
        self.refresh_display()
    
    def toggle_auto_scroll(self):
        """Toggle auto-scrolling"""
        self.auto_scroll = not self.auto_scroll
        if self.auto_scroll and len(self.lines) > self.max_lines:
            self.scroll_position = len(self.lines) - self.max_lines
        self.refresh_display()
    
    def save_log(self, filename: str = "display_log.json"):
        """Save current display content to file"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "total_lines": len(self.lines),
            "lines": []
        }
        
        for line_data in self.lines:
            log_entry = {
                "line_number": line_data["line_number"],
                "timestamp": line_data["timestamp"],
                "severity": line_data["severity"].value[0],
                "content": line_data["content"]
            }
            log_data["lines"].append(log_entry)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        self.add_line(f"📄 Log saved to {filename}", ErrorSeverity.SUCCESS)


class PremiumLogger:
    """Premium logging system with advanced display features"""
    
    def __init__(self, display: ScrollingDisplay):
        self.display = display
        self.log_history = []
        self.error_counts = {severity: 0 for severity in ErrorSeverity}
        
    def log(self, message: str, severity: ErrorSeverity = ErrorSeverity.INFO, 
            animate: bool = False, **kwargs):
        """Log a message with specified severity"""
        
        # Update error counts
        self.error_counts[severity] += 1
        
        # Add to history
        log_entry = {
            "timestamp": datetime.now(),
            "message": message,
            "severity": severity,
            "kwargs": kwargs
        }
        self.log_history.append(log_entry)
        
        # Display message
        self.display.add_line(message, severity, animate=animate)
        
        # Special handling for critical errors
        if severity == ErrorSeverity.CRITICAL:
            self._handle_critical_error(message)
    
    def _handle_critical_error(self, message: str):
        """Special handling for critical errors"""
        # Create attention-grabbing display
        DisplayEffects.typewriter_effect(
            f"🚨 CRITICAL ERROR DETECTED 🚨", 
            delay=0.05, 
            color="\033[91m"
        )
        
        # Flash effect
        for _ in range(3):
            print("\033[105m" + " " * 80 + "\033[0m")
            time.sleep(0.2)
            print("\033[K", end="")  # Clear line
            time.sleep(0.2)
        
        self.display.add_line("🔧 System attempting recovery...", ErrorSeverity.WARNING)
    
    def success(self, message: str, **kwargs):
        """Log success message"""
        self.log(f"✅ {message}", ErrorSeverity.SUCCESS, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.log(f"ℹ️ {message}", ErrorSeverity.INFO, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.log(f"⚠️ {message}", ErrorSeverity.WARNING, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.log(f"❌ {message}", ErrorSeverity.ERROR, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical error message"""
        self.log(f"🚨 {message}", ErrorSeverity.CRITICAL, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.log(f"🐛 {message}", ErrorSeverity.DEBUG, **kwargs)
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error counts"""
        return {severity.value[0]: count for severity, count in self.error_counts.items()}
    
    def display_error_summary(self):
        """Display beautiful error summary"""
        summary = self.get_error_summary()
        
        self.display.add_line("📊 Error Summary Report", ErrorSeverity.INFO)
        self.display.add_line("═" * 50, ErrorSeverity.INFO)
        
        for severity_name, count in summary.items():
            if count > 0:
                severity = next(s for s in ErrorSeverity if s.value[0] == severity_name)
                self.display.add_line(f"{severity.value[1]} {severity_name}: {count}", severity)
        
        self.display.add_line("═" * 50, ErrorSeverity.INFO)


class AdvancedProgressTracker:
    """Advanced progress tracking with beautiful animations"""
    
    def __init__(self, logger: PremiumLogger):
        self.logger = logger
        self.current_tasks = {}
        self.completed_tasks = []
    
    def start_task(self, task_id: str, task_name: str, total_steps: int = 100):
        """Start tracking a new task"""
        task = {
            "id": task_id,
            "name": task_name,
            "total_steps": total_steps,
            "current_step": 0,
            "start_time": datetime.now(),
            "status": "running"
        }
        
        self.current_tasks[task_id] = task
        self.logger.info(f"🚀 Started: {task_name}")
    
    def update_task(self, task_id: str, step: int, status_message: str = ""):
        """Update task progress"""
        if task_id not in self.current_tasks:
            return
        
        task = self.current_tasks[task_id]
        task["current_step"] = step
        
        # Calculate progress
        progress = (step / task["total_steps"]) * 100
        
        # Create beautiful progress bar
        DisplayEffects.progress_bar_beautiful(
            step, task["total_steps"],
            prefix=f"📊 {task['name'][:20]}",
            suffix=f"{status_message[:30]}"
        )
        
        # Log milestone progress
        if step % (task["total_steps"] // 10) == 0:  # Every 10%
            self.logger.info(f"🎯 {task['name']}: {progress:.0f}% complete")
    
    def complete_task(self, task_id: str, success: bool = True):
        """Mark task as completed"""
        if task_id not in self.current_tasks:
            return
        
        task = self.current_tasks[task_id]
        task["end_time"] = datetime.now()
        task["status"] = "completed" if success else "failed"
        
        duration = (task["end_time"] - task["start_time"]).total_seconds()
        
        if success:
            self.logger.success(f"✅ Completed: {task['name']} ({duration:.1f}s)")
        else:
            self.logger.error(f"❌ Failed: {task['name']} ({duration:.1f}s)")
        
        # Move to completed tasks
        self.completed_tasks.append(task)
        del self.current_tasks[task_id]
    
    def display_active_tasks(self):
        """Display all active tasks"""
        if not self.current_tasks:
            self.logger.info("📝 No active tasks")
            return
        
        self.logger.info("📋 Active Tasks:")
        for task in self.current_tasks.values():
            progress = (task["current_step"] / task["total_steps"]) * 100
            elapsed = (datetime.now() - task["start_time"]).total_seconds()
            
            self.logger.info(
                f"  🔄 {task['name']}: {progress:.1f}% ({elapsed:.1f}s elapsed)"
            )


class BeautifulMenuSystem:
    """Beautiful menu system with scrolling support"""
    
    def __init__(self, logger: PremiumLogger):
        self.logger = logger
        self.menu_stack = []
        self.current_menu = None
    
    def create_menu(self, title: str, options: List[Dict[str, Any]], 
                   description: str = ""):
        """Create a beautiful menu"""
        menu = {
            "title": title,
            "description": description,
            "options": options,
            "selected_index": 0
        }
        return menu
    
    def display_menu(self, menu: Dict[str, Any]):
        """Display menu with beautiful formatting"""
        
        # Clear screen and display header
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Menu title with decorations
        title_color = "\033[96m"
        border_color = "\033[94m"
        reset = "\033[0m"
        
        border = border_color + "═" * 60 + reset
        title = f"{title_color}🎯 {menu['title']} 🎯{reset}"
        
        print(border)
        print(f"{title:^80}")
        if menu["description"]:
            desc = f"{border_color}{menu['description']}{reset}"
            print(f"{desc:^80}")
        print(border)
        print()
        
        # Display options
        for i, option in enumerate(menu["options"]):
            prefix = "👉" if i == menu["selected_index"] else "  "
            option_color = "\033[93m" if i == menu["selected_index"] else "\033[97m"
            
            option_text = f"{prefix} {option_color}{option['label']}{reset}"
            
            if "description" in option:
                option_text += f" {border_color}- {option['description']}{reset}"
            
            print(option_text)
        
        print()
        print(f"{border_color}Use ↑/↓ to navigate, Enter to select, 'q' to quit{reset}")
        print(border)
    
    def show_menu(self, menu: Dict[str, Any]) -> Optional[Any]:
        """Show interactive menu and return selected option"""
        self.current_menu = menu
        
        while True:
            self.display_menu(menu)
            
            try:
                # Get user input (simplified for demo)
                choice = input(f"\n👉 Select option (1-{len(menu['options'])}): ").strip()
                
                if choice.lower() == 'q':
                    return None
                
                try:
                    index = int(choice) - 1
                    if 0 <= index < len(menu["options"]):
                        selected_option = menu["options"][index]
                        self.logger.success(f"Selected: {selected_option['label']}")
                        return selected_option
                    else:
                        self.logger.warning("Invalid option number")
                        
                except ValueError:
                    self.logger.warning("Please enter a valid number")
                    
            except KeyboardInterrupt:
                self.logger.info("Menu cancelled by user")
                return None
            except Exception as e:
                self.logger.error(f"Menu error: {str(e)}")


def demo_premium_display_system():
    """Demonstration of the premium display system"""
    
    # Initialize systems
    display = ScrollingDisplay(max_lines=15, width=80)
    logger = PremiumLogger(display)
    progress_tracker = AdvancedProgressTracker(logger)
    menu_system = BeautifulMenuSystem(logger)
    
    # Welcome message with typewriter effect
    DisplayEffects.typewriter_effect(
        "🎉 Welcome to NICEGOLD ProjectP Premium Display System! 🎉",
        delay=0.05,
        color="\033[96m"
    )
    
    time.sleep(1)
    
    # Demo different message types
    logger.success("System initialization completed successfully!")
    logger.info("Loading configuration files...")
    logger.warning("Some optional components are not available")
    logger.error("Failed to connect to external data source")
    logger.debug("Debug: Memory usage: 45.2MB")
    
    # Demo progress tracking
    progress_tracker.start_task("data_load", "Loading Market Data", 50)
    
    for i in range(51):
        progress_tracker.update_task("data_load", i, f"Processing batch {i+1}/50")
        time.sleep(0.1)
    
    progress_tracker.complete_task("data_load", success=True)
    
    # Demo critical error
    logger.critical("Database connection lost - attempting recovery")
    
    # Show error summary
    logger.display_error_summary()
    
    # Demo menu system
    main_menu = menu_system.create_menu(
        title="NICEGOLD ProjectP Main Menu",
        description="Select an option to continue",
        options=[
            {"label": "🚀 Full Pipeline", "action": "full_pipeline", 
             "description": "Run complete analysis pipeline"},
            {"label": "📊 Data Analysis", "action": "data_analysis",
             "description": "Advanced data analysis tools"},
            {"label": "⚙️ Settings", "action": "settings",
             "description": "Configure system settings"},
            {"label": "❌ Exit", "action": "exit",
             "description": "Exit the application"}
        ]
    )
    
    selected = menu_system.show_menu(main_menu)
    if selected:
        logger.success(f"Executing: {selected['action']}")
    
    # Save log
    display.save_log("premium_display_demo.json")
    
    logger.success("Premium Display System demo completed!")


if __name__ == "__main__":
    demo_premium_display_system()
