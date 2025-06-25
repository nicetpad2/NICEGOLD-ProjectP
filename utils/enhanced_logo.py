#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Logo System for NICEGOLD ProjectP v2.0
Modern, beautiful and professional logos for all modes
"""

import random
import time
from datetime import datetime

from utils.colors import Colors, colorize


class ProjectPLogo:
    """Enhanced logo system with multiple styles and animations"""
    
    @staticmethod
    def get_modern_logo():
        """Get the main modern ProjectP logo"""
        return f"""
{colorize('╔═══════════════════════════════════════════════════════════════════════════════════════╗', Colors.BRIGHT_CYAN)}
{colorize('║', Colors.BRIGHT_CYAN)}                                                                                   {colorize('║', Colors.BRIGHT_CYAN)}
{colorize('║', Colors.BRIGHT_CYAN)}    {colorize('██████╗ ██████╗  ██████╗      ██╗███████╗ ██████╗████████╗██████╗ ', Colors.BRIGHT_MAGENTA)}    {colorize('║', Colors.BRIGHT_CYAN)}
{colorize('║', Colors.BRIGHT_CYAN)}    {colorize('██╔══██╗██╔══██╗██╔═══██╗     ██║██╔════╝██╔════╝╚══██╔══╝██╔══██╗', Colors.BRIGHT_MAGENTA)}    {colorize('║', Colors.BRIGHT_CYAN)}
{colorize('║', Colors.BRIGHT_CYAN)}    {colorize('██████╔╝██████╔╝██║   ██║     ██║█████╗  ██║        ██║   ██████╔╝', Colors.BRIGHT_MAGENTA)}    {colorize('║', Colors.BRIGHT_CYAN)}
{colorize('║', Colors.BRIGHT_CYAN)}    {colorize('██╔═══╝ ██╔══██╗██║   ██║██   ██║██╔══╝  ██║        ██║   ██╔═══╝ ', Colors.BRIGHT_MAGENTA)}    {colorize('║', Colors.BRIGHT_CYAN)}
{colorize('║', Colors.BRIGHT_CYAN)}    {colorize('██║     ██║  ██║╚██████╔╝╚█████╔╝███████╗╚██████╗   ██║   ██║     ', Colors.BRIGHT_MAGENTA)}    {colorize('║', Colors.BRIGHT_CYAN)}
{colorize('║', Colors.BRIGHT_CYAN)}    {colorize('╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝   ╚═╝     ', Colors.BRIGHT_MAGENTA)}    {colorize('║', Colors.BRIGHT_CYAN)}
{colorize('║', Colors.BRIGHT_CYAN)}                                                                                   {colorize('║', Colors.BRIGHT_CYAN)}
{colorize('║', Colors.BRIGHT_CYAN)}                           {colorize('🚀 NICEGOLD ENTERPRISE v2.0', Colors.BOLD + Colors.BRIGHT_YELLOW)}                          {colorize('║', Colors.BRIGHT_CYAN)}
{colorize('║', Colors.BRIGHT_CYAN)}                        {colorize('Professional AI Trading System', Colors.BRIGHT_WHITE)}                         {colorize('║', Colors.BRIGHT_CYAN)}
{colorize('║', Colors.BRIGHT_CYAN)}                                                                                   {colorize('║', Colors.BRIGHT_CYAN)}
{colorize('║', Colors.BRIGHT_CYAN)}  {colorize('💎 Machine Learning', Colors.BRIGHT_YELLOW)}  {colorize('📊 Real-time Analytics', Colors.BRIGHT_GREEN)}  {colorize('🎯 Smart Trading', Colors.BRIGHT_BLUE)}  {colorize('⚡ High Performance', Colors.BRIGHT_RED)}  {colorize('║', Colors.BRIGHT_CYAN)}
{colorize('║', Colors.BRIGHT_CYAN)}                                                                                   {colorize('║', Colors.BRIGHT_CYAN)}
{colorize('╚═══════════════════════════════════════════════════════════════════════════════════════╝', Colors.BRIGHT_CYAN)}
"""

    @staticmethod
    def get_compact_logo():
        """Get compact logo for space-limited displays"""
        return f"""
{colorize('╔═══════════════════════════════════════════════════════════════════════╗', Colors.BRIGHT_CYAN)}
{colorize('║', Colors.BRIGHT_CYAN)}  {colorize('ProjectP', Colors.BOLD + Colors.BRIGHT_MAGENTA)} {colorize('v2.0', Colors.BRIGHT_YELLOW)} - {colorize('NICEGOLD Enterprise', Colors.BRIGHT_WHITE)} {colorize('🚀', Colors.BRIGHT_GREEN)}  {colorize('║', Colors.BRIGHT_CYAN)}
{colorize('║', Colors.BRIGHT_CYAN)}  {colorize('Professional AI Trading System', Colors.BRIGHT_CYAN)}                     {colorize('║', Colors.BRIGHT_CYAN)}
{colorize('╚═══════════════════════════════════════════════════════════════════════╝', Colors.BRIGHT_CYAN)}
"""

    @staticmethod
    def get_ascii_logo():
        """Get stylized ASCII logo"""
        return f"""
{colorize('     ██████╗ ██████╗  ██████╗      ██╗███████╗ ██████╗████████╗██████╗ ', Colors.BRIGHT_MAGENTA)}
{colorize('     ██╔══██╗██╔══██╗██╔═══██╗     ██║██╔════╝██╔════╝╚══██╔══╝██╔══██╗', Colors.BRIGHT_MAGENTA)}
{colorize('     ██████╔╝██████╔╝██║   ██║     ██║█████╗  ██║        ██║   ██████╔╝', Colors.BRIGHT_MAGENTA)}
{colorize('     ██╔═══╝ ██╔══██╗██║   ██║██   ██║██╔══╝  ██║        ██║   ██╔═══╝ ', Colors.BRIGHT_MAGENTA)}
{colorize('     ██║     ██║  ██║╚██████╔╝╚█████╔╝███████╗╚██████╗   ██║   ██║     ', Colors.BRIGHT_MAGENTA)}
{colorize('     ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝   ╚═╝     ', Colors.BRIGHT_MAGENTA)}
                                                                              
                   {colorize('🚀 NICEGOLD ENTERPRISE TRADING SYSTEM', Colors.BOLD + Colors.BRIGHT_YELLOW)}
                        {colorize('Professional AI Trading Platform v2.0', Colors.BRIGHT_WHITE)}
"""

    @staticmethod
    def get_minimal_logo():
        """Get minimal logo for terminal headers"""
        return f"{colorize('ProjectP', Colors.BOLD + Colors.BRIGHT_MAGENTA)} {colorize('v2.0', Colors.BRIGHT_YELLOW)} | {colorize('NICEGOLD Enterprise', Colors.BRIGHT_CYAN)} {colorize('🚀', Colors.BRIGHT_GREEN)}"

    @staticmethod
    def get_startup_animation():
        """Get animated startup sequence"""
        frames = [
            f"    {colorize('●○○', Colors.BRIGHT_RED)} {colorize('Initializing ProjectP...', Colors.WHITE)}",
            f"    {colorize('●●○', Colors.BRIGHT_YELLOW)} {colorize('Loading Enterprise Modules...', Colors.WHITE)}",
            f"    {colorize('●●●', Colors.BRIGHT_GREEN)} {colorize('System Ready!', Colors.WHITE)}"
        ]
        return frames

    @staticmethod
    def print_welcome_sequence():
        """Print animated welcome sequence"""
        print("\n")
        
        # Animated loading
        frames = ProjectPLogo.get_startup_animation()
        for frame in frames:
            print(frame)
            time.sleep(0.8)
        
        print("\n")
        
        # Main logo
        print(ProjectPLogo.get_modern_logo())
        
        # Welcome message
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"    {colorize('Welcome to the Future of Trading!', Colors.BOLD + Colors.BRIGHT_GREEN)}")
        print(f"    {colorize('System initialized at:', Colors.BRIGHT_CYAN)} {colorize(current_time, Colors.WHITE)}")
        print()

    @staticmethod
    def get_footer():
        """Get footer with system info"""
        return f"""
{colorize('─' * 90, Colors.DIM)}
{colorize('© 2025 NICEGOLD Enterprise', Colors.DIM)} | {colorize('Professional Trading Platform', Colors.DIM)} | {colorize('Built with ❤️  in Thailand', Colors.DIM)}
{colorize('─' * 90, Colors.DIM)}
"""

    @staticmethod
    def get_status_header(title="SYSTEM STATUS", mode="normal"):
        """Get status header with different modes"""
        modes = {
            "normal": Colors.BRIGHT_CYAN,
            "success": Colors.BRIGHT_GREEN,
            "warning": Colors.BRIGHT_YELLOW,
            "error": Colors.BRIGHT_RED,
            "info": Colors.BRIGHT_BLUE
        }
        
        color = modes.get(mode, Colors.BRIGHT_CYAN)
        
        return f"""
{colorize('╔═══════════════════════════════════════════════════════════════════════╗', color)}
{colorize('║', color)} {colorize(f'📊 {title}', Colors.BOLD + color):^69} {colorize('║', color)}
{colorize('╠═══════════════════════════════════════════════════════════════════════╣', color)}
"""

    @staticmethod
    def get_section_header(title, icon="🔧", color=Colors.BRIGHT_CYAN):
        """Get section header"""
        return f"""
{colorize('┌─────────────────────────────────────────────────────────────────────┐', color)}
{colorize('│', color)} {colorize(f'{icon} {title}', Colors.BOLD + color):^67} {colorize('│', color)}
{colorize('└─────────────────────────────────────────────────────────────────────┘', color)}
"""

    @staticmethod
    def get_loading_animation(message="Processing", duration=2.0):
        """Get loading animation with spinner"""
        chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        end_time = time.time() + duration
        i = 0
        
        while time.time() < end_time:
            char = chars[i % len(chars)]
            print(f"\r{colorize(char, Colors.BRIGHT_CYAN)} {colorize(message, Colors.WHITE)}", end="", flush=True)
            time.sleep(0.1)
            i += 1
        
        print(f"\r{colorize('✅', Colors.BRIGHT_GREEN)} {colorize(message, Colors.WHITE)} - {colorize('เสร็จสิ้น!', Colors.BRIGHT_GREEN)}")

    @staticmethod
    def print_mode_banner(mode_name, description="", color=Colors.BRIGHT_MAGENTA):
        """Print mode banner"""
        banner = f"""
{colorize('╔═══════════════════════════════════════════════════════════════════════╗', color)}
{colorize('║', color)} {colorize(f'🎯 {mode_name.upper()} MODE', Colors.BOLD + color):^67} {colorize('║', color)}
{colorize('║', color)} {colorize(description, Colors.WHITE):^67} {colorize('║', color)}
{colorize('╚═══════════════════════════════════════════════════════════════════════╝', color)}
"""
        print(banner)
