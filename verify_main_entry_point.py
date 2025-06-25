#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - Main Entry Point Verification
═══════════════════════════════════════════════════

This script verifies that ProjectP_refactored.py is properly set up
as the main entry point for the NICEGOLD ProjectP system.
"""

import os
import subprocess
import sys
from pathlib import Path


# ANSI color codes
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_header():
    """Print verification header"""
    print(f"{Colors.CYAN}╔" + "═" * 70 + "╗")
    print(f"║{' ' * 70}║")
    print(f"║{'🔍 NICEGOLD ProjectP - Entry Point Verification'.center(70)}║")
    print(f"║{' ' * 70}║")
    print(f"╚" + "═" * 70 + "╝{Colors.END}")
    print()


def check_file_exists(file_path, description):
    """Check if a file exists"""
    if file_path.exists():
        print(f"{Colors.GREEN}✅ {description}: {file_path}{Colors.END}")
        return True
    else:
        print(f"{Colors.RED}❌ {description}: {file_path} (NOT FOUND){Colors.END}")
        return False


def check_file_executable(file_path, description):
    """Check if a file is executable"""
    if file_path.exists() and os.access(file_path, os.X_OK):
        print(f"{Colors.GREEN}✅ {description}: Executable{Colors.END}")
        return True
    elif file_path.exists():
        print(
            f"{Colors.YELLOW}⚠️ {description}: Not executable (chmod +x needed){Colors.END}"
        )
        return False
    else:
        print(f"{Colors.RED}❌ {description}: File not found{Colors.END}")
        return False


def run_command(command, description):
    """Run a command and check the result"""
    try:
        print(f"{Colors.BLUE}🔍 Testing: {description}{Colors.END}")
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            print(f"{Colors.GREEN}✅ {description}: SUCCESS{Colors.END}")
            # Show first few lines of output
            output_lines = result.stdout.strip().split("\n")[:3]
            for line in output_lines:
                print(f"   {Colors.WHITE}{line}{Colors.END}")
            return True
        else:
            print(f"{Colors.RED}❌ {description}: FAILED{Colors.END}")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"{Colors.YELLOW}⚠️ {description}: TIMEOUT{Colors.END}")
        return False
    except Exception as e:
        print(f"{Colors.RED}❌ {description}: ERROR - {e}{Colors.END}")
        return False


def check_directory_structure():
    """Check the modular directory structure"""
    project_root = Path.cwd()

    directories = [
        ("src/", "Source directory"),
        ("src/core/", "Core modules"),
        ("src/ui/", "UI modules"),
        ("src/system/", "System modules"),
        ("src/commands/", "Command modules"),
        ("src/api/", "API modules"),
    ]

    all_good = True
    print(f"\n{Colors.PURPLE}📁 Directory Structure Check:{Colors.END}")

    for dir_path, description in directories:
        full_path = project_root / dir_path
        if check_file_exists(full_path, description):
            continue
        else:
            all_good = False

    return all_good


def check_key_files():
    """Check for key files"""
    project_root = Path.cwd()

    files = [
        ("ProjectP_refactored.py", "Main entry point"),
        ("start_nicegold.sh", "Startup script"),
        ("src/core/__init__.py", "Core module init"),
        ("src/ui/__init__.py", "UI module init"),
        ("src/system/__init__.py", "System module init"),
        ("src/commands/__init__.py", "Commands module init"),
        ("src/api/__init__.py", "API module init"),
    ]

    all_good = True
    print(f"\n{Colors.PURPLE}📄 Key Files Check:{Colors.END}")

    for file_path, description in files:
        full_path = project_root / file_path
        if not check_file_exists(full_path, description):
            all_good = False

    return all_good


def check_executability():
    """Check if main files are executable"""
    project_root = Path.cwd()

    files = [
        (project_root / "ProjectP_refactored.py", "Main entry point"),
        (project_root / "start_nicegold.sh", "Startup script"),
    ]

    all_good = True
    print(f"\n{Colors.PURPLE}🔧 Executability Check:{Colors.END}")

    for file_path, description in files:
        if not check_file_executable(file_path, description):
            all_good = False

    return all_good


def test_command_line_options():
    """Test command line options"""
    commands = [
        ("python ProjectP_refactored.py --version", "Version command"),
        ("python ProjectP_refactored.py --help", "Help command"),
    ]

    all_good = True
    print(f"\n{Colors.PURPLE}🧪 Command Line Tests:{Colors.END}")

    for command, description in commands:
        if not run_command(command, description):
            all_good = False

    return all_good


def test_startup_script():
    """Test the startup script"""
    commands = [
        ("./start_nicegold.sh version", "Startup script version"),
        ("./start_nicegold.sh help", "Startup script help"),
    ]

    all_good = True
    print(f"\n{Colors.PURPLE}🚀 Startup Script Tests:{Colors.END}")

    for command, description in commands:
        if not run_command(command, description):
            all_good = False

    return all_good


def check_python_imports():
    """Test Python imports"""
    import_tests = [
        "import sys; sys.path.insert(0, 'src'); from core.colors import Colors",
        "import sys; sys.path.insert(0, 'src'); from ui.animations import print_logo",
        "import sys; sys.path.insert(0, 'src'); from system.health_monitor import SystemHealthMonitor",
    ]

    all_good = True
    print(f"\n{Colors.PURPLE}🐍 Python Import Tests:{Colors.END}")

    for i, import_test in enumerate(import_tests, 1):
        command = f'python -c "{import_test}"'
        if not run_command(command, f"Import test {i}"):
            all_good = False

    return all_good


def main():
    """Main verification function"""
    print_header()

    print(f"{Colors.BLUE}📍 Project Directory: {Path.cwd()}{Colors.END}")
    print(f"{Colors.BLUE}🐍 Python Version: {sys.version.split()[0]}{Colors.END}")
    print()

    # Run all checks
    checks = [
        ("Directory Structure", check_directory_structure),
        ("Key Files", check_key_files),
        ("Executability", check_executability),
        ("Command Line Options", test_command_line_options),
        ("Startup Script", test_startup_script),
        ("Python Imports", check_python_imports),
    ]

    results = []
    for check_name, check_function in checks:
        try:
            result = check_function()
            results.append((check_name, result))
        except Exception as e:
            print(f"{Colors.RED}❌ {check_name}: ERROR - {e}{Colors.END}")
            results.append((check_name, False))

    # Summary
    print(f"\n{Colors.BOLD}📊 VERIFICATION SUMMARY:{Colors.END}")
    print("═" * 50)

    passed = 0
    total = len(results)

    for check_name, result in results:
        if result:
            print(f"{Colors.GREEN}✅ {check_name}: PASSED{Colors.END}")
            passed += 1
        else:
            print(f"{Colors.RED}❌ {check_name}: FAILED{Colors.END}")

    print("═" * 50)

    if passed == total:
        print(
            f"{Colors.GREEN}{Colors.BOLD}🎉 ALL CHECKS PASSED! ({passed}/{total}){Colors.END}"
        )
        print(
            f"{Colors.GREEN}✅ ProjectP_refactored.py is ready to use as the main entry point!{Colors.END}"
        )
        print(f"\n{Colors.CYAN}🚀 Quick Start Commands:{Colors.END}")
        print(f"   {Colors.WHITE}python ProjectP_refactored.py{Colors.END}")
        print(f"   {Colors.WHITE}./start_nicegold.sh{Colors.END}")
    else:
        print(
            f"{Colors.YELLOW}⚠️ SOME CHECKS FAILED ({passed}/{total} passed){Colors.END}"
        )
        print(
            f"{Colors.YELLOW}💡 Please fix the issues above before using the system.{Colors.END}"
        )

    print(f"\n{Colors.CYAN}📚 Documentation:{Colors.END}")
    print(f"   - {Colors.WHITE}MAIN_ENTRY_POINT_GUIDE.md{Colors.END}")
    print(f"   - {Colors.WHITE}QUICK_START.md{Colors.END}")
    print(f"   - {Colors.WHITE}REFACTORING_COMPLETION_REPORT.md{Colors.END}")


if __name__ == "__main__":
    main()
