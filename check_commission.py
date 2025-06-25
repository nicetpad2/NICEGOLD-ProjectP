#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Commission Check - Tests commission setting in the main files
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


# Color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    END = "\033[0m"


def check_commission_in_files():
    """Check commission setting in key files"""
    print(
        f"{Colors.BOLD}{Colors.CYAN}🔍 COMMISSION VERIFICATION IN SOURCE FILES{Colors.END}"
    )
    print("=" * 60)

    files_to_check = [
        "src/commands/pipeline_commands.py",
        "src/commands/advanced_results_summary.py",
        "src/strategy.py",
        "src/cost.py",
    ]

    commission_found = {}

    for file_path in files_to_check:
        full_path = project_root / file_path
        if full_path.exists():
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Check for commission patterns
                if "0.07" in content:
                    commission_found[file_path] = True
                    print(
                        f"{Colors.GREEN}✅ {file_path}: Commission 0.07 found{Colors.END}"
                    )

                    # Look for specific patterns
                    if "mini lot" in content.lower():
                        print(
                            f"   {Colors.BLUE}📌 Contains 'mini lot' reference{Colors.END}"
                        )
                    if "0.01 lot" in content:
                        print(
                            f"   {Colors.BLUE}📌 Contains '0.01 lot' reference{Colors.END}"
                        )

                else:
                    commission_found[file_path] = False
                    print(
                        f"{Colors.YELLOW}⚠️ {file_path}: Commission 0.07 not found{Colors.END}"
                    )

            except Exception as e:
                print(
                    f"{Colors.RED}❌ {file_path}: Error reading file - {e}{Colors.END}"
                )
        else:
            print(f"{Colors.RED}❌ {file_path}: File not found{Colors.END}")

    return commission_found


def check_commission_display():
    """Check commission display format"""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}💰 COMMISSION DISPLAY FORMAT{Colors.END}")
    print("-" * 40)

    commission = 0.07
    starting_capital = 100
    trades = 85

    print(
        f"• Commission Rate: {Colors.WHITE}${commission:.2f}{Colors.END} per 0.01 lot (mini lot)"
    )
    print(f"• Starting Capital: {Colors.GREEN}${starting_capital}{Colors.END}")
    print(f"• Sample Trades: {Colors.CYAN}{trades}{Colors.END}")
    print(f"• Total Commission: {Colors.RED}${trades * commission:.2f}{Colors.END}")
    print(
        f"• Commission Impact: {Colors.RED}{(trades * commission / starting_capital * 100):.2f}%{Colors.END}"
    )


def check_environment():
    """Check Python environment"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}🐍 ENVIRONMENT CHECK{Colors.END}")
    print("-" * 30)
    print(f"• Python Version: {Colors.WHITE}{sys.version.split()[0]}{Colors.END}")
    print(f"• Project Root: {Colors.WHITE}{project_root}{Colors.END}")
    print(f"• Working Directory: {Colors.WHITE}{os.getcwd()}{Colors.END}")


def main():
    """Main check function"""
    print(
        f"{Colors.BOLD}{Colors.GREEN}🧪 NICEGOLD ProjectP - Commission Verification{Colors.END}"
    )
    print("=" * 70)

    # Check environment
    check_environment()

    # Check files
    commission_results = check_commission_in_files()

    # Show display format
    check_commission_display()

    # Summary
    print(f"\n{Colors.BOLD}{Colors.GREEN}📊 VERIFICATION SUMMARY{Colors.END}")
    print("=" * 40)

    total_files = len(commission_results)
    found_files = sum(commission_results.values())

    print(f"• Files Checked: {Colors.CYAN}{total_files}{Colors.END}")
    print(f"• Files with Commission: {Colors.GREEN}{found_files}{Colors.END}")
    print(
        f"• Success Rate: {Colors.YELLOW}{(found_files/total_files*100):.1f}%{Colors.END}"
    )

    if found_files == total_files:
        print(
            f"\n{Colors.BOLD}{Colors.GREEN}🎯 COMMISSION VERIFICATION: PASSED!{Colors.END}"
        )
    else:
        print(
            f"\n{Colors.BOLD}{Colors.YELLOW}⚠️ COMMISSION VERIFICATION: PARTIAL{Colors.END}"
        )

    return found_files == total_files


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
