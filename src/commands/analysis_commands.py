# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - Analysis Commands
════════════════════════════════════════════════════════════════════════════════

Data analysis and statistical commands.

Author: NICEGOLD Team
Version: 3.0
Created: 2025-01-05
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import from parent modules
sys.path.append(str(Path(__file__).parent.parent))
from core.colors import Colors, colorize


class AnalysisCommands:
    """Handler for data analysis and statistical commands"""

    def __init__(self, project_root: Path, csv_manager=None, logger=None):
        self.project_root = project_root
        self.csv_manager = csv_manager
        self.logger = logger
        self.python_cmd = [sys.executable]

    def run_command(
        self, command: List[str], description: str, capture_output: bool = False
    ) -> bool:
        """Execute a command with proper error handling"""
        try:
            print(f"\n{colorize('⚡ ' + description, Colors.BRIGHT_CYAN)}")
            print(f"{colorize('═' * 60, Colors.DIM)}")

            # Ensure we're in the right directory
            os.chdir(self.project_root)

            if capture_output:
                result = subprocess.run(
                    command, capture_output=True, text=True, cwd=self.project_root
                )
                if result.returncode == 0:
                    print(
                        f"{colorize('✅ Command completed successfully', Colors.BRIGHT_GREEN)}"
                    )
                    if result.stdout:
                        print(result.stdout)
                    return True
                else:
                    print(f"{colorize('❌ Command failed', Colors.BRIGHT_RED)}")
                    if result.stderr:
                        print(result.stderr)
                    return False
            else:
                result = subprocess.run(command, cwd=self.project_root)
                return result.returncode == 0

        except Exception as e:
            print(f"{colorize('❌ Error executing command:', Colors.BRIGHT_RED)} {e}")
            if self.logger:
                self.logger.error(
                    f"Command execution failed: {description}", "ANALYSIS", str(e), e
                )
            return False

    def data_analysis_statistics(self) -> bool:
        """Perform data analysis and statistics with real data"""
        print(f"{colorize('📊 Starting Data Analysis...', Colors.BRIGHT_BLUE)}")

        if self.csv_manager:
            try:
                # Show detailed analysis of all CSV files
                self.csv_manager.print_validation_report()

                # Get and analyze best CSV file
                best_csv = self.csv_manager.get_best_csv_file()
                if best_csv:
                    print(
                        f"{colorize('✅ Analyzing:', Colors.BRIGHT_GREEN)} {best_csv}"
                    )

                    # Load and show basic statistics
                    df = self.csv_manager.validate_and_standardize_csv(best_csv)

                    print(
                        f"\n{colorize('📈 Dataset Overview:', Colors.BOLD + Colors.BRIGHT_BLUE)}"
                    )
                    print(f"  • {colorize('Rows:', Colors.BRIGHT_CYAN)} {len(df):,}")
                    print(
                        f"  • {colorize('Columns:', Colors.BRIGHT_CYAN)} {len(df.columns)}"
                    )
                    print(
                        f"  • {colorize('Memory Usage:', Colors.BRIGHT_CYAN)} {df.memory_usage().sum() / 1024**2:.2f} MB"
                    )

                    # Show column information
                    print(
                        f"\n{colorize('📋 Column Information:', Colors.BOLD + Colors.BRIGHT_BLUE)}"
                    )
                    for col in df.columns:
                        dtype = str(df[col].dtype)
                        null_count = df[col].isnull().sum()
                        print(
                            f"  • {colorize(col, Colors.BRIGHT_WHITE)}: {dtype} ({null_count} nulls)"
                        )

                    # Show basic statistics
                    print(
                        f"\n{colorize('📊 Statistical Summary:', Colors.BOLD + Colors.BRIGHT_BLUE)}"
                    )
                    print(df.describe())

                else:
                    print(
                        f"{colorize('❌ No suitable CSV files found', Colors.BRIGHT_RED)}"
                    )
                    return False

            except Exception as e:
                print(
                    f"{colorize('❌ Data analysis failed:', Colors.BRIGHT_RED)} {str(e)}"
                )
                return False
        else:
            print(f"{colorize('❌ CSV Manager not available', Colors.BRIGHT_RED)}")
            return False

        return True

    def performance_analysis(self) -> bool:
        """Run performance analysis"""
        print(
            f"{colorize('📈 Starting Performance Analysis...', Colors.BRIGHT_MAGENTA)}"
        )

        return self.run_command(
            [
                "python",
                "-c",
                "from src.analysis import analyze_performance; analyze_performance()",
            ],
            "Performance Analysis",
        )

    def risk_analysis(self) -> bool:
        """Run risk management analysis"""
        print(f"{colorize('⚠️ Starting Risk Analysis...', Colors.BRIGHT_YELLOW)}")

        return self.run_command(
            [
                "python",
                "-c",
                "from src.risk_management import analyze_risk; analyze_risk()",
            ],
            "Risk Management Analysis",
        )

    def model_comparison(self) -> bool:
        """Run model comparison analysis"""
        print(f"{colorize('🤖 Starting Model Comparison...', Colors.BRIGHT_CYAN)}")

        return self.run_command(
            [
                "python",
                "-c",
                "from src.model_comparison import compare_models; compare_models()",
            ],
            "Model Comparison Analysis",
        )

    def hyperparameter_sweep(self) -> bool:
        """Run hyperparameter optimization sweep"""
        print(f"{colorize('⚙️ Starting Hyperparameter Sweep...', Colors.BRIGHT_BLUE)}")
        print(f"{colorize('This may take several minutes...', Colors.DIM)}")

        return self.run_command(
            ["python", "tuning/hyperparameter_sweep.py"],
            "Hyperparameter Optimization Sweep",
        )

    def threshold_optimization(self) -> bool:
        """Run threshold optimization"""
        print(
            f"{colorize('🎯 Starting Threshold Optimization...', Colors.BRIGHT_GREEN)}"
        )

        return self.run_command(
            ["python", "threshold_optimization.py"], "Decision Threshold Optimization"
        )

    def comprehensive_analysis(self) -> bool:
        """Run comprehensive analysis suite"""
        print(
            f"{colorize('🔍 Starting Comprehensive Analysis...', Colors.BOLD + Colors.BRIGHT_BLUE)}"
        )
        print(f"{colorize('Running all analysis modules...', Colors.DIM)}")

        success = True

        # Run each analysis component
        analyses = [
            ("Data Statistics", self.data_analysis_statistics),
            ("Performance Analysis", self.performance_analysis),
            ("Risk Analysis", self.risk_analysis),
            ("Model Comparison", self.model_comparison),
        ]

        for name, func in analyses:
            print(f"\n{colorize(f'Running {name}...', Colors.BRIGHT_CYAN)}")
            if not func():
                print(f"{colorize(f'❌ {name} failed', Colors.BRIGHT_RED)}")
                success = False
            else:
                print(f"{colorize(f'✅ {name} completed', Colors.BRIGHT_GREEN)}")

        if success:
            print(
                f"\n{colorize('🎉 Comprehensive analysis completed successfully!', Colors.BOLD + Colors.BRIGHT_GREEN)}"
            )
        else:
            print(
                f"\n{colorize('⚠️ Some analyses failed - check logs for details', Colors.BRIGHT_YELLOW)}"
            )

        return success
