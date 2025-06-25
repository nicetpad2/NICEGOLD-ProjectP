# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - Trading Commands
════════════════════════════════════════════════════════════════════════════════

Trading-related commands and live simulation.

Author: NICEGOLD Team
Version: 3.0
Created: 2025-01-05
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import from parent modules
sys.path.append(str(Path(__file__).parent.parent))
from core.colors import Colors, colorize


class TradingCommands:
    """Handler for trading-related commands"""

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
                    f"Command execution failed: {description}", "TRADING", str(e), e
                )
            return False

    def start_live_simulation(self) -> bool:
        """Start live trading simulation"""
        print(
            f"{colorize('🔴 Starting Live Trading Simulation...', Colors.BRIGHT_RED)}"
        )
        print(
            f"{colorize('⚠️  This will run in the background. Use monitoring tools to track progress.', Colors.BRIGHT_YELLOW)}"
        )

        return self.run_command(
            [
                "python",
                "-c",
                "from src.live_trading import start_live_simulation; start_live_simulation()",
            ],
            "Live Trading Simulation",
        )

    def start_monitoring(self) -> bool:
        """Start system monitoring"""
        print(f"{colorize('📊 Starting System Monitoring...', Colors.BRIGHT_BLUE)}")
        print(
            f"{colorize('Monitor will track system performance and trading metrics.', Colors.DIM)}"
        )

        return self.run_command(
            ["python", "-c", "from src.monitor import start_monitor; start_monitor()"],
            "System Monitoring",
        )

    def realistic_backtest_live(self) -> bool:
        """Run realistic backtest with live simulation"""
        print(
            f"{colorize('📈 Starting Realistic Backtest (Live Mode)...', Colors.BRIGHT_MAGENTA)}"
        )
        print(
            f"{colorize('Live simulation with real-time data processing', Colors.DIM)}"
        )

        return self.run_command(
            ["python", "main.py", "--mode", "realistic_backtest_live"],
            "Realistic Backtest - Live Simulation",
        )

    def paper_trading(self) -> bool:
        """Start paper trading session"""
        print(f"{colorize('📋 Starting Paper Trading...', Colors.BRIGHT_GREEN)}")
        print(f"{colorize('Virtual trading with real market conditions', Colors.DIM)}")

        # Create paper trading script
        paper_trading_code = f"""
import sys
import os
sys.path.append('.')
os.chdir('{self.project_root}')

print('📋 NICEGOLD Paper Trading Session')
print('=' * 50)

try:
    # Import required modules
    from src.live_trading import start_live_simulation
    import threading
    import time
    
    print('🔄 Initializing paper trading environment...')
    
    # Start live simulation in paper trading mode
    def run_paper_trading():
        print('📊 Starting paper trading simulation...')
        start_live_simulation()
    
    # Start monitoring
    def run_monitoring():
        print('📈 Starting trading monitor...')
        from src.monitor import start_monitor
        start_monitor()
    
    # Run both in parallel
    trading_thread = threading.Thread(target=run_paper_trading)
    monitor_thread = threading.Thread(target=run_monitoring)
    
    trading_thread.start()
    time.sleep(2)  # Give trading a head start
    monitor_thread.start()
    
    print('✅ Paper trading session started successfully!')
    print('📊 Monitoring dashboard running in parallel')
    print('⏹️  Press Ctrl+C to stop')
    
    trading_thread.join()
    monitor_thread.join()

except ImportError as e:
    print(f'❌ Import error: {{e}}')
    print('💡 Please ensure all trading modules are available')
except Exception as e:
    print(f'❌ Paper trading error: {{e}}')
    import traceback
    traceback.print_exc()
"""

        return self.run_command(
            self.python_cmd + ["-c", paper_trading_code], "Paper Trading Session"
        )

    def trading_signals_analysis(self) -> bool:
        """Analyze trading signals"""
        print(f"{colorize('📡 Analyzing Trading Signals...', Colors.BRIGHT_CYAN)}")

        # Create signals analysis script
        signals_code = f"""
import sys
import os
import pandas as pd
import numpy as np
sys.path.append('.')
os.chdir('{self.project_root}')

print('📡 NICEGOLD Trading Signals Analysis')
print('=' * 50)

try:
    # Check for recent predictions/signals
    import glob
    
    # Look for signal files
    signal_files = glob.glob('output*/predictions*.csv') + glob.glob('signals/*.csv')
    
    if signal_files:
        print(f'📊 Found {{len(signal_files)}} signal files')
        
        for file in signal_files[:5]:  # Analyze recent files
            print(f'\\n📈 Analyzing: {{file}}')
            df = pd.read_csv(file)
            
            if 'prediction' in df.columns or 'signal' in df.columns:
                signal_col = 'prediction' if 'prediction' in df.columns else 'signal'
                
                # Basic signal statistics
                total_signals = len(df)
                buy_signals = (df[signal_col] == 1).sum() if signal_col in df.columns else 0
                sell_signals = (df[signal_col] == 0).sum() if signal_col in df.columns else 0
                
                print(f'  • Total signals: {{total_signals:,}}')
                print(f'  • Buy signals: {{buy_signals:,}} ({{buy_signals/total_signals*100:.1f}}%)')
                print(f'  • Sell signals: {{sell_signals:,}} ({{sell_signals/total_signals*100:.1f}}%)')
                
                # Recent signals (last 100)
                if len(df) > 100:
                    recent = df.tail(100)
                    recent_buy = (recent[signal_col] == 1).sum()
                    print(f'  • Recent buy signals (last 100): {{recent_buy}} ({{recent_buy/100*100:.1f}}%)')
            else:
                print(f'  • Signal column not found in {{file}}')
    else:
        print('❌ No signal files found')
        print('💡 Run a prediction pipeline first to generate signals')

except Exception as e:
    print(f'❌ Signals analysis error: {{e}}')
    import traceback
    traceback.print_exc()
"""

        return self.run_command(
            self.python_cmd + ["-c", signals_code], "Trading Signals Analysis"
        )

    def performance_metrics(self) -> bool:
        """Calculate trading performance metrics"""
        print(
            f"{colorize('📊 Calculating Performance Metrics...', Colors.BRIGHT_YELLOW)}"
        )

        # Create performance metrics script
        metrics_code = f"""
import sys
import os
import pandas as pd
import numpy as np
sys.path.append('.')
os.chdir('{self.project_root}')

print('📊 NICEGOLD Performance Metrics')
print('=' * 50)

try:
    import glob
    
    # Look for results files
    result_files = glob.glob('output*/results*.csv') + glob.glob('output*/evaluation*.csv')
    
    if result_files:
        print(f'📈 Found {{len(result_files)}} result files')
        
        for file in result_files[:3]:  # Analyze recent files
            print(f'\\n📊 Analyzing: {{file}}')
            df = pd.read_csv(file)
            
            print(f'  • Columns: {{list(df.columns)}}')
            print(f'  • Shape: {{df.shape}}')
            
            # Look for common performance columns
            perf_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'profit', 'return']
            found_cols = [col for col in df.columns if any(perf in col.lower() for perf in perf_cols)]
            
            if found_cols:
                print(f'  • Performance metrics found: {{found_cols}}')
                for col in found_cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        print(f'    - {{col}}: {{df[col].iloc[-1]:.4f}}' if len(df) > 0 else f'    - {{col}}: No data')
            else:
                print('  • No standard performance metrics found')
    else:
        print('❌ No result files found')
        print('💡 Run a pipeline first to generate performance data')

except Exception as e:
    print(f'❌ Performance metrics error: {{e}}')
    import traceback
    traceback.print_exc()
"""

        return self.run_command(
            self.python_cmd + ["-c", metrics_code], "Performance Metrics Calculation"
        )
