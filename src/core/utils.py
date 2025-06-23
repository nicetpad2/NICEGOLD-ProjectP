"""
🛠️ Utility Functions
===================

ฟังก์ชันเสริมต่างๆ ที่ใช้ร่วมกันในระบบ
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


class TableParser:
    """Parser สำหรับข้อมูลตาราง"""
    
    @staticmethod
    def parse_table(text: str) -> List[List[str]]:
        """Parse table data from text with error handling"""
        try:
            pattern = r'\|\s*(.+?)\s*\|'
            matches = re.findall(pattern, text)
            return [match.split('|') for match in matches]
        except Exception as e:
            print(f"❌ Table parsing failed: {e}")
            return []


class DataGenerator:
    """สร้างข้อมูลสำหรับการทดสอบ"""
    
    @staticmethod
    def create_synthetic_financial_data(n_samples: int = 10000) -> pd.DataFrame:
        """Create realistic synthetic financial data"""
        try:
            np.random.seed(42)  # For reproducibility
            
            # Generate realistic price data
            dates = pd.date_range('2020-01-01', periods=n_samples, freq='5min')
            
            # Price simulation with realistic market behavior
            returns = np.random.normal(0, 0.01, n_samples)  # 1% volatility
            returns[0] = 0  # First return is zero
            
            prices = 100 * np.exp(np.cumsum(returns))  # Geometric Brownian motion
            
            # OHLCV data
            high = prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples)))
            low = prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples)))
            volume = np.random.exponential(100000, n_samples)
            
            df = pd.DataFrame({
                'timestamp': dates,
                'Open': prices,
                'High': high,
                'Low': low,
                'Close': prices + np.random.normal(0, 0.1, n_samples),
                'Volume': volume
            })
            
            # Technical indicators
            df['returns'] = df['Close'].pct_change().fillna(0)
            df['sma_10'] = df['Close'].rolling(10, min_periods=1).mean()
            df['sma_30'] = df['Close'].rolling(30, min_periods=1).mean()
            df['rsi'] = 50 + np.random.normal(0, 15, n_samples)  # Simplified RSI
            df['rsi'] = np.clip(df['rsi'], 0, 100)  # Keep RSI in valid range
            df['macd'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
            df['momentum'] = df['Close'].pct_change(5).fillna(0)
            df['volatility'] = df['returns'].rolling(20, min_periods=1).std().fillna(0)
            
            # Create extremely imbalanced target (similar to real trading signals)
            signal_prob = np.random.rand(n_samples)
            target = np.zeros(n_samples)
            
            # Extreme imbalance: ~0.5% buy signals, ~0.3% sell signals, rest neutral
            buy_threshold = 0.005  # 0.5%
            sell_threshold = 0.997  # 0.3% (99.7% - 100%)
            
            target[signal_prob < buy_threshold] = 1  # Buy signals
            target[signal_prob > sell_threshold] = -1  # Sell signals
            # Rest remain 0 (Hold)
            
            df['target'] = target
            df['signal'] = target  # Alias for compatibility
            
            # Remove any NaN values
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            print(f"✅ Created synthetic data: {df.shape[0]} samples with {(target != 0).sum()} signals")
            
            return df
            
        except Exception as e:
            print(f"❌ Synthetic data creation failed: {e}")
            # Return minimal viable dataset
            minimal_df = pd.DataFrame({
                'feature_1': np.random.randn(100),
                'feature_2': np.random.randn(100),
                'target': np.random.choice([0, 1], 100, p=[0.9, 0.1])
            })
            return minimal_df


class LogManager:
    """จัดการ logging ขั้นสูง"""
    
    def __init__(self):
        self.log_issues: List[Dict[str, Any]] = []
    
    def enhanced_pro_log(self, msg: str, tag: Optional[str] = None, level: str = "info") -> None:
        """Enhanced logging with issue tracking"""
        important_levels = ["error", "critical", "warning"]
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if level in important_levels or (tag and tag.lower() in ("result", "summary", "important")):
            color = ""
            reset = ""
            
            try:
                from colorama import Fore, Style
                color = Fore.RED if level == 'error' else Fore.YELLOW if level == 'warning' else Fore.GREEN
                reset = Style.RESET_ALL
            except ImportError:
                pass
            
            print(f"{color}[{timestamp}] [{level.upper()}] {tag or 'LOG'}: {msg}{reset}")
            
            self.log_issues.append({
                "level": level,
                "tag": tag or "LOG",
                "msg": msg,
                "timestamp": timestamp
            })
        else:
            print(f"[{timestamp}] [{level.upper()}] {tag or 'LOG'}: {msg}")
    
    def print_log_summary(self) -> None:
        """Print summary of logged issues"""
        if not self.log_issues:
            try:
                from colorama import Fore, Style
                print(f"{Fore.GREEN}✅ No critical issues found in logs!{Style.RESET_ALL}")
            except ImportError:
                print("✅ No critical issues found in logs!")
            return
        
        try:
            # Try rich table if available
            from rich.console import Console
            from rich.table import Table
            
            console = Console()
            table = Table(title="Log Issues Summary")
            table.add_column("Time", style="cyan")
            table.add_column("Level", style="red")
            table.add_column("Tag", style="yellow")
            table.add_column("Message", style="white")
            
            for issue in self.log_issues:
                table.add_row(
                    str(issue.get("timestamp", "")),
                    str(issue.get("level", "")),
                    str(issue.get("tag", "")),
                    str(issue.get("msg", ""))
                )
            
            console.print(table)
            
        except ImportError:
            # Fallback to simple table
            try:
                from colorama import Fore, Style
                print(f"{Fore.YELLOW}📊 LOG ISSUES SUMMARY:{Style.RESET_ALL}")
            except ImportError:
                print("📊 LOG ISSUES SUMMARY:")
                
            for i, issue in enumerate(self.log_issues, 1):
                timestamp = issue.get("timestamp", "")
                level = issue.get("level", "")
                tag = issue.get("tag", "")
                msg = issue.get("msg", "")
                print(f"{i}. [{timestamp}] [{level.upper()}] {tag}: {msg}")
    
    def save_log_to_file(self, filepath: str) -> None:
        """Save log issues to file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.log_issues, f, indent=2, ensure_ascii=False)
            print(f"✅ Log saved to {filepath}")
        except Exception as e:
            print(f"❌ Failed to save log: {e}")


class ResultManager:
    """จัดการผลลัพธ์และการบันทึก"""
    
    @staticmethod
    def save_results(results: Dict[str, Any], filepath: str) -> bool:
        """Save results to JSON file"""
        try:
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"✅ Results saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
            return False
    
    @staticmethod
    def load_results(filepath: str) -> Optional[Dict[str, Any]]:
        """Load results from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            print(f"✅ Results loaded from {filepath}")
            return results
            
        except Exception as e:
            print(f"❌ Failed to load results: {e}")
            return None


# Singleton instances
table_parser = TableParser()
data_generator = DataGenerator()
log_manager = LogManager()
result_manager = ResultManager()
