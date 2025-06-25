#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 ADVANCED DATA QUALITY PIPELINE
Enhanced data validation, quality scoring, and multi-timeframe analysis
for NICEGOLD ProjectP v2.1

Author: NICEGOLD Enterprise
Date: June 25, 2025
"""

import warnings
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Rich imports with fallback
RICH_AVAILABLE = False
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    pass


class AdvancedDataPipeline:
    """🚀 Advanced Data Quality Pipeline with Multi-timeframe Analysis"""
    
    def __init__(self, console_output: bool = True):
        self.console = Console() if RICH_AVAILABLE else None
        self.console_output = console_output
        self.quality_scores = {}
        self.data_statistics = {}
        
    def validate_data_quality(self, df: pd.DataFrame,
                              symbol: str = "GOLD") -> Dict[str, Any]:
        """
        🔍 ทำการตรวจสอบคุณภาพข้อมูลอย่างครอบคลัว
        """
        if self.console_output and RICH_AVAILABLE:
            self._show_quality_check_header(symbol)
        
        quality_report = {
            'symbol': symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_records': len(df),
            'completeness_score': 0,
            'consistency_score': 0,
            'outlier_score': 0,
            'overall_quality': 0,
            'issues_found': [],
            'recommendations': []
        }
        
        # 1. Data Completeness Check
        completeness = self._check_data_completeness(df)
        quality_report['completeness_score'] = completeness['score']
        quality_report['missing_data'] = completeness['details']
        
        # 2. Data Consistency Validation
        consistency = self._check_data_consistency(df)
        quality_report['consistency_score'] = consistency['score']
        quality_report['consistency_issues'] = consistency['issues']
        
        # 3. Outlier Detection
        outliers = self._detect_outliers(df)
        quality_report['outlier_score'] = outliers['score']
        quality_report['outlier_details'] = outliers['details']
        
        # 4. Calculate Overall Quality Score
        quality_report['overall_quality'] = (
            quality_report['completeness_score'] * 0.4 +
            quality_report['consistency_score'] * 0.3 +
            quality_report['outlier_score'] * 0.3
        )
        
        # 5. Generate Recommendations
        quality_report['recommendations'] = self._generate_recommendations(quality_report)
        
        if self.console_output:
            self._display_quality_report(quality_report)
        
        return quality_report
    
    def _check_data_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """ตรวจสอบความสมบูรณ์ของข้อมูล"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness_ratio = 1 - (missing_cells / total_cells)
        
        missing_by_column = df.isnull().sum()
        problematic_columns = missing_by_column[missing_by_column > len(df) * 0.1].to_dict()
        
        return {
            'score': min(100, completeness_ratio * 100),
            'details': {
                'total_cells': total_cells,
                'missing_cells': missing_cells,
                'completeness_ratio': completeness_ratio,
                'problematic_columns': problematic_columns
            }
        }
    
    def _check_data_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """ตรวจสอบความสอดคล้องของข้อมูล"""
        issues = []
        consistency_score = 100
        
        # Check for price inconsistencies (High < Low, etc.)
        if 'high' in df.columns and 'low' in df.columns:
            invalid_hl = (df['high'] < df['low']).sum()
            if invalid_hl > 0:
                issues.append(f"High < Low in {invalid_hl} records")
                consistency_score -= min(20, invalid_hl / len(df) * 100)
        
        # Check for extreme price movements
        if 'close' in df.columns:
            price_changes = df['close'].pct_change().abs()
            extreme_changes = (price_changes > 0.1).sum()  # >10% change
            if extreme_changes > len(df) * 0.01:  # >1% of data
                issues.append(f"Extreme price movements detected: {extreme_changes}")
                consistency_score -= min(15, extreme_changes / len(df) * 100)
        
        # Check for volume anomalies
        if 'volume' in df.columns:
            zero_volume = (df['volume'] == 0).sum()
            if zero_volume > len(df) * 0.05:  # >5% zero volume
                issues.append(f"High zero volume records: {zero_volume}")
                consistency_score -= min(10, zero_volume / len(df) * 50)
        
        return {
            'score': max(0, consistency_score),
            'issues': issues
        }
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """ตรวจหาค่าผิดปกติในข้อมูล"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_details = {}
        total_outliers = 0
        
        for col in numeric_columns:
            if col in df.columns and not df[col].empty:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_ratio = outliers / len(df)
                
                outlier_details[col] = {
                    'count': outliers,
                    'ratio': outlier_ratio,
                    'bounds': (lower_bound, upper_bound)
                }
                total_outliers += outliers
        
        outlier_score = max(0, 100 - (total_outliers / len(df) * 100))
        
        return {
            'score': outlier_score,
            'details': outlier_details
        }
    
    def _generate_recommendations(self, quality_report: Dict[str, Any]) -> List[str]:
        """สร้างคำแนะนำสำหรับการปรับปรุงคุณภาพข้อมูล"""
        recommendations = []
        
        if quality_report['completeness_score'] < 95:
            recommendations.append("🔧 ใช้ advanced imputation methods (KNN, interpolation)")
        
        if quality_report['consistency_score'] < 90:
            recommendations.append("⚠️ ตรวจสอบและแก้ไขข้อมูลที่ไม่สอดคล้อง")
        
        if quality_report['outlier_score'] < 85:
            recommendations.append("📊 ใช้ robust scaling หรือ outlier treatment")
        
        if quality_report['overall_quality'] < 85:
            recommendations.append("🎯 พิจารณาใช้ additional data sources")
        
        return recommendations
    
    def multi_timeframe_analysis(self, df: pd.DataFrame, timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        📈 วิเคราะห์ข้อมูลหลายไทม์เฟรม
        """
        if timeframes is None:
            timeframes = ['5T', '15T', '1H', '4H', '1D']  # 5min, 15min, 1hour, 4hour, daily
        
        if self.console_output and RICH_AVAILABLE:
            self._show_timeframe_analysis_header(timeframes)
        
        multi_tf_data = {}
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            elif 'datetime' in df.columns:
                df = df.set_index('datetime')
        
        for tf in timeframes:
            try:
                # Resample to different timeframes
                resampled = df.resample(tf).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum' if 'volume' in df.columns else 'mean'
                }).dropna()
                
                # Add timeframe-specific features
                resampled = self._add_timeframe_features(resampled, tf)
                multi_tf_data[tf] = resampled
                
                if self.console_output and not RICH_AVAILABLE:
                    print(f"✅ Processed {tf} timeframe: {len(resampled)} bars")
                    
            except Exception as e:
                if self.console_output:
                    print(f"⚠️ Error processing {tf}: {str(e)}")
        
        return multi_tf_data
    
    def _add_timeframe_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """เพิ่มฟีเจอร์เฉพาะไทม์เฟรม"""
        df = df.copy()
        
        # Basic technical indicators
        df[f'sma_20_{timeframe}'] = df['close'].rolling(20).mean()
        df[f'sma_50_{timeframe}'] = df['close'].rolling(50).mean()
        df[f'rsi_{timeframe}'] = self._calculate_rsi(df['close'])
        df[f'volatility_{timeframe}'] = df['close'].rolling(20).std()
        
        # Price momentum
        df[f'momentum_{timeframe}'] = df['close'].pct_change(10)
        df[f'price_range_{timeframe}'] = (df['high'] - df['low']) / df['close']
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """คำนวณ RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def impute_missing_data(self, df: pd.DataFrame, method: str = 'advanced') -> pd.DataFrame:
        """
        🔧 ทำการ imputation ข้อมูลที่ขาดหาย
        """
        df_clean = df.copy()
        
        if method == 'advanced':
            # Use forward fill for time series data
            df_clean = df_clean.fillna(method='ffill')
            # Use backward fill for remaining
            df_clean = df_clean.fillna(method='bfill')
            # Use interpolation for numeric columns
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method='linear')
        elif method == 'median':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        
        return df_clean
    
    def _show_quality_check_header(self, symbol: str):
        """แสดงหัวข้อการตรวจสอบคุณภาพ"""
        if RICH_AVAILABLE:
            header = Panel(
                f"[bold cyan]🔍 DATA QUALITY ANALYSIS[/bold cyan]\n"
                f"[yellow]Symbol:[/yellow] {symbol}\n"
                f"[yellow]Timestamp:[/yellow] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                border_style="cyan"
            )
            self.console.print(header)
    
    def _show_timeframe_analysis_header(self, timeframes: List[str]):
        """แสดงหัวข้อการวิเคราะห์หลายไทม์เฟรม"""
        if RICH_AVAILABLE:
            header = Panel(
                f"[bold green]📈 MULTI-TIMEFRAME ANALYSIS[/bold green]\n"
                f"[yellow]Timeframes:[/yellow] {', '.join(timeframes)}",
                border_style="green"
            )
            self.console.print(header)
    
    def _display_quality_report(self, report: Dict[str, Any]):
        """แสดงรายงานคุณภาพข้อมูล"""
        if RICH_AVAILABLE:
            # Create quality score table
            table = Table(title="📊 Data Quality Report", border_style="blue")
            table.add_column("Metric", style="cyan")
            table.add_column("Score", style="green")
            table.add_column("Status", style="yellow")
            
            # Add scores
            metrics = [
                ("Completeness", f"{report['completeness_score']:.1f}%"),
                ("Consistency", f"{report['consistency_score']:.1f}%"),
                ("Outlier Score", f"{report['outlier_score']:.1f}%"),
                ("Overall Quality", f"{report['overall_quality']:.1f}%")
            ]
            
            for metric, score in metrics:
                score_val = float(score.replace('%', ''))
                if score_val >= 90:
                    status = "🟢 Excellent"
                elif score_val >= 80:
                    status = "🟡 Good"
                elif score_val >= 70:
                    status = "🟠 Fair"
                else:
                    status = "🔴 Poor"
                table.add_row(metric, score, status)
            
            self.console.print(table)
            
            # Show recommendations if any
            if report['recommendations']:
                rec_panel = Panel(
                    "\n".join(f"• {rec}" for rec in report['recommendations']),
                    title="💡 Recommendations",
                    border_style="yellow"
                )
                self.console.print(rec_panel)
        else:
            # Fallback display
            print(f"\n📊 DATA QUALITY REPORT")
            print(f"═" * 50)
            print(f"Completeness: {report['completeness_score']:.1f}%")
            print(f"Consistency:  {report['consistency_score']:.1f}%")
            print(f"Outlier Score: {report['outlier_score']:.1f}%")
            print(f"Overall Quality: {report['overall_quality']:.1f}%")
            print(f"═" * 50)
            
            if report['recommendations']:
                print("💡 Recommendations:")
                for rec in report['recommendations']:
                    print(f"  • {rec}")


if __name__ == "__main__":
    # Demo/Test the Advanced Data Pipeline
    print("🚀 NICEGOLD ProjectP - Advanced Data Pipeline Demo")
    
    # Create sample data for testing
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1H')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': 2000 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'high': 2000 + np.cumsum(np.random.randn(len(dates)) * 0.5) + np.random.rand(len(dates)) * 5,
        'low': 2000 + np.cumsum(np.random.randn(len(dates)) * 0.5) - np.random.rand(len(dates)) * 5,
        'close': 2000 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    # Introduce some data quality issues for testing
    sample_data.loc[100:110, 'close'] = np.nan  # Missing data
    sample_data.loc[200, 'high'] = sample_data.loc[200, 'low'] - 10  # Inconsistent data
    sample_data.loc[300:305, 'volume'] = 0  # Zero volume
    
    # Test the pipeline
    pipeline = AdvancedDataPipeline()
    
    # 1. Test data quality validation
    quality_report = pipeline.validate_data_quality(sample_data, "XAUUSD")
    
    # 2. Test multi-timeframe analysis
    sample_data_indexed = sample_data.set_index('timestamp')
    multi_tf_data = pipeline.multi_timeframe_analysis(sample_data_indexed)
    
    print(f"\n✅ Demo completed successfully!")
    print(f"📊 Quality Score: {quality_report['overall_quality']:.1f}%")
    print(f"📈 Timeframes processed: {len(multi_tf_data)}")
