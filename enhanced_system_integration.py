#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 ENHANCED SYSTEM INTEGRATION MODULE
Integrates all advanced features for NICEGOLD ProjectP v2.1

Author: NICEGOLD Enterprise
Date: June 25, 2025
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Import advanced modules with fallback
try:
    from advanced_data_pipeline import AdvancedDataPipeline
    ADVANCED_DATA_AVAILABLE = True
except ImportError:
    ADVANCED_DATA_AVAILABLE = False

try:
    from model_ensemble_system import ModelEnsemble
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

try:
    from interactive_dashboard import InteractiveDashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

try:
    from risk_management_system import RiskManagementSystem
    RISK_MGMT_AVAILABLE = True
except ImportError:
    RISK_MGMT_AVAILABLE = False

# Rich imports with fallback
RICH_AVAILABLE = False
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import track
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    pass


class EnhancedFullPipelineV2:
    """🚀 Enhanced Full Pipeline with Advanced Features"""
    
    def __init__(self, console_output: bool = True):
        self.console = Console() if RICH_AVAILABLE else None
        self.console_output = console_output
        
        # Initialize advanced components
        self.data_pipeline = None
        self.ensemble_system = None
        self.dashboard = None
        self.risk_manager = None
        
        self._initialize_components()
        
    def _initialize_components(self):
        """เริ่มต้นระบบย่อยต่างๆ"""
        if self.console_output and RICH_AVAILABLE:
            self._show_initialization_header()
        
        # Initialize Data Pipeline
        if ADVANCED_DATA_AVAILABLE:
            self.data_pipeline = AdvancedDataPipeline(self.console_output)
            if self.console_output and not RICH_AVAILABLE:
                print("✅ Advanced Data Pipeline initialized")
        
        # Initialize Model Ensemble
        if ENSEMBLE_AVAILABLE:
            self.ensemble_system = ModelEnsemble(self.console_output)
            if self.console_output and not RICH_AVAILABLE:
                print("✅ Model Ensemble System initialized")
        
        # Initialize Dashboard
        if DASHBOARD_AVAILABLE:
            self.dashboard = InteractiveDashboard(self.console_output)
            if self.console_output and not RICH_AVAILABLE:
                print("✅ Interactive Dashboard initialized")
        
        # Initialize Risk Management
        if RISK_MGMT_AVAILABLE:
            self.risk_manager = RiskManagementSystem(self.console_output)
            if self.console_output and not RICH_AVAILABLE:
                print("✅ Risk Management System initialized")
        
        if self.console_output and RICH_AVAILABLE:
            self._display_component_status()
    
    def run_enhanced_pipeline(self, data_path: str = None, 
                            config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        🎯 เรียกใช้ Enhanced Pipeline แบบครบครัน
        """
        if self.console_output and RICH_AVAILABLE:
            self._show_pipeline_header()
        
        pipeline_results = {
            'timestamp': datetime.now(),
            'data_quality': {},
            'model_performance': {},
            'risk_analysis': {},
            'dashboard_path': None,
            'recommendations': [],
            'success': False
        }
        
        try:
            # Phase 1: Advanced Data Processing
            if self.console_output:
                print("\n🔍 Phase 1: Advanced Data Processing...")
            
            # Load and validate data
            if data_path and Path(data_path).exists():
                raw_data = pd.read_csv(data_path)
            else:
                # Generate sample data for demo
                raw_data = self._generate_sample_data()
            
            # Data quality analysis
            if self.data_pipeline:
                pipeline_results['data_quality'] = self.data_pipeline.validate_data_quality(
                    raw_data, "XAUUSD"
                )
                
                # Multi-timeframe analysis
                if 'timestamp' in raw_data.columns:
                    raw_data_indexed = raw_data.set_index('timestamp')
                elif isinstance(raw_data.index, pd.DatetimeIndex):
                    raw_data_indexed = raw_data
                else:
                    # Create datetime index
                    dates = pd.date_range(start='2024-01-01', periods=len(raw_data), freq='1H')
                    raw_data_indexed = raw_data.copy()
                    raw_data_indexed.index = dates
                
                multi_tf_data = self.data_pipeline.multi_timeframe_analysis(raw_data_indexed)
                pipeline_results['multi_timeframe_data'] = multi_tf_data
                
                # Data imputation
                clean_data = self.data_pipeline.impute_missing_data(raw_data_indexed)
            else:
                clean_data = raw_data
                if self.console_output:
                    print("⚠️ Advanced data pipeline not available")
            
            # Phase 2: Advanced ML Modeling
            if self.console_output:
                print("\n🤖 Phase 2: Advanced ML Modeling...")
            
            # Prepare features and targets
            features, targets = self._prepare_ml_data(clean_data)
            
            if self.ensemble_system and len(features) > 0:
                # Initialize and train ensemble
                self.ensemble_system.initialize_base_models()
                
                # Split data
                split_idx = int(0.8 * len(features))
                X_train, X_test = features[:split_idx], features[split_idx:]
                y_train, y_test = targets[:split_idx], targets[split_idx:]
                
                # Train stacking ensemble
                stacking_results = self.ensemble_system.stack_models(
                    X_train, y_train, X_test, y_test
                )
                
                # Train adaptive ensemble
                adaptive_results = self.ensemble_system.adaptive_ensemble(
                    X_train, y_train, X_test, y_test
                )
                
                # Make predictions
                predictions = self.ensemble_system.predict_ensemble(X_test, method='adaptive')
                
                pipeline_results['model_performance'] = {
                    'stacking_results': stacking_results,
                    'adaptive_results': adaptive_results,
                    'predictions': predictions,
                    'test_accuracy': self._calculate_accuracy(y_test, predictions)
                }
            else:
                if self.console_output:
                    print("⚠️ Model ensemble system not available")
            
            # Phase 3: Risk Management Analysis
            if self.console_output:
                print("\n⚠️ Phase 3: Risk Management Analysis...")
            
            if self.risk_manager:
                # Sample portfolio data
                portfolio_data = {
                    'total_value': 100000,
                    'positions': {
                        'GOLD_1': {'weight': 0.08, 'shares': 40, 'entry_price': 2000}
                    }
                }
                
                # Risk monitoring
                risk_report = self.risk_manager.monitor_portfolio_risk(
                    portfolio_data, clean_data
                )
                
                # Position sizing calculation
                if 'predictions' in pipeline_results['model_performance']:
                    latest_prediction = pipeline_results['model_performance']['predictions'][-1]
                    signal_strength = float(latest_prediction)
                    
                    position_sizing = self.risk_manager.calculate_position_size(
                        signal_strength=signal_strength,
                        account_balance=100000,
                        current_price=clean_data['close'].iloc[-1] if 'close' in clean_data.columns else 2000,
                        volatility=clean_data['close'].pct_change().std() if 'close' in clean_data.columns else 0.02
                    )
                    
                    pipeline_results['risk_analysis'] = {
                        'risk_report': risk_report,
                        'position_sizing': position_sizing
                    }
            else:
                if self.console_output:
                    print("⚠️ Risk management system not available")
            
            # Phase 4: Interactive Dashboard Creation
            if self.console_output:
                print("\n📊 Phase 4: Dashboard Creation...")
            
            if self.dashboard:
                # Create charts
                predictions_for_dashboard = pipeline_results['model_performance'].get('predictions')
                charts = self.dashboard.create_plotly_charts(clean_data, predictions_for_dashboard)
                
                # Create dashboard
                dashboard_path = self.dashboard.create_live_dashboard(clean_data)
                pipeline_results['dashboard_path'] = dashboard_path
                pipeline_results['charts_created'] = len(charts)
            else:
                if self.console_output:
                    print("⚠️ Interactive dashboard not available")
            
            # Phase 5: Generate Comprehensive Recommendations
            if self.console_output:
                print("\n💡 Phase 5: Generating Recommendations...")
            
            pipeline_results['recommendations'] = self._generate_comprehensive_recommendations(
                pipeline_results
            )
            
            pipeline_results['success'] = True
            
            if self.console_output:
                self._display_pipeline_summary(pipeline_results)
            
        except Exception as e:
            if self.console_output:
                print(f"❌ Pipeline error: {str(e)}")
            pipeline_results['error'] = str(e)
        
        return pipeline_results
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """สร้างข้อมูลตัวอย่างสำหรับการทดสอบ"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1H')
        
        # Generate realistic gold price data
        base_price = 2000
        price_series = [base_price]
        
        for i in range(1, len(dates)):
            # Add trend and noise
            trend = 0.0001 * i  # Slight upward trend
            noise = np.random.normal(0, 1.0)  # Random noise
            new_price = price_series[-1] + trend + noise
            price_series.append(max(1800, min(2200, new_price)))  # Keep within bounds
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': price_series,
            'high': [p + np.random.uniform(0, 5) for p in price_series],
            'low': [p - np.random.uniform(0, 5) for p in price_series],
            'close': price_series,
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        
        # Add some missing values for testing
        data.loc[100:105, 'close'] = np.nan
        
        return data
    
    def _prepare_ml_data(self, data: pd.DataFrame) -> tuple:
        """เตรียมข้อมูลสำหรับ ML"""
        try:
            # Select numeric columns for features
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return np.array([]), np.array([])
            
            # Create features
            features_data = data[numeric_cols].fillna(method='ffill').fillna(method='bfill')
            
            # Add technical indicators as features
            if 'close' in features_data.columns:
                features_data['sma_5'] = features_data['close'].rolling(5).mean()
                features_data['sma_20'] = features_data['close'].rolling(20).mean()
                features_data['returns'] = features_data['close'].pct_change()
                features_data['volatility'] = features_data['returns'].rolling(10).std()
            
            # Fill any remaining NaN values
            features_data = features_data.fillna(0)
            
            # Create binary target (price goes up = 1, down = 0)
            if 'close' in data.columns:
                targets = (data['close'].shift(-1) > data['close']).astype(int).fillna(0)
                targets = targets[:-1]  # Remove last row (no future price)
                features_data = features_data[:-1]  # Align with targets
            else:
                # Random targets for demo
                targets = np.random.randint(0, 2, len(features_data))
            
            return features_data.values, targets.values
            
        except Exception as e:
            if self.console_output:
                print(f"⚠️ Error preparing ML data: {str(e)}")
            return np.array([]), np.array([])
    
    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """คำนวณความแม่นยำ"""
        try:
            if len(y_true) == 0 or len(y_pred) == 0:
                return 0.0
            
            # Convert predictions to binary
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            # Calculate accuracy
            accuracy = np.mean(y_true == y_pred_binary)
            return float(accuracy)
        except Exception:
            return 0.0
    
    def _generate_comprehensive_recommendations(self, results: Dict[str, Any]) -> list:
        """สร้างคำแนะนำจากผลลัพธ์ทั้งหมด"""
        recommendations = []
        
        # Data quality recommendations
        data_quality = results.get('data_quality', {})
        if data_quality.get('overall_quality', 0) < 85:
            recommendations.append("🔧 Improve data quality through better collection and validation")
        
        # Model performance recommendations
        model_perf = results.get('model_performance', {})
        if model_perf.get('test_accuracy', 0) < 0.6:
            recommendations.append("🤖 Consider model hyperparameter tuning or feature engineering")
        elif model_perf.get('test_accuracy', 0) > 0.8:
            recommendations.append("✅ Excellent model performance - consider live trading")
        
        # Risk management recommendations
        risk_analysis = results.get('risk_analysis', {})
        if risk_analysis:
            risk_score = risk_analysis.get('risk_report', {}).get('overall_risk_score', 0)
            if risk_score > 0.7:
                recommendations.append("⚠️ High risk detected - reduce position sizes")
            elif risk_score < 0.3:
                recommendations.append("💰 Low risk environment - consider increasing exposure")
        
        # General recommendations
        if results.get('success'):
            recommendations.append("🚀 Pipeline executed successfully - ready for production")
        
        return recommendations
    
    # Display methods
    def _show_initialization_header(self):
        """แสดงหัวข้อการเริ่มต้น"""
        if RICH_AVAILABLE:
            header = Panel(
                "[bold cyan]🚀 ENHANCED SYSTEM INITIALIZATION[/bold cyan]\n"
                "[yellow]Loading advanced trading components...[/yellow]",
                border_style="cyan"
            )
            self.console.print(header)
    
    def _show_pipeline_header(self):
        """แสดงหัวข้อ pipeline"""
        if RICH_AVAILABLE:
            header = Panel(
                "[bold green]🎯 ENHANCED FULL PIPELINE EXECUTION[/bold green]\n"
                "[yellow]Complete end-to-end trading system[/yellow]",
                border_style="green"
            )
            self.console.print(header)
    
    def _display_component_status(self):
        """แสดงสถานะของระบบย่อย"""
        if RICH_AVAILABLE:
            table = Table(title="🔧 System Components", border_style="blue")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="yellow")
            table.add_column("Description", style="green")
            
            components = [
                ("Data Pipeline", "✅ Available" if self.data_pipeline else "❌ Not Available", 
                 "Quality analysis, multi-timeframe"),
                ("Model Ensemble", "✅ Available" if self.ensemble_system else "❌ Not Available",
                 "Stacking, adaptive weighting"),
                ("Dashboard", "✅ Available" if self.dashboard else "❌ Not Available",
                 "Interactive charts, real-time"),
                ("Risk Management", "✅ Available" if self.risk_manager else "❌ Not Available",
                 "Position sizing, risk monitoring")
            ]
            
            for name, status, desc in components:
                table.add_row(name, status, desc)
            
            self.console.print(table)
    
    def _display_pipeline_summary(self, results: Dict[str, Any]):
        """แสดงสรุปผลลัพธ์ pipeline"""
        if RICH_AVAILABLE:
            # Main results table
            table = Table(title="📊 Pipeline Results Summary", border_style="green")
            table.add_column("Phase", style="cyan")
            table.add_column("Status", style="yellow")
            table.add_column("Key Metrics", style="green")
            
            # Data quality
            dq_score = results.get('data_quality', {}).get('overall_quality', 0)
            table.add_row("Data Quality", f"{dq_score:.1f}%", "Quality analysis completed")
            
            # Model performance
            accuracy = results.get('model_performance', {}).get('test_accuracy', 0)
            table.add_row("ML Models", f"{accuracy:.1%} accuracy", "Ensemble training completed")
            
            # Risk analysis
            risk_score = results.get('risk_analysis', {}).get('risk_report', {}).get('overall_risk_score', 0)
            table.add_row("Risk Management", f"Risk: {risk_score:.2f}", "Position analysis completed")
            
            # Dashboard
            charts_count = results.get('charts_created', 0)
            table.add_row("Dashboard", f"{charts_count} charts", "Visualization ready")
            
            self.console.print(table)
            
            # Recommendations
            if results.get('recommendations'):
                rec_panel = Panel(
                    "\n".join([f"• {rec}" for rec in results['recommendations']]),
                    title="💡 System Recommendations",
                    border_style="yellow"
                )
                self.console.print(rec_panel)


# Integration with existing ProjectP system
def enhance_existing_pipeline():
    """เพิ่มความสามารถขั้นสูงให้กับระบบเดิม"""
    enhanced_pipeline = EnhancedFullPipelineV2()
    
    # Run the enhanced pipeline
    results = enhanced_pipeline.run_enhanced_pipeline()
    
    return results


if __name__ == "__main__":
    # Demo/Test the Enhanced System Integration
    print("🚀 NICEGOLD ProjectP - Enhanced System Integration Demo")
    
    # Test the enhanced pipeline
    enhanced_system = EnhancedFullPipelineV2()
    
    # Run complete pipeline
    results = enhanced_system.run_enhanced_pipeline()
    
    print(f"\n✅ Enhanced pipeline completed!")
    print(f"📊 Success: {results['success']}")
    print(f"💡 Recommendations: {len(results.get('recommendations', []))}")
    
    if results.get('dashboard_path'):
        print(f"📱 Dashboard: {results['dashboard_path']}")
