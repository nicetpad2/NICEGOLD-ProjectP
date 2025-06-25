#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔗 INTEGRATION UPDATER FOR PROJECTP.PY
Updates the main ProjectP.py with enhanced menu options
for NICEGOLD ProjectP v2.1

Author: NICEGOLD Enterprise
Date: June 25, 2025
"""

import os
import sys
from pathlib import Path


def update_projectp_with_enhancements():
    """อัปเดต ProjectP.py ให้รองรับฟีเจอร์ขั้นสูงใหม่"""
    
    # Get the project directory
    project_dir = Path(__file__).parent
    projectp_path = project_dir / "ProjectP.py"
    
    if not projectp_path.exists():
        print("❌ ProjectP.py not found!")
        return False
    
    # Read current ProjectP.py content
    with open(projectp_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define the new menu options to add
    new_menu_options = '''
        # Enhanced Features Menu
        elif choice == '7':
            print("\\n🚀 Enhanced Features Menu")
            print("=" * 50)
            print("1. 🔍 Advanced Data Quality Analysis")
            print("2. 🤖 Model Ensemble System") 
            print("3. 📊 Interactive Dashboard")
            print("4. ⚠️ Risk Management System")
            print("5. 🎯 Complete Enhanced Pipeline")
            print("0. ← Back to Main Menu")
            
            enhanced_choice = input("\\nSelect enhanced feature (0-5): ").strip()
            
            if enhanced_choice == '1':
                run_advanced_data_pipeline()
            elif enhanced_choice == '2':
                run_model_ensemble_system()
            elif enhanced_choice == '3':
                run_interactive_dashboard()
            elif enhanced_choice == '4':
                run_risk_management_system()
            elif enhanced_choice == '5':
                run_complete_enhanced_pipeline()
            elif enhanced_choice == '0':
                continue
            else:
                print("❌ Invalid choice. Please try again.")
'''
    
    # Define the enhanced function implementations
    enhanced_functions = '''

# Enhanced Feature Functions
def run_advanced_data_pipeline():
    """🔍 Run Advanced Data Quality Analysis"""
    try:
        from advanced_data_pipeline import AdvancedDataPipeline
        
        print("\\n🔍 Starting Advanced Data Quality Analysis...")
        pipeline = AdvancedDataPipeline()
        
        # Check if we have existing data
        data_files = list(Path(".").glob("*.csv"))
        if data_files:
            data_path = data_files[0]
            print(f"📁 Loading data from: {data_path}")
            import pandas as pd
            data = pd.read_csv(data_path)
        else:
            print("📊 Generating sample data for analysis...")
            # Generate sample data
            import pandas as pd
            import numpy as np
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1H')
            data = pd.DataFrame({
                'timestamp': dates,
                'open': 2000 + np.cumsum(np.random.randn(len(dates)) * 0.5),
                'high': 2000 + np.cumsum(np.random.randn(len(dates)) * 0.5) + np.random.rand(len(dates)) * 5,
                'low': 2000 + np.cumsum(np.random.randn(len(dates)) * 0.5) - np.random.rand(len(dates)) * 5,
                'close': 2000 + np.cumsum(np.random.randn(len(dates)) * 0.5),
                'volume': np.random.randint(1000, 10000, len(dates))
            })
        
        # Run data quality analysis
        quality_report = pipeline.validate_data_quality(data, "XAUUSD")
        
        # Multi-timeframe analysis
        if 'timestamp' in data.columns:
            data_indexed = data.set_index('timestamp')
        else:
            dates = pd.date_range(start='2024-01-01', periods=len(data), freq='1H')
            data_indexed = data.copy()
            data_indexed.index = dates
        
        multi_tf_data = pipeline.multi_timeframe_analysis(data_indexed)
        
        print(f"\\n✅ Data Quality Analysis Complete!")
        print(f"📊 Overall Quality Score: {quality_report['overall_quality']:.1f}%")
        print(f"📈 Timeframes Analyzed: {len(multi_tf_data)}")
        
        input("\\nPress Enter to continue...")
        
    except ImportError:
        print("❌ Advanced Data Pipeline not available. Please check installation.")
        input("Press Enter to continue...")
    except Exception as e:
        print(f"❌ Error running advanced data pipeline: {str(e)}")
        input("Press Enter to continue...")


def run_model_ensemble_system():
    """🤖 Run Model Ensemble System"""
    try:
        from model_ensemble_system import ModelEnsemble
        import numpy as np
        
        print("\\n🤖 Starting Model Ensemble System...")
        ensemble = ModelEnsemble()
        
        # Generate sample ML data
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] * 0.5 + X[:, 2] * 0.3 + 
             np.random.randn(n_samples) * 0.1 > 0).astype(int)
        
        # Split data
        split_idx = int(0.8 * n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print("📊 Training ensemble models...")
        
        # Train stacking ensemble
        stacking_results = ensemble.stack_models(X_train, y_train, X_test, y_test)
        
        # Train adaptive ensemble  
        adaptive_results = ensemble.adaptive_ensemble(X_train, y_train, X_test, y_test)
        
        print(f"\\n✅ Model Ensemble Training Complete!")
        print(f"🎯 Stacking AUC: {stacking_results.get('ensemble_score', 'N/A')}")
        print(f"🧠 Adaptive AUC: {adaptive_results.get('ensemble_score', 'N/A')}")
        
        input("\\nPress Enter to continue...")
        
    except ImportError:
        print("❌ Model Ensemble System not available. Please check installation.")
        input("Press Enter to continue...")
    except Exception as e:
        print(f"❌ Error running model ensemble: {str(e)}")
        input("Press Enter to continue...")


def run_interactive_dashboard():
    """📊 Run Interactive Dashboard"""
    try:
        from interactive_dashboard import InteractiveDashboard
        import pandas as pd
        import numpy as np
        
        print("\\n📊 Creating Interactive Dashboard...")
        dashboard = InteractiveDashboard()
        
        # Generate sample data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1H')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'open': 2000 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            'high': 2000 + np.cumsum(np.random.randn(len(dates)) * 0.5) + np.random.rand(len(dates)) * 5,
            'low': 2000 + np.cumsum(np.random.randn(len(dates)) * 0.5) - np.random.rand(len(dates)) * 5,
            'close': 2000 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        # Generate sample predictions
        predictions = np.random.random(len(data))
        
        print("🎨 Creating interactive charts...")
        charts = dashboard.create_plotly_charts(data, predictions)
        
        print("📱 Generating dashboard...")
        dashboard_path = dashboard.create_live_dashboard(data)
        
        print(f"\\n✅ Interactive Dashboard Created!")
        print(f"📊 Charts Created: {len(charts)}")
        if isinstance(dashboard_path, str):
            print(f"📱 Dashboard saved to: {dashboard_path}")
            print("🌐 Open the HTML file in your browser to view the dashboard")
        
        input("\\nPress Enter to continue...")
        
    except ImportError:
        print("❌ Interactive Dashboard not available. Please install plotly: pip install plotly")
        input("Press Enter to continue...")
    except Exception as e:
        print(f"❌ Error creating dashboard: {str(e)}")
        input("Press Enter to continue...")


def run_risk_management_system():
    """⚠️ Run Risk Management System"""
    try:
        from risk_management_system import RiskManagementSystem
        import pandas as pd
        import numpy as np
        
        print("\\n⚠️ Starting Risk Management Analysis...")
        risk_mgr = RiskManagementSystem()
        
        # Generate sample market data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1H')
        np.random.seed(42)
        
        market_data = pd.DataFrame({
            'close': 2000 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        # Sample portfolio data
        portfolio_data = {
            'total_value': 100000,
            'positions': {
                'GOLD_1': {'weight': 0.08, 'shares': 40, 'entry_price': 2000},
                'GOLD_2': {'weight': 0.05, 'shares': 25, 'entry_price': 1995}
            }
        }
        
        print("💰 Calculating optimal position size...")
        position_size = risk_mgr.calculate_position_size(
            signal_strength=0.75,
            account_balance=100000,
            current_price=2000,
            volatility=0.2
        )
        
        print("📊 Analyzing portfolio risk...")
        risk_report = risk_mgr.monitor_portfolio_risk(portfolio_data, market_data)
        
        print(f"\\n✅ Risk Management Analysis Complete!")
        print(f"💰 Recommended Position Size: ${position_size['dollar_amount']:,.2f}")
        print(f"📊 Portfolio Risk Score: {risk_report['overall_risk_score']:.2f}")
        print(f"⚠️ Risk Alerts: {len(risk_report['alerts'])}")
        
        input("\\nPress Enter to continue...")
        
    except ImportError:
        print("❌ Risk Management System not available. Please check installation.")
        input("Press Enter to continue...")
    except Exception as e:
        print(f"❌ Error running risk management: {str(e)}")
        input("Press Enter to continue...")


def run_complete_enhanced_pipeline():
    """🎯 Run Complete Enhanced Pipeline"""
    try:
        from enhanced_system_integration import EnhancedFullPipelineV2
        
        print("\\n🎯 Starting Complete Enhanced Pipeline...")
        print("This may take a few minutes to complete all phases...")
        
        enhanced_system = EnhancedFullPipelineV2()
        
        # Run the complete pipeline
        results = enhanced_system.run_enhanced_pipeline()
        
        print(f"\\n✅ Enhanced Pipeline Completed!")
        print(f"📊 Success: {results['success']}")
        print(f"🔍 Data Quality: {results.get('data_quality', {}).get('overall_quality', 0):.1f}%")
        
        model_perf = results.get('model_performance', {})
        if model_perf:
            print(f"🤖 Model Accuracy: {model_perf.get('test_accuracy', 0):.1%}")
        
        risk_analysis = results.get('risk_analysis', {})
        if risk_analysis:
            risk_score = risk_analysis.get('risk_report', {}).get('overall_risk_score', 0)
            print(f"⚠️ Risk Score: {risk_score:.2f}")
        
        if results.get('dashboard_path'):
            print(f"📱 Dashboard: {results['dashboard_path']}")
        
        print(f"💡 Recommendations: {len(results.get('recommendations', []))}")
        for rec in results.get('recommendations', [])[:3]:  # Show first 3
            print(f"   • {rec}")
        
        input("\\nPress Enter to continue...")
        
    except ImportError:
        print("❌ Enhanced System Integration not available. Please check installation.")
        input("Press Enter to continue...")
    except Exception as e:
        print(f"❌ Error running enhanced pipeline: {str(e)}")
        input("Press Enter to continue...")
'''
    
    # Check if enhancements are already added
    if "Enhanced Features Menu" in content:
        print("✅ ProjectP.py already contains enhanced features!")
        return True
    
    # Find the main menu section and add our new option
    main_menu_pattern = '''        print("6. 🔧 System Configuration")
        print("0. 🚪 Exit")'''
    
    new_main_menu = '''        print("6. 🔧 System Configuration")
        print("7. 🚀 Enhanced Features")
        print("0. 🚪 Exit")'''
    
    if main_menu_pattern in content:
        content = content.replace(main_menu_pattern, new_main_menu)
        print("✅ Updated main menu")
    else:
        print("⚠️ Could not find main menu pattern")
    
    # Find where to insert the new menu option logic
    # Look for the end of the main menu handling
    insert_point = content.find("        elif choice == '0':")
    
    if insert_point != -1:
        # Insert our new menu option before the exit option
        content = content[:insert_point] + new_menu_options + "\\n" + content[insert_point:]
        print("✅ Added enhanced features menu")
    else:
        print("⚠️ Could not find insertion point for menu options")
    
    # Add the enhanced functions at the end of the file, before if __name__ == "__main__":
    main_check_pattern = 'if __name__ == "__main__":'
    if main_check_pattern in content:
        insert_point = content.rfind(main_check_pattern)
        content = content[:insert_point] + enhanced_functions + "\\n\\n" + content[insert_point:]
        print("✅ Added enhanced function implementations")
    else:
        # If no main check, just append
        content += enhanced_functions
        print("✅ Appended enhanced function implementations")
    
    # Write the updated content back
    try:
        # Create backup
        backup_path = projectp_path.with_suffix('.py.backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(open(projectp_path, 'r', encoding='utf-8').read())
        print(f"📁 Backup created: {backup_path}")
        
        # Write updated content
        with open(projectp_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ ProjectP.py updated successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error writing file: {str(e)}")
        return False


if __name__ == "__main__":
    print("🔗 NICEGOLD ProjectP - Integration Updater")
    print("=" * 50)
    
    success = update_projectp_with_enhancements()
    
    if success:
        print("\\n🚀 Integration completed successfully!")
        print("\\n📋 What's new in ProjectP.py:")
        print("  • 🔍 Advanced Data Quality Analysis")
        print("  • 🤖 Model Ensemble System") 
        print("  • 📊 Interactive Dashboard")
        print("  • ⚠️ Risk Management System")
        print("  • 🎯 Complete Enhanced Pipeline")
        print("\\n💡 Run ProjectP.py and select option '7' to access enhanced features!")
    else:
        print("\\n❌ Integration failed. Please check the errors above.")
