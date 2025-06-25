# NICEGOLD ProjectP v2.0 - README

## 🚀 Professional AI Trading System

New modular architecture for better maintainability and development.

### 📁 Project Structure

```
NICEGOLD-ProjectP/
├── ProjectP.py                  # Main entry point
├── core/                        # Core modules
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── system.py               # System utilities
│   ├── menu_operations.py      # Menu implementations
│   └── menu_interface.py       # User interface
├── utils/                       # Utility modules
│   ├── __init__.py
│   └── colors.py               # Color formatting
├── datacsv/                    # Input data folder
├── output_default/             # Results folder
├── models/                     # Trained models
├── logs/                       # Log files
├── config.yaml                 # Configuration file
└── setup_new.py               # Setup script
```

### 🛠️ Installation

1. **Quick Setup:**
   ```bash
   python setup_new.py
   ```

2. **Manual Installation:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   pip install joblib pyyaml tqdm requests psutil
   pip install streamlit fastapi uvicorn  # Optional web features
   pip install xgboost lightgbm optuna    # Optional ML packages
   ```

### 🚀 Usage

1. **Start the application:**
   ```bash
   python ProjectP.py
   ```

2. **Add your data:**
   - Place CSV files in the `datacsv/` folder
   - Supported formats: OHLCV data with Date/Time column

3. **Use the menu:**
   - Option 1: Full Pipeline (complete ML workflow)
   - Option 3: Data Analysis (explore your data)
   - Option 8: Train Models (train ML models)
   - Option 14: Web Dashboard (Streamlit interface)

### 🔧 Key Features

- **Modular Design:** Separated into logical modules
- **Easy Configuration:** YAML-based configuration
- **Multiple ML Models:** Random Forest, XGBoost, LightGBM
- **Web Interface:** Streamlit dashboard
- **System Monitoring:** Built-in health checks
- **Error Handling:** Robust error management
- **Beautiful UI:** Colorized terminal interface

### 📊 Menu Options

| Option | Feature | Description |
|--------|---------|-------------|
| 1 | 🚀 Full Pipeline | Complete ML trading workflow |
| 3 | 📊 Data Analysis | Comprehensive data exploration |
| 4 | 🔧 Quick Test | System functionality test |
| 8 | 🤖 Train Models | Machine learning model training |
| 10 | 🎯 Backtest | Trading strategy backtesting |
| 14 | 🌐 Web Dashboard | Streamlit web interface |
| 22 | 🔍 Health Check | System status monitoring |
| 23 | 📦 Install Dependencies | Package installation |
| 24 | 🧹 Clean System | Clean temporary files |

### 🔄 Development

The new architecture makes it easy to:

- **Add new features:** Extend `menu_operations.py`
- **Modify UI:** Update `menu_interface.py`
- **Change configuration:** Edit `config.yaml`
- **Add utilities:** Extend modules in `utils/`

### 🐛 Troubleshooting

1. **Import errors:** Run `python setup_new.py`
2. **Missing packages:** Check requirements and install manually
3. **Permission errors:** Ensure write access to project folder
4. **Data issues:** Verify CSV format in `datacsv/` folder

### 📝 Configuration

Edit `config.yaml` to customize:

```yaml
project:
  name: "NICEGOLD ProjectP"
  version: "2.0.0"

data:
  input_folder: "datacsv"
  output_folder: "output_default"

trading:
  initial_balance: 10000
  max_position_size: 0.1

ml:
  models: ["RandomForest", "XGBoost", "LightGBM"]
  test_size: 0.2
```

### ✅ Benefits of New Architecture

- **Maintainable:** Clear separation of concerns
- **Extensible:** Easy to add new features
- **Testable:** Modular components
- **Readable:** Well-documented code
- **Scalable:** Can grow with requirements

---

**NICEGOLD ProjectP v2.0** - Professional AI Trading System
