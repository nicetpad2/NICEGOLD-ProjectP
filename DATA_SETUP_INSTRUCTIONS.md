# 📊 NICEGOLD ProjectP - Data Setup Instructions

## ⚠️ IMPORTANT: Data Files Required

The `datacsv/` folder is **NOT included** in this repository due to large file sizes. You need to set up your own data files to run the system.

## 📁 Required Data Structure

Create the following folder structure in your project directory:

```
NICEGOLD-ProjectP/
├── datacsv/                    # ← CREATE THIS FOLDER
│   ├── XAUUSD_M1.csv          # ← Required: Main 1-minute XAUUSD data
│   ├── XAUUSD_M15.csv         # ← Optional: 15-minute XAUUSD data
│   └── [other_data_files.csv] # ← Optional: Additional data files
├── ProjectP.py
├── core/
└── ...
```

## 📋 Required Data Format

### Primary File: `XAUUSD_M1.csv`

The system expects CSV files with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `Time` | Timestamp | `2024-01-01 00:00:00` |
| `Open` | Opening price | `2063.45` |
| `High` | Highest price | `2065.12` |
| `Low` | Lowest price | `2062.89` |
| `Close` | Closing price | `2064.78` |
| `Volume` | Trading volume | `1250` |

### Sample Data Format:
```csv
Time,Open,High,Low,Close,Volume
2024-01-01 00:00:00,2063.45,2065.12,2062.89,2064.78,1250
2024-01-01 00:01:00,2064.78,2066.23,2063.45,2065.67,890
...
```

## 🔗 Data Sources

### Recommended Free Data Sources:

1. **MetaTrader 5 (MT5)**
   - Free XAUUSD historical data
   - Export to CSV format
   - High quality 1-minute data

2. **Yahoo Finance**
   - Use yfinance Python library
   - Limited historical data
   - Free but may have gaps

3. **Alpha Vantage**
   - Free API with registration
   - Good quality data
   - Rate limited

4. **Investing.com**
   - Download historical data
   - Multiple timeframes
   - Manual download required

### Premium Data Sources:

1. **Quandl**
2. **Bloomberg API**
3. **Refinitiv (Reuters)**
4. **Interactive Brokers**

## 🛠️ Quick Setup with Sample Data

### Option 1: Create Sample Data (for testing)

```python
# Run this to create sample data for testing
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create sample XAUUSD data
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
dates = pd.date_range(start_date, end_date, freq='1min')

# Generate realistic XAUUSD price data
np.random.seed(42)
base_price = 2000
price_changes = np.random.normal(0, 0.5, len(dates))
prices = base_price + np.cumsum(price_changes)

# Create OHLCV data
sample_data = []
for i, date in enumerate(dates):
    open_price = prices[i]
    close_price = prices[i] + np.random.normal(0, 0.2)
    high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.3))
    low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.3))
    volume = np.random.randint(100, 2000)
    
    sample_data.append({
        'Time': date.strftime('%Y-%m-%d %H:%M:%S'),
        'Open': round(open_price, 2),
        'High': round(high_price, 2),
        'Low': round(low_price, 2),
        'Close': round(close_price, 2),
        'Volume': volume
    })

# Save to CSV
df = pd.DataFrame(sample_data)
df.to_csv('datacsv/XAUUSD_M1.csv', index=False)
print("✅ Sample data created: datacsv/XAUUSD_M1.csv")
```

### Option 2: Download from Yahoo Finance

```python
import yfinance as yf
import pandas as pd
from pathlib import Path

# Create datacsv folder
Path('datacsv').mkdir(exist_ok=True)

# Download XAUUSD data (Gold USD)
ticker = "GC=F"  # Gold futures
data = yf.download(ticker, start="2024-01-01", end="2024-12-31", interval="1m")

# Format for our system
data.reset_index(inplace=True)
data.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
data = data[['Time', 'Open', 'High', 'Low', 'Close', 'Volume']]
data['Time'] = data['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Save to CSV
data.to_csv('datacsv/XAUUSD_M1.csv', index=False)
print("✅ Yahoo Finance data downloaded: datacsv/XAUUSD_M1.csv")
```

## ✅ Verification

After setting up your data, verify it works:

```bash
# Run the system
python ProjectP.py

# Choose option 3 (Quick Test) to verify data loading
# The system should detect your CSV files automatically
```

## 📊 System Behavior

- **Data Detection**: System automatically scans `datacsv/` folder
- **Primary File**: Looks for `XAUUSD_M1.csv` first
- **Fallback**: Uses any available CSV file in the folder
- **Real Data Only**: No dummy data generation
- **Validation**: Automatic data quality checks

## 🚫 What NOT to Include

- ❌ Do not commit large CSV files to git
- ❌ Do not include personal trading data
- ❌ Do not share proprietary data sources
- ❌ Do not include API keys in data files

## 🆘 Troubleshooting

### "No data files found"
1. Check if `datacsv/` folder exists
2. Verify CSV files are in correct format
3. Ensure at least one CSV file is present

### "Data validation failed"
1. Check column names match expected format
2. Verify timestamp format: `YYYY-MM-DD HH:MM:SS`
3. Ensure numeric data in OHLCV columns

### "Permission errors"
1. Check folder write permissions
2. Ensure CSV files are not locked by other programs

---

**📝 Note**: This system is designed for **historical data analysis only**. No live trading capabilities are included for safety.

**🔒 Security**: Keep your data files local and do not share proprietary market data.
