    from main import main as run_main
from pathlib import Path
                        from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import sys; sys.path.append('.')
def handle_menu_choice_new(choice):
    """
    Handle menu choice with beautiful interface and robust CSV management.
    All options now use only real CSV data from datacsv/ folder.
    """
    python_cmd = [sys.executable]

    # Ensure we're in the right directory
    os.chdir(PROJECT_ROOT)

    # Handle exit commands
    if choice.lower() in ["0", "q", "quit", "exit"]:
        print(
            f"{colorize('👋 Thank you for using NICEGOLD ProjectP!', Colors.BRIGHT_MAGENTA)}"
        )
        return False

    # Initialize CSV Manager for data - related options
    csv_manager = None
    if choice in ["1", "2", "3", "4", "5", "6", "7", "9", "10"]:  # Data - related options
        try:
            if RobustCSVManager:
                csv_manager = RobustCSVManager()
                print(
                    f"{colorize('📁 Initializing CSV data management...', Colors.BRIGHT_CYAN)}"
                )
            else:
                print(
                    f"{colorize('⚠️  CSV Manager not available - continuing with basic functionality', Colors.BRIGHT_YELLOW)}"
                )
        except Exception as e:
            print(
                f"{colorize('❌ CSV Manager initialization failed:', Colors.BRIGHT_RED)} {str(e)}"
            )
            print(
                f"{colorize('ℹ️  Continuing with basic functionality...', Colors.BRIGHT_BLUE)}"
            )

    if choice == "1":
        # Full Pipeline Run
        print(f"{colorize('🚀 Starting Full Pipeline...', Colors.BRIGHT_GREEN)}")

        if csv_manager:
            try:
                # Show CSV validation report
                csv_manager.print_validation_report()

                # Get best CSV file
                best_csv = csv_manager.get_best_csv_file()
                if not best_csv:
                    print(
                        f"{colorize('❌ No suitable CSV files found in datacsv/', Colors.BRIGHT_RED)}"
                    )
                    print(
                        f"{colorize('📁 Please add valid trading data CSV files to datacsv/ folder', Colors.BRIGHT_YELLOW)}"
                    )
                    return True

                print(
                    f"{colorize('✅ Using best CSV file:', Colors.BRIGHT_GREEN)} {best_csv}"
                )

                # Process the CSV
                df = csv_manager.validate_and_standardize_csv(best_csv)
                print(
                    f"{colorize('📊 Data loaded successfully:', Colors.BRIGHT_GREEN)} {len(df)} rows, {len(df.columns)} columns"
                )

                # Save processed data for pipeline
                processed_path = "datacsv/processed_data.csv"
                df.to_csv(processed_path, index = False)
                print(
                    f"{colorize('💾 Processed data saved to:', Colors.BRIGHT_GREEN)} {processed_path}"
                )

            except Exception as e:
                print(
                    f"{colorize('❌ CSV processing failed:', Colors.BRIGHT_RED)} {str(e)}"
                )
                return True

        return run_command(
            python_cmd
            + [
                " - c", 
                f"""
print('🚀 Starting NICEGOLD Full Pipeline...')
try:
    print('📊 Loading configuration and data...')
    run_main()
    print('✅ Full pipeline completed successfully!')
except Exception as e:
    print(f'❌ Pipeline error: {{e}}')
    print('ℹ️  Check logs for detailed error information')
""", 
            ], 
            "Full Pipeline", 
        )

    elif choice == "2":
        # Data Analysis & Statistics
        print(f"{colorize('📊 Starting Data Analysis...', Colors.BRIGHT_BLUE)}")

        if csv_manager:
            try:
                # Show detailed analysis of all CSV files
                csv_manager.print_validation_report()

                # Analyze best file in detail
                best_csv = csv_manager.get_best_csv_file()
                if best_csv:
                    df = csv_manager.validate_and_standardize_csv(best_csv)

                    print(
                        f"\n{colorize('📈 DETAILED DATA ANALYSIS', Colors.BRIGHT_MAGENTA)}"
                    )
                    print(f"{colorize(' = ' * 50, Colors.BRIGHT_MAGENTA)}")
                    print(f"📊 Dataset: {best_csv}")
                    print(f"📏 Shape: {df.shape}")
                    print(f"📅 Date Range: {df['Time'].min()} to {df['Time'].max()}")
                    print(f"🔢 Columns: {list(df.columns)}")

                    # Basic statistics
                    numeric_cols = df.select_dtypes(include = ["number"]).columns
                    if len(numeric_cols) > 0:
                        print(
                            f"\n{colorize('📊 Statistical Summary:', Colors.BRIGHT_CYAN)}"
                        )
                        print(df[numeric_cols].describe().round(4))

                    # Data quality check
                    print(
                        f"\n{colorize('🔍 Data Quality Check:', Colors.BRIGHT_YELLOW)}"
                    )
                    print(f"Missing Values: {df.isnull().sum().sum()}")
                    print(f"Duplicate Rows: {df.duplicated().sum()}")
                    print(f"Data Types: {df.dtypes.to_dict()}")

                    return True
                else:
                    print(
                        f"{colorize('❌ No suitable CSV files found', Colors.BRIGHT_RED)}"
                    )
                    return True

            except Exception as e:
                print(f"{colorize('❌ Analysis failed:', Colors.BRIGHT_RED)} {str(e)}")
                return True

        return run_command(
            python_cmd
            + [
                " - c", 
                """

print('📊 Starting Data Analysis...')

# Try to load real data
data_files = ['datacsv/XAUUSD_M1_clean.csv', 'datacsv/XAUUSD_M1.csv', 'datacsv/XAUUSD_M15.csv']
df = None

for file_path in data_files:
    if Path(file_path).exists():
        try:
            print(f'📂 Loading {file_path}...')
            df = pd.read_csv(file_path, nrows = 10000)  # Load sample for analysis
            print(f'✅ Loaded {len(df)} rows from {file_path}')
            break
        except Exception as e:
            print(f'⚠️  Failed to load {file_path}: {e}')
            continue

if df is not None:
    print(f'\\n📊 Dataset Analysis:')
    print(f'Shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    print(f'\\nData Types:\\n{df.dtypes}')
    print(f'\\nMissing Values:\\n{df.isnull().sum()}')

    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include = [np.number]).columns
    if len(numeric_cols) > 0:
        print(f'\\n📈 Statistical Summary:')
        print(df[numeric_cols].describe())

    print('✅ Data analysis completed!')
else:
    print('❌ No data files found in datacsv/')
    print('📁 Please add CSV files to datacsv/ folder')
""", 
            ], 
            "Data Analysis", 
        )

    elif choice == "3":
        # Quick Test with Real Data
        print(f"{colorize('⚡ Running Quick Test...', Colors.BRIGHT_YELLOW)}")

        if csv_manager:
            try:
                best_csv = csv_manager.get_best_csv_file()
                if best_csv:
                    # Quick test with real data
                    df = csv_manager.validate_and_standardize_csv(best_csv)
                    print(
                        f"{colorize('⚡ Quick test using real data:', Colors.BRIGHT_GREEN)} {len(df)} rows"
                    )

                    # Basic ML test
                    if all(
                        col in df.columns
                        for col in ["Open", "High", "Low", "Close", "Volume"]
                    ):

                        # Prepare features (use only first 1000 rows for quick test)
                        test_df = df.head(1000).copy()
                        X = test_df[["Open", "High", "Low", "Volume"]].fillna(0)
                        y = (test_df["Close"] > test_df["Open"]).astype(int)

                        # Quick model test
                        model = RandomForestClassifier(n_estimators = 10, random_state = 42)
                        model.fit(X, y)
                        score = model.score(X, y)

                        print(
                            f"{colorize('✅ Quick ML test completed:', Colors.BRIGHT_GREEN)} Accuracy {score:.3f}"
                        )
                        print(
                            f"{colorize('📊 Test features:', Colors.BRIGHT_CYAN)} {list(X.columns)}"
                        )
                        print(
                            f"{colorize('🎯 Prediction target:', Colors.BRIGHT_CYAN)} Price direction (Up/Down)"
                        )

                        return True
                    else:
                        print(
                            f"{colorize('⚠️  Missing required columns for ML test', Colors.BRIGHT_YELLOW)}"
                        )
                        return True

            except Exception as e:
                print(
                    f"{colorize('❌ Quick test failed:', Colors.BRIGHT_RED)} {str(e)}"
                )
                return True

        return run_command(
            python_cmd
            + [
                " - c", 
                """

print('⚡ Quick Test: Initializing...')

# Try to use real data first
data_files = ['datacsv/XAUUSD_M1_clean.csv', 'datacsv/XAUUSD_M1.csv', 'datacsv/XAUUSD_M15.csv']
df = None

for file_path in data_files:
    if Path(file_path).exists():
        try:
            print(f'📂 Loading {file_path} for quick test...')
            df = pd.read_csv(file_path, nrows = 100)  # Small sample for quick test
            print(f'✅ Loaded {len(df)} rows from {file_path}')
            break
        except Exception as e:
            print(f'⚠️  Failed to load {file_path}: {e}')
            continue

if df is not None and len(df) > 10:
    print(f'📊 Using real data for quick test')
    print(f'Shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')

    # Test with real data
    if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
        print('🤖 Testing ML components with real data...')

        X = df[['Open', 'High', 'Low']].fillna(method = 'ffill').fillna(0)
        y = (df['Close'] > df['Open']).astype(int)

        if len(X) > 5:
            model = RandomForestClassifier(n_estimators = 5, random_state = 42)
            model.fit(X, y)
            score = model.score(X, y)
            print(f'✅ Model accuracy with real data: {score:.3f}')
        else:
            print('⚠️  Insufficient data for ML test')
    else:
        print('⚠️  Missing required columns in real data')
else:
    print('⚠️  No real data available, running basic test')

print('⚡ Quick test completed!')
""", 
            ], 
            "Quick Test", 
        )

    elif choice == "4":
        # CSV Data Validation & Management
        print(
            f"{colorize('📊 CSV Data Validation & Management...', Colors.BRIGHT_BLUE)}"
        )

        if csv_manager:
            try:
                # Comprehensive CSV analysis
                print(
                    f"{colorize('🔍 Analyzing all CSV files in datacsv/', Colors.BRIGHT_CYAN)}"
                )
                csv_manager.print_validation_report()

                # Process each CSV file
                report = csv_manager.get_validation_report()

                print(f"\n{colorize('⚙️ Processing CSV Files:', Colors.BRIGHT_MAGENTA)}")
                print(f"{colorize(' = ' * 50, Colors.BRIGHT_MAGENTA)}")

                for file_name, file_info in report["files"].items():
                    try:
                        print(
                            f"\n{colorize(f'📄 Processing {file_name}...', Colors.BRIGHT_YELLOW)}"
                        )

                        file_path = str(csv_manager.datacsv_path / file_name)
                        df = csv_manager.validate_and_standardize_csv(file_path)

                        print(
                            f"  ✅ Success: {len(df)} rows, {len(df.columns)} columns"
                        )
                        print(
                            f"  📅 Date range: {df['Time'].min()} to {df['Time'].max()}"
                        )
                        print(f"  📊 Columns: {list(df.columns)}")

                        # Save cleaned version
                        output_path = f"datacsv/cleaned_{file_name}"
                        df.to_csv(output_path, index = False)
                        print(f"  💾 Cleaned version saved: {output_path}")

                    except Exception as e:
                        print(f"  ❌ Error processing {file_name}: {str(e)}")

                print(
                    f"\n{colorize('✅ CSV validation and management completed!', Colors.BRIGHT_GREEN)}"
                )
                return True

            except Exception as e:
                print(
                    f"{colorize('❌ CSV management failed:', Colors.BRIGHT_RED)} {str(e)}"
                )
                return True

        else:
            print(f"{colorize('❌ CSV Manager not available', Colors.BRIGHT_RED)}")
            return True

    elif choice == "5":
        # Feature Engineering with Real Data
        print(f"{colorize('⚙️ Feature Engineering...', Colors.BRIGHT_CYAN)}")

        if csv_manager:
            try:
                best_csv = csv_manager.get_best_csv_file()
                if best_csv:
                    df = csv_manager.validate_and_standardize_csv(best_csv)

                    print(
                        f"{colorize('⚙️ Engineering features from real data...', Colors.BRIGHT_CYAN)}"
                    )
                    print(f"📊 Source: {best_csv}")
                    print(f"📏 Data shape: {df.shape}")

                    # Feature engineering with real data
                    if all(
                        col in df.columns
                        for col in ["Open", "High", "Low", "Close", "Volume"]
                    ):
                        # Price - based features
                        df["returns"] = df["Close"].pct_change()
                        df["hl_spread"] = (df["High"] - df["Low"]) / df["Close"]
                        df["oc_spread"] = (df["Close"] - df["Open"]) / df["Open"]
                        df["price_position"] = (df["Close"] - df["Low"]) / (
                            df["High"] - df["Low"]
                        )

                        # Moving averages
                        df["sma_5"] = df["Close"].rolling(5).mean()
                        df["sma_20"] = df["Close"].rolling(20).mean()
                        df["sma_50"] = df["Close"].rolling(50).mean()

                        # Volatility measures
                        df["volatility_10"] = df["returns"].rolling(10).std()
                        df["volatility_30"] = df["returns"].rolling(30).std()

                        # RSI calculation
                        delta = df["Close"].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                        loss = ( - delta.where(delta < 0, 0)).rolling(14).mean()
                        rs = gain / loss
                        df["rsi"] = 100 - (100 / (1 + rs))

                        # Bollinger Bands
                        sma_20 = df["Close"].rolling(20).mean()
                        std_20 = df["Close"].rolling(20).std()
                        df["bb_upper"] = sma_20 + (2 * std_20)
                        df["bb_lower"] = sma_20 - (2 * std_20)
                        df["bb_position"] = (df["Close"] - df["bb_lower"]) / (
                            df["bb_upper"] - df["bb_lower"]
                        )

                        # Clean data
                        df_clean = df.dropna()

                        # Save engineered features
                        features_path = "datacsv/engineered_features.csv"
                        df_clean.to_csv(features_path, index = False)

                        print(
                            f"{colorize('✅ Feature engineering completed!', Colors.BRIGHT_GREEN)}"
                        )
                        print(
                            f"📊 Original columns: {len(csv_manager.validate_and_standardize_csv(best_csv).columns)}"
                        )
                        print(f"📊 Engineered columns: {len(df_clean.columns)}")
                        print(f"📊 Final data shape: {df_clean.shape}")
                        print(f"💾 Saved to: {features_path}")

                        # Show new features
                        new_features = [
                            col
                            for col in df_clean.columns
                            if col
                            not in [
                                "Time", 
                                "Open", 
                                "High", 
                                "Low", 
                                "Close", 
                                "Volume", 
                                "target", 
                            ]
                        ]
                        print(f"🎯 New features: {new_features}")

                        return True
                    else:
                        print(
                            f"{colorize('❌ Missing required columns for feature engineering', Colors.BRIGHT_RED)}"
                        )
                        return True

            except Exception as e:
                print(
                    f"{colorize('❌ Feature engineering failed:', Colors.BRIGHT_RED)} {str(e)}"
                )
                return True

        print(
            f"{colorize('⚠️  Using fallback feature engineering...', Colors.BRIGHT_YELLOW)}"
        )
        return run_command(
            python_cmd
            + [
                " - c", 
                """

print('⚙️ Starting Feature Engineering...')

# Try to load real data
data_files = ['datacsv/XAUUSD_M1_clean.csv', 'datacsv/XAUUSD_M1.csv', 'datacsv/XAUUSD_M15.csv']
df = None

for file_path in data_files:
    if Path(file_path).exists():
        try:
            print(f'📂 Loading {file_path}...')
            df = pd.read_csv(file_path, nrows = 5000)  # Load reasonable sample
            print(f'✅ Loaded {len(df)} rows from {file_path}')
            break
        except Exception as e:
            print(f'⚠️  Failed to load {file_path}: {e}')
            continue

if df is not None and all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
    print('🔧 Creating technical indicators from real data...')

    # Basic features
    df['returns'] = df['Close'].pct_change()
    df['hl_spread'] = (df['High'] - df['Low']) / df['Close']
    df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])

    # Moving averages
    df['sma_5'] = df['Close'].rolling(5).mean()
    df['sma_20'] = df['Close'].rolling(20).mean()

    # Volatility
    df['volatility'] = df['returns'].rolling(10).std()

    # RSI simulation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = ( - delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Clean data
    df = df.dropna()

    print(f'✅ Feature engineering completed with real data!')
    print(f'📊 Output data shape: {df.shape}')
    print(f'🎯 Features created: {len(df.columns)} columns')
    print(f'📋 Features: {list(df.columns)}')

    # Save results
    df.to_csv('datacsv/features_output.csv', index = False)
    print('💾 Features saved to datacsv/features_output.csv')
else:
    print('❌ No suitable data found for feature engineering')
    print('📁 Please add valid CSV files to datacsv/ folder')
""", 
            ], 
            "Feature Engineering", 
        )

    # Continue with other choices...
    else:
        print(
            f"{colorize('⚠️  Invalid choice. Please select a valid option.', Colors.BRIGHT_YELLOW)}"
        )
        return True