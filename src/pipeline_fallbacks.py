"""
Pipeline fallback functions for when imports fail
Provides basic functionality to keep the system running
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)

def safe_load_csv_auto_fallback(file_path: str, row_limit: Optional[int] = None, **kwargs: Any) -> pd.DataFrame:
    """
    Fallback function for safe_load_csv_auto when main import fails
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ไม่พบไฟล์ {file_path}")
        
        # Basic CSV loading with error handling
        df = pd.read_csv(file_path, nrows=row_limit, **kwargs)
        
        # Handle BOM in column names
        if len(df.columns) > 0 and df.columns[0].startswith('\ufeff'):
            df.columns = [col.replace('\ufeff', '') for col in df.columns]
        
        # Basic datetime detection and conversion
        datetime_columns = []
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                try:
                    # Try to convert first few values to check if it's datetime
                    test_sample = df[col].dropna().head(3)
                    if not test_sample.empty:
                        pd.to_datetime(test_sample.iloc[0])
                        datetime_columns.append(col)
                except:
                    pass
        
        # Convert detected datetime columns
        for col in datetime_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        # Set index to first datetime column if available
        if datetime_columns:
            try:
                df = df.set_index(datetime_columns[0])
            except:
                pass
        
        return df
        
    except Exception as e:
        logger.error(f"Fallback CSV loading failed: {e}")
        return pd.DataFrame()

def basic_feature_engineering_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic feature engineering when advanced functions are not available
    """
    if df.empty:
        return df
    
    try:
        # Basic technical indicators if OHLC data is available
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            # Simple moving averages
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            
            # Price changes
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_Abs'] = df['Price_Change'].abs()
            
            # High-Low spread
            df['HL_Spread'] = (df['High'] - df['Low']) / df['Close']
            
            # Volume indicators if available
            if 'Volume' in df.columns:
                df['Volume_SMA'] = df['Volume'].rolling(window=10).mean()
                df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Fill missing values
        df = df.ffill().bfill()
        
        return df
        
    except Exception as e:
        logger.error(f"Basic feature engineering failed: {e}")
        return df

def basic_model_training_fallback(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Basic model training when advanced ML is not available
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train simple model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        return {
            'model': model,
            'accuracy': accuracy,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'method': 'fallback_random_forest'
        }
        
    except Exception as e:
        logger.error(f"Basic model training failed: {e}")
        return {
            'model': None,
            'accuracy': 0.5,
            'auc': 0.5,
            'predictions': None,
            'probabilities': None,
            'method': 'fallback_dummy',
            'error': str(e)
        }

def create_basic_signals_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create basic trading signals when advanced signal generation fails
    """
    if df.empty or 'Close' not in df.columns:
        return df
    
    try:
        # Simple moving average crossover
        df['SMA_Fast'] = df['Close'].rolling(window=10).mean()
        df['SMA_Slow'] = df['Close'].rolling(window=20).mean()
        
        # Generate signals
        df['Signal'] = 0
        df.loc[df['SMA_Fast'] > df['SMA_Slow'], 'Signal'] = 1
        df.loc[df['SMA_Fast'] < df['SMA_Slow'], 'Signal'] = -1
        
        # Signal strength (simple momentum)
        df['Signal_Strength'] = df['Close'].pct_change().rolling(window=5).mean()
        
        return df
        
    except Exception as e:
        logger.error(f"Basic signal creation failed: {e}")
        return df

def basic_risk_management_fallback(df: pd.DataFrame) -> Dict[str, float]:
    """
    Basic risk management when advanced functions are not available
    """
    try:
        if 'Close' not in df.columns:
            return {'risk_score': 0.5, 'position_size': 0.01}
        
        # Simple volatility-based risk
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std()
        
        # Risk score based on volatility
        risk_score = min(volatility * 100, 1.0)  # Cap at 1.0
        
        # Position size inversely related to risk
        position_size = max(0.01, min(0.1, 0.05 / (risk_score + 0.01)))
        
        return {
            'risk_score': risk_score,
            'position_size': position_size,
            'volatility': volatility,
            'method': 'fallback_volatility'
        }
        
    except Exception as e:
        logger.error(f"Basic risk management failed: {e}")
        return {
            'risk_score': 0.5,
            'position_size': 0.01,
            'method': 'fallback_dummy',
            'error': str(e)
        }

# Create a fallback pipeline registry
FALLBACK_FUNCTIONS = {
    'safe_load_csv_auto': safe_load_csv_auto_fallback,
    'feature_engineering': basic_feature_engineering_fallback,
    'model_training': basic_model_training_fallback,
    'signal_generation': create_basic_signals_fallback,
    'risk_management': basic_risk_management_fallback
}

def get_fallback_function(function_name: str):
    """Get a fallback function by name"""
    return FALLBACK_FUNCTIONS.get(function_name, lambda *args, **kwargs: None)

logger.info("✅ Pipeline fallback functions loaded successfully")
