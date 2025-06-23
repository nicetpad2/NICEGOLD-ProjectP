"""
Advanced Feature Engineering for AUC Improvement
เพิ่มฟีเจอร์ขั้นสูงเพื่อปรับปรุง AUC จาก 0.516 ให้สูงขึ้น

Focus: Market Microstructure, Regime Detection, Multi-timeframe Analysis
"""

import pandas as pd
import numpy as np
import ta
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    def __init__(self):
        self.feature_groups = []
        
    def add_market_microstructure_features(self, df):
        """เพิ่มฟีเจอร์ Market Microstructure"""
        features = df.copy()
        
        # 1. Price Action Features
        features['body_size'] = abs(features['Close'] - features['Open'])
        features['upper_wick'] = features['High'] - features[['Open', 'Close']].max(axis=1)
        features['lower_wick'] = features[['Open', 'Close']].min(axis=1) - features['Low']
        features['total_range'] = features['High'] - features['Low']
        
        # 2. Relative Features
        features['body_to_range_ratio'] = features['body_size'] / (features['total_range'] + 1e-8)
        features['upper_wick_ratio'] = features['upper_wick'] / (features['total_range'] + 1e-8)
        features['lower_wick_ratio'] = features['lower_wick'] / (features['total_range'] + 1e-8)
        
        # 3. Pattern Recognition
        features['doji'] = (features['body_size'] < features['total_range'] * 0.1).astype(int)
        features['hammer'] = (
            (features['lower_wick'] > features['body_size'] * 2) & 
            (features['upper_wick'] < features['body_size'] * 0.5)
        ).astype(int)
        features['shooting_star'] = (
            (features['upper_wick'] > features['body_size'] * 2) & 
            (features['lower_wick'] < features['body_size'] * 0.5)
        ).astype(int)
        
        self.feature_groups.append("Market Microstructure")
        return features
    
    def add_regime_detection_features(self, df):
        """เพิ่มฟีเจอร์ Market Regime Detection"""
        features = df.copy()
        
        # 1. Volatility Regimes
        features['returns'] = features['Close'].pct_change()
        features['volatility_5'] = features['returns'].rolling(5).std()
        features['volatility_20'] = features['returns'].rolling(20).std()
        features['volatility_ratio'] = features['volatility_5'] / (features['volatility_20'] + 1e-8)
        
        # Volatility Regime Classification
        vol_quantiles = features['volatility_20'].quantile([0.33, 0.67])
        features['vol_regime'] = pd.cut(
            features['volatility_20'], 
            bins=[-np.inf, vol_quantiles.iloc[0], vol_quantiles.iloc[1], np.inf],
            labels=[0, 1, 2]  # Low, Medium, High volatility
        ).astype(float)
        
        # 2. Trend Regimes
        features['sma_10'] = features['Close'].rolling(10).mean()
        features['sma_50'] = features['Close'].rolling(50).mean()
        features['trend_strength'] = (features['sma_10'] - features['sma_50']) / features['Close']
        
        # Trend Classification
        features['trend_regime'] = np.where(
            features['trend_strength'] > 0.001, 1,  # Uptrend
            np.where(features['trend_strength'] < -0.001, -1, 0)  # Downtrend vs Sideways
        )
        
        # 3. Market Stress Indicators
        features['max_drawdown_5'] = features['Close'].rolling(5).apply(
            lambda x: (x.min() - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0
        )
        features['max_runup_5'] = features['Close'].rolling(5).apply(
            lambda x: (x.max() - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0
        )
        
        self.feature_groups.append("Regime Detection")
        return features
    
    def add_multi_timeframe_features(self, df):
        """เพิ่มฟีเจอร์ Multi-timeframe Analysis"""
        features = df.copy()
        
        # 1. Multiple Moving Averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = features['Close'].rolling(period).mean()
            features[f'ema_{period}'] = features['Close'].ewm(span=period).mean()
            features[f'price_vs_sma_{period}'] = (features['Close'] - features[f'sma_{period}']) / features['Close']
        
        # 2. Bollinger Bands
        for period in [10, 20]:
            sma = features['Close'].rolling(period).mean()
            std = features['Close'].rolling(period).std()
            features[f'bb_upper_{period}'] = sma + (2 * std)
            features[f'bb_lower_{period}'] = sma - (2 * std)
            features[f'bb_position_{period}'] = (features['Close'] - features[f'bb_lower_{period}']) / (
                features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'] + 1e-8
            )
        
        # 3. RSI Multiple Timeframes
        for period in [7, 14, 21]:
            features[f'rsi_{period}'] = ta.momentum.RSIIndicator(
                features['Close'], window=period
            ).rsi()
            
        # 4. MACD Features
        macd = ta.trend.MACD(features['Close'])
        features['macd'] = macd.macd()
        features['macd_signal'] = macd.macd_signal()
        features['macd_histogram'] = macd.macd_diff()
        
        self.feature_groups.append("Multi-timeframe")
        return features
    
    def add_momentum_features(self, df):
        """เพิ่มฟีเจอร์ Momentum และ Mean Reversion"""
        features = df.copy()
        
        # 1. Rate of Change
        for period in [1, 3, 5, 10]:
            features[f'roc_{period}'] = features['Close'].pct_change(period)
            features[f'roc_{period}_normalized'] = zscore(features[f'roc_{period}'].fillna(0))
        
        # 2. Acceleration
        features['momentum_5'] = features['Close'].rolling(5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0
        )
        features['acceleration_5'] = features['momentum_5'].diff()
        
        # 3. Relative Strength
        features['rs_vs_sma20'] = features['Close'] / features['Close'].rolling(20).mean()
        features['rs_rank_10'] = features['Close'].rolling(10).rank(pct=True)
        
        # 4. Mean Reversion Indicators
        features['deviation_from_mean'] = (features['Close'] - features['Close'].rolling(20).mean()) / (
            features['Close'].rolling(20).std() + 1e-8
        )
        features['mean_reversion_signal'] = np.where(
            abs(features['deviation_from_mean']) > 2, 
            -np.sign(features['deviation_from_mean']), 0
        )
        
        self.feature_groups.append("Momentum & Mean Reversion")
        return features
    
    def add_volume_features(self, df):
        """เพิ่มฟีเจอร์ Volume Analysis (หากมี Volume)"""
        features = df.copy()
        
        if 'Volume' in features.columns:
            # 1. Volume Indicators
            features['volume_sma_10'] = features['Volume'].rolling(10).mean()
            features['volume_ratio'] = features['Volume'] / (features['volume_sma_10'] + 1e-8)
            
            # 2. Price-Volume Relationship
            features['price_volume_trend'] = (
                (features['Close'] - features['Close'].shift(1)) / features['Close'].shift(1) *
                features['Volume']
            ).fillna(0)
            
            # 3. On-Balance Volume
            obv = []
            obv_value = 0
            for i in range(len(features)):
                if i == 0:
                    obv.append(0)
                else:
                    if features['Close'].iloc[i] > features['Close'].iloc[i-1]:
                        obv_value += features['Volume'].iloc[i]
                    elif features['Close'].iloc[i] < features['Close'].iloc[i-1]:
                        obv_value -= features['Volume'].iloc[i]
                    obv.append(obv_value)
            
            features['obv'] = obv
            features['obv_sma_10'] = pd.Series(obv).rolling(10).mean()
            
            self.feature_groups.append("Volume Analysis")
        else:
            # Create synthetic volume indicators based on price action
            features['synthetic_volume'] = features['total_range'] * abs(features['returns'])
            features['volume_proxy'] = features['synthetic_volume'].rolling(10).mean()
            self.feature_groups.append("Synthetic Volume")
            
        return features
    
    def add_statistical_features(self, df):
        """เพิ่มฟีเจอร์ Statistical และ Distribution"""
        features = df.copy()
        
        # 1. Statistical Moments
        for window in [5, 10, 20]:
            returns_window = features['returns'].rolling(window)
            features[f'skewness_{window}'] = returns_window.skew()
            features[f'kurtosis_{window}'] = returns_window.kurt()
            
        # 2. Percentile Features
        for window in [10, 20]:
            features[f'percentile_rank_{window}'] = features['Close'].rolling(window).rank(pct=True)
            
        # 3. Entropy and Complexity
        features['price_entropy_10'] = features['Close'].rolling(10).apply(
            lambda x: -sum(p * np.log2(p + 1e-8) for p in np.histogram(x, bins=5, density=True)[0] if p > 0)
        )
        
        # 4. Fractal Dimension (simplified)
        features['fractal_dimension_5'] = features['Close'].rolling(5).apply(
            lambda x: len(x) / (len(x) + sum(abs(np.diff(x))))
        )
        
        self.feature_groups.append("Statistical Features")
        return features
    
    def add_interaction_features(self, df):
        """เพิ่มฟีเจอร์ Interaction"""
        features = df.copy()
        
        # Key interactions that might be meaningful for trading
        if 'rsi_14' in features.columns and 'volatility_ratio' in features.columns:
            features['rsi_volatility_interaction'] = features['rsi_14'] * features['volatility_ratio']
            
        if 'macd_histogram' in features.columns and 'trend_regime' in features.columns:
            features['macd_trend_interaction'] = features['macd_histogram'] * features['trend_regime']
            
        if 'bb_position_20' in features.columns and 'momentum_5' in features.columns:
            features['bb_momentum_interaction'] = features['bb_position_20'] * features['momentum_5']
        
        self.feature_groups.append("Interaction Features")
        return features
    
    def create_target_variable(self, df, method='future_return', periods=5):
        """สร้าง Target Variable ที่ดีขึ้น"""
        
        if method == 'future_return':
            # Use future return as target
            future_return = df['Close'].shift(-periods) / df['Close'] - 1
            target = (future_return > 0).astype(int)
            
        elif method == 'bollinger_breakout':
            # Bollinger Band breakout as target
            sma_20 = df['Close'].rolling(20).mean()
            std_20 = df['Close'].rolling(20).std()
            bb_upper = sma_20 + (2 * std_20)
            bb_lower = sma_20 - (2 * std_20)
            
            future_high = df['High'].rolling(periods).max().shift(-periods)
            future_low = df['Low'].rolling(periods).min().shift(-periods)
            
            target = np.where(
                future_high > bb_upper, 1,  # Bullish breakout
                np.where(future_low < bb_lower, 0, np.nan)  # Bearish breakout
            )
            
        elif method == 'momentum_persistence':
            # Momentum persistence as target
            current_momentum = df['Close'].rolling(5).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0
            )
            future_momentum = current_momentum.shift(-periods)
            target = (current_momentum * future_momentum > 0).astype(int)
            
        return pd.Series(target, index=df.index, name='target')
    
    def engineer_all_features(self, df, target_method='future_return'):
        """รันการสร้างฟีเจอร์ทั้งหมด"""
        print("🔧 Starting Advanced Feature Engineering...")
        
        # Ensure we have basic OHLC columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Apply all feature engineering steps
        features = df.copy()
        
        print("   Adding Market Microstructure features...")
        features = self.add_market_microstructure_features(features)
        
        print("   Adding Regime Detection features...")
        features = self.add_regime_detection_features(features)
        
        print("   Adding Multi-timeframe features...")
        features = self.add_multi_timeframe_features(features)
        
        print("   Adding Momentum features...")
        features = self.add_momentum_features(features)
        
        print("   Adding Volume features...")
        features = self.add_volume_features(features)
        
        print("   Adding Statistical features...")
        features = self.add_statistical_features(features)
        
        print("   Adding Interaction features...")
        features = self.add_interaction_features(features)
        
        print("   Creating improved target variable...")
        features['target'] = self.create_target_variable(features, method=target_method)
        
        # Clean up features
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        print(f"✅ Feature engineering complete!")
        print(f"📊 Total features: {len(features.columns)}")
        print(f"🎯 Feature groups: {', '.join(self.feature_groups)}")
        
        return features
    
    def get_feature_importance_summary(self, df):
        """สรุปความสำคัญของฟีเจอร์"""
        if 'target' not in df.columns:
            print("⚠️ No target variable found for importance calculation")
            return None
            
        from sklearn.feature_selection import mutual_info_classif
        
        feature_cols = [col for col in df.columns if col != 'target']
        X = df[feature_cols].fillna(0)
        y = df['target'].fillna(0)
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        print(f"\n📈 Top 15 Most Important Features:")
        print(importance_df.head(15).to_string(index=False))
        
        return importance_df

def main():
    """Test the advanced feature engineering"""
    # Load sample data
    try:
        df = pd.read_csv("XAUUSD_M1.csv", nrows=5000)
    except FileNotFoundError:
        print("⚠️ XAUUSD_M1.csv not found. Creating sample data...")
        # Create sample OHLC data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
        prices = 2000 + np.cumsum(np.random.randn(1000) * 0.1)
        
        df = pd.DataFrame({
            'Open': prices + np.random.randn(1000) * 0.05,
            'High': prices + abs(np.random.randn(1000)) * 0.1,
            'Low': prices - abs(np.random.randn(1000)) * 0.1,
            'Close': prices,
            'Volume': np.random.randint(100, 1000, 1000)
        }, index=dates)
    
    # Initialize feature engineer
    engineer = AdvancedFeatureEngineer()
    
    # Create all features
    enhanced_df = engineer.engineer_all_features(df, target_method='future_return')
    
    # Get feature importance
    importance = engineer.get_feature_importance_summary(enhanced_df)
    
    # Save enhanced dataset
    enhanced_df.to_parquet('output_default/enhanced_features.parquet')
    print(f"\n💾 Saved enhanced features to: output_default/enhanced_features.parquet")
    print(f"📊 Dataset shape: {enhanced_df.shape}")
    
    return enhanced_df

if __name__ == "__main__":
    enhanced_data = main()
