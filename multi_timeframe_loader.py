"""
Multi-Timeframe Data Loader - ขั้นเทพ
=====================================

โหลดและผสมข้อมูลจากหลาย timeframe เพื่อการวิเคราะห์ที่แม่นยำขั้นสูง
- M1 (1 นาที): Short-term signals, Scalping, Noise detection
- M15 (15 นาที): Medium-term trends, Swing signals
- ผสมข้อมูลแบบ intelligent alignment
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

class MultiTimeframeLoader:
    """โหลดและผสมข้อมูลจากหลาย timeframe ขั้นเทพ"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = base_path
        self.m1_path = os.path.join(base_path, "XAUUSD_M1.csv")
        self.m15_path = os.path.join(base_path, "XAUUSD_M15.csv")
        
    def load_timeframe_data(self, timeframe: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        """โหลดข้อมูลตาม timeframe"""
        
        if timeframe == "M1":
            path = self.m1_path
            print(f"🔄 Loading M1 data from {path}")
        elif timeframe == "M15":
            path = self.m15_path
            print(f"🔄 Loading M15 data from {path}")
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
            
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        
        try:
            # โหลดข้อมูล
            if max_rows:
                df = pd.read_csv(path, nrows=max_rows)
                print(f"📊 Loaded {len(df):,} rows (limited)")
            else:
                df = pd.read_csv(path)
                print(f"📊 Loaded {len(df):,} rows (full dataset)")
            
            # แปลงเวลาสำหรับ M15 (แก้ไขปีพุทธศักราช)
            if timeframe == "M15":
                print("🔧 Converting Buddhist era to Christian era...")
                df['Timestamp'] = df['Timestamp'].str.replace(r'^25(\d{2})', 
                    lambda m: f'20{int(m.group(1)) - 43}', regex=True)
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                df.rename(columns={'Timestamp': 'Time'}, inplace=True)
            else:
                df['Time'] = pd.to_datetime(df['Time'])
            
            df = df.sort_values('Time').reset_index(drop=True)
            
            print(f"📅 Data range: {df['Time'].min()} to {df['Time'].max()}")
            print(f"💰 Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading {timeframe} data: {e}")
            raise
    
    def create_basic_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """สร้างฟีเจอร์พื้นฐานสำหรับแต่ละ timeframe"""
        print(f"🔧 Creating {timeframe} features...")
        
        df = df.copy()
        prefix = timeframe.lower()
        
        try:
            # 1. Price Features
            df[f'{prefix}_price_change'] = df['Close'] - df['Open']
            df[f'{prefix}_price_change_pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
            df[f'{prefix}_high_low_spread'] = df['High'] - df['Low']
            df[f'{prefix}_close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
            
            # 2. Moving Averages
            for period in [5, 10, 20]:
                df[f'{prefix}_ma_{period}'] = df['Close'].rolling(window=period).mean()
                df[f'{prefix}_ma_{period}_diff'] = df['Close'] - df[f'{prefix}_ma_{period}']
                df[f'{prefix}_ma_{period}_slope'] = df[f'{prefix}_ma_{period}'].diff()
            
            # 3. Volatility
            df[f'{prefix}_volatility'] = df['Close'].rolling(window=20).std()
            df[f'{prefix}_atr'] = df[f'{prefix}_high_low_spread'].rolling(window=14).mean()
            
            # 4. Momentum (RSI-like)
            price_change = df['Close'].diff()
            gain = price_change.where(price_change > 0, 0)
            loss = -price_change.where(price_change < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            df[f'{prefix}_rsi'] = 100 - (100 / (1 + rs))
            
            # 5. Volume analysis (if available)
            if 'Volume' in df.columns:
                df[f'{prefix}_volume_ma'] = df['Volume'].rolling(window=20).mean()
                df[f'{prefix}_volume_ratio'] = df['Volume'] / df[f'{prefix}_volume_ma']
            
            # 6. Trend Strength
            df[f'{prefix}_trend_strength'] = abs(df[f'{prefix}_ma_5'] - df[f'{prefix}_ma_20']) / df['Close'] * 100
            
            return df
            
        except Exception as e:
            print(f"❌ Error creating {timeframe} features: {e}")
            raise
    
    def create_multi_timeframe_features(self, m1_df: pd.DataFrame, m15_df: pd.DataFrame) -> pd.DataFrame:
        """สร้างฟีเจอร์ Multi-timeframe ขั้นเทพ"""
        print("🧠 Creating ADVANCED Multi-timeframe features...")
        
        try:
            # ปรับ M1 และ M15 ให้ match กันด้วยเวลา
            print("🔄 Aligning M1 and M15 timeframes...")
            
            # สร้าง reference time จาก M15 
            m15_df = m15_df.copy()
            m15_df['Time_rounded'] = m15_df['Time'].dt.floor('15T')
            
            # สร้าง M1 aggregated ให้ match กับ M15
            m1_df = m1_df.copy()
            m1_df['Time_rounded'] = m1_df['Time'].dt.floor('15T')
            
            # 1. M1 Aggregations for M15 periods
            print("📊 Creating M1 aggregations for M15 alignment...")
            m1_agg = m1_df.groupby('Time_rounded').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum' if 'Volume' in m1_df.columns else 'count',
                'm1_volatility': 'mean',
                'm1_rsi': 'mean',
                'm1_atr': 'mean',
                'm1_trend_strength': 'mean'
            }).reset_index()
              # Rename aggregated M1 features
            new_columns = ['Time_rounded']
            for col in m1_agg.columns[1:]:  # Skip Time_rounded
                if col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    new_columns.append(f'm1_agg_{col}')
                else:
                    new_columns.append(f'm1_agg_{col}')
            
            m1_agg.columns = new_columns
            
            # 2. Merge M15 with aggregated M1
            print("🔗 Merging M15 with M1 aggregations...")
            merged_df = pd.merge(m15_df, m1_agg, on='Time_rounded', how='left')
            
            # 3. Multi-timeframe Analysis Features
            print("🎯 Creating multi-timeframe analysis features...")
              # Trend Alignment
            merged_df['mtf_trend_alignment'] = np.where(
                (merged_df['m15_ma_5'] > merged_df['m15_ma_20']) & 
                (merged_df['m1_agg_Close'] > merged_df['m1_agg_Open']), 1,
                np.where(
                    (merged_df['m15_ma_5'] < merged_df['m15_ma_20']) & 
                    (merged_df['m1_agg_Close'] < merged_df['m1_agg_Open']), -1, 0
                )
            )
              # Volatility Convergence
            merged_df['mtf_volatility_ratio'] = merged_df['m1_agg_m1_volatility'] / merged_df['m15_volatility']
              # Price Momentum Across Timeframes
            merged_df['mtf_momentum_m1'] = (merged_df['m1_agg_Close'] - merged_df['m1_agg_Open']) / merged_df['m1_agg_Open'] * 100
            merged_df['mtf_momentum_m15'] = (merged_df['Close'] - merged_df['Open']) / merged_df['Open'] * 100
            merged_df['mtf_momentum_alignment'] = merged_df['mtf_momentum_m1'] * merged_df['mtf_momentum_m15']
            
            # RSI Divergence Detection
            merged_df['mtf_rsi_divergence'] = abs(merged_df['m15_rsi'] - merged_df['m1_agg_m1_rsi'])
            
            # Volume Confirmation
            if 'Volume' in m1_df.columns:
                merged_df['mtf_volume_confirmation'] = merged_df['m1_agg_Volume'] / merged_df['Volume']
            
            # 4. Support/Resistance Level Analysis
            print("📈 Creating support/resistance analysis...")
            
            # M15 levels
            merged_df['m15_support'] = merged_df['Low'].rolling(window=20).min()
            merged_df['m15_resistance'] = merged_df['High'].rolling(window=20).max()
            
            # M1 levels (aggregated)
            merged_df['m1_support'] = merged_df['m1_agg_Low']
            merged_df['m1_resistance'] = merged_df['m1_agg_High']
            
            # Level breakthrough detection
            merged_df['mtf_breakout_strength'] = np.where(
                merged_df['Close'] > merged_df['m15_resistance'], 1,
                np.where(merged_df['Close'] < merged_df['m15_support'], -1, 0)
            )
            
            # 5. Advanced Pattern Recognition
            print("🔍 Creating pattern recognition features...")
              # Doji pattern across timeframes
            m15_doji = abs(merged_df['Close'] - merged_df['Open']) / (merged_df['High'] - merged_df['Low']) < 0.1
            m1_doji = abs(merged_df['m1_agg_Close'] - merged_df['m1_agg_Open']) / (merged_df['m1_agg_High'] - merged_df['m1_agg_Low']) < 0.1
            merged_df['mtf_doji_pattern'] = (m15_doji & m1_doji).astype(int)
            
            # Engulfing pattern
            merged_df['mtf_engulfing'] = np.where(
                (merged_df['Close'] > merged_df['Open']) & 
                (merged_df['m1_agg_Close'] > merged_df['m1_agg_Open']) &
                (merged_df['Close'] > merged_df['Open'].shift(1)) &
                (merged_df['Open'] < merged_df['Close'].shift(1)), 1,
                np.where(
                    (merged_df['Close'] < merged_df['Open']) & 
                    (merged_df['m1_agg_Close'] < merged_df['m1_agg_Open']) &
                    (merged_df['Close'] < merged_df['Open'].shift(1)) &
                    (merged_df['Open'] > merged_df['Close'].shift(1)), -1, 0
                )
            )
              # Market Regime Detection
            print("🌊 Creating market regime detection...")
            
            # Trend strength across timeframes
            merged_df['mtf_trend_strength'] = (merged_df['m15_trend_strength'] + merged_df['m1_agg_m1_trend_strength']) / 2
            
            # Market regime classification
            merged_df['mtf_market_regime'] = np.where(
                merged_df['mtf_trend_strength'] > 2, 'trending',
                np.where(merged_df['mtf_trend_strength'] < 0.5, 'ranging', 'transitioning')
            )
            
            # Regime-based signals
            regime_mapping = {'trending': 2, 'transitioning': 1, 'ranging': 0}
            merged_df['mtf_regime_signal'] = merged_df['mtf_market_regime'].map(regime_mapping)
            
            print(f"✅ Created {len([col for col in merged_df.columns if 'mtf_' in col])} multi-timeframe features")
            
            return merged_df
            
        except Exception as e:
            print(f"❌ Error creating multi-timeframe features: {e}")
            raise
    
    def create_trading_targets(self, df: pd.DataFrame, future_periods: int = 5, 
                             profit_threshold: float = 0.12) -> pd.DataFrame:
        """สร้าง trading targets สำหรับ multi-timeframe"""
        print(f"🎯 Creating multi-timeframe trading targets (periods: {future_periods}, threshold: {profit_threshold}%)")
        
        try:
            df = df.copy()
            
            # คำนวณ future returns
            df['future_return'] = (df['Close'].shift(-future_periods) - df['Close']) / df['Close'] * 100
            
            # สร้าง target แบบ 3 class
            df['target'] = np.where(
                df['future_return'] > profit_threshold, 1,  # Buy signal
                np.where(df['future_return'] < -profit_threshold, 2, 0)  # Sell signal, Hold
            )
            
            # Binary target สำหรับ direction
            df['target_binary'] = np.where(df['future_return'] > 0, 1, 0)
            
            # Continuous target
            df['target_return'] = df['future_return']
            
            # Target distribution
            target_counts = df['target'].value_counts().sort_index()
            print(f"🎯 Target distribution:")
            print(f"   Hold (0): {target_counts.get(0, 0):,} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
            print(f"   Buy (1):  {target_counts.get(1, 0):,} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")
            print(f"   Sell (2): {target_counts.get(2, 0):,} ({target_counts.get(2, 0)/len(df)*100:.1f}%)")
            
            return df
            
        except Exception as e:
            print(f"❌ Error creating targets: {e}")
            raise

    def load_multi_timeframe_data(self, 
                                 max_rows_m1: Optional[int] = None,
                                 max_rows_m15: Optional[int] = None,
                                 future_periods: int = 5,
                                 profit_threshold: float = 0.12) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """โหลดและผสมข้อมูล Multi-timeframe ขั้นเทพ"""
        
        print(f"🚀 Loading ADVANCED Multi-timeframe data...")
        print(f"   M1 max_rows: {max_rows_m1}")
        print(f"   M15 max_rows: {max_rows_m15}")
        
        try:
            # 1. โหลดข้อมูลทั้ง 2 timeframe
            m1_df = self.load_timeframe_data("M1", max_rows_m1)
            m15_df = self.load_timeframe_data("M15", max_rows_m15)
            
            # 2. สร้างฟีเจอร์พื้นฐานสำหรับแต่ละ timeframe
            m1_df = self.create_basic_features(m1_df, "M1")
            m15_df = self.create_basic_features(m15_df, "M15")
            
            # 3. ผสมข้อมูลและสร้างฟีเจอร์ multi-timeframe
            merged_df = self.create_multi_timeframe_features(m1_df, m15_df)
            
            # 4. สร้าง trading targets
            merged_df = self.create_trading_targets(merged_df, future_periods, profit_threshold)
            
            # 5. ทำความสะอาดข้อมูล
            initial_len = len(merged_df)
            merged_df = merged_df.dropna()
            final_len = len(merged_df)
            print(f"🧹 Cleaned data: {initial_len:,} → {final_len:,} rows ({final_len/initial_len*100:.1f}% retained)")
            
            # 6. สรุปข้อมูล
            feature_cols = [col for col in merged_df.columns if col not in 
                          ['Time', 'Time_rounded', 'target', 'target_binary', 'target_return', 'future_return']]
            
            mtf_features = [col for col in feature_cols if 'mtf_' in col]
            m1_features = [col for col in feature_cols if 'm1_' in col]
            m15_features = [col for col in feature_cols if 'm15_' in col]
            
            info = {
                'total_rows': final_len,
                'total_features': len(feature_cols),
                'mtf_features': len(mtf_features),
                'm1_features': len(m1_features),
                'm15_features': len(m15_features),
                'time_range': (merged_df['Time'].min(), merged_df['Time'].max()),
                'price_range': (merged_df['Close'].min(), merged_df['Close'].max()),
                'feature_columns': feature_cols,
                'target_columns': ['target', 'target_binary', 'target_return'],
                'target_distribution': merged_df['target'].value_counts().to_dict(),
                'profit_threshold': profit_threshold,
                'future_periods': future_periods,
                'data_source': 'Multi-timeframe (M1 + M15) Real Market Data'
            }
            
            print("✅ Multi-timeframe data preparation completed!")
            print(f"📊 Total Features: {len(feature_cols)}")
            print(f"   🔥 Multi-timeframe: {len(mtf_features)}")
            print(f"   ⚡ M1 features: {len(m1_features)}")
            print(f"   📈 M15 features: {len(m15_features)}")
            print(f"📊 Samples: {final_len:,}")
            
            return merged_df, info
            
        except Exception as e:
            print(f"❌ Error in multi-timeframe data loading: {e}")
            raise

def load_multi_timeframe_trading_data(max_rows_m1: Optional[int] = None,
                                     max_rows_m15: Optional[int] = None,
                                     future_periods: int = 5,
                                     profit_threshold: float = 0.12) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main function สำหรับโหลดข้อมูล Multi-timeframe ขั้นเทพ
    
    Parameters:
        max_rows_m1: จำกัดจำนวนแถว M1 (None = ไม่จำกัด)
        max_rows_m15: จำกัดจำนวนแถว M15 (None = ไม่จำกัด)  
        future_periods: จำนวน periods สำหรับ target
        profit_threshold: เกณฑ์กำไร % สำหรับ buy/sell signals
    
    Returns:
        df: DataFrame ที่ผสม M1 + M15 + Multi-timeframe features
        info: ข้อมูลสถิติและรายละเอียด
    """
    
    loader = MultiTimeframeLoader()
    return loader.load_multi_timeframe_data(
        max_rows_m1=max_rows_m1,
        max_rows_m15=max_rows_m15,
        future_periods=future_periods,
        profit_threshold=profit_threshold
    )

if __name__ == "__main__":
    # ทดสอบการโหลดข้อมูล Multi-timeframe
    print("🧪 Testing Multi-timeframe Data Loader...")
    
    df, info = load_multi_timeframe_trading_data(
        max_rows_m1=5000,  # ทดสอบด้วย 5k แถว
        max_rows_m15=1000,  # ทดสอบด้วย 1k แถว
        profit_threshold=0.15
    )
    
    print(f"\n🎉 Multi-timeframe data ready!")
    print(f"📊 Shape: {df.shape}")
    print(f"🔥 MTF Features: {info['mtf_features']}")
