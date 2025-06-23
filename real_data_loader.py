"""
Real Data Loader - โหลดข้อมูลทองคำจริงแทนข้อมูลตัวอย่าง
===========================================

โหลดข้อมูล XAUUSD จริงและเตรียมสำหรับการเทรนโมเดล ML
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class RealDataLoader:
    """โหลดและประมวลผลข้อมูลทองคำจริง"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = base_path
        self.m1_path = os.path.join(base_path, "XAUUSD_M1.csv")
        self.m15_path = os.path.join(base_path, "XAUUSD_M15.csv")
        
    def load_m1_data(self, max_rows: Optional[int] = None) -> pd.DataFrame:
        """โหลดข้อมูล M1 (1 นาที)"""
        print(f"🔄 Loading M1 data from {self.m1_path}")
        
        if not os.path.exists(self.m1_path):
            raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูล M1: {self.m1_path}")
        
        try:
            # โหลดข้อมูล
            if max_rows:
                df = pd.read_csv(self.m1_path, nrows=max_rows)
                print(f"📊 Loaded {len(df):,} rows (limited)")
            else:
                df = pd.read_csv(self.m1_path)
                print(f"📊 Loaded {len(df):,} rows (full dataset)")
            
            # แปลงเวลา
            df['Time'] = pd.to_datetime(df['Time'])
            df = df.sort_values('Time').reset_index(drop=True)
            
            print(f"📅 Data range: {df['Time'].min()} to {df['Time'].max()}")
            print(f"💰 Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading M1 data: {e}")
            raise
    
    def load_m15_data(self, max_rows: Optional[int] = None) -> pd.DataFrame:
        """โหลดข้อมูล M15 (15 นาที)"""
        print(f"🔄 Loading M15 data from {self.m15_path}")
        
        if not os.path.exists(self.m15_path):
            raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูล M15: {self.m15_path}")
        
        try:
            # โหลดข้อมูล
            if max_rows:
                df = pd.read_csv(self.m15_path, nrows=max_rows)
                print(f"📊 Loaded {len(df):,} rows (limited)")
            else:
                df = pd.read_csv(self.m15_path)
                print(f"📊 Loaded {len(df):,} rows (full dataset)")            # แปลงเวลา (แก้ไขปีพุทธศักราชเป็นคริสต์ศักราช: 25XX -> 20XX)
            print("🔧 Converting Buddhist era to Christian era...")
            # แปลง 2563 -> 2020, 2564 -> 2021, etc.
            df['Timestamp'] = df['Timestamp'].str.replace(r'^25(\d{2})', lambda m: f'20{int(m.group(1)) - 43}', regex=True)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df = df.sort_values('Timestamp').reset_index(drop=True)
            df.rename(columns={'Timestamp': 'Time'}, inplace=True)
            
            print(f"📅 Data range: {df['Time'].min()} to {df['Time'].max()}")
            print(f"💰 Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading M15 data: {e}")
            raise
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """สร้างฟีเจอร์เทคนิคัลสำหรับการเทรดทองคำ"""
        print("🔧 Creating technical features...")
        
        df = df.copy()
        
        try:
            # 1. Price Features
            df['price_change'] = df['Close'] - df['Open']
            df['price_change_pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
            df['high_low_spread'] = df['High'] - df['Low']
            df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
            
            # 2. Moving Averages
            for period in [5, 10, 20, 50]:
                df[f'ma_{period}'] = df['Close'].rolling(window=period).mean()
                df[f'ma_{period}_diff'] = df['Close'] - df[f'ma_{period}']
                df[f'ma_{period}_slope'] = df[f'ma_{period}'].diff()
            
            # 3. Volatility Features
            df['volatility_5'] = df['Close'].rolling(window=5).std()
            df['volatility_20'] = df['Close'].rolling(window=20).std()
            df['volume_ma'] = df['Volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma']
            
            # 4. Price Momentum
            for period in [3, 5, 10]:
                df[f'momentum_{period}'] = df['Close'].pct_change(periods=period) * 100
                df[f'roc_{period}'] = ((df['Close'] / df['Close'].shift(period)) - 1) * 100
            
            # 5. Support/Resistance
            df['high_20'] = df['High'].rolling(window=20).max()
            df['low_20'] = df['Low'].rolling(window=20).min()
            df['resistance_distance'] = (df['high_20'] - df['Close']) / df['Close'] * 100
            df['support_distance'] = (df['Close'] - df['low_20']) / df['Close'] * 100
            
            # 6. Time Features
            df['hour'] = df['Time'].dt.hour
            df['day_of_week'] = df['Time'].dt.dayofweek
            df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] <= 8)).astype(int)
            df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)
            df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] <= 22)).astype(int)
            
            # 7. RSI-like indicator
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            print(f"✅ Created {len([col for col in df.columns if col not in ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']])} features")
            
            return df
            
        except Exception as e:
            print(f"❌ Error creating features: {e}")
            raise
    
    def create_trading_targets(self, df: pd.DataFrame, 
                             future_periods: int = 5,
                             profit_threshold: float = 0.1) -> pd.DataFrame:
        """สร้าง target สำหรับการเทรด (Buy/Sell/Hold)"""
        print(f"🎯 Creating trading targets (look ahead: {future_periods} periods, threshold: {profit_threshold}%)")
        
        df = df.copy()
        
        try:
            # คำนวณราคาในอนาคต
            df['future_high'] = df['High'].shift(-future_periods).rolling(window=future_periods).max()
            df['future_low'] = df['Low'].shift(-future_periods).rolling(window=future_periods).min()
            
            # คำนวณ potential return
            df['potential_profit_long'] = ((df['future_high'] - df['Close']) / df['Close']) * 100
            df['potential_profit_short'] = ((df['Close'] - df['future_low']) / df['Close']) * 100
            
            # สร้าง target
            conditions = [
                (df['potential_profit_long'] >= profit_threshold) & 
                (df['potential_profit_long'] > df['potential_profit_short']),
                
                (df['potential_profit_short'] >= profit_threshold) & 
                (df['potential_profit_short'] > df['potential_profit_long'])
            ]
            
            choices = [1, 2]  # 1=Buy, 2=Sell
            df['target'] = np.select(conditions, choices, default=0)  # 0=Hold
            
            # Binary target (มีการเทรดหรือไม่)
            df['target_binary'] = (df['target'] != 0).astype(int)
            
            # Regression target (expected return)
            df['target_return'] = np.where(
                df['target'] == 1, df['potential_profit_long'],
                np.where(df['target'] == 2, df['potential_profit_short'], 0)
            )
            
            # Statistics
            target_counts = df['target'].value_counts().sort_index()
            print("🎯 Target distribution:")
            print(f"   Hold (0): {target_counts.get(0, 0):,} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
            print(f"   Buy (1):  {target_counts.get(1, 0):,} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")
            print(f"   Sell (2): {target_counts.get(2, 0):,} ({target_counts.get(2, 0)/len(df)*100:.1f}%)")
            
            # Drop helper columns
            df.drop(['future_high', 'future_low', 'potential_profit_long', 'potential_profit_short'], 
                   axis=1, inplace=True)
            
            return df
            
        except Exception as e:
            print(f"❌ Error creating targets: {e}")
            raise
    
    def prepare_ml_data(self, 
                       timeframe: str = "M15",
                       max_rows: Optional[int] = 50000,
                       future_periods: int = 5,
                       profit_threshold: float = 0.1) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """เตรียมข้อมูลสำหรับ ML พร้อมทั้งแยก features และ targets"""
        
        print(f"🚀 Preparing ML data (timeframe: {timeframe}, max_rows: {max_rows})")
        
        try:
            # โหลดข้อมูล
            if timeframe.upper() == "M1":
                df = self.load_m1_data(max_rows=max_rows)
            elif timeframe.upper() == "M15":
                df = self.load_m15_data(max_rows=max_rows)
            else:
                raise ValueError(f"Unsupported timeframe: {timeframe}")
            
            # สร้างฟีเจอร์
            df = self.create_features(df)
            
            # สร้าง targets
            df = self.create_trading_targets(df, future_periods, profit_threshold)
            
            # ลบ NaN
            initial_len = len(df)
            df = df.dropna()
            final_len = len(df)
            print(f"🧹 Cleaned data: {initial_len:,} → {final_len:,} rows ({final_len/initial_len*100:.1f}% retained)")
            
            # แยก features และ targets
            feature_cols = [col for col in df.columns if col not in 
                          ['Time', 'target', 'target_binary', 'target_return']]
            target_cols = ['target', 'target_binary', 'target_return']
            
            # ข้อมูลสถิติ
            info = {
                'total_rows': final_len,
                'feature_count': len(feature_cols),
                'time_range': (df['Time'].min(), df['Time'].max()),
                'price_range': (df['Close'].min(), df['Close'].max()),
                'feature_columns': feature_cols,
                'target_columns': target_cols,
                'target_distribution': df['target'].value_counts().to_dict(),
                'timeframe': timeframe,
                'profit_threshold': profit_threshold,
                'future_periods': future_periods
            }
            
            print("✅ Data preparation completed!")
            print(f"📊 Features: {len(feature_cols)}")
            print(f"📊 Samples: {final_len:,}")
            print(f"📊 Targets: {len(target_cols)}")
            
            return df, info
            
        except Exception as e:
            print(f"❌ Error preparing ML data: {e}")
            raise


def load_real_trading_data(timeframe: str = "M15", 
                          max_rows: Optional[int] = 50000,
                          profit_threshold: float = 0.1) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    ฟังก์ชันหลักสำหรับโหลดข้อมูลจริง
    
    Args:
        timeframe: "M1" หรือ "M15" 
        max_rows: จำนวนแถวสูงสุด (None = ทั้งหมด)
        profit_threshold: เกณฑ์กำไรสำหรับสร้าง target (%)
    
    Returns:
        df: DataFrame พร้อมฟีเจอร์และ targets
        info: ข้อมูลสถิติ
    """
    loader = RealDataLoader()
    return loader.prepare_ml_data(
        timeframe=timeframe,
        max_rows=max_rows,
        profit_threshold=profit_threshold
    )


if __name__ == "__main__":
    # ทดสอบโหลดข้อมูล
    print("=" * 60)
    print("🧪 Testing Real Data Loader")
    print("=" * 60)
    
    try:
        # ทดสอบ M15 (ข้อมูลน้อยกว่า)
        df, info = load_real_trading_data(
            timeframe="M15",
            max_rows=10000,  # ทดสอบแค่ 10k แถว
            profit_threshold=0.15
        )
        
        print("\n📋 Data Summary:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)[:10]}...")  # แสดงแค่ 10 คอลัมน์แรก
        print(f"Target distribution: {info['target_distribution']}")
        
        # แสดงตัวอย่างข้อมูล
        print("\n📊 Sample data:")
        print(df[['Time', 'Close', 'target', 'target_binary']].head())
        
        print("\n✅ Real data loader test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
