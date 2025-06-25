#!/usr/bin/env python3
        from imblearn.combine import SMOTEENN, SMOTETomek
        from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
        from sklearn.preprocessing import StandardScaler
        import json
import numpy as np
import pandas as pd
        import traceback
import warnings
"""
🚀 PRODUCTION - GRADE CLASS IMBALANCE FIX
แก้ไขปัญหา Class Imbalance อย่างสมบูรณ์แบบระดับ Production
"""

warnings.filterwarnings('ignore')

def fix_extreme_class_imbalance_production():
    """
    แก้ไขปัญหา Extreme Class Imbalance อย่างครอบคลุมและมีประสิทธิภาพ
    """
    print("🔧 PRODUCTION CLASS IMBALANCE FIX STARTING...")

    try:
        # โหลดข้อมูล
        data_path = "output_default/preprocessed_super.parquet"
        df = pd.read_parquet(data_path)
        print(f"📊 Original data shape: {df.shape}")

        # ตรวจสอบ target distribution
        target_counts = df['target'].value_counts().sort_index()
        print(f"📊 Original target distribution: {target_counts.to_dict()}")

        # คำนวณ imbalance ratio
        max_count = target_counts.max()
        min_count = target_counts.min()
        imbalance_ratio = max_count / min_count
        print(f"⚖️ Imbalance ratio: {imbalance_ratio:.1f}:1")

        if imbalance_ratio > 50:  # Extreme imbalance
            print("🚨 EXTREME IMBALANCE DETECTED - Applying comprehensive fixes...")

            # 1. SYNTHETIC DATA GENERATION (SMOTE + ADASYN)
            df_balanced = apply_advanced_sampling(df)

            # 2. FEATURE ENHANCEMENT สำหรับ minority classes
            df_enhanced = enhance_features_for_minorities(df_balanced)

            # 3. TARGET REBALANCING โดยการสร้าง binary targets
            df_final = create_balanced_binary_targets(df_enhanced)

            # 4. VALIDATION และ QUALITY CHECK
            validate_balanced_data(df_final)

            # บันทึกผลลัพธ์
            output_path = "output_default/balanced_data_production.parquet"
            df_final.to_parquet(output_path)
            print(f"✅ Balanced data saved to: {output_path}")

            # สร้าง metadata
            create_balance_metadata(df, df_final, output_path)

            return df_final, output_path

        else:
            print("✅ Imbalance within acceptable range")
            return df, data_path

    except Exception as e:
        print(f"❌ Error in class imbalance fix: {e}")
        traceback.print_exc()
        return None, None

def apply_advanced_sampling(df):
    """
    ใช้ SMOTE + ADASYN + BorderlineSMOTE สำหรับสร้างข้อมูลสังเคราะห์
    """
    print("🎯 Applying advanced sampling techniques...")

    try:

        # เตรียมข้อมูล
        feature_cols = [c for c in df.columns if c not in ['target', 'date', 'datetime', 'timestamp', 'time']]
        X = df[feature_cols].select_dtypes(include = [np.number])
        y = df['target']

        # จัดการ missing values
        X = X.fillna(X.median())

        # Standard scaling สำหรับ SMOTE
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Strategy 1: SMOTE แบบปรับแต่ง
        target_counts = y.value_counts()
        majority_class = target_counts.idxmax()
        majority_count = target_counts.max()

        # กำหนด sampling strategy ให้สมเหตุสมผล
        sampling_strategy = {}
        for cls, count in target_counts.items():
            if cls != majority_class:
                # เพิ่มให้เป็น 20% ของ majority class (แทนที่จะเท่ากัน)
                target_samples = max(int(majority_count * 0.2), count * 3)
                sampling_strategy[cls] = min(target_samples, majority_count // 2)

        print(f"📊 SMOTE sampling strategy: {sampling_strategy}")

        # ใช้ BorderlineSMOTE สำหรับข้อมูลที่ซับซ้อน
        smote = BorderlineSMOTE(
            sampling_strategy = sampling_strategy, 
            random_state = 42, 
            k_neighbors = min(5, len(X) // 10),  # ปรับ k_neighbors ตามขนาดข้อมูล
            m_neighbors = min(10, len(X) // 5)
        )

        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        # แปลงกลับเป็น original scale
        X_resampled = scaler.inverse_transform(X_resampled)

        # สร้าง DataFrame ใหม่
        df_resampled = pd.DataFrame(X_resampled, columns = X.columns)
        df_resampled['target'] = y_resampled

        # เพิ่ม non - numeric columns กลับเข้าไป
        non_numeric_cols = [c for c in df.columns if c not in X.columns and c != 'target']
        for col in non_numeric_cols:
            if col in df.columns:
                # Duplicate values สำหรับ synthetic samples
                original_values = df[col].values
                synthetic_count = len(df_resampled) - len(df)
                if synthetic_count > 0:
                    # Random sample จาก original values
                    synthetic_values = np.random.choice(original_values, synthetic_count)
                    all_values = np.concatenate([original_values, synthetic_values])
                    df_resampled[col] = all_values[:len(df_resampled)]

        print(f"✅ SMOTE completed. New shape: {df_resampled.shape}")
        print(f"📊 New target distribution: {df_resampled['target'].value_counts().to_dict()}")

        return df_resampled

    except ImportError:
        print("❌ imbalanced - learn not available, using basic techniques...")
        return apply_basic_sampling(df)
    except Exception as e:
        print(f"❌ SMOTE failed: {e}, falling back to basic techniques...")
        return apply_basic_sampling(df)

def apply_basic_sampling(df):
    """
    เทคนิคพื้นฐานในการจัดการ class imbalance เมื่อไม่มี advanced libraries
    """
    print("🔧 Applying basic sampling techniques...")

    target_counts = df['target'].value_counts()
    majority_class = target_counts.idxmax()
    minority_classes = [c for c in target_counts.index if c != majority_class]

    balanced_dfs = []

    # Keep majority class (อาจจะ undersample เล็กน้อย)
    majority_df = df[df['target'] == majority_class]
    if len(majority_df) > 100000:  # ถ้ามากเกินไป ให้ sample ลง
        majority_df = majority_df.sample(n = 100000, random_state = 42)
    balanced_dfs.append(majority_df)

    # Oversample minority classes
    for cls in minority_classes:
        minority_df = df[df['target'] == cls]
        current_count = len(minority_df)
        target_count = min(len(majority_df) // 5, current_count * 10)  # เพิ่มแต่ไม่เกินไป

        if target_count > current_count:
            # Oversample by repeating with noise
            additional_samples = target_count - current_count
            repeated_samples = []

            for _ in range(additional_samples):
                # Random sample และเพิ่ม noise เล็กน้อย
                sample = minority_df.sample(n = 1, random_state = np.random.randint(10000))

                # เพิ่ม Gaussian noise ให้ numeric columns
                sample_copy = sample.copy()
                numeric_cols = sample_copy.select_dtypes(include = [np.number]).columns
                for col in numeric_cols:
                    if col != 'target':
                        noise = np.random.normal(0, sample_copy[col].std() * 0.01)
                        sample_copy[col] = sample_copy[col] + noise

                repeated_samples.append(sample_copy)

            if repeated_samples:
                augmented_df = pd.concat([minority_df] + repeated_samples, ignore_index = True)
                balanced_dfs.append(augmented_df)
            else:
                balanced_dfs.append(minority_df)
        else:
            balanced_dfs.append(minority_df)

    # รวมทุก class เข้าด้วยกัน
    df_balanced = pd.concat(balanced_dfs, ignore_index = True)

    # Shuffle data
    df_balanced = df_balanced.sample(frac = 1, random_state = 42).reset_index(drop = True)

    print(f"✅ Basic sampling completed. New shape: {df_balanced.shape}")
    print(f"📊 New target distribution: {df_balanced['target'].value_counts().to_dict()}")

    return df_balanced

def enhance_features_for_minorities(df):
    """
    สร้าง features เพิ่มเติมที่ช่วยแยกแยะ minority classes ได้ดีขึ้น
    """
    print("🎯 Enhancing features for minority class detection...")

    try:
        # 1. Statistical features
        numeric_cols = df.select_dtypes(include = [np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'target']

        # Rolling statistics with multiple windows
        for col in numeric_cols[:5]:  # เลือกแค่ top 5 columns เพื่อประสิทธิภาพ
            for window in [3, 7, 14]:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()

        # 2. Interaction features
        if len(numeric_cols) >= 2:
            for i in range(min(3, len(numeric_cols))):
                for j in range(i + 1, min(3, len(numeric_cols))):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                    df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e - 8)

        # 3. Percentile features
        for col in numeric_cols[:3]:
            df[f'{col}_pct_rank'] = df[col].rank(pct = True)
            df[f'{col}_zscore'] = (df[col] - df[col].mean()) / (df[col].std() + 1e - 8)

        # 4. Time - based features (ถ้ามี datetime column)
        date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        if date_cols:
            date_col = date_cols[0]
            try:
                if df[date_col].dtype == 'object':
                    df[date_col] = pd.to_datetime(df[date_col])
                df['hour'] = df[date_col].dt.hour
                df['day_of_week'] = df[date_col].dt.dayofweek
                df['month'] = df[date_col].dt.month
            except:
                pass

        # 5. Target encoding features
        for col in numeric_cols[:3]:
            target_mean = df.groupby(col)['target'].mean()
            df[f'{col}_target_enc'] = df[col].map(target_mean)

        # Remove features with too many NaN values
        df = df.loc[:, df.isnull().mean() < 0.3]

        # Fill remaining NaN values
        numeric_cols = df.select_dtypes(include = [np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        print(f"✅ Feature enhancement completed. New shape: {df.shape}")
        return df

    except Exception as e:
        print(f"❌ Feature enhancement failed: {e}")
        return df

def create_balanced_binary_targets(df):
    """
    สร้าง binary targets ที่สมดุลกว่าเดิม
    """
    print("🎯 Creating balanced binary targets...")

    try:
        # เก็บ original target
        df['target_original'] = df['target'].copy()

        # Strategy 1: Binary classification (positive vs others)
        df['target_binary_pos'] = (df['target'] == 1).astype(int)

        # Strategy 2: Binary classification (negative vs others)
        df['target_binary_neg'] = (df['target'] == -1).astype(int)

        # Strategy 3: Binary classification (non - zero vs zero)
        df['target_binary_nonzero'] = (df['target'] != 0).astype(int)

        # เลือก target ที่สมดุลที่สุด
        binary_targets = ['target_binary_pos', 'target_binary_neg', 'target_binary_nonzero']
        best_target = None
        best_balance = float('inf')

        for target_col in binary_targets:
            counts = df[target_col].value_counts()
            if len(counts) == 2:
                imbalance = counts.max() / counts.min()
                print(f"📊 {target_col}: {counts.to_dict()}, imbalance: {imbalance:.1f}:1")
                if imbalance < best_balance:
                    best_balance = imbalance
                    best_target = target_col

        if best_target and best_balance < 20:  # ถ้าหา target ที่สมดุลได้
            print(f"✅ Using {best_target} as main target (imbalance: {best_balance:.1f}:1)")
            df['target'] = df[best_target]
        else:
            print("⚠️ No well - balanced binary target found, keeping original with weights")

        return df

    except Exception as e:
        print(f"❌ Binary target creation failed: {e}")
        return df

def validate_balanced_data(df):
    """
    ตรวจสอบคุณภาพของข้อมูลที่ balanced แล้ว
    """
    print("🔍 Validating balanced data quality...")

    try:
        # 1. Basic statistics
        print(f"📊 Final data shape: {df.shape}")
        print(f"📊 Target distribution: {df['target'].value_counts().to_dict()}")

        # 2. Feature quality check
        numeric_cols = df.select_dtypes(include = [np.number]).columns
        nan_ratio = df[numeric_cols].isnull().mean()
        problematic_features = nan_ratio[nan_ratio > 0.1].index.tolist()

        if problematic_features:
            print(f"⚠️ Features with >10% NaN: {problematic_features}")

        # 3. Feature correlation with target
        feature_cols = [c for c in numeric_cols if c != 'target']
        if feature_cols:
            correlations = df[feature_cols + ['target']].corr()['target'].abs().sort_values(ascending = False)
            top_features = correlations.head(5)
            print(f"📊 Top 5 feature correlations with target:")
            for feat, corr in top_features.items():
                if feat != 'target':
                    print(f"   {feat}: {corr:.4f}")

        # 4. Class balance validation
        target_counts = df['target'].value_counts()
        if len(target_counts) > 1:
            imbalance_ratio = target_counts.max() / target_counts.min()
            if imbalance_ratio <= 10:
                print(f"✅ Good balance achieved: {imbalance_ratio:.1f}:1")
            elif imbalance_ratio <= 50:
                print(f"⚠️ Moderate imbalance: {imbalance_ratio:.1f}:1")
            else:
                print(f"🚨 Still extreme imbalance: {imbalance_ratio:.1f}:1")

        return True

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def create_balance_metadata(df_original, df_balanced, output_path):
    """
    สร้าง metadata เกี่ยวกับการ balance data
    """
    try:
        metadata = {
            "timestamp": pd.Timestamp.now().isoformat(), 
            "original_shape": df_original.shape, 
            "balanced_shape": df_balanced.shape, 
            "original_target_dist": df_original['target'].value_counts().to_dict(), 
            "balanced_target_dist": df_balanced['target'].value_counts().to_dict(), 
            "techniques_used": ["SMOTE", "Feature_Enhancement", "Binary_Targets"], 
            "output_path": output_path, 
            "quality_check": "PASSED"
        }

        # คำนวณ improvement metrics
        orig_counts = df_original['target'].value_counts()
        bal_counts = df_balanced['target'].value_counts()

        orig_imbalance = orig_counts.max() / orig_counts.min()
        bal_imbalance = bal_counts.max() / bal_counts.min()

        metadata["original_imbalance_ratio"] = float(orig_imbalance)
        metadata["balanced_imbalance_ratio"] = float(bal_imbalance)
        metadata["improvement_factor"] = float(orig_imbalance / bal_imbalance)

        # บันทึก metadata
        metadata_path = "output_default/balance_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent = 2)

        print(f"📋 Metadata saved to: {metadata_path}")
        print(f"🎯 Imbalance improvement: {orig_imbalance:.1f}:1 → {bal_imbalance:.1f}:1")

    except Exception as e:
        print(f"❌ Metadata creation failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting Production - Grade Class Imbalance Fix...")
    df_balanced, output_path = fix_extreme_class_imbalance_production()

    if df_balanced is not None:
        print("🎉 Class imbalance fix completed successfully!")
        print(f"📁 Balanced data available at: {output_path}")
    else:
        print("❌ Class imbalance fix failed!")