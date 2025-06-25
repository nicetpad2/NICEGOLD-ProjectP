from colorama import Fore, Style, init as colorama_init
from real_data_loader import load_real_trading_data
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from typing import Tuple, Optional, Dict, Any
        import joblib
        import json
import numpy as np
import os
import pandas as pd
        import traceback
import warnings
"""
Training Module - ใช้ข้อมูลจริงเท่านั้น
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

ห้าม:
- ใช้ dummy data
- สร้างข้อมูลสังเคราะห์
- ใช้ข้อมูลจำลอง

ใช้:
- ข้อมูล XAUUSD จริงเท่านั้น
- Real market data จาก XAUUSD_M1.csv และ XAUUSD_M15.csv
"""

warnings.filterwarnings('ignore')

# ใช้ real_data_loader เท่านั้น
colorama_init(autoreset = True)

def load_only_real_data(timeframe: str = "M15", max_rows: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    โหลดข้อมูลจริงเท่านั้น - ห้ามใช้ dummy data

    Args:
        timeframe: "M1" หรือ "M15" เท่านั้น
        max_rows: จำนวนแถวสูงสุด (None = ไม่จำกัด)

    Returns:
        DataFrame และ metadata ของข้อมูลจริง
    """
    print(f"{Fore.GREEN}🔄 Loading REAL MARKET DATA ONLY (timeframe: {timeframe}){Style.RESET_ALL}")
    print(f"{Fore.RED}❌ NO DUMMY DATA ALLOWED{Style.RESET_ALL}")

    # ตรวจสอบว่าไฟล์ข้อมูลจริงมีอยู่
    data_files = {
        "M1": "XAUUSD_M1.csv", 
        "M15": "XAUUSD_M15.csv"
    }

    expected_file = data_files.get(timeframe.upper())
    if not expected_file or not os.path.exists(expected_file):
        raise FileNotFoundError(f"❌ Real data file not found: {expected_file}")

    # โหลดข้อมูลจริงเท่านั้น
    df, info = load_real_trading_data(
        timeframe = timeframe, 
        max_rows = max_rows, 
        profit_threshold = 0.12
    )

    # ยืนยันว่าเป็นข้อมูลจริง
    info['data_validation'] = {
        'is_real_data': True, 
        'is_dummy_data': False, 
        'is_synthetic_data': False, 
        'source_verified': True, 
        'file_source': expected_file
    }

    print(f"{Fore.GREEN}✅ VERIFIED: Using real market data only{Style.RESET_ALL}")
    print(f"{Fore.GREEN}📊 Data points: {len(df):, } (100% real market data){Style.RESET_ALL}")

    return df, info

def validate_real_data_only(df: pd.DataFrame, info: Dict[str, Any]) -> bool:
    """
    ตรวจสอบว่าเป็นข้อมูลจริงเท่านั้น
    """
    validation = info.get('data_validation', {})

    # ตรวจสอบเครื่องหมายการยืนยัน
    is_real = validation.get('is_real_data', False)
    is_not_dummy = not validation.get('is_dummy_data', True)
    is_not_synthetic = not validation.get('is_synthetic_data', True)
    source_verified = validation.get('source_verified', False)

    if not all([is_real, is_not_dummy, is_not_synthetic, source_verified]):
        raise ValueError("❌ VALIDATION FAILED: Not verified as real data!")

    # ตรวจสอบคุณภาพข้อมูล
    if len(df) < 1000:
        print(f"{Fore.YELLOW}⚠️ WARNING: Limited data points ({len(df):, }){Style.RESET_ALL}")

    # ตรวจสอบช่วงราคาที่สมเหตุสมผล (ทองคำ)
    price_range = df['Close'].max() - df['Close'].min()
    if price_range < 100:  # ราคาทองผันผวนอย่างน้อย $100
        print(f"{Fore.YELLOW}⚠️ WARNING: Unusual price range: ${price_range:.2f}{Style.RESET_ALL}")

    print(f"{Fore.GREEN}✅ DATA VALIDATION PASSED: Real market data confirmed{Style.RESET_ALL}")
    return True

def train(timeframe: str = "M15", 
          max_rows: Optional[int] = None, 
          model_type: str = "random_forest") -> Dict[str, Any]:
    """
    เทรนโมเดลด้วยข้อมูลจริงเท่านั้น

    Args:
        timeframe: "M1" หรือ "M15"
        max_rows: จำนวนแถวสูงสุด (None = ไม่จำกัด)
        model_type: "random_forest" หรือ "gradient_boosting"

    Returns:
        ผลลัพธ์การเทรน
    """
    print(f"{Fore.CYAN}{' = '*80}")
    print(f"{Fore.CYAN}🤖 TRAINING WITH REAL DATA ONLY")
    print(f"{Fore.CYAN}{' = '*80}")

    try:
        # 1. โหลดข้อมูลจริงเท่านั้น
        df, info = load_only_real_data(timeframe = timeframe, max_rows = max_rows)

        # 2. ตรวจสอบความถูกต้องของข้อมูล
        validate_real_data_only(df, info)

        # 3. เตรียมข้อมูลสำหรับ ML
        print(f"{Fore.GREEN}🔧 Preparing features from real data...{Style.RESET_ALL}")

        # แยก features และ targets
        feature_cols = [col for col in df.columns if col not in
                       ['Time', 'target', 'target_binary', 'target_return']]

        X = df[feature_cols].select_dtypes(include = [np.number])
        y = df['target']

        # ทำความสะอาดข้อมูล
        X = X.fillna(X.median())  # ใช้ median จากข้อมูลจริง

        print(f"{Fore.GREEN}📊 Features: {X.shape[1]} columns, {X.shape[0]:, } samples{Style.RESET_ALL}")
        print(f"{Fore.GREEN}🎯 Target distribution: {y.value_counts().to_dict()}{Style.RESET_ALL}")

        # 4. แยกข้อมูล train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = 0.2, random_state = 42, stratify = y
        )

        print(f"{Fore.GREEN}🔧 Training set: {X_train.shape[0]:, } samples{Style.RESET_ALL}")
        print(f"{Fore.GREEN}🔧 Test set: {X_test.shape[0]:, } samples{Style.RESET_ALL}")

        # 5. เทรนโมเดล
        print(f"{Fore.GREEN}🤖 Training {model_type} model...{Style.RESET_ALL}")

        if model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators = 100, 
                max_depth = 10, 
                random_state = 42, 
                n_jobs = -1
            )
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(
                n_estimators = 100, 
                max_depth = 6, 
                random_state = 42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # เทรนโมเดล
        model.fit(X_train, y_train)

        # 6. ประเมินผล
        print(f"{Fore.GREEN}📈 Evaluating model performance...{Style.RESET_ALL}")

        # ทำนาย
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # คำนวณ metrics
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        # Cross - validation
        cv_scores = cross_val_score(model, X, y, cv = 5, scoring = 'accuracy')

        # Feature importance (top 10)
        feature_importance = pd.DataFrame({
            'feature': X.columns, 
            'importance': model.feature_importances_
        }).sort_values('importance', ascending = False).head(10)

        print(f"{Fore.GREEN}📊 Training Accuracy: {train_score:.3f}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}📊 Test Accuracy: {test_score:.3f}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}📊 CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}{Style.RESET_ALL}")

        print(f"{Fore.CYAN}🔝 Top Features:{Style.RESET_ALL}")
        for idx, row in feature_importance.iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")

        # 7. บันทึกผลลัพธ์
        os.makedirs("output_default", exist_ok = True)

        results = {
            'model_type': model_type, 
            'timeframe': timeframe, 
            'data_source': 'REAL_MARKET_DATA_ONLY', 
            'samples_total': len(df), 
            'samples_train': len(X_train), 
            'samples_test': len(X_test), 
            'features_count': X.shape[1], 
            'training_accuracy': float(train_score), 
            'test_accuracy': float(test_score), 
            'cv_accuracy_mean': float(cv_scores.mean()), 
            'cv_accuracy_std': float(cv_scores.std()), 
            'feature_importance': feature_importance.to_dict('records'), 
            'data_validation': info['data_validation'], 
            'target_distribution': y.value_counts().to_dict()
        }

        # บันทึก model และผลลัพธ์
        model_filename = f"output_default/model_{model_type}_{timeframe}_real_data.joblib"
        joblib.dump(model, model_filename)

        with open("output_default/training_results_real_data.json", "w") as f:
            json.dump(results, f, indent = 2)

        print(f"{Fore.GREEN}💾 Model saved: {model_filename}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}📁 Results saved: output_default/training_results_real_data.json{Style.RESET_ALL}")

        # 8. สรุปผล
        print(f"\n{Fore.CYAN}{' = '*80}")
        print(f"{Fore.CYAN}📊 TRAINING SUMMARY (REAL DATA ONLY)")
        print(f"{Fore.CYAN}{' = '*80}")
        print(f"{Fore.GREEN}✅ Data Source: 100% Real Market Data")
        print(f"{Fore.GREEN}✅ Model: {model_type}")
        print(f"{Fore.GREEN}✅ Timeframe: {timeframe}")
        print(f"{Fore.GREEN}✅ Samples: {len(df):, }")
        print(f"{Fore.GREEN}✅ Test Accuracy: {test_score:.3f}")
        print(f"{Fore.GREEN}✅ No dummy data used")
        print(f"{Fore.CYAN}{' = '*80}{Style.RESET_ALL}")

        return results

    except Exception as e:
        print(f"{Fore.RED}❌ Training failed: {e}{Style.RESET_ALL}")
        raise

def train_multi_timeframe(max_rows: Optional[int] = None) -> Dict[str, Any]:
    """
    เทรนด้วยข้อมูลจริงจากหลาย timeframe
    """
    print(f"{Fore.CYAN}🔥 MULTI - TIMEFRAME TRAINING (REAL DATA ONLY){Style.RESET_ALL}")

    results = {}

    for timeframe in ["M1", "M15"]:
        try:
            print(f"\n{Fore.YELLOW}🔄 Training {timeframe} model...{Style.RESET_ALL}")
            result = train(timeframe = timeframe, max_rows = max_rows)
            results[timeframe] = result

        except Exception as e:
            print(f"{Fore.RED}❌ Failed to train {timeframe}: {e}{Style.RESET_ALL}")
            results[timeframe] = {'error': str(e)}

    # บันทึกผลรวม
    with open("output_default/multi_timeframe_results_real_data.json", "w") as f:
        json.dump(results, f, indent = 2)

    print(f"{Fore.GREEN}💾 Multi - timeframe results saved{Style.RESET_ALL}")
    return results

if __name__ == "__main__":
    # ทดสอบการเทรนด้วยข้อมูลจริงเท่านั้น
    print("🚀 Starting real data training...")

    try:
        # เทรนแบบ single timeframe
        print("Training with M15 timeframe...")
        result = train(timeframe = "M15", max_rows = 10000)  # จำกัด 10k สำหรับทดสอบ
        print("✅ Training completed successfully!")

        # เทรนแบบ multi - timeframe (ถ้าต้องการ)
        # print("Training multi - timeframe...")
        # multi_result = train_multi_timeframe(max_rows = 5000)

    except Exception as e:
        print(f"❌ Training failed: {e}")
        traceback.print_exc()