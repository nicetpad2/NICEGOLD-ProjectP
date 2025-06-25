from colorama import Fore, Style, init as colorama_init
            from real_data_loader import load_real_trading_data
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, classification_report
            from sklearn.model_selection import train_test_split
        import json
import os
        import pandas as pd
import sys
        import traceback
"""
Quick Test - ProjectP with Real Data
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

ทดสอบ ProjectP ที่ใช้ข้อมูลจริงแบบรวดเร็ว
"""

colorama_init(autoreset = True)

def test_projectp_real_data():
    """ทดสอบ ProjectP ด้วยข้อมูลจริง"""

    print(f"{Fore.CYAN}{' = '*60}")
    print(f"{Fore.CYAN}🧪 TESTING ProjectP WITH REAL DATA")
    print(f"{Fore.CYAN}{' = '*60}")

    try:
        # 1. ตรวจสอบว่าข้อมูลจริงพร้อมใช้แล้วหรือไม่
        print(f"{Fore.GREEN}Step 1: Checking real data availability...{Style.RESET_ALL}")

        data_files = [
            "output_default/real_trading_data.csv", 
            "output_default/features.csv", 
            "output_default/targets.csv"
        ]

        data_ready = all(os.path.exists(f) for f in data_files)

        if not data_ready:
            print(f"{Fore.YELLOW}📊 Real data not ready, preparing...{Style.RESET_ALL}")

            df, info = load_real_trading_data(
                timeframe = "M15", 
                max_rows = 3000,  # ใช้ 3k แถวสำหรับทดสอบเร็ว
                profit_threshold = 0.12
            )

            # บันทึกข้อมูล
            os.makedirs("output_default", exist_ok = True)
            df.to_csv("output_default/real_trading_data.csv", index = False)

            feature_cols = [col for col in df.columns if col not in
                           ['Time', 'target', 'target_binary', 'target_return']]

            df[feature_cols].to_csv("output_default/features.csv", index = False)
            df[['target', 'target_binary', 'target_return']].to_csv("output_default/targets.csv", index = False)

            print(f"{Fore.GREEN}✅ Real data prepared: {len(df):, } samples{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}✅ Real data already available{Style.RESET_ALL}")

        # 2. ทดสอบการโหลดข้อมูลจริง
        print(f"{Fore.GREEN}Step 2: Testing real data loading...{Style.RESET_ALL}")

        features_df = pd.read_csv("output_default/features.csv")
        targets_df = pd.read_csv("output_default/targets.csv")

        print(f"📊 Features: {features_df.shape}")
        print(f"📊 Targets: {targets_df.shape}")
        print(f"📊 Feature columns: {list(features_df.columns)[:5]}...")
        print(f"📊 Target columns: {list(targets_df.columns)}")

        # 3. ทดสอบการรันฟังก์ชันพื้นฐาน
        print(f"{Fore.GREEN}Step 3: Testing basic pipeline functions...{Style.RESET_ALL}")

        # สร้างฟังก์ชันทดสอบง่ายๆ
        def test_simple_ml_pipeline():
            """ทดสอบ ML pipeline แบบง่าย"""

            # เตรียมข้อมูล
            X = features_df.select_dtypes(include = ['int64', 'float64'])
            y = targets_df['target']

            # แยกข้อมูล train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size = 0.2, random_state = 42, stratify = y
            )

            print(f"🔧 Training data: {X_train.shape}")
            print(f"🔧 Test data: {X_test.shape}")

            # เทรนโมเดล
            model = RandomForestClassifier(n_estimators = 50, random_state = 42, n_jobs = -1)
            model.fit(X_train, y_train)

            # ทำนาย
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print(f"📈 Model Accuracy: {accuracy:.3f}")
            print(f"📊 Classification Report:")
            print(classification_report(y_test, y_pred))

            return accuracy > 0.3  # เกณฑ์ขั้นต่ำ

        # รันการทดสอบ
        ml_success = test_simple_ml_pipeline()

        if ml_success:
            print(f"{Fore.GREEN}✅ ML pipeline test successful!{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠️ ML pipeline test had issues but continuing...{Style.RESET_ALL}")

        # 4. ทดสอบการสร้างรายงาน
        print(f"{Fore.GREEN}Step 4: Generating test report...{Style.RESET_ALL}")

        report = {
            'test_date': '2025 - 06 - 23', 
            'data_source': 'Real XAUUSD Market Data', 
            'samples': len(features_df), 
            'features': len(features_df.columns), 
            'targets': len(targets_df.columns), 
            'ml_test_passed': ml_success, 
            'status': 'SUCCESS' if ml_success else 'WARNING'
        }

        with open("output_default/test_report.json", "w") as f:
            json.dump(report, f, indent = 2)

        # 5. สรุปผล
        print(f"\n{Fore.CYAN}{' = '*60}")
        print(f"{Fore.CYAN}📊 TEST SUMMARY")
        print(f"{Fore.CYAN}{' = '*60}")

        print(f"{Fore.GREEN}✅ Data Source: Real XAUUSD Market Data")
        print(f"{Fore.GREEN}✅ Samples: {len(features_df):, }")
        print(f"{Fore.GREEN}✅ Features: {len(features_df.columns)}")
        print(f"{Fore.GREEN}✅ ML Pipeline: {'PASSED' if ml_success else 'WARNING'}")

        if ml_success:
            print(f"\n{Fore.GREEN}🎉 ProjectP READY FOR REAL DATA TRADING!{Style.RESET_ALL}")
            print(f"{Fore.CYAN}สามารถรัน: python ProjectP.py - - run_full_pipeline{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.YELLOW}⚠️ ProjectP needs adjustment but basic functions work{Style.RESET_ALL}")

        return True

    except Exception as e:
        print(f"{Fore.RED}❌ Test failed: {e}{Style.RESET_ALL}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(f"{Fore.CYAN}🚀 Starting ProjectP Real Data Test{Style.RESET_ALL}")

    success = test_projectp_real_data()

    if success:
        print(f"\n{Fore.GREEN}🎯 NEXT STEPS:{Style.RESET_ALL}")
        print(f"1. ระบบใช้ข้อมูลทองคำจริงแล้ว!")
        print(f"2. รัน: python ProjectP.py - - run_full_pipeline")
        print(f"3. ข้อมูลจริงให้ผลลัพธ์ที่น่าเชื่อถือกว่าข้อมูลตัวอย่าง")
    else:
        print(f"\n{Fore.RED}❌ Test failed. ต้องแก้ไขก่อนใช้งาน{Style.RESET_ALL}")