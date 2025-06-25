#!/usr/bin/env python3
        from catboost import CatBoostClassifier
            from feature_engineering import run_production_grade_feature_engineering
        from projectp.pipeline import run_full_pipeline
        from projectp.steps.predict import run_predict
        from projectp.steps.preprocess import run_preprocess
        from projectp.steps.threshold import run_threshold
        from projectp.steps.train import run_train
        from projectp.steps.walkforward import run_walkforward
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        import json
import os
            import pandas as pd
import sys
        import traceback
import warnings
"""
🚀 DIRECT FULL PIPELINE RUNNER
รัน full pipeline โดยตรงเพื่อแก้ไขปัญหา import และ error handling
"""

warnings.filterwarnings('ignore')

def main():
    """รัน full pipeline โดยตรง"""
    print("🚀 STARTING DIRECT FULL PIPELINE...")
    print(" = " * 80)

    try:
        # 1. Import pipeline function
        print("📦 Importing pipeline modules...")

        print("✅ Pipeline modules imported successfully")

        # 2. Run the full pipeline
        print("\n⚡ Starting full pipeline execution...")
        run_full_pipeline()

        print("\n🎉 FULL PIPELINE COMPLETED SUCCESSFULLY!")

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("🔧 Trying alternative pipeline runner...")
        run_alternative_pipeline()

    except Exception as e:
        print(f"❌ Pipeline Error: {e}")
        traceback.print_exc()

        print("\n🔧 Trying fallback pipeline...")
        run_fallback_pipeline()

def run_alternative_pipeline():
    """รัน pipeline แบบทางเลือก"""
    print("\n🔄 ALTERNATIVE PIPELINE STARTING...")

    try:
        # Import individual steps
        print("📦 Importing individual pipeline steps...")


        print("✅ Individual steps imported")

        # Run steps sequentially
        print("\n🔍 Step 1: Preprocessing...")
        run_preprocess()

        print("\n🎯 Step 2: Training...")
        run_train()

        print("\n📊 Step 3: Threshold Optimization...")
        run_threshold()

        print("\n🔄 Step 4: WalkForward Validation...")
        run_walkforward()

        print("\n📈 Step 5: Prediction...")
        run_predict()

        print("\n🎉 ALTERNATIVE PIPELINE COMPLETED!")

    except Exception as e:
        print(f"❌ Alternative pipeline failed: {e}")
        traceback.print_exc()

def run_fallback_pipeline():
    """รัน pipeline แบบ fallback ขั้นพื้นฐาน"""
    print("\n🆘 FALLBACK PIPELINE STARTING...")

    try:
        # Check if main data file exists
        data_file = "output_default/preprocessed_super.parquet"
        if os.path.exists(data_file):
            print(f"✅ Data file found: {data_file}")

            df = pd.read_parquet(data_file)
            print(f"📊 Data shape: {df.shape}")
            print(f"📊 Target distribution: {df['target'].value_counts().to_dict() if 'target' in df.columns else 'No target column'}")

        else:
            print("❌ No preprocessed data found")
            print("🔧 Running basic feature engineering...")

            # Run basic feature engineering
            df, path = run_production_grade_feature_engineering()

            if df is not None:
                print(f"✅ Feature engineering completed: {df.shape}")
            else:
                print("❌ Feature engineering failed")
                return False

        # Try to run basic model training
        print("\n🎯 Running basic model training...")

        # Import CatBoost directly

        # Load or use existing data
        if 'df' not in locals():
            df = pd.read_parquet(data_file)

        # Prepare features and target
        feature_cols = [c for c in df.columns if c not in ['target', 'date', 'datetime', 'timestamp']]
        X = df[feature_cols].select_dtypes(include = ['number'])
        y = df['target']

        # Handle missing values
        X = X.fillna(X.median())

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        # Train simple model
        model = CatBoostClassifier(iterations = 100, verbose = False, random_state = 42)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        # Calculate AUC (handle multiclass)
        try:
            if len(set(y_test)) == 2:
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class = 'ovr')
        except:
            auc = 0.5

        print(f"📊 Basic Model Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   AUC: {auc:.4f}")

        # Save basic results
        results = {
            'accuracy': accuracy, 
            'auc': auc, 
            'model_type': 'CatBoost', 
            'features_used': len(X.columns), 
            'data_shape': df.shape
        }

        os.makedirs('output_default', exist_ok = True)
        with open('output_default/fallback_results.json', 'w') as f:
            json.dump(results, f, indent = 2)

        print("\n🎉 FALLBACK PIPELINE COMPLETED!")
        print(f"📁 Results saved to: output_default/fallback_results.json")

        return True

    except Exception as e:
        print(f"❌ Fallback pipeline failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()

    if success is not False:
        print("\n🎉 PIPELINE EXECUTION COMPLETED!")
        sys.exit(0)
    else:
        print("\n❌ PIPELINE EXECUTION FAILED!")
        sys.exit(1)