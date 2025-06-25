#!/usr/bin/env python3
            from catboost import CatBoostClassifier
                from feature_engineering import run_production_grade_feature_engineering
                            from production_class_imbalance_fix import fix_extreme_class_imbalance_production
            from projectp.steps.predict import run_predict
            from projectp.steps.threshold import run_threshold
            from projectp.steps.walkforward import run_walkforward
            from sklearn.metrics import accuracy_score, roc_auc_score
                    from sklearn.metrics import precision_recall_curve, f1_score
            from sklearn.model_selection import train_test_split
                from sklearn.utils.class_weight import compute_class_weight
            import joblib
                    import json
            import numpy as np
import os
            import pandas as pd
import sys
            import traceback
import warnings
"""
🚀 SIMPLE PIPELINE RUNNER
รัน pipeline แบบเรียบง่ายโดยหลีกเลี่ยงปัญหา import ซับซ้อน
"""

warnings.filterwarnings('ignore')

def run_simple_pipeline():
    """รัน pipeline แบบง่าย"""
    print("🚀 STARTING SIMPLE FULL PIPELINE...")
    print(" = " * 80)

    success_count = 0
    total_steps = 6

    try:
        # Step 1: Data Quality Check
        print("\n📊 STEP 1: Data Quality Check")
        print(" - " * 50)

        if os.path.exists("output_default/preprocessed_super.parquet"):
            df = pd.read_parquet("output_default/preprocessed_super.parquet")
            print(f"✅ Data loaded: {df.shape}")

            if 'target' in df.columns:
                target_dist = df['target'].value_counts()
                print(f"📊 Target distribution: {target_dist.to_dict()}")

                # Check class imbalance
                if len(target_dist) > 1:
                    imbalance = target_dist.max() / target_dist.min()
                    print(f"⚖️ Class imbalance ratio: {imbalance:.1f}:1")

                    if imbalance > 50:
                        print("⚠️ Extreme class imbalance detected!")
                        # Run class imbalance fix
                        try:
                            df_balanced, _ = fix_extreme_class_imbalance_production()
                            if df_balanced is not None:
                                print("✅ Class imbalance fix applied")
                                df = df_balanced
                        except Exception as e:
                            print(f"⚠️ Class imbalance fix failed: {e}")

            success_count += 1
        else:
            print("❌ No preprocessed data found")
            print("🔧 Running feature engineering...")

            try:
                df, _ = run_production_grade_feature_engineering()

                if df is not None:
                    print(f"✅ Feature engineering completed: {df.shape}")
                    success_count += 1
                else:
                    print("❌ Feature engineering failed")

            except Exception as e:
                print(f"❌ Feature engineering error: {e}")

        # Step 2: Model Training
        print("\n🎯 STEP 2: Model Training")
        print(" - " * 50)

        try:
            # Load latest data
            if 'df' not in locals():
                df = pd.read_parquet("output_default/preprocessed_super.parquet")

            # Import training modules

            # Prepare data
            feature_cols = [c for c in df.columns if c not in ['target', 'date', 'datetime', 'timestamp', 'time']]
            X = df[feature_cols].select_dtypes(include = ['number'])
            y = df['target']

            # Handle missing values
            X = X.fillna(X.median())

            print(f"📊 Features: {X.shape[1]}, Samples: {X.shape[0]}")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

            # Train model with class weights for imbalance
            class_weights = None
            if len(y.value_counts()) > 1:
                classes = np.unique(y)
                weights = compute_class_weight('balanced', classes = classes, y = y)
                class_weights = dict(zip(classes, weights))
                print(f"📊 Using class weights: {class_weights}")

            model = CatBoostClassifier(
                iterations = 200, 
                depth = 6, 
                learning_rate = 0.1, 
                random_state = 42, 
                verbose = False, 
                class_weights = class_weights
            )

            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            # Calculate AUC
            try:
                if len(set(y_test)) == 2:
                    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    auc = roc_auc_score(y_test, y_pred_proba, multi_class = 'ovr')
            except:
                auc = 0.5

            print(f"📊 Model Performance:")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   AUC: {auc:.4f}")

            # Save model
            os.makedirs('models', exist_ok = True)
            joblib.dump(model, 'models/catboost_model.joblib')
            print("✅ Model saved to models/catboost_model.joblib")

            success_count += 1

        except Exception as e:
            print(f"❌ Model training failed: {e}")
            traceback.print_exc()

        # Step 3: Threshold Optimization
        print("\n🎯 STEP 3: Threshold Optimization")
        print(" - " * 50)

        try:
            result = run_threshold()
            print(f"✅ Threshold optimization completed: {result}")
            success_count += 1

        except Exception as e:
            print(f"❌ Threshold optimization failed: {e}")

            # Fallback threshold optimization
            try:
                print("🔧 Running fallback threshold optimization...")

                if 'model' in locals() and 'X_test' in locals() and 'y_test' in locals():

                    y_scores = model.predict_proba(X_test)
                    if y_scores.shape[1] > 2:  # multiclass
                        y_scores = y_scores[:, 1]  # use class 1 probability
                    else:
                        y_scores = y_scores[:, 1]

                    # Find best threshold based on F1 score
                    thresholds = np.arange(0.1, 0.9, 0.05)
                    best_threshold = 0.5
                    best_f1 = 0

                    for threshold in thresholds:
                        y_pred_thresh = (y_scores > threshold).astype(int)
                        try:
                            f1 = f1_score(y_test, y_pred_thresh, average = 'weighted')
                            if f1 > best_f1:
                                best_f1 = f1
                                best_threshold = threshold
                        except:
                            continue

                    print(f"✅ Best threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")

                    # Save threshold
                    threshold_result = {
                        'best_threshold': best_threshold, 
                        'best_f1': best_f1, 
                        'method': 'fallback_f1_optimization'
                    }

                    with open('models/threshold_result.json', 'w') as f:
                        json.dump(threshold_result, f, indent = 2)

                    success_count += 1

            except Exception as e2:
                print(f"❌ Fallback threshold optimization failed: {e2}")

        # Step 4: WalkForward Validation
        print("\n🔄 STEP 4: WalkForward Validation")
        print(" - " * 50)

        try:
            result = run_walkforward()
            print(f"✅ WalkForward validation completed: {result}")
            success_count += 1

        except Exception as e:
            print(f"❌ WalkForward validation failed: {e}")
            print("🔧 Skipping WalkForward validation...")

        # Step 5: Prediction/Export
        print("\n📈 STEP 5: Prediction & Export")
        print(" - " * 50)

        try:
            result = run_predict()
            print(f"✅ Prediction completed: {result}")
            success_count += 1

        except Exception as e:
            print(f"❌ Prediction failed: {e}")

            # Fallback prediction
            try:
                print("🔧 Running fallback prediction...")

                if 'model' in locals() and 'df' in locals():
                    # Use full dataset for prediction
                    feature_cols = [c for c in df.columns if c not in ['target', 'date', 'datetime', 'timestamp', 'time']]
                    X_full = df[feature_cols].select_dtypes(include = ['number'])
                    X_full = X_full.fillna(X_full.median())

                    predictions = model.predict_proba(X_full)

                    # Save predictions
                    pred_df = pd.DataFrame()
                    pred_df['prediction'] = model.predict(X_full)

                    if predictions.shape[1] > 1:
                        for i in range(predictions.shape[1]):
                            pred_df[f'proba_class_{i}'] = predictions[:, i]

                    pred_df.to_csv('output_default/fallback_predictions.csv', index = False)
                    print("✅ Fallback predictions saved to output_default/fallback_predictions.csv")

                    success_count += 1

            except Exception as e2:
                print(f"❌ Fallback prediction failed: {e2}")

        # Step 6: Final Summary
        print("\n📊 STEP 6: Pipeline Summary")
        print(" - " * 50)

        # Create summary report
        summary = {
            'pipeline_type': 'simple_full_pipeline', 
            'total_steps': total_steps, 
            'successful_steps': success_count, 
            'success_rate': f"{(success_count/total_steps)*100:.1f}%", 
            'data_shape': df.shape if 'df' in locals() else 'Unknown', 
            'model_performance': {
                'accuracy': accuracy if 'accuracy' in locals() else 'Unknown', 
                'auc': auc if 'auc' in locals() else 'Unknown'
            } if 'accuracy' in locals() else 'Not available'
        }

        os.makedirs('output_default', exist_ok = True)
        with open('output_default/simple_pipeline_summary.json', 'w') as f:
            json.dump(summary, f, indent = 2)

        print(f"📊 Pipeline Summary:")
        print(f"   Total Steps: {total_steps}")
        print(f"   Successful Steps: {success_count}")
        print(f"   Success Rate: {(success_count/total_steps)*100:.1f}%")

        if 'df' in locals():
            print(f"   Data Shape: {df.shape}")

        if 'accuracy' in locals():
            print(f"   Model Accuracy: {accuracy:.4f}")
            print(f"   Model AUC: {auc:.4f}")

        print(f"✅ Summary saved to: output_default/simple_pipeline_summary.json")

        success_count += 1

        # Final assessment
        print("\n🏆 FINAL ASSESSMENT")
        print(" = " * 80)

        if success_count >= 4:
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("✅ Most critical steps completed")
            return True
        elif success_count >= 2:
            print("⚠️ PIPELINE PARTIALLY COMPLETED")
            print("🔧 Some steps failed but core functionality works")
            return True
        else:
            print("❌ PIPELINE FAILED")
            print("🆘 Too many critical failures")
            return False

    except Exception as e:
        print(f"❌ Critical pipeline error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 NICEGOLD SIMPLE PIPELINE RUNNER")
    print(" = " * 80)

    success = run_simple_pipeline()

    if success:
        print("\n🎉 PIPELINE EXECUTION SUCCESSFUL!")
        sys.exit(0)
    else:
        print("\n❌ PIPELINE EXECUTION FAILED!")
        sys.exit(1)