#!/usr/bin/env python3
"""
Production-Ready NICEGOLD Pipeline Runner
=========================================
Windows-compatible, Unicode-safe, Error-resistant
"""

import os
import sys
import traceback
import warnings
from pathlib import Path

# Configure encoding for Windows compatibility
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

warnings.filterwarnings("ignore")

def safe_print(msg, level="INFO"):
    """Safe printing with fallback for Unicode issues"""
    try:
        print(f"[{level}] {msg}")
    except UnicodeEncodeError:
        # Fallback to ASCII-safe printing
        safe_msg = msg.encode('ascii', errors='ignore').decode('ascii')
        print(f"[{level}] {safe_msg}")

def create_directory_safely(path):
    """Create directory with error handling"""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        safe_print(f"Failed to create directory {path}: {e}", "ERROR")
        return False

def run_production_pipeline():
    """Production-ready pipeline execution"""
    safe_print("NICEGOLD PRODUCTION PIPELINE STARTING", "INFO")
    safe_print("=" * 80, "INFO")
    
    try:
        # Ensure output directory exists
        if not create_directory_safely("output_default"):
            safe_print("Cannot create output directory!", "ERROR")
            return False
        
        # Step 1: Load and validate data
        safe_print("STEP 1: Data Loading and Validation", "INFO")
        safe_print("-" * 50, "INFO")
        
        try:
            import pandas as pd
            import numpy as np
            
            # Try to load main data file
            data_files = ["XAUUSD_M1.parquet", "XAUUSD_M1.csv", "XAUUSD_M15.parquet", "XAUUSD_M15.csv"]
            df = None
            
            for file in data_files:
                if os.path.exists(file):
                    try:
                        if file.endswith('.parquet'):
                            df = pd.read_parquet(file)
                        else:
                            df = pd.read_csv(file)
                        safe_print(f"Loaded data from {file}: {df.shape}", "SUCCESS")
                        break
                    except Exception as e:
                        safe_print(f"Failed to load {file}: {e}", "WARNING")
                        continue
            
            if df is None:
                safe_print("No data files found! Cannot proceed.", "ERROR")
                return False
                
            # Validate required columns
            required_cols = ['Open', 'High', 'Low', 'Close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                safe_print(f"Missing required columns: {missing_cols}", "ERROR")
                return False
                
            safe_print(f"Data validation successful: {df.shape}", "SUCCESS")
            
        except Exception as e:
            safe_print(f"Data loading failed: {e}", "ERROR")
            return False
        
        # Step 2: Feature Engineering        safe_print("STEP 2: Feature Engineering", "INFO")
        safe_print("-" * 50, "INFO")
        
        try:
            # Import and run production feature engineering
            from production_class_imbalance_fix_clean import run_production_class_imbalance_fix
            
            safe_print("Running production-grade feature engineering...", "INFO")
            df_processed = run_production_class_imbalance_fix(df)
            
            if df_processed is not None:
                safe_print(f"Feature engineering completed: {df_processed.shape}", "SUCCESS")
                
                # Save processed data
                output_path = "output_default/production_processed.parquet"
                df_processed.to_parquet(output_path)
                safe_print(f"Processed data saved to: {output_path}", "SUCCESS")
            else:
                safe_print("Feature engineering failed", "ERROR")
                return False
                
        except Exception as e:
            safe_print(f"Feature engineering failed: {e}", "ERROR")
            safe_print(f"Traceback: {traceback.format_exc()}", "DEBUG")
            return False
        
        # Step 3: Model Training
        safe_print("STEP 3: Model Training", "INFO")
        safe_print("-" * 50, "INFO")
        
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import roc_auc_score, accuracy_score
            
            # Prepare features and target
            feature_cols = [col for col in df_processed.columns if col not in ['target', 'Date', 'timestamp']]
            X = df_processed[feature_cols].fillna(0)
            y = df_processed['target'].fillna(0)
            
            # Check class distribution
            class_counts = y.value_counts()
            safe_print(f"Class distribution: {class_counts.to_dict()}", "INFO")
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            safe_print("Training model...", "INFO")
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            
            auc = roc_auc_score(y_test, y_pred_proba)
            acc = accuracy_score(y_test, y_pred)
            
            safe_print(f"Model Performance - AUC: {auc:.4f}, Accuracy: {acc:.4f}", "SUCCESS")
            
            # Save model
            import joblib
            model_path = "output_default/production_model.pkl"
            joblib.dump(model, model_path)
            safe_print(f"Model saved to: {model_path}", "SUCCESS")
            
        except Exception as e:
            safe_print(f"Model training failed: {e}", "ERROR")
            safe_print(f"Traceback: {traceback.format_exc()}", "DEBUG")
            return False
        
        # Step 4: Walk-Forward Validation
        safe_print("STEP 4: Walk-Forward Validation", "INFO")
        safe_print("-" * 50, "INFO")
        
        try:
            # Simple walk-forward validation
            n_splits = 5
            split_size = len(df_processed) // n_splits
            
            auc_scores = []
            acc_scores = []
            
            for i in range(n_splits - 1):
                train_start = 0
                train_end = (i + 1) * split_size
                test_start = train_end
                test_end = test_start + split_size
                
                X_wf_train = X.iloc[train_start:train_end]
                y_wf_train = y.iloc[train_start:train_end]
                X_wf_test = X.iloc[test_start:test_end]
                y_wf_test = y.iloc[test_start:test_end]
                
                # Train fold model
                fold_model = RandomForestClassifier(
                    n_estimators=50, max_depth=8, random_state=42
                )
                fold_model.fit(X_wf_train, y_wf_train)
                
                # Predict
                y_wf_pred_proba = fold_model.predict_proba(X_wf_test)[:, 1]
                y_wf_pred = fold_model.predict(X_wf_test)
                
                fold_auc = roc_auc_score(y_wf_test, y_wf_pred_proba)
                fold_acc = accuracy_score(y_wf_test, y_wf_pred)
                
                auc_scores.append(fold_auc)
                acc_scores.append(fold_acc)
                
                safe_print(f"Fold {i+1}: AUC={fold_auc:.4f}, ACC={fold_acc:.4f}", "INFO")
            
            mean_auc = np.mean(auc_scores)
            mean_acc = np.mean(acc_scores)
            
            safe_print(f"Walk-Forward Results - Mean AUC: {mean_auc:.4f}, Mean ACC: {mean_acc:.4f}", "SUCCESS")
            
            # Save walk-forward results
            wf_results = {
                'fold_auc_scores': auc_scores,
                'fold_acc_scores': acc_scores,
                'mean_auc': mean_auc,
                'mean_acc': mean_acc
            }
            
            import json
            results_path = "output_default/walkforward_results.json"
            with open(results_path, 'w') as f:
                json.dump(wf_results, f, indent=2)
            safe_print(f"Walk-forward results saved to: {results_path}", "SUCCESS")
            
        except Exception as e:
            safe_print(f"Walk-forward validation failed: {e}", "ERROR")
            safe_print(f"Traceback: {traceback.format_exc()}", "DEBUG")
            return False
        
        # Step 5: Generate Predictions
        safe_print("STEP 5: Generate Final Predictions", "INFO")
        safe_print("-" * 50, "INFO")
        
        try:
            # Generate predictions for the full dataset
            predictions = model.predict_proba(X)[:, 1]
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'prediction_proba': predictions,
                'prediction_binary': (predictions > 0.5).astype(int),
                'actual_target': y
            })
            
            # Save predictions
            pred_path = "output_default/production_predictions.csv"
            results_df.to_csv(pred_path, index=False)
            safe_print(f"Predictions saved to: {pred_path}", "SUCCESS")
            
        except Exception as e:
            safe_print(f"Prediction generation failed: {e}", "ERROR")
            return False
        
        # Final Summary
        safe_print("=" * 80, "INFO")
        safe_print("PRODUCTION PIPELINE COMPLETED SUCCESSFULLY", "SUCCESS")
        safe_print("=" * 80, "INFO")
        safe_print(f"Final Model AUC: {auc:.4f}", "INFO")
        safe_print(f"Walk-Forward Mean AUC: {mean_auc:.4f}", "INFO")
        safe_print(f"Data Shape: {df_processed.shape}", "INFO")
        safe_print(f"Feature Count: {len(feature_cols)}", "INFO")
        
        # Create success flag file
        with open("output_default/pipeline_success.txt", 'w') as f:
            f.write(f"Pipeline completed successfully at {pd.Timestamp.now()}\n")
            f.write(f"Final AUC: {auc:.4f}\n")
            f.write(f"Walk-Forward AUC: {mean_auc:.4f}\n")
        
        return True
        
    except Exception as e:
        safe_print(f"CRITICAL PIPELINE FAILURE: {e}", "ERROR")
        safe_print(f"Traceback: {traceback.format_exc()}", "DEBUG")
        return False

def main():
    """Main entry point"""
    try:
        success = run_production_pipeline()
        if success:
            safe_print("PIPELINE EXECUTION: SUCCESS", "SUCCESS")
            return 0
        else:
            safe_print("PIPELINE EXECUTION: FAILED", "ERROR")
            return 1
    except Exception as e:
        safe_print(f"MAIN EXECUTION FAILED: {e}", "ERROR")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
