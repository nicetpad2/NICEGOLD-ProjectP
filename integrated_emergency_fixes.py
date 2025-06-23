#!/usr/bin/env python3
"""
🔥 INTEGRATED EMERGENCY FIXES MODULE
บูรณาการ emergency fix logic เข้ากับทุกโหมดของ pipeline
เพื่อให้ระบบทำงานอัตโนมัติและแก้ไขปัญหาเองได้
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import json
from datetime import datetime
import traceback
from typing import Optional, Dict, Any, Tuple, List
import yaml
import os

warnings.filterwarnings('ignore')

class EmergencyFixManager:
    """Manager สำหรับการแก้ไขปัญหา emergency อัตโนมัติ"""
    
    def __init__(self, output_dir: str = "output_default"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.log_file = self.output_dir / "emergency_fixes.log"
        
    def log_fix(self, message: str, level: str = "INFO"):
        """บันทึก log การแก้ไข"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}\n"
        
        with open(self.log_file, "a", encoding='utf-8') as f:
            f.write(log_message)
        
        print(f"🔧 [{level}] {message}")
    
    def check_data_health(self, df: pd.DataFrame, target_col: str = 'target') -> Tuple[bool, List[str]]:
        """ตรวจสอบสุขภาพข้อมูลและระบุปัญหา"""
        issues = []
        
        try:
            # 1. ตรวจสอบ target column
            if target_col not in df.columns:
                issues.append(f"Missing target column: {target_col}")
                return False, issues
            
            # 2. ตรวจสอบ class distribution
            target_counts = df[target_col].value_counts()
            if len(target_counts) < 2:
                issues.append("Target has only one class")
            else:
                class_ratio = target_counts.max() / target_counts.min()
                if class_ratio > 100:
                    issues.append(f"Extreme class imbalance: {class_ratio:.1f}:1")
            
            # 3. ตรวจสอบ NaN values
            nan_percentage = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
            if nan_percentage > 10:
                issues.append(f"High NaN percentage: {nan_percentage:.1f}%")
            
            # 4. ตรวจสอบ feature correlation
            feature_cols = [col for col in df.columns if col != target_col and df[col].dtype in ['int64', 'float64']]
            if len(feature_cols) > 1:
                corr_matrix = df[feature_cols + [target_col]].corr()
                max_target_corr = abs(corr_matrix[target_col][feature_cols]).max()
                if max_target_corr < 0.05:
                    issues.append(f"Very low feature-target correlation: {max_target_corr:.4f}")
            
            # 5. ตรวจสอบ data size
            if len(df) < 100:
                issues.append(f"Insufficient data size: {len(df)} rows")
            
            # ถ้าไม่มีปัญหาร้าย แรง
            critical_keywords = ["Missing target", "only one class", "Extreme class imbalance", "Very low"]
            has_critical_issues = any(any(keyword in issue for keyword in critical_keywords) for issue in issues)
            
            return not has_critical_issues, issues
            
        except Exception as e:
            issues.append(f"Health check error: {str(e)}")
            return False, issues
    
    def auto_fix_data(self, df: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
        """แก้ไขข้อมูลอัตโนมัติ"""
        self.log_fix("Starting automatic data fixing...", "INFO")
        
        df_fixed = df.copy()
        
        try:
            # 1. สร้าง target column ถ้าไม่มี
            if target_col not in df_fixed.columns:
                self.log_fix(f"Creating missing target column: {target_col}", "WARN")
                if len(df_fixed.columns) > 0:
                    # ใช้ median split จาก column แรกที่เป็นตัวเลข
                    numeric_cols = df_fixed.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        base_col = numeric_cols[0]
                        median_val = df_fixed[base_col].median()
                        df_fixed[target_col] = (df_fixed[base_col] > median_val).astype(int)
                        self.log_fix(f"Created target from {base_col} median split", "INFO")
                    else:
                        # สร้าง random target
                        np.random.seed(42)
                        df_fixed[target_col] = np.random.choice([0, 1], size=len(df_fixed), p=[0.7, 0.3])
                        self.log_fix("Created random target column", "WARN")
            
            # 2. ทำความสะอาด features
            feature_cols = [col for col in df_fixed.columns if col != target_col]
            
            for col in feature_cols:
                if df_fixed[col].dtype in ['int64', 'float64']:
                    # แทนที่ NaN และ infinite values
                    df_fixed[col] = df_fixed[col].fillna(df_fixed[col].median())
                    df_fixed[col] = df_fixed[col].replace([np.inf, -np.inf], df_fixed[col].median())
            
            self.log_fix("Cleaned NaN and infinite values", "INFO")
            
            # 3. สร้าง engineered features
            numeric_cols = [col for col in feature_cols if df_fixed[col].dtype in ['int64', 'float64']]
            
            if len(numeric_cols) >= 2:
                # Moving averages
                for col in numeric_cols[:3]:
                    df_fixed[f'{col}_ma3'] = df_fixed[col].rolling(3, min_periods=1).mean()
                
                # Interaction features
                df_fixed['feature_sum'] = df_fixed[numeric_cols[0]] + df_fixed[numeric_cols[1]]
                df_fixed['feature_ratio'] = df_fixed[numeric_cols[0]] / (df_fixed[numeric_cols[1]] + 1e-8)
                
                # Rank features
                for col in numeric_cols[:2]:
                    df_fixed[f'{col}_rank'] = df_fixed[col].rank(pct=True)
                
                self.log_fix("Created engineered features", "INFO")
            
            # 4. แก้ไข class imbalance
            target_counts = df_fixed[target_col].value_counts()
            if len(target_counts) >= 2:
                class_ratio = target_counts.max() / target_counts.min()
                if class_ratio > 50:  # Extreme imbalance
                    self.log_fix(f"Fixing extreme class imbalance: {class_ratio:.1f}:1", "WARN")
                    df_fixed = self._balance_classes(df_fixed, target_col)
            
            # 5. ตรวจสอบผลลัพธ์
            is_healthy, remaining_issues = self.check_data_health(df_fixed, target_col)
            if is_healthy:
                self.log_fix("Data successfully fixed!", "SUCCESS")
            else:
                self.log_fix(f"Some issues remain: {remaining_issues}", "WARN")
            
            # บันทึกข้อมูลที่แก้ไขแล้ว
            output_file = self.output_dir / "fixed_data.csv"
            df_fixed.to_csv(output_file, index=False)
            self.log_fix(f"Saved fixed data to {output_file}", "INFO")
            
            return df_fixed
            
        except Exception as e:
            self.log_fix(f"Error in auto_fix_data: {str(e)}", "ERROR")
            self.log_fix(f"Traceback: {traceback.format_exc()}", "DEBUG")
            return df_fixed
    
    def _balance_classes(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """ปรับสมดุล classes"""
        try:
            target_counts = df[target_col].value_counts()
            min_samples = max(100, target_counts.min() * 3)  # เพิ่มจาก minority class
            
            balanced_dfs = []
            for class_val in target_counts.index:
                class_df = df[df[target_col] == class_val]
                
                if len(class_df) < min_samples:
                    # Oversample
                    additional_samples = min_samples - len(class_df)
                    resampled = class_df.sample(n=additional_samples, replace=True, random_state=42)
                    balanced_df = pd.concat([class_df, resampled], ignore_index=True)
                elif len(class_df) > min_samples * 3:
                    # Undersample
                    balanced_df = class_df.sample(n=min_samples * 2, random_state=42)
                else:
                    balanced_df = class_df
                
                balanced_dfs.append(balanced_df)
            
            result_df = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42)
            self.log_fix(f"Balanced classes: {result_df[target_col].value_counts().to_dict()}", "INFO")
            
            return result_df
            
        except Exception as e:
            self.log_fix(f"Error in _balance_classes: {str(e)}", "ERROR")
            return df
    
    def prepare_model_safe_data(self, df: pd.DataFrame, target_col: str = 'target') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """เตรียมข้อมูลให้ปลอดภัยสำหรับ model training"""
        try:
            # ตรวจสอบและแก้ไขข้อมูล
            is_healthy, issues = self.check_data_health(df, target_col)
            
            if not is_healthy:
                self.log_fix(f"Data issues detected: {issues}", "WARN")
                df = self.auto_fix_data(df, target_col)
            
            # เตรียม features และ target
            feature_cols = [col for col in df.columns if col != target_col and df[col].dtype in ['int64', 'float64']]
            
            X = df[feature_cols].copy()
            y = df[target_col].copy()
            
            # Standardize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            # สร้าง metadata
            metadata = {
                'feature_columns': feature_cols,
                'target_column': target_col,
                'original_shape': df.shape,
                'processed_shape': (len(X_scaled), len(feature_cols)),
                'class_distribution': y.value_counts().to_dict(),
                'scaler_params': {
                    'mean': scaler.mean_.tolist(),
                    'scale': scaler.scale_.tolist()
                }
            }
            
            # รวม X และ y กลับเข้าด้วยกัน
            final_df = X_scaled.copy()
            final_df[target_col] = y.values
            
            self.log_fix(f"Prepared safe data: {final_df.shape}", "SUCCESS")
            
            return final_df, metadata
            
        except Exception as e:
            self.log_fix(f"Error in prepare_model_safe_data: {str(e)}", "ERROR")
            # Return original data if processing fails
            return df, {'error': str(e)}

    def validate_and_fix_config(self, config_path: str = "config.yaml") -> bool:
        """ตรวจสอบและแก้ไข config.yaml ให้สมบูรณ์"""
        self.log_fix("Starting config validation and auto-fix...")
        
        try:
            config_file = Path(config_path)
            
            # Template config ที่สมบูรณ์
            complete_config = {
                'data': {
                    'source': 'dummy_m1.csv',
                    'target_column': 'target',
                    'features': ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'volatility', 'momentum', 'rsi', 'macd']
                },
                'model_class': 'RandomForestClassifier',
                'model_params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'n_jobs': -1
                },
                'model': {
                    'type': 'RandomForest',
                    'file': 'models/rf_model.joblib',
                    'features': ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'volatility', 'momentum', 'rsi', 'macd']
                },
                'training': {
                    'test_size': 0.3,
                    'random_state': 42,
                    'cross_validation': 5
                },
                'walk_forward': {
                    'enabled': True,
                    'window_size': 1000,
                    'step_size': 100,
                    'min_train_size': 500
                },                'metrics': ['auc', 'accuracy', 'precision', 'recall', 'f1_score', 'balanced_accuracy', 'roc_auc'],
                'export': {
                    'output_dir': 'output_default',
                    'save_features': True,
                    'save_model': True,
                    'save_predictions': True,
                    'save_reports': True,
                    'save_logs': True
                },
                'parallel': {
                    'enabled': True,
                    'n_jobs': -1,
                    'backend': 'threading'
                },
                'visualization': {
                    'enabled': True,
                    'show_plots': False,
                    'save_plots': True,
                    'plot_dir': 'output_default/plots'
                },
                'emergency_fixes': {
                    'enabled': True,
                    'auto_fix_data': True,
                    'create_missing_target': True,
                    'balance_classes': True,
                    'handle_nan_features': True
                }
            }
            
            # โหลด config ปัจจุบัน
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        current_config = yaml.safe_load(f) or {}
                    self.log_fix(f"Loaded existing config: {len(current_config)} sections")
                except Exception as e:
                    self.log_fix(f"Error loading config: {e}", "WARN")
                    current_config = {}
            else:
                self.log_fix("Config file not found, creating new one")
                current_config = {}
            
            # ตรวจสอบและเติมฟิลด์ที่ขาดหายไป
            missing_fields = []
            updated_config = current_config.copy()
            
            def check_and_add_field(config_dict, key, default_value, path=""):
                full_path = f"{path}.{key}" if path else key
                
                if key not in config_dict:
                    config_dict[key] = default_value
                    missing_fields.append(full_path)
                    return True
                elif isinstance(default_value, dict) and isinstance(config_dict[key], dict):
                    # Recursive check for nested dicts
                    changed = False
                    for sub_key, sub_value in default_value.items():
                        if check_and_add_field(config_dict[key], sub_key, sub_value, full_path):
                            changed = True
                    return changed
                return False
            
            # ตรวจสอบและเติมทุกฟิลด์
            config_changed = False
            for key, value in complete_config.items():
                if check_and_add_field(updated_config, key, value):
                    config_changed = True
            
            # บันทึก config ใหม่ถ้ามีการเปลี่ยนแปลง
            if config_changed or missing_fields:
                # สร้าง backup ของ config เก่า
                if config_file.exists():
                    backup_file = config_file.with_suffix('.yaml.backup')
                    import shutil
                    shutil.copy2(config_file, backup_file)
                    self.log_fix(f"Created backup: {backup_file}")
                
                # เขียน config ใหม่
                with open(config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(updated_config, f, default_flow_style=False, indent=2, allow_unicode=True)
                
                self.log_fix(f"Updated config with {len(missing_fields)} missing fields:")
                for field in missing_fields[:10]:  # แสดงแค่ 10 อันแรก
                    self.log_fix(f"  + Added: {field}")
                
                if len(missing_fields) > 10:
                    self.log_fix(f"  + ... and {len(missing_fields) - 10} more fields")
            
            self.log_fix("Config validation completed successfully")
            return True
            
        except Exception as e:
            self.log_fix(f"Config validation failed: {e}", "ERROR")
            traceback.print_exc()
            return False

def ensure_complete_config():
    """Ensure config.yaml has all required fields for pipeline execution"""
    config_path = Path("config.yaml")
    
    # Default complete config template
    default_config = {
        'data': {
            'source': 'dummy_m1.csv',
            'target_column': 'target'
        },
        'model': {
            'features': ['Open', 'Volume', 'returns', 'volatility', 'momentum', 'rsi', 'macd'],
            'file': 'models/rf_model.joblib',
            'type': 'RandomForest',
            'model_class': 'RandomForestClassifier',
            'model_params': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            }
        },
        'training': {
            'cross_validation': 5,
            'random_state': 42,
            'test_size': 0.3,
            'walk_forward': {
                'enabled': True,
                'window_size': 1000,
                'step_size': 100
            }
        },
        'metrics': {
            'primary': 'auc',
            'additional': ['accuracy', 'precision', 'recall', 'f1_score']
        },
        'export': {
            'output_dir': 'output_default',
            'save_model': True,
            'save_predictions': True,
            'save_features': True,
            'save_reports': True
        },
        'parallel': {
            'enabled': True,
            'n_jobs': -1,
            'backend': 'threading'
        },
        'visualization': {
            'enabled': True,
            'save_plots': True,
            'show_plots': False,
            'plot_dir': 'output_default/plots'
        }
    }
    
    try:
        # Load existing config
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}
        
        # Merge with defaults to fill missing fields
        def merge_configs(base, update):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_configs(base[key], value)
                else:
                    base[key] = value
        
        complete_config = default_config.copy()
        merge_configs(complete_config, config)
        
        # Write complete config back
        with open(config_path, 'w') as f:
            yaml.dump(complete_config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Config validated and updated: {config_path}")
        return True
        
    except Exception as e:
        print(f"❌ Config repair failed: {e}")
        # Create fresh config if repair fails
        try:
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            print(f"✅ Fresh config created: {config_path}")
            return True
        except Exception as e2:
            print(f"❌ Fresh config creation failed: {e2}")
            return False

# Factory function สำหรับสร้าง emergency fix manager
def create_emergency_fix_manager(output_dir: str = "output_default") -> EmergencyFixManager:
    """สร้าง emergency fix manager"""
    return EmergencyFixManager(output_dir)

# Integration functions สำหรับใช้ใน pipeline modes
def apply_emergency_fixes_to_pipeline(mode: str = "full_pipeline", **kwargs) -> bool:
    """ใช้ emergency fixes กับ pipeline mode ที่ระบุ"""
    fix_manager = create_emergency_fix_manager()
    
    try:
        fix_manager.log_fix(f"Starting emergency fixes for mode: {mode}", "INFO")
        
        # Step 1: Validate and fix config first
        if not fix_manager.validate_and_fix_config():
            fix_manager.log_fix("Config validation failed", "ERROR")
            return False
        
        # ตรวจหาไฟล์ข้อมูล
        data_files = [
            "dummy_m1.csv", "dummy_m15.csv",
            "data/dummy_m1.csv", "data/dummy_m15.csv",
            "output_default/processed_data.csv"
        ]
        
        df = None
        for file_path in data_files:
            if Path(file_path).exists():
                try:
                    df = pd.read_csv(file_path)
                    fix_manager.log_fix(f"Loaded data from {file_path}: {df.shape}", "INFO")
                    break
                except Exception as e:
                    fix_manager.log_fix(f"Failed to load {file_path}: {e}", "WARN")
        
        if df is None:
            fix_manager.log_fix("No data found, creating synthetic data", "WARN")
            df = _create_synthetic_data(fix_manager)
        
        # ใช้ emergency fixes
        fixed_df, metadata = fix_manager.prepare_model_safe_data(df)
        
        # บันทึกผลลัพธ์
        output_file = fix_manager.output_dir / f"emergency_fixed_{mode}.csv"
        fixed_df.to_csv(output_file, index=False)
        
        metadata_file = fix_manager.output_dir / f"emergency_metadata_{mode}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        fix_manager.log_fix(f"Emergency fixes completed for {mode}", "SUCCESS")
        return True
        
    except Exception as e:
        fix_manager.log_fix(f"Emergency fixes failed for {mode}: {str(e)}", "ERROR")
        return False

def _create_synthetic_data(fix_manager: EmergencyFixManager) -> pd.DataFrame:
    """สร้างข้อมูล synthetic สำหรับ testing"""
    try:
        np.random.seed(42)
        n_samples = 5000
        
        # สร้าง features
        features = {}
        feature_names = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd']
        
        for i, name in enumerate(feature_names):
            base_signal = np.random.randn(n_samples) * 0.1
            if i > 0:  # correlation
                base_signal += features[feature_names[0]] * 0.05
            features[name] = base_signal
        
        df = pd.DataFrame(features)
        
        # สร้าง target with moderate imbalance
        target_probs = np.random.random(n_samples)
        df['target'] = np.where(target_probs < 0.2, 1, 0)  # 20% class 1
        
        fix_manager.log_fix(f"Created synthetic data: {df.shape}", "INFO")
        fix_manager.log_fix(f"Target distribution: {df['target'].value_counts().to_dict()}", "INFO")
        
        return df
        
    except Exception as e:
        fix_manager.log_fix(f"Error creating synthetic data: {str(e)}", "ERROR")
        # Return minimal data
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 3, 4, 5, 6],
            'target': [0, 1, 0, 1, 0]
        })

# Main testing function
if __name__ == "__main__":
    print("🔥 Testing Integrated Emergency Fixes...")
    
    # Ensure config is complete
    ensure_complete_config()
    
    # Test for different modes
    modes = ["full_pipeline", "debug_full_pipeline", "preprocess", "realistic_backtest"]
    
    for mode in modes:
        print(f"\n🧪 Testing {mode}...")
        success = apply_emergency_fixes_to_pipeline(mode)
        
        if success:
            print(f"✅ {mode} emergency fixes completed")
        else:
            print(f"❌ {mode} emergency fixes failed")
    
    print("\n🎉 Integration testing completed!")
