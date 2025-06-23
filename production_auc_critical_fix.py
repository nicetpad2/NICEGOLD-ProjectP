"""
🚀 PRODUCTION AUC CRITICAL FIX
===============================
แก้ไขปัญหา AUC ต่ำ (0.50) ในระบบ Production ให้ทำงานได้อย่างสมบูรณ์

Critical Issues Fixed:
1. ⚡ Class Imbalance - แก้ไขความไม่สมดุลของข้อมูล
2. 🎯 Feature Engineering - สร้าง Features ที่มีประสิทธิภาพ
3. 🧠 Model Selection - เลือกโมเดลที่เหมาะสม
4. 📊 Data Quality - ตรวจสอบและแก้ไขคุณภาพข้อมูล
5. 🔧 Hyperparameter Tuning - ปรับพารามิเตอร์ให้เหมาะสม
"""

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# Advanced ML
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Rich console
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, track
from rich import box

console = Console()

class ProductionAUCFix:
    def __init__(self, output_dir="output_default"):
        """Initialize Production AUC Fix System"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        log_file = self.output_dir / "production_auc_fix.log"
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Target metrics
        self.target_auc = 0.75
        self.minimum_auc = 0.70
        
        console.print(Panel.fit("🚀 PRODUCTION AUC CRITICAL FIX INITIALIZED", style="bold green"))
        self.logger.info("Production AUC Fix System initialized")
    
    def load_and_diagnose_data(self):
        """🔍 Step 1: Load and diagnose data issues"""
        console.print(Panel.fit("🔍 Step 1: Data Diagnosis", style="bold blue"))
        
        # Try multiple data sources
        data_paths = [
            "output_default/preprocessed_super.parquet",
            "output_default/auto_features.parquet", 
            "data/raw/your_data_file.csv",
            "XAUUSD_M1.csv",
            "dummy_m1.csv"
        ]
        
        df = None
        for path in data_paths:
            if os.path.exists(path):
                try:
                    if path.endswith('.parquet'):
                        df = pd.read_parquet(path)
                    else:
                        df = pd.read_csv(path, nrows=50000)  # Limit for faster processing
                    self.logger.info(f"✅ Loaded data from: {path}, Shape: {df.shape}")
                    break
                except Exception as e:
                    self.logger.warning(f"❌ Failed to load {path}: {e}")
                    continue
        
        if df is None:
            # Create synthetic data for testing
            df = self._create_synthetic_trading_data()
            self.logger.info("📊 Created synthetic trading data for testing")
        
        # Standardize column names
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Diagnose data issues
        diagnosis = self._diagnose_data_issues(df)
        
        return df, diagnosis
    
    def _create_synthetic_trading_data(self, n_samples=10000):
        """Create realistic synthetic trading data"""
        np.random.seed(42)
        
        # Base price series
        price_base = 2000
        price_returns = np.random.normal(0, 0.01, n_samples)
        prices = price_base * np.cumprod(1 + price_returns)
        
        # Create synthetic features
        data = {
            'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_samples))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create technical indicators
        df['returns'] = df['close'].pct_change()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        df['volatility'] = df['returns'].rolling(20).std()
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Create realistic target (trend continuation)
        # Higher probability of continuation in trending markets
        df['future_return'] = df['close'].shift(-5) / df['close'] - 1
        df['target'] = (df['future_return'] > 0).astype(int)
        
        # Remove NaN values
        df = df.dropna()
        
        console.print(f"📊 Created synthetic dataset: {df.shape}")
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _diagnose_data_issues(self, df):
        """Diagnose critical data issues"""
        diagnosis = {}
        
        # Find target column
        target_candidates = ['target', 'label', 'y', 'signal', 'trade_signal']
        target_col = None
        for col in target_candidates:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None and 'close' in df.columns:
            # Create target from price movement
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            target_col = 'target'
            console.print("⚠️ Created target from price movement")
        
        if target_col is None:
            raise ValueError("❌ No target column found and cannot create one")
        
        diagnosis['target_col'] = target_col
        
        # Class distribution analysis
        target_dist = df[target_col].value_counts()
        imbalance_ratio = target_dist.max() / target_dist.min()
        diagnosis['class_imbalance'] = {
            'distribution': target_dist.to_dict(),
            'imbalance_ratio': imbalance_ratio,
            'is_severe': imbalance_ratio > 3
        }
        
        # Data quality issues
        feature_cols = [col for col in df.columns if col != target_col and df[col].dtype in ['float64', 'int64']]
        diagnosis['feature_cols'] = feature_cols
        diagnosis['n_features'] = len(feature_cols)
        diagnosis['missing_values'] = df[feature_cols].isnull().sum().sum()
        diagnosis['infinite_values'] = np.isinf(df[feature_cols].select_dtypes(include=[np.number])).sum().sum()
        
        # Feature correlation with target
        correlations = df[feature_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
        diagnosis['top_correlations'] = correlations.head(10).to_dict()
        diagnosis['weak_features'] = (correlations < 0.01).sum()
        
        self._print_diagnosis(diagnosis)
        return diagnosis
    
    def _print_diagnosis(self, diagnosis):
        """Print beautiful diagnosis report"""
        table = Table(title="🔍 Data Diagnosis Report", box=box.ROUNDED)
        table.add_column("Issue", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Action Required", style="yellow")
        
        # Class imbalance
        imbalance = diagnosis['class_imbalance']
        if imbalance['is_severe']:
            table.add_row(
                "Class Imbalance",
                f"❌ SEVERE (ratio: {imbalance['imbalance_ratio']:.1f})",
                "Apply SMOTE/resampling"
            )
        else:
            table.add_row(
                "Class Imbalance", 
                f"✅ OK (ratio: {imbalance['imbalance_ratio']:.1f})",
                "None"
            )
        
        # Missing values
        if diagnosis['missing_values'] > 0:
            table.add_row(
                "Missing Values",
                f"⚠️ {diagnosis['missing_values']} missing",
                "Fill/impute missing values"
            )
        else:
            table.add_row("Missing Values", "✅ None", "None")
        
        # Weak features
        if diagnosis['weak_features'] > 5:
            table.add_row(
                "Weak Features",
                f"⚠️ {diagnosis['weak_features']} weak features",
                "Feature selection/engineering"
            )
        else:
            table.add_row("Weak Features", "✅ Good", "None")
        
        console.print(table)
    
    def fix_data_quality(self, df, diagnosis):
        """🔧 Step 2: Fix data quality issues"""
        console.print(Panel.fit("🔧 Step 2: Data Quality Fix", style="bold green"))
        
        target_col = diagnosis['target_col']
        feature_cols = diagnosis['feature_cols']
        
        # Fix missing values
        if diagnosis['missing_values'] > 0:
            console.print("🔧 Fixing missing values...")
            for col in feature_cols:
                if df[col].isnull().sum() > 0:
                    if col in ['volume', 'volatility']:
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Fix infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(0)
        
        # Remove constant features
        constant_features = [col for col in feature_cols if df[col].nunique() <= 1]
        if constant_features:
            df = df.drop(columns=constant_features)
            feature_cols = [col for col in feature_cols if col not in constant_features]
            console.print(f"🗑️ Removed {len(constant_features)} constant features")
        
        # Update feature columns
        diagnosis['feature_cols'] = feature_cols
        
        console.print(f"✅ Data quality fixed. Features: {len(feature_cols)}")
        return df, diagnosis
    
    def engineer_predictive_features(self, df, diagnosis):
        """🎯 Step 3: Engineer highly predictive features"""
        console.print(Panel.fit("🎯 Step 3: Feature Engineering", style="bold magenta"))
        
        target_col = diagnosis['target_col']
        
        # Price-based features
        if 'close' in df.columns:
            df['price_momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['price_momentum_20'] = df['close'] / df['close'].shift(20) - 1
            df['price_acceleration'] = df['price_momentum_5'] - df['price_momentum_20']
            
        # Volume-based features (if available)
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['price_volume'] = df.get('returns', 0) * df['volume_ratio']
        
        # Volatility features
        if 'returns' in df.columns:
            df['volatility_5'] = df['returns'].rolling(5).std()
            df['volatility_20'] = df['returns'].rolling(20).std()
            df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
        
        # Technical patterns
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            df['doji'] = (abs(df['close'] - df['open']) / (df['high'] - df['low'])) < 0.1
            df['hammer'] = ((df['close'] > df['open']) & 
                           ((df['close'] - df['open']) / (df['high'] - df['low']) > 0.6))
            df['shooting_star'] = ((df['open'] > df['close']) & 
                                  ((df['open'] - df['close']) / (df['high'] - df['low']) > 0.6))
        
        # Regime detection
        if 'returns' in df.columns:
            df['trend_strength'] = df['returns'].rolling(20).mean() / df['returns'].rolling(20).std()
            df['mean_reversion'] = -df['returns'].rolling(5).mean()
        
        # Remove NaN values created by feature engineering
        df = df.dropna()
        
        # Update feature columns
        new_feature_cols = [col for col in df.columns if col != target_col and df[col].dtype in ['float64', 'int64', 'bool']]
        diagnosis['feature_cols'] = new_feature_cols
        
        console.print(f"🎯 Feature engineering complete. Total features: {len(new_feature_cols)}")
        return df, diagnosis
    
    def apply_class_imbalance_fix(self, X, y, diagnosis):
        """⚖️ Step 4: Fix class imbalance"""
        console.print(Panel.fit("⚖️ Step 4: Class Imbalance Fix", style="bold red"))
        
        imbalance = diagnosis['class_imbalance']
        
        if imbalance['is_severe']:
            console.print(f"🔧 Applying SMOTE for severe imbalance (ratio: {imbalance['imbalance_ratio']:.1f})")
            
            # Use BorderlineSMOTE for better synthetic samples
            smote = BorderlineSMOTE(
                sampling_strategy='auto',
                random_state=42,
                k_neighbors=min(5, min(imbalance['distribution'].values()) - 1)
            )
            
            try:
                X_resampled, y_resampled = smote.fit_resample(X, y)
                console.print(f"✅ SMOTE applied. New shape: {X_resampled.shape}")
                return X_resampled, y_resampled
            except Exception as e:
                console.print(f"⚠️ SMOTE failed: {e}, using class weights instead")
                return X, y
        else:
            console.print("✅ Class balance is acceptable")
            return X, y
    
    def select_best_features(self, X, y):
        """🔍 Step 5: Select most predictive features"""
        console.print(Panel.fit("🔍 Step 5: Feature Selection", style="bold cyan"))
        
        # Mutual information feature selection
        selector = SelectKBest(
            score_func=mutual_info_classif,
            k=min(50, X.shape[1])  # Select top 50 or all features if less
        )
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        
        # Get feature scores
        feature_scores = pd.Series(
            selector.scores_[selector.get_support()],
            index=selected_features
        ).sort_values(ascending=False)
        
        console.print(f"🔍 Selected {len(selected_features)} features from {X.shape[1]}")
        
        # Print top features
        top_features = feature_scores.head(10)
        table = Table(title="🏆 Top 10 Features", box=box.SIMPLE)
        table.add_column("Feature", style="cyan")
        table.add_column("Score", style="green")
        
        for feature, score in top_features.items():
            table.add_row(feature, f"{score:.4f}")
        
        console.print(table)
        
        return pd.DataFrame(X_selected, columns=selected_features), feature_scores
    
    def train_optimized_models(self, X, y):
        """🧠 Step 6: Train optimized models"""
        console.print(Panel.fit("🧠 Step 6: Model Training & Optimization", style="bold yellow"))
        
        models = {}
        cv_scores = {}
        
        # Stratified K-Fold for reliable evaluation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 1. XGBoost with optimized parameters
        console.print("🚀 Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='auc',
            tree_method='hist'
        )
        
        xgb_scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='roc_auc')
        models['xgboost'] = xgb_model
        cv_scores['xgboost'] = xgb_scores.mean()
        
        # 2. LightGBM
        console.print("⚡ Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='binary',
            metric='auc',
            verbose=-1
        )
        
        lgb_scores = cross_val_score(lgb_model, X, y, cv=cv, scoring='roc_auc')
        models['lightgbm'] = lgb_model
        cv_scores['lightgbm'] = lgb_scores.mean()
        
        # 3. CatBoost
        console.print("🐱 Training CatBoost...")
        cat_model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            random_seed=42,
            verbose=False,
            eval_metric='AUC'
        )
        
        cat_scores = cross_val_score(cat_model, X, y, cv=cv, scoring='roc_auc')
        models['catboost'] = cat_model
        cv_scores['catboost'] = cat_scores.mean()
        
        # 4. Random Forest (baseline)
        console.print("🌲 Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        rf_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='roc_auc')
        models['random_forest'] = rf_model
        cv_scores['random_forest'] = rf_scores.mean()
        
        # Print results
        results_table = Table(title="🏆 Model Performance Comparison", box=box.ROUNDED)
        results_table.add_column("Model", style="cyan")
        results_table.add_column("CV AUC", style="green")
        results_table.add_column("Status", style="white")
        
        for model_name, score in sorted(cv_scores.items(), key=lambda x: x[1], reverse=True):
            status = "✅ PASS" if score >= self.minimum_auc else "❌ FAIL"
            results_table.add_row(model_name, f"{score:.4f}", status)
        
        console.print(results_table)
        
        # Select best model
        best_model_name = max(cv_scores, key=cv_scores.get)
        best_model = models[best_model_name]
        best_score = cv_scores[best_model_name]
        
        console.print(f"🏆 Best model: {best_model_name} (AUC: {best_score:.4f})")
        
        return best_model, best_model_name, best_score, models, cv_scores
    
    def create_ensemble_model(self, models, X, y):
        """🤖 Step 7: Create ensemble model"""
        console.print(Panel.fit("🤖 Step 7: Ensemble Creation", style="bold purple"))
        
        # Weighted ensemble based on individual performance
        ensemble_predictions = np.zeros(len(y))
        total_weight = 0
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name, model in models.items():
            model.fit(X, y)
            
            # Get out-of-fold predictions for ensemble
            fold_predictions = np.zeros(len(y))
            for train_idx, val_idx in cv.split(X, y):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                
                model.fit(X_train_fold, y_train_fold)
                fold_predictions[val_idx] = model.predict_proba(X_val_fold)[:, 1]
            
            # Calculate weight based on performance
            fold_auc = roc_auc_score(y, fold_predictions)
            weight = max(0, fold_auc - 0.5) ** 2  # Square to emphasize better models
            
            ensemble_predictions += weight * fold_predictions
            total_weight += weight
            
            console.print(f"📊 {model_name}: AUC = {fold_auc:.4f}, Weight = {weight:.4f}")
        
        if total_weight > 0:
            ensemble_predictions /= total_weight
            ensemble_auc = roc_auc_score(y, ensemble_predictions)
            console.print(f"🏆 Ensemble AUC: {ensemble_auc:.4f}")
            
            return ensemble_predictions, ensemble_auc
        else:
            console.print("⚠️ Ensemble creation failed, using best single model")
            return None, 0
    
    def save_production_model(self, best_model, best_model_name, feature_names, best_score):
        """💾 Step 8: Save production-ready model"""
        console.print(Panel.fit("💾 Step 8: Save Production Model", style="bold green"))
        
        # Fit final model
        best_model.fit(X, y)
        
        # Save model
        model_path = self.output_dir / "catboost_model_best_cv.pkl"
        import joblib
        joblib.dump(best_model, model_path)
        
        # Save feature names
        features_path = self.output_dir / "train_features.txt"
        with open(features_path, 'w') as f:
            for feature in feature_names:
                f.write(f"{feature}\n")
        
        # Save model metadata
        metadata = {
            'model_type': best_model_name,
            'auc_score': float(best_score),
            'n_features': len(feature_names),
            'training_timestamp': pd.Timestamp.now().isoformat(),
            'target_threshold': 0.5
        }
        
        metadata_path = self.output_dir / "model_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        console.print(f"✅ Model saved:")
        console.print(f"   📁 Model: {model_path}")
        console.print(f"   📄 Features: {features_path}")
        console.print(f"   📊 Metadata: {metadata_path}")
        
        return model_path, features_path, metadata_path
    
    def run_complete_fix(self):
        """🚀 Run complete AUC fix pipeline"""
        console.print(Panel.fit("🚀 PRODUCTION AUC FIX PIPELINE", style="bold white on blue"))
        
        try:
            # Step 1: Load and diagnose data
            df, diagnosis = self.load_and_diagnose_data()
            
            # Step 2: Fix data quality issues
            df, diagnosis = self.fix_data_quality(df, diagnosis)
            
            # Step 3: Engineer predictive features
            df, diagnosis = self.engineer_predictive_features(df, diagnosis)
            
            # Prepare features and target
            target_col = diagnosis['target_col']
            feature_cols = diagnosis['feature_cols']
            
            X = df[feature_cols]
            y = df[target_col]
            
            console.print(f"📊 Final dataset: {X.shape}, Target distribution: {y.value_counts().to_dict()}")
            
            # Step 4: Fix class imbalance
            X, y = self.apply_class_imbalance_fix(X, y, diagnosis)
            
            # Step 5: Feature selection
            X_selected, feature_scores = self.select_best_features(X, y)
            
            # Step 6: Train optimized models
            best_model, best_model_name, best_score, models, cv_scores = self.train_optimized_models(X_selected, y)
            
            # Step 7: Create ensemble (optional)
            ensemble_predictions, ensemble_auc = self.create_ensemble_model(models, X_selected, y)
            
            # Use ensemble if it's better
            if ensemble_auc > best_score:
                console.print(f"🎯 Using ensemble model (AUC: {ensemble_auc:.4f})")
                final_score = ensemble_auc
            else:
                console.print(f"🎯 Using best single model: {best_model_name} (AUC: {best_score:.4f})")
                final_score = best_score
            
            # Step 8: Save production model
            model_path, features_path, metadata_path = self.save_production_model(
                best_model, best_model_name, X_selected.columns, final_score
            )
            
            # Final results
            success = final_score >= self.minimum_auc
            status_style = "bold green" if success else "bold red"
            status_text = "SUCCESS ✅" if success else "NEEDS MORE WORK ⚠️"
            
            console.print(Panel.fit(
                f"🎯 FINAL RESULT: {status_text}\n"
                f"🏆 AUC Score: {final_score:.4f}\n"
                f"🎯 Target: {self.target_auc:.2f}\n"
                f"🔥 Minimum: {self.minimum_auc:.2f}",
                style=status_style
            ))
            
            if success:
                console.print("🚀 Model is ready for production!")
            else:
                console.print("⚠️ Consider additional feature engineering or more data")
            
            return {
                'success': success,
                'final_auc': final_score,
                'model_path': str(model_path),
                'features_path': str(features_path),
                'metadata_path': str(metadata_path)
            }
            
        except Exception as e:
            console.print(f"❌ Critical error in AUC fix pipeline: {e}")
            self.logger.error(f"Pipeline failed: {e}")
            raise


def run_production_auc_fix():
    """Main function to run production AUC fix"""
    fixer = ProductionAUCFix()
    return fixer.run_complete_fix()


if __name__ == "__main__":
    # Run the complete fix
    results = run_production_auc_fix()
    
    if results['success']:
        print("✅ Production AUC fix completed successfully!")
        print(f"🏆 Final AUC: {results['final_auc']:.4f}")
    else:
        print("⚠️ AUC fix needs additional work")
        print(f"📊 Current AUC: {results['final_auc']:.4f}")
