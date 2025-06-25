

# Add project root to path
# Seamless integration of advanced protection system with existing trading pipeline
# 🔗 ProjectP ML Protection Integration
from datetime import datetime
    from ml_protection_system import MLProtectionSystem, ProtectionLevel, ProtectionResult
from pathlib import Path
    from tracking import EnterpriseTracker
from typing import Dict, Any, Optional, Tuple
import joblib
import logging
import numpy as np
import os
import pandas as pd
import sys
import warnings
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    PROTECTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML Protection system not available: {e}")
    PROTECTION_AVAILABLE = False

class ProjectPProtectionIntegration:
    """Integration layer for ProjectP with ML Protection System"""

    def __init__(self, protection_level: str = "enterprise", 
                 config_path: Optional[str] = None, 
                 enable_tracking: bool = True):
        """
        Initialize protection integration

        Args:
            protection_level: Protection level (basic, standard, aggressive, enterprise)
            config_path: Path to protection config file
            enable_tracking: Enable experiment tracking
        """
        self.protection_level = ProtectionLevel(protection_level.lower())
        self.enable_tracking = enable_tracking

        # Initialize protection system if available
        if PROTECTION_AVAILABLE:
            config_file = config_path or "ml_protection_config.yaml"
            self.protection_system = MLProtectionSystem(
                protection_level = self.protection_level, 
                config_path = config_file if Path(config_file).exists() else None
            )
        else:
            self.protection_system = None
            warnings.warn("Protection system not available. Running without protection.")

        # Setup logging
        self.logger = self._setup_logging()

        # Protection results storage
        self.last_protection_result: Optional[ProtectionResult] = None
        self.protection_history = []

    def _setup_logging(self) -> logging.Logger:
        """Setup protection integration logging"""
        logger = logging.getLogger('projectp_protection')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - ProjectP Protection - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def protect_data_pipeline(self, data: pd.DataFrame, 
                             target_col: str = 'target', 
                             timestamp_col: str = 'timestamp', 
                             stage: str = "preprocessing") -> pd.DataFrame:
        """
        Protect data in the ML pipeline

        Args:
            data: Input DataFrame
            target_col: Target column name
            timestamp_col: Timestamp column name
            stage: Pipeline stage name

        Returns:
            Protected DataFrame
        """
        if not PROTECTION_AVAILABLE or self.protection_system is None:
            self.logger.warning("Protection system not available. Returning original data.")
            return data

        self.logger.info(f"🛡️ Applying ML protection at {stage} stage...")

        try:
            # Run protection analysis
            protection_result = self.protection_system.protect_dataset(
                data = data, 
                target_col = target_col, 
                timestamp_col = timestamp_col, 
                model = None
            )

            # Store results
            self.last_protection_result = protection_result
            self.protection_history.append({
                'timestamp': datetime.now(), 
                'stage': stage, 
                'result': protection_result, 
                'data_shape': data.shape
            })

            # Log protection results
            self.logger.info(f"Protection analysis complete:")
            self.logger.info(f"  - Noise Score: {protection_result.noise_score:.3f}")
            self.logger.info(f"  - Leakage Score: {protection_result.leakage_score:.3f}")
            self.logger.info(f"  - Overall Clean: {protection_result.is_clean}")

            if protection_result.issues_found:
                self.logger.warning(f"Issues detected: {protection_result.issues_found}")
                for rec in protection_result.recommendations:
                    self.logger.info(f"Recommendation: {rec}")

            # Return cleaned data
            if protection_result.cleaned_data is not None:
                cleaned_shape = protection_result.cleaned_data.shape
                original_shape = data.shape

                if cleaned_shape != original_shape:
                    self.logger.info(f"Data cleaned: {original_shape} → {cleaned_shape}")

                return protection_result.cleaned_data
            else:
                return data

        except Exception as e:
            self.logger.error(f"Error in protection pipeline: {str(e)}")
            return data

    def protect_model_training(self, X: pd.DataFrame, y: pd.Series, 
                              model: Any, timestamp_col: str = 'timestamp') -> Dict[str, Any]:
        """
        Protect model training process

        Args:
            X: Feature matrix
            y: Target vector
            model: ML model to be trained
            timestamp_col: Timestamp column name

        Returns:
            Protection analysis results
        """
        if not PROTECTION_AVAILABLE or self.protection_system is None:
            self.logger.warning("Protection system not available. Skipping model protection.")
            return {'protected': False, 'reason': 'system_unavailable'}

        self.logger.info("🛡️ Applying model training protection...")

        try:
            # Combine X and y for analysis
            data = X.copy()
            data['target'] = y

            # Run comprehensive protection
            protection_result = self.protection_system.protect_dataset(
                data = data, 
                target_col = 'target', 
                timestamp_col = timestamp_col, 
                model = model
            )

            # Store results
            self.last_protection_result = protection_result

            # Check if training should proceed
            should_train = self._should_proceed_with_training(protection_result)

            protection_summary = {
                'protected': True, 
                'should_train': should_train, 
                'noise_score': protection_result.noise_score, 
                'leakage_score': protection_result.leakage_score, 
                'overfitting_score': protection_result.overfitting_score, 
                'is_clean': protection_result.is_clean, 
                'issues_found': protection_result.issues_found, 
                'recommendations': protection_result.recommendations, 
                'feature_report': protection_result.feature_report
            }

            if not should_train:
                self.logger.warning("⚠️ Training halted due to protection concerns!")
                self.logger.warning("Issues must be resolved before proceeding.")

            return protection_summary

        except Exception as e:
            self.logger.error(f"Error in model protection: {str(e)}")
            return {'protected': False, 'error': str(e)}

    def _should_proceed_with_training(self, result: ProtectionResult) -> bool:
        """Determine if training should proceed based on protection results"""

        # Define safety thresholds based on protection level
        thresholds = {
            ProtectionLevel.BASIC: {'noise': 0.3, 'leakage': 0.2, 'overfitting': 0.5}, 
            ProtectionLevel.STANDARD: {'noise': 0.2, 'leakage': 0.15, 'overfitting': 0.4}, 
            ProtectionLevel.AGGRESSIVE: {'noise': 0.15, 'leakage': 0.1, 'overfitting': 0.3}, 
            ProtectionLevel.ENTERPRISE: {'noise': 0.1, 'leakage': 0.05, 'overfitting': 0.2}
        }

        threshold = thresholds[self.protection_level]

        # Check each protection score against thresholds
        if result.noise_score > threshold['noise']:
            self.logger.warning(f"Noise score ({result.noise_score:.3f}) exceeds threshold ({threshold['noise']})")
            return False

        if result.leakage_score > threshold['leakage']:
            self.logger.warning(f"Leakage score ({result.leakage_score:.3f}) exceeds threshold ({threshold['leakage']})")
            return False

        if result.overfitting_score > threshold['overfitting']:
            self.logger.warning(f"Overfitting score ({result.overfitting_score:.3f}) exceeds threshold ({threshold['overfitting']})")
            return False

        return True

    def generate_protection_report(self, output_dir: str = "./reports") -> str:
        """Generate comprehensive protection report"""

        if self.last_protection_result is None:
            self.logger.warning("No protection results available for reporting.")
            return ""

        # Ensure output directory exists
        Path(output_dir).mkdir(parents = True, exist_ok = True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = Path(output_dir) / f"projectp_protection_report_{timestamp}.html"

        if PROTECTION_AVAILABLE and self.protection_system:
            report_file = self.protection_system.generate_protection_report(
                self.last_protection_result, 
                str(report_path)
            )
            self.logger.info(f"📊 Protection report generated: {report_file}")
            return report_file
        else:
            self.logger.warning("Cannot generate report: Protection system not available.")
            return ""

    def get_protection_summary(self) -> Dict[str, Any]:
        """Get summary of all protection activities"""

        if not self.protection_history:
            return {'status': 'no_protection_run'}

        latest_result = self.last_protection_result

        summary = {
            'total_protection_runs': len(self.protection_history), 
            'protection_level': self.protection_level.value, 
            'latest_analysis': {
                'timestamp': self.protection_history[ - 1]['timestamp'].isoformat(), 
                'stage': self.protection_history[ - 1]['stage'], 
                'is_clean': latest_result.is_clean if latest_result else False, 
                'scores': {
                    'noise': latest_result.noise_score if latest_result else 0, 
                    'leakage': latest_result.leakage_score if latest_result else 0, 
                    'overfitting': latest_result.overfitting_score if latest_result else 0
                } if latest_result else {}
            }, 
            'protection_history': [
                {
                    'timestamp': entry['timestamp'].isoformat(), 
                    'stage': entry['stage'], 
                    'data_shape': entry['data_shape']
                }
                for entry in self.protection_history
            ]
        }

        return summary

def integrate_with_projectp():
    """Main integration function for ProjectP.py"""

    # This function can be called from ProjectP.py to enable protection
    protection = ProjectPProtectionIntegration(
        protection_level = "enterprise",  # Can be configured
        enable_tracking = True
    )

    print("🛡️ ML Protection System integrated with ProjectP!")
    print(f"Protection Level: {protection.protection_level.value}")
    print("Ready to protect your ML pipeline.")

    return protection

# Example integration patterns for ProjectP.py
def example_usage_in_projectp():
    """Example of how to integrate protection in ProjectP.py"""

    # Initialize protection
    protection = ProjectPProtectionIntegration(protection_level = "enterprise")

    # In data loading/preprocessing stage
    def load_and_protect_data(file_path: str) -> pd.DataFrame:
        # Load data as usual
        data = pd.read_csv(file_path)

        # Apply protection
        protected_data = protection.protect_data_pipeline(
            data = data, 
            target_col = 'target', 
            timestamp_col = 'timestamp', 
            stage = "data_loading"
        )

        return protected_data

    # In feature engineering stage
    def feature_engineering_with_protection(data: pd.DataFrame) -> pd.DataFrame:
        # Your existing feature engineering
        engineered_data = apply_feature_engineering(data)  # Your function

        # Protect engineered features
        protected_features = protection.protect_data_pipeline(
            data = engineered_data, 
            target_col = 'target', 
            timestamp_col = 'timestamp', 
            stage = "feature_engineering"
        )

        return protected_features

    # In model training stage
    def train_model_with_protection(X: pd.DataFrame, y: pd.Series, model):
        # Run protection analysis
        protection_result = protection.protect_model_training(
            X = X, y = y, model = model
        )

        # Only train if protection passes
        if protection_result.get('should_train', False):
            print("✅ Protection passed. Training model...")
            model.fit(X, y)
            return model
        else:
            print("❌ Protection failed. Aborting training.")
            print("Issues:", protection_result.get('issues_found', []))
            return None

    # Generate final report
    def generate_final_report():
        report_path = protection.generate_protection_report()
        summary = protection.get_protection_summary()

        print(f"📊 Protection Summary: {summary}")
        print(f"📄 Detailed Report: {report_path}")

# Integration hook for existing ProjectP.py
def apply_protection_to_existing_projectp(data_df: pd.DataFrame, 
                                        target_col: str = 'target', 
                                        timestamp_col: str = 'timestamp', 
                                        model: Any = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Drop - in function to add protection to existing ProjectP.py

    Args:
        data_df: Your existing DataFrame
        target_col: Target column name
        timestamp_col: Timestamp column name
        model: Your trained model (optional)

    Returns:
        Tuple of (protected_dataframe, protection_summary)
    """

    print("🛡️ Applying ML Protection to your data...")

    # Initialize protection system
    protection = ProjectPProtectionIntegration(protection_level = "enterprise")

    # Protect data
    protected_data = protection.protect_data_pipeline(
        data = data_df, 
        target_col = target_col, 
        timestamp_col = timestamp_col, 
        stage = "projectp_integration"
    )

    # If model provided, run model protection
    protection_summary = {}
    if model is not None:
        X = protected_data.drop([target_col], axis = 1) if target_col in protected_data.columns else protected_data
        y = protected_data[target_col] if target_col in protected_data.columns else None

        if y is not None:
            protection_summary = protection.protect_model_training(X, y, model)

    # Get overall summary
    overall_summary = protection.get_protection_summary()
    protection_summary.update(overall_summary)

    print("✅ Protection analysis complete!")
    print(f"Data shape: {data_df.shape} → {protected_data.shape}")

    if protection.last_protection_result:
        print(f"Cleanliness: {'✅ CLEAN' if protection.last_protection_result.is_clean else '⚠️ ISSUES'}")
        print(f"Noise Score: {protection.last_protection_result.noise_score:.3f}")
        print(f"Leakage Score: {protection.last_protection_result.leakage_score:.3f}")

    return protected_data, protection_summary

if __name__ == "__main__":
    # Test integration
    protection = integrate_with_projectp()

    # Example test with dummy data
    print("\n🧪 Testing with sample data...")
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2024 - 01 - 01', periods = 100, freq = 'H'), 
        'feature1': np.random.randn(100), 
        'feature2': np.random.randn(100), 
        'target': np.random.randint(0, 2, 100)
    })

    protected_data, summary = apply_protection_to_existing_projectp(test_data)
    print(f"\n📊 Test Summary: {summary['latest_analysis']}")