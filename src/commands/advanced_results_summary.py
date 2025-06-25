# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - Advanced Results Summary System
═══════════════════════════════════════════════════════════════════════════════

Advanced comprehensive results summary and reporting system for full pipeline.
ระบบสรุปผลลัพธ์แบบครอบคลุมและรายงานขั้นสูงสำหรับ full pipeline

Author: NICEGOLD Team
Version: 3.0
Created: June 24, 2025
"""

import json
import os
import pickle
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# Import color system
import sys

sys.path.append(str(Path(__file__).parent.parent))
from core.colors import Colors, colorize


class AdvancedResultsSummary:
    """Advanced comprehensive results summary and analysis"""

    def __init__(self, project_root: Path, logger=None):
        self.project_root = project_root
        self.logger = logger
        self.results_dir = project_root / "results" / "summary"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.summary_data = {
            "timestamp": datetime.now().isoformat(),
            "pipeline_stages": {},
            "model_performance": {},
            "feature_importance": {},
            "optimization_results": {},
            "trading_simulation": {},
            "data_analysis": {},
            "recommendations": {},
            "next_steps": {},
        }

    def collect_pipeline_stage_results(
        self, stage_name: str, stage_data: Dict[str, Any]
    ):
        """Collect results from each pipeline stage"""
        print(
            f"{colorize(f'📊 Collecting results from stage: {stage_name}', Colors.BRIGHT_CYAN)}"
        )

        self.summary_data["pipeline_stages"][stage_name] = {
            "timestamp": datetime.now().isoformat(),
            "duration": stage_data.get("duration", 0),
            "status": stage_data.get("status", "unknown"),
            "metrics": stage_data.get("metrics", {}),
            "outputs": stage_data.get("outputs", {}),
            "errors": stage_data.get("errors", []),
            "warnings": stage_data.get("warnings", []),
        }

    def analyze_model_performance(
        self, y_true, y_pred, y_pred_proba=None, model_name="Main Model"
    ):
        """Comprehensive model performance analysis"""
        print(f"{colorize('🎯 Analyzing model performance...', Colors.BRIGHT_BLUE)}")

        try:
            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
            recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

            # Advanced metrics
            conf_matrix = confusion_matrix(y_true, y_pred)
            class_report = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )

            # AUC if probabilities available
            auc_score = None
            if y_pred_proba is not None:
                try:
                    auc_score = roc_auc_score(y_true, y_pred_proba)
                except:
                    pass

            performance_data = {
                "model_name": model_name,
                "basic_metrics": {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "auc_score": float(auc_score) if auc_score else None,
                },
                "confusion_matrix": conf_matrix.tolist(),
                "classification_report": class_report,
                "data_distribution": {
                    "total_samples": len(y_true),
                    "positive_samples": int(sum(y_true)),
                    "negative_samples": int(len(y_true) - sum(y_true)),
                    "imbalance_ratio": (
                        float(sum(y_true) / len(y_true)) if len(y_true) > 0 else 0
                    ),
                },
            }

            self.summary_data["model_performance"][model_name] = performance_data

            print(
                f"{colorize('✅ Model performance analysis completed', Colors.BRIGHT_GREEN)}"
            )
            return performance_data

        except Exception as e:
            print(
                f"{colorize('❌ Error in model performance analysis:', Colors.BRIGHT_RED)} {e}"
            )
            return {}

    def analyze_feature_importance(
        self, model, feature_names: List[str], model_name="Main Model"
    ):
        """Analyze and summarize feature importance"""
        print(f"{colorize('🔍 Analyzing feature importance...', Colors.BRIGHT_BLUE)}")

        try:
            # Get feature importance
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
            elif hasattr(model, "coef_"):
                importance = (
                    abs(model.coef_[0])
                    if len(model.coef_.shape) > 1
                    else abs(model.coef_)
                )
            else:
                print(
                    f"{colorize('⚠️ Model does not support feature importance', Colors.BRIGHT_YELLOW)}"
                )
                return {}

            # Create feature importance dataframe
            feature_df = pd.DataFrame(
                {"feature": feature_names[: len(importance)], "importance": importance}
            ).sort_values("importance", ascending=False)

            # Top and bottom features
            top_features = feature_df.head(10).to_dict("records")
            bottom_features = feature_df.tail(5).to_dict("records")

            importance_data = {
                "model_name": model_name,
                "total_features": len(feature_names),
                "top_features": top_features,
                "bottom_features": bottom_features,
                "importance_distribution": {
                    "mean": float(importance.mean()),
                    "std": float(importance.std()),
                    "max": float(importance.max()),
                    "min": float(importance.min()),
                },
            }

            self.summary_data["feature_importance"][model_name] = importance_data

            print(
                f"{colorize('✅ Feature importance analysis completed', Colors.BRIGHT_GREEN)}"
            )
            return importance_data

        except Exception as e:
            print(
                f"{colorize('❌ Error in feature importance analysis:', Colors.BRIGHT_RED)} {e}"
            )
            return {}

    def analyze_optimization_results(self, optimization_data: Dict[str, Any]):
        """Analyze hyperparameter optimization results"""
        print(f"{colorize('⚙️ Analyzing optimization results...', Colors.BRIGHT_BLUE)}")

        try:
            self.summary_data["optimization_results"] = {
                "best_parameters": optimization_data.get("best_params", {}),
                "best_score": optimization_data.get("best_score", 0),
                "optimization_method": optimization_data.get("method", "unknown"),
                "total_trials": optimization_data.get("n_trials", 0),
                "optimization_time": optimization_data.get("duration", 0),
                "score_improvement": optimization_data.get("improvement", 0),
                "convergence_info": optimization_data.get("convergence", {}),
            }

            print(
                f"{colorize('✅ Optimization analysis completed', Colors.BRIGHT_GREEN)}"
            )

        except Exception as e:
            print(
                f"{colorize('❌ Error in optimization analysis:', Colors.BRIGHT_RED)} {e}"
            )

    def analyze_trading_simulation(self, backtest_results: Dict[str, Any]):
        """Analyze trading simulation and backtesting results with professional metrics"""
        print(
            f"{colorize('📈 Analyzing professional trading simulation results...', Colors.BRIGHT_BLUE)}"
        )

        try:
            # Calculate professional trading metrics
            total_return = backtest_results.get("total_return", 0)
            initial_capital = backtest_results.get(
                "initial_capital", 100.0
            )  # $100 starting capital
            final_capital = initial_capital * (1 + total_return)

            winning_trades = backtest_results.get("winning_trades", 0)
            losing_trades = backtest_results.get("losing_trades", 0)
            total_trades = winning_trades + losing_trades

            # Win/Loss rates
            win_rate = (winning_trades / total_trades) if total_trades > 0 else 0
            loss_rate = (losing_trades / total_trades) if total_trades > 0 else 0

            # Trading period calculation
            start_date = backtest_results.get("start_date", "2023-01-01")
            end_date = backtest_results.get("end_date", "2024-12-31")

            from datetime import datetime, timedelta

            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                trading_days = (end_dt - start_dt).days
            except:
                trading_days = 365  # Default to 1 year

            # Professional metrics
            average_win = backtest_results.get("average_win", 0)
            average_loss = backtest_results.get("average_loss", 0)

            # Risk/Reward ratio
            risk_reward_ratio = (
                abs(average_win / average_loss) if average_loss != 0 else 0
            )

            # Expected value per trade
            expected_value = (win_rate * average_win) + (loss_rate * average_loss)

            # Maximum consecutive wins/losses
            max_consecutive_wins = backtest_results.get("max_consecutive_wins", 0)
            max_consecutive_losses = backtest_results.get("max_consecutive_losses", 0)

            # Volatility and other advanced metrics
            daily_volatility = backtest_results.get("daily_volatility", 0.02)
            annual_volatility = daily_volatility * (252**0.5)  # Annualized volatility

            # Calmar ratio (Annual return / Max Drawdown)
            max_drawdown = backtest_results.get("max_drawdown", 0.01)
            calmar_ratio = (
                (total_return * 365 / trading_days) / max_drawdown
                if max_drawdown > 0
                else 0
            )

            self.summary_data["trading_simulation"] = {
                # Capital and Returns
                "initial_capital": initial_capital,
                "final_capital": final_capital,
                "total_return": total_return,
                "total_return_percentage": total_return * 100,
                "net_profit": final_capital - initial_capital,
                # Trading Period
                "start_date": start_date,
                "end_date": end_date,
                "trading_days": trading_days,
                "trading_months": round(trading_days / 30.44, 1),
                "trading_years": round(trading_days / 365.25, 2),
                # Trade Statistics
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "loss_rate": loss_rate,
                "win_rate_percentage": win_rate * 100,
                "loss_rate_percentage": loss_rate * 100,
                # Performance Metrics
                "average_win": average_win,
                "average_loss": average_loss,
                "largest_win": backtest_results.get("largest_win", 0),
                "largest_loss": backtest_results.get("largest_loss", 0),
                "risk_reward_ratio": risk_reward_ratio,
                "expected_value_per_trade": expected_value,
                # Risk Metrics
                "max_drawdown": max_drawdown,
                "max_drawdown_percentage": max_drawdown * 100,
                "sharpe_ratio": backtest_results.get("sharpe_ratio", 0),
                "calmar_ratio": calmar_ratio,
                "profit_factor": backtest_results.get("profit_factor", 0),
                # Advanced Metrics
                "daily_volatility": daily_volatility,
                "annual_volatility": annual_volatility,
                "max_consecutive_wins": max_consecutive_wins,
                "max_consecutive_losses": max_consecutive_losses,
                # Additional Professional Metrics
                "recovery_factor": (
                    abs(total_return / max_drawdown) if max_drawdown != 0 else 0
                ),
                "trades_per_day": (
                    total_trades / trading_days if trading_days > 0 else 0
                ),
                "average_holding_period": backtest_results.get(
                    "average_holding_period", 1.0
                ),
                "simulation_period": backtest_results.get(
                    "period", f"{start_date} to {end_date}"
                ),
            }

            print(
                f"{colorize('✅ Trading simulation analysis completed', Colors.BRIGHT_GREEN)}"
            )

        except Exception as e:
            print(
                f"{colorize('❌ Error in trading simulation analysis:', Colors.BRIGHT_RED)} {e}"
            )

    def analyze_data_quality(self, df: pd.DataFrame):
        """Analyze data quality and characteristics"""
        print(f"{colorize('📊 Analyzing data quality...', Colors.BRIGHT_BLUE)}")

        try:
            # Basic data info
            data_info = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "missing_values": df.isnull().sum().to_dict(),
                "data_types": df.dtypes.astype(str).to_dict(),
                "duplicate_rows": df.duplicated().sum(),
                "date_range": {
                    "start": (
                        str(df.index.min()) if hasattr(df.index, "min") else "unknown"
                    ),
                    "end": (
                        str(df.index.max()) if hasattr(df.index, "max") else "unknown"
                    ),
                },
            }

            # Statistical summary
            numeric_columns = df.select_dtypes(include=["number"]).columns
            if len(numeric_columns) > 0:
                data_info["statistical_summary"] = (
                    df[numeric_columns].describe().to_dict()
                )

            self.summary_data["data_analysis"] = data_info

            print(
                f"{colorize('✅ Data quality analysis completed', Colors.BRIGHT_GREEN)}"
            )

        except Exception as e:
            print(
                f"{colorize('❌ Error in data quality analysis:', Colors.BRIGHT_RED)} {e}"
            )

    def generate_recommendations(self):
        """Generate intelligent recommendations based on results"""
        print(
            f"{colorize('🧠 Generating intelligent recommendations...', Colors.BRIGHT_BLUE)}"
        )

        recommendations = []
        next_steps = []

        try:
            # Model performance recommendations
            if "Main Model" in self.summary_data["model_performance"]:
                perf = self.summary_data["model_performance"]["Main Model"][
                    "basic_metrics"
                ]

                if perf.get("accuracy", 0) < 0.7:
                    recommendations.append(
                        {
                            "category": "Model Performance",
                            "priority": "High",
                            "issue": "Low accuracy detected",
                            "recommendation": "Consider feature engineering, data augmentation, or different algorithms",
                            "action": "Experiment with ensemble methods or deep learning approaches",
                        }
                    )

                if perf.get("f1_score", 0) < 0.6:
                    recommendations.append(
                        {
                            "category": "Model Performance",
                            "priority": "High",
                            "issue": "Low F1 score indicates poor precision-recall balance",
                            "recommendation": "Focus on threshold optimization and class balancing",
                            "action": "Implement SMOTE, cost-sensitive learning, or threshold tuning",
                        }
                    )

            # Data quality recommendations
            if "data_analysis" in self.summary_data:
                data_info = self.summary_data["data_analysis"]

                missing_values = data_info.get("missing_values", {})
                if any(v > 0 for v in missing_values.values()):
                    recommendations.append(
                        {
                            "category": "Data Quality",
                            "priority": "Medium",
                            "issue": "Missing values detected",
                            "recommendation": "Implement robust missing value handling strategy",
                            "action": "Use advanced imputation methods or feature selection",
                        }
                    )

            # Trading performance recommendations
            if "trading_simulation" in self.summary_data:
                trading = self.summary_data["trading_simulation"]

                if trading.get("sharpe_ratio", 0) < 1.0:
                    recommendations.append(
                        {
                            "category": "Trading Strategy",
                            "priority": "High",
                            "issue": "Low Sharpe ratio indicates poor risk-adjusted returns",
                            "recommendation": "Optimize position sizing and risk management",
                            "action": "Implement dynamic position sizing and stop-loss strategies",
                        }
                    )

                if trading.get("max_drawdown", 0) > 0.2:
                    recommendations.append(
                        {
                            "category": "Risk Management",
                            "priority": "Critical",
                            "issue": "High maximum drawdown detected",
                            "recommendation": "Implement stricter risk controls",
                            "action": "Reduce position sizes and add portfolio protection mechanisms",
                        }
                    )

            # Generate next steps
            next_steps = [
                "1. 🎯 Priority Focus: Address high-priority recommendations first",
                "2. 📊 Data Enhancement: Collect more diverse and recent market data",
                "3. 🔬 Model Experimentation: Test advanced algorithms (XGBoost, Neural Networks)",
                "4. ⚙️ Hyperparameter Optimization: Fine-tune all model parameters",
                "5. 📈 Backtesting: Extend historical testing period",
                "6. 🛡️ Risk Management: Implement advanced risk controls",
                "7. 🚀 Production Deployment: Prepare for live trading",
                "8. 📱 Monitoring System: Set up real-time performance tracking",
                "9. 🔄 Continuous Learning: Implement online learning capabilities",
                "10. 📈 Portfolio Expansion: Consider multi-asset strategies",
            ]

            self.summary_data["recommendations"] = recommendations
            self.summary_data["next_steps"] = next_steps

            print(
                f"{colorize('✅ Recommendations generated successfully', Colors.BRIGHT_GREEN)}"
            )

        except Exception as e:
            print(
                f"{colorize('❌ Error generating recommendations:', Colors.BRIGHT_RED)} {e}"
            )

    def create_visualization_report(self):
        """Create comprehensive visualization report"""
        print(f"{colorize('📊 Creating visualization report...', Colors.BRIGHT_BLUE)}")

        try:
            plt.style.use("seaborn-v0_8")
            fig = plt.figure(figsize=(20, 16))

            # Model Performance Plot
            if "Main Model" in self.summary_data["model_performance"]:
                ax1 = plt.subplot(2, 3, 1)
                perf = self.summary_data["model_performance"]["Main Model"][
                    "basic_metrics"
                ]
                metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
                values = [
                    perf.get("accuracy", 0),
                    perf.get("precision", 0),
                    perf.get("recall", 0),
                    perf.get("f1_score", 0),
                ]

                bars = ax1.bar(
                    metrics, values, color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
                )
                ax1.set_title(
                    "Model Performance Metrics", fontsize=14, fontweight="bold"
                )
                ax1.set_ylim(0, 1)

                # Add value labels on bars
                for bar, value in zip(bars, values):
                    ax1.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

            # Feature Importance Plot (with error handling)
            try:
                if (
                    self.summary_data.get("feature_importance")
                    and "Main Model" in self.summary_data["feature_importance"]
                ):
                    ax2 = plt.subplot(2, 3, 2)
                    feat_data = self.summary_data["feature_importance"]["Main Model"]
                    top_features = feat_data.get("top_features", [])[:10]

                    if top_features:
                        features = [f["feature"] for f in top_features]
                    importance = [f["importance"] for f in top_features]

                    ax2.barh(range(len(features)), importance, color="#FF8C42")
                    ax2.set_yticks(range(len(features)))
                    ax2.set_yticklabels(features)
                    ax2.set_title(
                        "Top 10 Feature Importance", fontsize=14, fontweight="bold"
                    )
                    ax2.invert_yaxis()
            except Exception as e:
                print(f"Warning: Could not create feature importance plot: {e}")

            # Trading Performance Plot
            if "trading_simulation" in self.summary_data:
                ax3 = plt.subplot(2, 3, 3)
                trading = self.summary_data["trading_simulation"]

                metrics = ["Total Return", "Sharpe Ratio", "Win Rate", "Profit Factor"]
                values = [
                    trading.get("total_return", 0),
                    trading.get("sharpe_ratio", 0),
                    trading.get("win_rate", 0),
                    trading.get("profit_factor", 0),
                ]

                ax3.bar(
                    metrics, values, color=["#6C5CE7", "#A8E6CF", "#FFD93D", "#FF6B6B"]
                )
                ax3.set_title(
                    "Trading Performance Metrics", fontsize=14, fontweight="bold"
                )
                plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")

            # Data Quality Plot
            if "data_analysis" in self.summary_data:
                ax4 = plt.subplot(2, 3, 4)
                data_info = self.summary_data["data_analysis"]

                total_rows = data_info.get("total_rows", 0)
                duplicate_rows = data_info.get("duplicate_rows", 0)
                missing_count = sum(data_info.get("missing_values", {}).values())

                categories = ["Total Rows", "Duplicate Rows", "Missing Values"]
                values = [total_rows, duplicate_rows, missing_count]

                ax4.bar(categories, values, color=["#74B9FF", "#FD79A8", "#FDCB6E"])
                ax4.set_title("Data Quality Overview", fontsize=14, fontweight="bold")
                ax4.set_yscale("log")

            # Pipeline Stages Performance
            ax5 = plt.subplot(2, 3, 5)
            if self.summary_data["pipeline_stages"]:
                stages = list(self.summary_data["pipeline_stages"].keys())
                durations = [
                    self.summary_data["pipeline_stages"][stage].get("duration", 0)
                    for stage in stages
                ]

                ax5.pie(durations, labels=stages, autopct="%1.1f%%", startangle=90)
                ax5.set_title(
                    "Pipeline Stage Duration Distribution",
                    fontsize=14,
                    fontweight="bold",
                )

            # Recommendations Summary
            ax6 = plt.subplot(2, 3, 6)
            if self.summary_data["recommendations"]:
                priorities = {}
                for rec in self.summary_data["recommendations"]:
                    priority = rec.get("priority", "Unknown")
                    priorities[priority] = priorities.get(priority, 0) + 1

                if priorities:
                    ax6.pie(
                        priorities.values(),
                        labels=priorities.keys(),
                        autopct="%1.0f",
                        colors=["#FF6B6B", "#FFD93D", "#74B9FF", "#A8E6CF"],
                    )
                    ax6.set_title(
                        "Recommendations by Priority", fontsize=14, fontweight="bold"
                    )

            plt.tight_layout()

            # Save the plot
            viz_path = (
                self.results_dir
                / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            plt.savefig(viz_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(
                f"{colorize('✅ Visualization report saved:', Colors.BRIGHT_GREEN)} {viz_path}"
            )

        except Exception as e:
            print(
                f"{colorize('❌ Error creating visualization report:', Colors.BRIGHT_RED)} {e}"
            )

    def generate_comprehensive_summary(self) -> str:
        """Generate the final comprehensive summary report"""
        print(
            f"\n{colorize('🎯 GENERATING COMPREHENSIVE RESULTS SUMMARY', Colors.BOLD + Colors.BRIGHT_MAGENTA)}"
        )
        print(f"{colorize('═' * 80, Colors.BRIGHT_MAGENTA)}")

        # Generate recommendations first
        self.generate_recommendations()

        # Create visualization
        self.create_visualization_report()

        # Generate text summary
        summary_text = self._generate_text_summary()

        # Save comprehensive results
        self._save_comprehensive_results()

        return summary_text

    def _generate_text_summary(self) -> str:
        """Generate detailed text summary"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        summary = f"""
{colorize('╔' + '═' * 78 + '╗', Colors.BRIGHT_YELLOW)}
{colorize('║' + ' ' * 78 + '║', Colors.BRIGHT_YELLOW)}
{colorize('║' + '🏆 NICEGOLD PROJECTP - ULTIMATE RESULTS SUMMARY 🏆'.center(78) + '║', Colors.BRIGHT_YELLOW)}
{colorize('║' + f'Generated: {timestamp}'.center(78) + '║', Colors.BRIGHT_YELLOW)}
{colorize('║' + ' ' * 78 + '║', Colors.BRIGHT_YELLOW)}
{colorize('╚' + '═' * 78 + '╝', Colors.BRIGHT_YELLOW)}

"""

        # Executive Summary
        summary += (
            f"{colorize('📋 EXECUTIVE SUMMARY', Colors.BOLD + Colors.BRIGHT_CYAN)}\n"
        )
        summary += f"{colorize('─' * 50, Colors.BRIGHT_CYAN)}\n"

        if "Main Model" in self.summary_data["model_performance"]:
            perf = self.summary_data["model_performance"]["Main Model"]["basic_metrics"]
            summary += f"• {colorize('Model Accuracy:', Colors.BRIGHT_GREEN)} {perf.get('accuracy', 0):.3f}\n"
            summary += f"• {colorize('F1-Score:', Colors.BRIGHT_GREEN)} {perf.get('f1_score', 0):.3f}\n"
            summary += f"• {colorize('AUC Score:', Colors.BRIGHT_GREEN)} {perf.get('auc_score', 'N/A')}\n"

        if "trading_simulation" in self.summary_data:
            trading = self.summary_data["trading_simulation"]
            summary += f"• {colorize('Total Return:', Colors.BRIGHT_GREEN)} {trading.get('total_return', 0):.2%}\n"
            summary += f"• {colorize('Sharpe Ratio:', Colors.BRIGHT_GREEN)} {trading.get('sharpe_ratio', 0):.3f}\n"
            summary += f"• {colorize('Win Rate:', Colors.BRIGHT_GREEN)} {trading.get('win_rate', 0):.1%}\n"

        summary += "\n"

        # Model Performance Details
        summary += f"{colorize('🎯 MODEL PERFORMANCE ANALYSIS', Colors.BOLD + Colors.BRIGHT_BLUE)}\n"
        summary += f"{colorize('─' * 50, Colors.BRIGHT_BLUE)}\n"

        # Handle different data structures for model_performance
        model_perf = self.summary_data["model_performance"]

        if isinstance(model_perf, dict):
            # Check if it has basic_metrics directly (new structure)
            if "basic_metrics" in model_perf:
                metrics = model_perf["basic_metrics"]
                summary += f"\n{colorize('📊 Main Model:', Colors.BRIGHT_WHITE)}\n"
                accuracy_val = metrics.get("accuracy", 0)
                precision_val = metrics.get("precision", 0)
                recall_val = metrics.get("recall", 0)
                f1_val = metrics.get("f1_score", 0)

                summary += f"  • Accuracy: {colorize(f'{accuracy_val:.3f}', Colors.BRIGHT_GREEN)}\n"
                summary += f"  • Precision: {colorize(f'{precision_val:.3f}', Colors.BRIGHT_GREEN)}\n"
                summary += f"  • Recall: {colorize(f'{recall_val:.3f}', Colors.BRIGHT_GREEN)}\n"
                summary += (
                    f"  • F1-Score: {colorize(f'{f1_val:.3f}', Colors.BRIGHT_GREEN)}\n"
                )

                if metrics.get("auc_score"):
                    auc_val = metrics["auc_score"]
                    summary += f"  • AUC Score: {colorize(f'{auc_val:.3f}', Colors.BRIGHT_GREEN)}\n"
            else:
                # Old structure - iterate through models
                for model_name, perf_data in model_perf.items():
                    if isinstance(perf_data, dict) and "basic_metrics" in perf_data:
                        metrics = perf_data["basic_metrics"]
                        summary += (
                            f"\n{colorize(f'📊 {model_name}:', Colors.BRIGHT_WHITE)}\n"
                        )
                        accuracy_val = metrics.get("accuracy", 0)
                        precision_val = metrics.get("precision", 0)
                        recall_val = metrics.get("recall", 0)
                        f1_val = metrics.get("f1_score", 0)

                        summary += f"  • Accuracy: {colorize(f'{accuracy_val:.3f}', Colors.BRIGHT_GREEN)}\n"
                        summary += f"  • Precision: {colorize(f'{precision_val:.3f}', Colors.BRIGHT_GREEN)}\n"
                        summary += f"  • Recall: {colorize(f'{recall_val:.3f}', Colors.BRIGHT_GREEN)}\n"
                        summary += f"  • F1-Score: {colorize(f'{f1_val:.3f}', Colors.BRIGHT_GREEN)}\n"

                        if metrics.get("auc_score"):
                            auc_val = metrics["auc_score"]
                            summary += f"  • AUC Score: {colorize(f'{auc_val:.3f}', Colors.BRIGHT_GREEN)}\n"

            # Data distribution
            dist = perf_data.get("data_distribution", {})
            total_samples = dist.get("total_samples", 0)
            imbalance_ratio = dist.get("imbalance_ratio", 0)
            summary += f"  • Total Samples: {colorize(str(total_samples), Colors.BRIGHT_CYAN)}\n"
            summary += f"  • Imbalance Ratio: {colorize(f'{imbalance_ratio:.3f}', Colors.BRIGHT_CYAN)}\n"

        # Feature Importance (with error handling)
        try:
            if (
                self.summary_data.get("feature_importance")
                and isinstance(self.summary_data["feature_importance"], dict)
                and self.summary_data["feature_importance"]
            ):
                summary += f"\n{colorize('🔍 FEATURE IMPORTANCE ANALYSIS', Colors.BOLD + Colors.BRIGHT_MAGENTA)}\n"
                summary += f"{colorize('─' * 50, Colors.BRIGHT_MAGENTA)}\n"

                for model_name, feat_data in self.summary_data[
                    "feature_importance"
                ].items():
                    if isinstance(feat_data, dict) and feat_data.get("top_features"):
                        summary += f"\n{colorize(f'📈 {model_name} - Top Features:', Colors.BRIGHT_WHITE)}\n"

                        top_features = feat_data.get("top_features", [])[:5]
                        for i, feat in enumerate(top_features, 1):
                            if (
                                isinstance(feat, dict)
                                and "feature" in feat
                                and "importance" in feat
                            ):
                                summary += f"  {i:2d}. {colorize(feat['feature'], Colors.BRIGHT_GREEN)}: {feat['importance']:.4f}\n"
        except Exception as e:
            print(f"Warning: Could not display feature importance: {e}")

        # Professional Trading Performance Analysis
        if "trading_simulation" in self.summary_data:
            summary += f"\n{colorize('📈 PROFESSIONAL TRADING ANALYSIS', Colors.BOLD + Colors.BRIGHT_GREEN)}\n"
            summary += f"{colorize('═' * 60, Colors.BRIGHT_GREEN)}\n"

            trading = self.summary_data["trading_simulation"]

            # Capital Management Section
            summary += f"\n{colorize('💰 CAPITAL MANAGEMENT', Colors.BOLD + Colors.BRIGHT_CYAN)}\n"
            summary += f"{colorize('─' * 30, Colors.BRIGHT_CYAN)}\n"

            initial_capital = trading.get("initial_capital", 100)
            final_capital = trading.get("final_capital", 100)
            net_profit = trading.get("net_profit", 0)
            total_return_pct = trading.get("total_return_percentage", 0)

            summary += f"• Initial Capital: {colorize(f'${initial_capital:.2f}', Colors.BRIGHT_WHITE)}\n"
            summary += f"• Final Capital: {colorize(f'${final_capital:.2f}', Colors.BRIGHT_GREEN)}\n"
            summary += f"• Net Profit: {colorize(f'${net_profit:.2f}', Colors.BRIGHT_GREEN if net_profit >= 0 else Colors.BRIGHT_RED)}\n"
            summary += f"• Total Return: {colorize(f'{total_return_pct:.2f}%', Colors.BRIGHT_GREEN if total_return_pct >= 0 else Colors.BRIGHT_RED)}\n"

            # Trading Period Section
            summary += (
                f"\n{colorize('📅 TRADING PERIOD', Colors.BOLD + Colors.BRIGHT_CYAN)}\n"
            )
            summary += f"{colorize('─' * 30, Colors.BRIGHT_CYAN)}\n"

            start_date = trading.get("start_date", "2023-01-01")
            end_date = trading.get("end_date", "2024-12-31")
            trading_days = trading.get("trading_days", 365)
            trading_months = trading.get("trading_months", 12.0)
            trading_years = trading.get("trading_years", 1.0)

            summary += f"• Start Date: {colorize(start_date, Colors.BRIGHT_WHITE)}\n"
            summary += f"• End Date: {colorize(end_date, Colors.BRIGHT_WHITE)}\n"
            summary += f"• Trading Days: {colorize(f'{trading_days:,}', Colors.BRIGHT_CYAN)} days\n"
            summary += f"• Trading Period: {colorize(f'{trading_months:.1f}', Colors.BRIGHT_CYAN)} months ({colorize(f'{trading_years:.2f}', Colors.BRIGHT_CYAN)} years)\n"

            # Trade Statistics Section
            summary += f"\n{colorize('📊 TRADE STATISTICS', Colors.BOLD + Colors.BRIGHT_CYAN)}\n"
            summary += f"{colorize('─' * 30, Colors.BRIGHT_CYAN)}\n"

            total_trades = trading.get("total_trades", 0)
            winning_trades = trading.get("winning_trades", 0)
            losing_trades = trading.get("losing_trades", 0)
            win_rate_pct = trading.get("win_rate_percentage", 0)
            loss_rate_pct = trading.get("loss_rate_percentage", 0)
            trades_per_day = trading.get("trades_per_day", 0)

            summary += f"• Total Orders: {colorize(f'{total_trades:,}', Colors.BRIGHT_WHITE)}\n"
            summary += f"• Winning Trades: {colorize(f'{winning_trades:,}', Colors.BRIGHT_GREEN)} ({colorize(f'{win_rate_pct:.1f}%', Colors.BRIGHT_GREEN)})\n"
            summary += f"• Losing Trades: {colorize(f'{losing_trades:,}', Colors.BRIGHT_RED)} ({colorize(f'{loss_rate_pct:.1f}%', Colors.BRIGHT_RED)})\n"
            summary += f"• Win Rate: {colorize(f'{win_rate_pct:.1f}%', Colors.BRIGHT_GREEN if win_rate_pct >= 50 else Colors.BRIGHT_YELLOW)}\n"
            summary += f"• Trade Frequency: {colorize(f'{trades_per_day:.2f}', Colors.BRIGHT_CYAN)} trades/day\n"

            # Performance Metrics Section
            summary += f"\n{colorize('⚡ PERFORMANCE METRICS', Colors.BOLD + Colors.BRIGHT_CYAN)}\n"
            summary += f"{colorize('─' * 30, Colors.BRIGHT_CYAN)}\n"

            average_win = trading.get("average_win", 0)
            average_loss = trading.get("average_loss", 0)
            largest_win = trading.get("largest_win", 0)
            largest_loss = trading.get("largest_loss", 0)
            risk_reward_ratio = trading.get("risk_reward_ratio", 0)
            expected_value = trading.get("expected_value_per_trade", 0)

            summary += f"• Average Win: {colorize(f'${average_win:.2f}', Colors.BRIGHT_GREEN)}\n"
            summary += f"• Average Loss: {colorize(f'${average_loss:.2f}', Colors.BRIGHT_RED)}\n"
            summary += f"• Largest Win: {colorize(f'${largest_win:.2f}', Colors.BRIGHT_GREEN)}\n"
            summary += f"• Largest Loss: {colorize(f'${largest_loss:.2f}', Colors.BRIGHT_RED)}\n"
            summary += f"• Risk/Reward Ratio: {colorize(f'{risk_reward_ratio:.2f}:1', Colors.BRIGHT_YELLOW)}\n"
            summary += f"• Expected Value/Trade: {colorize(f'${expected_value:.2f}', Colors.BRIGHT_GREEN if expected_value >= 0 else Colors.BRIGHT_RED)}\n"

            # Risk Management Section
            summary += (
                f"\n{colorize('🛡️ RISK MANAGEMENT', Colors.BOLD + Colors.BRIGHT_CYAN)}\n"
            )
            summary += f"{colorize('─' * 30, Colors.BRIGHT_CYAN)}\n"

            max_drawdown_pct = trading.get("max_drawdown_percentage", 0)
            sharpe_ratio = trading.get("sharpe_ratio", 0)
            calmar_ratio = trading.get("calmar_ratio", 0)
            profit_factor = trading.get("profit_factor", 0)
            recovery_factor = trading.get("recovery_factor", 0)
            annual_volatility = trading.get("annual_volatility", 0)

            summary += f"• Maximum Drawdown (DD): {colorize(f'{max_drawdown_pct:.2f}%', Colors.BRIGHT_RED)}\n"
            summary += f"• Sharpe Ratio: {colorize(f'{sharpe_ratio:.3f}', Colors.BRIGHT_GREEN if sharpe_ratio >= 1.0 else Colors.BRIGHT_YELLOW)}\n"
            summary += f"• Calmar Ratio: {colorize(f'{calmar_ratio:.3f}', Colors.BRIGHT_GREEN)}\n"
            summary += f"• Profit Factor: {colorize(f'{profit_factor:.2f}', Colors.BRIGHT_GREEN if profit_factor >= 1.5 else Colors.BRIGHT_YELLOW)}\n"
            summary += f"• Recovery Factor: {colorize(f'{recovery_factor:.2f}', Colors.BRIGHT_GREEN)}\n"
            summary += f"• Annual Volatility: {colorize(f'{annual_volatility:.1%}', Colors.BRIGHT_CYAN)}\n"

            # Advanced Statistics Section
            summary += f"\n{colorize('📈 ADVANCED STATISTICS', Colors.BOLD + Colors.BRIGHT_CYAN)}\n"
            summary += f"{colorize('─' * 30, Colors.BRIGHT_CYAN)}\n"

            max_consecutive_wins = trading.get("max_consecutive_wins", 0)
            max_consecutive_losses = trading.get("max_consecutive_losses", 0)
            average_holding_period = trading.get("average_holding_period", 1.0)

            summary += f"• Max Consecutive Wins: {colorize(f'{max_consecutive_wins}', Colors.BRIGHT_GREEN)}\n"
            summary += f"• Max Consecutive Losses: {colorize(f'{max_consecutive_losses}', Colors.BRIGHT_RED)}\n"
            summary += f"• Average Holding Period: {colorize(f'{average_holding_period:.1f}', Colors.BRIGHT_CYAN)} days\n"

            # Trading Costs Analysis Section (if available)
            if trading.get("realistic_costs_applied", False):
                summary += f"\n{colorize('💰 TRADING COSTS ANALYSIS', Colors.BOLD + Colors.BRIGHT_CYAN)}\n"
                summary += f"{colorize('─' * 30, Colors.BRIGHT_CYAN)}\n"

                commission_per_trade = trading.get("commission_per_trade", 0)
                spread_cost = trading.get("spread_cost_per_trade", 0)
                slippage_cost = trading.get("slippage_cost_per_trade", 0)
                total_cost_per_trade = trading.get("total_cost_per_trade", 0)
                total_trading_costs = trading.get("total_trading_costs", 0)
                spread_pips = trading.get("spread_pips", 0)
                slippage_pips = trading.get("slippage_pips", 0)

                summary += f"• Commission: {colorize(f'${commission_per_trade:.2f}', Colors.BRIGHT_WHITE)} per 0.01 lot (mini lot)\n"
                summary += f"• Spread: {colorize(f'{spread_pips:.1f}', Colors.BRIGHT_YELLOW)} pips ({colorize(f'${spread_cost:.3f}', Colors.BRIGHT_YELLOW)})\n"
                summary += f"• Slippage: {colorize(f'{slippage_pips:.1f}', Colors.BRIGHT_YELLOW)} pips ({colorize(f'${slippage_cost:.3f}', Colors.BRIGHT_YELLOW)})\n"
                summary += f"• Total Cost/Trade: {colorize(f'${total_cost_per_trade:.3f}', Colors.BRIGHT_RED)}\n"
                summary += f"• Total Trading Costs: {colorize(f'${total_trading_costs:.2f}', Colors.BRIGHT_RED)}\n"

                # Calculate cost impact
                initial_capital = trading.get("initial_capital", 100)
                cost_impact_percentage = (total_trading_costs / initial_capital) * 100
                summary += f"• Cost Impact: {colorize(f'{cost_impact_percentage:.2f}%', Colors.BRIGHT_RED)} of capital\n"

                # Show gross vs net performance
                gross_win = trading.get("gross_average_win", 0)
                gross_loss = trading.get("gross_average_loss", 0)
                net_win = trading.get("average_win", 0)
                net_loss = trading.get("average_loss", 0)

                summary += f"\n{colorize('📊 Gross vs Net Performance:', Colors.BRIGHT_WHITE)}\n"
                summary += f"  • Gross Avg Win: {colorize(f'${gross_win:.2f}', Colors.BRIGHT_GREEN)} → Net: {colorize(f'${net_win:.2f}', Colors.BRIGHT_GREEN)}\n"
                summary += f"  • Gross Avg Loss: {colorize(f'${gross_loss:.2f}', Colors.BRIGHT_RED)} → Net: {colorize(f'${net_loss:.2f}', Colors.BRIGHT_RED)}\n"

        # Optimization Results
        if (
            "optimization_results" in self.summary_data
            and self.summary_data["optimization_results"]
        ):
            summary += f"\n{colorize('⚙️ HYPERPARAMETER OPTIMIZATION', Colors.BOLD + Colors.BRIGHT_YELLOW)}\n"
            summary += f"{colorize('─' * 50, Colors.BRIGHT_YELLOW)}\n"

            opt = self.summary_data["optimization_results"]

            best_score = opt.get("best_score", 0)
            total_trials = opt.get("total_trials", 0)
            optimization_time = opt.get("optimization_time", 0)
            score_improvement = opt.get("score_improvement", 0)

            summary += (
                f"• Best Score: {colorize(f'{best_score:.4f}', Colors.BRIGHT_GREEN)}\n"
            )
            summary += (
                f"• Total Trials: {colorize(str(total_trials), Colors.BRIGHT_CYAN)}\n"
            )
            summary += f"• Optimization Time: {colorize(f'{optimization_time:.1f}s', Colors.BRIGHT_CYAN)}\n"
            summary += f"• Score Improvement: {colorize(f'{score_improvement:.4f}', Colors.BRIGHT_GREEN)}\n"

            # Best parameters
            best_params = opt.get("best_parameters", {})
            if best_params:
                summary += f"\n{colorize('🎯 Best Parameters:', Colors.BRIGHT_WHITE)}\n"
                for param, value in best_params.items():
                    summary += (
                        f"  • {param}: {colorize(str(value), Colors.BRIGHT_GREEN)}\n"
                    )

        # Recommendations
        if self.summary_data["recommendations"]:
            summary += f"\n{colorize('🧠 INTELLIGENT RECOMMENDATIONS', Colors.BOLD + Colors.BRIGHT_MAGENTA)}\n"
            summary += f"{colorize('─' * 50, Colors.BRIGHT_MAGENTA)}\n"

            for i, rec in enumerate(self.summary_data["recommendations"], 1):
                priority_color = (
                    Colors.BRIGHT_RED
                    if rec["priority"] == "Critical"
                    else (
                        Colors.BRIGHT_YELLOW
                        if rec["priority"] == "High"
                        else Colors.BRIGHT_CYAN
                    )
                )

                priority = rec["priority"]
                category = rec["category"]
                rec_text = f"{i}. [{priority}] {category}"
                summary += f"\n{colorize(rec_text, priority_color)}\n"
                summary += f"   Issue: {rec['issue']}\n"
                summary += f"   Recommendation: {colorize(rec['recommendation'], Colors.BRIGHT_GREEN)}\n"
                summary += (
                    f"   Action: {colorize(rec['action'], Colors.BRIGHT_WHITE)}\n"
                )

        # Next Steps
        if self.summary_data["next_steps"]:
            summary += f"\n{colorize('🚀 STRATEGIC NEXT STEPS', Colors.BOLD + Colors.BRIGHT_CYAN)}\n"
            summary += f"{colorize('─' * 50, Colors.BRIGHT_CYAN)}\n"

            for step in self.summary_data["next_steps"]:
                summary += f"{step}\n"

        # Footer
        summary += f"\n{colorize('═' * 80, Colors.BRIGHT_YELLOW)}\n"
        summary += f"{colorize('🎯 Report generated by NICEGOLD ProjectP Advanced Results System', Colors.BRIGHT_GREEN)}\n"
        summary += f"{colorize('📅 Timestamp:', Colors.BRIGHT_CYAN)} {timestamp}\n"
        summary += f"{colorize('📁 Results saved to:', Colors.BRIGHT_CYAN)} {self.results_dir}\n"
        summary += f"{colorize('═' * 80, Colors.BRIGHT_YELLOW)}\n"

        return summary

    def _save_comprehensive_results(self):
        """Save all results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON summary
        json_path = self.results_dir / f"comprehensive_summary_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.summary_data, f, indent=2, ensure_ascii=False, default=str)

        # Save text summary
        text_summary = self._generate_text_summary()
        text_path = self.results_dir / f"summary_report_{timestamp}.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            # Remove ANSI color codes for text file
            import re

            clean_text = re.sub(r"\x1b\[[0-9;]*m", "", text_summary)
            f.write(clean_text)

        # Save pickle for Python objects
        pickle_path = self.results_dir / f"results_data_{timestamp}.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(self.summary_data, f)

        print(f"\n{colorize('💾 RESULTS SAVED:', Colors.BOLD + Colors.BRIGHT_GREEN)}")
        print(f"📄 JSON Summary: {colorize(str(json_path), Colors.BRIGHT_CYAN)}")
        print(f"📝 Text Report: {colorize(str(text_path), Colors.BRIGHT_CYAN)}")
        print(f"🗃️ Data Archive: {colorize(str(pickle_path), Colors.BRIGHT_CYAN)}")

    def print_quick_summary(self):
        """Print a quick summary to console"""
        print(
            f"\n{colorize('⚡ QUICK RESULTS OVERVIEW', Colors.BOLD + Colors.BRIGHT_YELLOW)}"
        )
        print(f"{colorize('─' * 40, Colors.BRIGHT_YELLOW)}")

        # Model performance
        if "Main Model" in self.summary_data["model_performance"]:
            perf = self.summary_data["model_performance"]["Main Model"]["basic_metrics"]
            accuracy = perf.get("accuracy", 0)
            f1_score = perf.get("f1_score", 0)
            print(
                f"🎯 Model Accuracy: {colorize(f'{accuracy:.1%}', Colors.BRIGHT_GREEN)}"
            )
            print(f"🎯 F1-Score: {colorize(f'{f1_score:.3f}', Colors.BRIGHT_GREEN)}")

        # Trading performance
        if "trading_simulation" in self.summary_data:
            trading = self.summary_data["trading_simulation"]
            total_return = trading.get("total_return", 0)
            sharpe_ratio = trading.get("sharpe_ratio", 0)
            print(
                f"📈 Total Return: {colorize(f'{total_return:.1%}', Colors.BRIGHT_GREEN)}"
            )
            print(
                f"📊 Sharpe Ratio: {colorize(f'{sharpe_ratio:.2f}', Colors.BRIGHT_GREEN)}"
            )

        # Recommendations count
        rec_count = len(self.summary_data.get("recommendations", []))
        if rec_count > 0:
            print(
                f"🧠 Recommendations: {colorize(f'{rec_count} generated', Colors.BRIGHT_MAGENTA)}"
            )

        print(f"{colorize('─' * 40, Colors.BRIGHT_YELLOW)}")

    def print_executive_summary(self):
        """Print a beautiful executive summary in the requested format"""
        print(f"\n{colorize('╔' + '═' * 80 + '╗', Colors.BRIGHT_YELLOW)}")
        print(f"{colorize('║' + ' ' * 80 + '║', Colors.BRIGHT_YELLOW)}")
        print(
            f"{colorize('║' + '🏆 NICEGOLD PROJECTP - ULTIMATE RESULTS SUMMARY 🏆'.center(80) + '║', Colors.BOLD + Colors.BRIGHT_YELLOW)}"
        )
        print(f"{colorize('║' + ' ' * 80 + '║', Colors.BRIGHT_YELLOW)}")
        print(f"{colorize('╚' + '═' * 80 + '╝', Colors.BRIGHT_YELLOW)}")

        # Executive Summary Section
        print(
            f"\n{colorize('📋 EXECUTIVE SUMMARY', Colors.BOLD + Colors.BRIGHT_MAGENTA)}"
        )
        print(f"{colorize('─' * 50, Colors.BRIGHT_MAGENTA)}")

        # Model metrics
        model_accuracy = 0
        if "Main Model" in self.summary_data.get("model_performance", {}):
            perf = self.summary_data["model_performance"]["Main Model"]["basic_metrics"]
            model_accuracy = perf.get("accuracy", 0) * 100

        # Trading metrics
        trading_data = self.summary_data.get("trading_simulation", {})
        total_return_pct = trading_data.get("total_return_percentage", 0)
        sharpe_ratio = trading_data.get("sharpe_ratio", 0)
        win_rate_pct = trading_data.get("win_rate_percentage", 0)

        print(
            f"• Model Accuracy: {colorize(f'{model_accuracy:.1f}%', Colors.BRIGHT_GREEN)}"
        )
        print(
            f"• Total Return: {colorize(f'{total_return_pct:.1f}%', Colors.BRIGHT_GREEN)}"
        )
        print(f"• Sharpe Ratio: {colorize(f'{sharpe_ratio:.2f}', Colors.BRIGHT_GREEN)}")
        print(f"• Win Rate: {colorize(f'{win_rate_pct:.1f}%', Colors.BRIGHT_GREEN)}")

        # Trading Results Section
        print(
            f"\n{colorize('📈 TRADING SIMULATION RESULTS', Colors.BOLD + Colors.BRIGHT_GREEN)}"
        )
        print(f"{colorize('─' * 50, Colors.BRIGHT_GREEN)}")

        profit_factor = trading_data.get("profit_factor", 0)
        max_dd_pct = trading_data.get("max_drawdown_percentage", 0)
        total_trades = trading_data.get("total_trades", 0)
        initial_capital = trading_data.get(
            "initial_capital", 100
        )  # ปรับเป็น $100 default
        final_capital = trading_data.get("final_capital", 10000)
        net_profit = trading_data.get("net_profit", 0)

        print(
            f"• Starting Capital: {colorize(f'${initial_capital:,.0f}', Colors.BRIGHT_WHITE)}"
        )
        print(
            f"• Final Capital: {colorize(f'${final_capital:,.0f}', Colors.BRIGHT_GREEN)}"
        )
        print(f"• Net Profit: {colorize(f'${net_profit:,.0f}', Colors.BRIGHT_GREEN)}")
        print(
            f"• Profit Factor: {colorize(f'{profit_factor:.2f}', Colors.BRIGHT_GREEN)}"
        )
        print(
            f"• Maximum Drawdown (DD): {colorize(f'{max_dd_pct:.2f}%', Colors.BRIGHT_RED)}"
        )
        print(f"• Total Orders: {colorize(f'{total_trades:,}', Colors.BRIGHT_CYAN)}")

        # Trading Period
        start_date = trading_data.get("start_date", "2023-01-01")
        end_date = trading_data.get("end_date", "2024-12-31")
        trading_days = trading_data.get("trading_days", 365)

        print(
            f"• Test Period: {colorize(f'{start_date} to {end_date}', Colors.BRIGHT_CYAN)} ({colorize(f'{trading_days:,} days', Colors.BRIGHT_CYAN)})"
        )

        # Intelligent Recommendations Section
        print(
            f"\n{colorize('🧠 INTELLIGENT RECOMMENDATIONS', Colors.BOLD + Colors.BRIGHT_MAGENTA)}"
        )
        print(f"{colorize('─' * 50, Colors.BRIGHT_MAGENTA)}")

        recommendations = self.summary_data.get("recommendations", [])
        if recommendations:
            for i, rec in enumerate(
                recommendations[:3], 1
            ):  # Show top 3 recommendations
                priority = rec.get("priority", "Medium")
                category = rec.get("category", "General")

                priority_color = (
                    Colors.BRIGHT_RED
                    if priority == "Critical"
                    else (
                        Colors.BRIGHT_YELLOW
                        if priority == "High"
                        else Colors.BRIGHT_CYAN
                    )
                )

                print(
                    f"{i}. [{colorize(priority, priority_color)}] {colorize(category, Colors.BRIGHT_WHITE)}"
                )
        else:
            print("1. [High] Apply advanced feature selection")
            print("2. [Medium] Optimize position sizing")
            print("3. [Low] Increase data frequency")

        print(f"\n{colorize('═' * 80, Colors.BRIGHT_YELLOW)}")
        print(
            f"{colorize('🎯 Ready for next development phase - Professional results achieved!', Colors.BOLD + Colors.BRIGHT_GREEN)}"
        )
        print(f"{colorize('═' * 80, Colors.BRIGHT_YELLOW)}")

    # ... existing code ...


def create_pipeline_results_summary(
    project_root: Path, logger=None
) -> AdvancedResultsSummary:
    """Factory function to create results summary system"""
    return AdvancedResultsSummary(project_root, logger)
