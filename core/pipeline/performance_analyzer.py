#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - Performance Analyzer Module
Enterprise-grade performance analysis and reporting system
"""

import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Enterprise-grade performance analysis and reporting system
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize PerformanceAnalyzer

        Args:
            config: Configuration dictionary for performance analysis
        """
        self.config = config or self._get_default_config()
        self.analysis_results = {}
        self.charts = {}
        self.reports = {}

        # Setup logging
        self._setup_logging()

        # Setup plotting style
        self._setup_plotting()

        logger.info("PerformanceAnalyzer initialized successfully")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for performance analysis"""
        return {
            "risk_free_rate": 0.02,  # 2% risk-free rate
            "benchmark_return": 0.08,  # 8% benchmark return
            "confidence_level": 0.95,  # 95% confidence level for VaR
            "rolling_window": 30,  # Days for rolling metrics
            "chart_style": "seaborn-v0_8",
            "figure_size": (12, 8),
            "save_charts": True,
            "chart_dir": "charts",
            "report_dir": "reports",
            "verbose": True,
        }

    def _setup_logging(self):
        """Setup logging for performance analyzer"""
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    def _setup_plotting(self):
        """Setup plotting configuration"""
        chart_style = self.config.get("chart_style", "seaborn-v0_8")

        # Handle backward compatibility for seaborn style
        if chart_style == "seaborn":
            chart_style = "seaborn-v0_8"

        try:
            plt.style.use(chart_style)
        except OSError:
            # Fallback to a safe default style
            plt.style.use("default")
            logger.warning(
                f"Chart style '{chart_style}' not available, using 'default'"
            )

        sns.set_palette("husl")

    def analyze_model_performance(
        self, model_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze machine learning model performance

        Args:
            model_results: Results from ModelTrainer

        Returns:
            Dictionary containing model performance analysis
        """
        try:
            logger.info("Analyzing model performance")

            if not model_results:
                raise ValueError("Model results are empty")

            analysis = {
                "model_comparison": {},
                "best_model_analysis": {},
                "feature_importance": {},
                "performance_metrics": {},
                "recommendations": [],
            }

            # Compare all models
            model_comparison = {}
            for model_name, result in model_results.items():
                metrics = result.get("metrics", {})
                model_comparison[model_name] = {
                    "test_r2": metrics.get("test_r2", 0),
                    "test_rmse": metrics.get("test_rmse", 0),
                    "test_mae": metrics.get("test_mae", 0),
                    "cv_mean": result.get("cv_mean", 0),
                    "cv_std": result.get("cv_std", 0),
                    "overfitting_score": self._calculate_overfitting_score(metrics),
                }

            analysis["model_comparison"] = model_comparison

            # Find best model
            best_model = max(model_comparison.items(), key=lambda x: x[1]["test_r2"])
            best_model_name = best_model[0]

            analysis["best_model_analysis"] = {
                "name": best_model_name,
                "metrics": best_model[1],
                "performance_grade": self._grade_performance(best_model[1]["test_r2"]),
                "reliability_score": self._calculate_reliability_score(best_model[1]),
            }

            # Generate recommendations
            recommendations = self._generate_model_recommendations(
                model_comparison, best_model[1]
            )
            analysis["recommendations"] = recommendations

            # Performance summary
            analysis["performance_metrics"] = {
                "total_models_trained": len(model_results),
                "best_r2_score": best_model[1]["test_r2"],
                "average_r2_score": np.mean(
                    [m["test_r2"] for m in model_comparison.values()]
                ),
                "model_variance": np.std(
                    [m["test_r2"] for m in model_comparison.values()]
                ),
                "performance_threshold_met": best_model[1]["test_r2"] >= 0.7,
            }

            self.analysis_results["model_performance"] = analysis

            logger.info(
                f"Model performance analysis completed - Best model: {best_model_name}"
            )

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing model performance: {str(e)}")
            raise

    def analyze_backtest_performance(
        self,
        backtest_results: Dict[str, Any],
        portfolio_values: List[float] = None,
        trades: List[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Analyze backtesting performance

        Args:
            backtest_results: Results from Backtester
            portfolio_values: Portfolio value time series
            trades: List of trade records

        Returns:
            Dictionary containing backtest performance analysis
        """
        try:
            logger.info("Analyzing backtest performance")

            if not backtest_results:
                raise ValueError("Backtest results are empty")

            analysis = {
                "return_analysis": {},
                "risk_analysis": {},
                "trade_analysis": {},
                "benchmark_comparison": {},
                "performance_attribution": {},
                "recommendations": [],
            }

            # Return analysis
            total_return = backtest_results.get("total_return", 0)
            annual_return = backtest_results.get("annual_return", 0)

            analysis["return_analysis"] = {
                "total_return": total_return,
                "annualized_return": annual_return,
                "monthly_return": annual_return / 12,
                "return_grade": self._grade_return_performance(annual_return),
                "compound_growth_rate": (1 + total_return) ** (1 / 1)
                - 1,  # Simplified for 1 year
            }

            # Risk analysis
            volatility = backtest_results.get("volatility", 0)
            max_drawdown = backtest_results.get("max_drawdown", 0)
            sharpe_ratio = backtest_results.get("sharpe_ratio", 0)

            analysis["risk_analysis"] = {
                "volatility": volatility,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "calmar_ratio": backtest_results.get("calmar_ratio", 0),
                "risk_grade": self._grade_risk_metrics(sharpe_ratio, max_drawdown),
                "risk_adjusted_return": (
                    annual_return / volatility if volatility > 0 else 0
                ),
            }

            # Trade analysis
            if trades:
                trade_analysis = self._analyze_trades(trades)
                analysis["trade_analysis"] = trade_analysis
            else:
                analysis["trade_analysis"] = {
                    "total_trades": backtest_results.get("total_trades", 0),
                    "win_rate": backtest_results.get("win_rate", 0),
                    "profit_factor": backtest_results.get("profit_factor", 0),
                    "average_win": backtest_results.get("average_win", 0),
                    "average_loss": backtest_results.get("average_loss", 0),
                }

            # Benchmark comparison
            benchmark_return = self.config.get("benchmark_return", 0.08)
            alpha = annual_return - benchmark_return

            analysis["benchmark_comparison"] = {
                "benchmark_return": benchmark_return,
                "strategy_return": annual_return,
                "alpha": alpha,
                "outperformance": alpha > 0,
                "relative_performance": (
                    annual_return / benchmark_return if benchmark_return > 0 else 0
                ),
            }

            # Performance attribution
            if portfolio_values:
                attribution = self._calculate_performance_attribution(portfolio_values)
                analysis["performance_attribution"] = attribution

            # Generate recommendations
            recommendations = self._generate_backtest_recommendations(analysis)
            analysis["recommendations"] = recommendations

            self.analysis_results["backtest_performance"] = analysis

            logger.info(f"Backtest performance analysis completed - Alpha: {alpha:.2%}")

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing backtest performance: {str(e)}")
            raise

    def _calculate_overfitting_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overfitting score based on train/test performance gap"""
        train_r2 = metrics.get("train_r2", 0)
        test_r2 = metrics.get("test_r2", 0)

        if train_r2 == 0:
            return 0

        gap = (train_r2 - test_r2) / train_r2
        return max(0, min(1, gap))  # Normalize to 0-1

    def _calculate_reliability_score(self, metrics: Dict[str, float]) -> float:
        """Calculate model reliability score"""
        test_r2 = metrics.get("test_r2", 0)
        cv_std = metrics.get("cv_std", 0)
        overfitting = metrics.get("overfitting_score", 0)

        # Reliability decreases with high CV std and overfitting
        reliability = test_r2 * (1 - cv_std) * (1 - overfitting)
        return max(0, min(1, reliability))

    def _grade_performance(self, r2_score: float) -> str:
        """Grade model performance based on R2 score"""
        if r2_score >= 0.9:
            return "Excellent"
        elif r2_score >= 0.8:
            return "Very Good"
        elif r2_score >= 0.7:
            return "Good"
        elif r2_score >= 0.6:
            return "Fair"
        else:
            return "Poor"

    def _grade_return_performance(self, annual_return: float) -> str:
        """Grade return performance"""
        if annual_return >= 0.20:
            return "Excellent"
        elif annual_return >= 0.15:
            return "Very Good"
        elif annual_return >= 0.10:
            return "Good"
        elif annual_return >= 0.05:
            return "Fair"
        else:
            return "Poor"

    def _grade_risk_metrics(self, sharpe_ratio: float, max_drawdown: float) -> str:
        """Grade risk metrics"""
        if sharpe_ratio >= 2.0 and max_drawdown <= 0.05:
            return "Excellent"
        elif sharpe_ratio >= 1.5 and max_drawdown <= 0.10:
            return "Very Good"
        elif sharpe_ratio >= 1.0 and max_drawdown <= 0.15:
            return "Good"
        elif sharpe_ratio >= 0.5 and max_drawdown <= 0.25:
            return "Fair"
        else:
            return "Poor"

    def _analyze_trades(self, trades: List[Dict]) -> Dict[str, Any]:
        """Analyze individual trades in detail"""
        closed_trades = [t for t in trades if t.get("type") == "close"]

        if not closed_trades:
            return {}

        pnls = [t.get("pnl", 0) for t in closed_trades]
        returns = [t.get("return", 0) for t in closed_trades]
        durations = [t.get("days_held", 0) for t in closed_trades]

        winning_trades = [t for t in closed_trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in closed_trades if t.get("pnl", 0) < 0]

        analysis = {
            "total_trades": len(closed_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": (
                len(winning_trades) / len(closed_trades) if closed_trades else 0
            ),
            "total_pnl": sum(pnls),
            "average_pnl": np.mean(pnls),
            "median_pnl": np.median(pnls),
            "pnl_std": np.std(pnls),
            "best_trade": max(pnls) if pnls else 0,
            "worst_trade": min(pnls) if pnls else 0,
            "average_return": np.mean(returns),
            "return_volatility": np.std(returns),
            "average_duration": np.mean(durations),
            "profit_factor": (
                abs(
                    sum(t.get("pnl", 0) for t in winning_trades)
                    / sum(t.get("pnl", 0) for t in losing_trades)
                )
                if losing_trades
                else float("inf")
            ),
        }

        return analysis

    def _calculate_performance_attribution(
        self, portfolio_values: List[float]
    ) -> Dict[str, Any]:
        """Calculate performance attribution metrics"""
        if len(portfolio_values) < 2:
            return {}

        returns = pd.Series(portfolio_values).pct_change().dropna()

        attribution = {
            "total_periods": len(portfolio_values),
            "positive_periods": len(returns[returns > 0]),
            "negative_periods": len(returns[returns < 0]),
            "average_positive_return": (
                returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
            ),
            "average_negative_return": (
                returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
            ),
            "return_skewness": returns.skew(),
            "return_kurtosis": returns.kurtosis(),
            "hit_ratio": (
                len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
            ),
        }

        return attribution

    def _generate_model_recommendations(
        self, model_comparison: Dict[str, Any], best_model_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for model improvement"""
        recommendations = []

        best_r2 = best_model_metrics["test_r2"]
        overfitting = best_model_metrics["overfitting_score"]
        cv_std = best_model_metrics["cv_std"]

        if best_r2 < 0.7:
            recommendations.append(
                "Consider feature engineering or additional data sources to improve model performance"
            )

        if overfitting > 0.2:
            recommendations.append(
                "High overfitting detected - consider regularization or simpler models"
            )

        if cv_std > 0.1:
            recommendations.append(
                "High cross-validation variance - consider ensemble methods or more data"
            )

        if best_r2 > 0.8:
            recommendations.append(
                "Excellent model performance - ready for production deployment"
            )

        # Model diversity recommendations
        r2_scores = [m["test_r2"] for m in model_comparison.values()]
        if np.std(r2_scores) < 0.05:
            recommendations.append(
                "Low model diversity - consider different algorithm types"
            )

        return recommendations

    def _generate_backtest_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for strategy improvement"""
        recommendations = []

        return_analysis = analysis.get("return_analysis", {})
        risk_analysis = analysis.get("risk_analysis", {})
        trade_analysis = analysis.get("trade_analysis", {})
        benchmark_comparison = analysis.get("benchmark_comparison", {})

        annual_return = return_analysis.get("annualized_return", 0)
        sharpe_ratio = risk_analysis.get("sharpe_ratio", 0)
        max_drawdown = risk_analysis.get("max_drawdown", 0)
        win_rate = trade_analysis.get("win_rate", 0)
        alpha = benchmark_comparison.get("alpha", 0)

        if alpha < 0:
            recommendations.append(
                "Strategy underperforming benchmark - consider parameter optimization"
            )

        if sharpe_ratio < 1.0:
            recommendations.append(
                "Low risk-adjusted returns - review position sizing and risk management"
            )

        if max_drawdown > 0.2:
            recommendations.append(
                "High maximum drawdown - implement stricter stop-loss mechanisms"
            )

        if win_rate < 0.4:
            recommendations.append(
                "Low win rate - review entry/exit signals and market conditions"
            )

        if win_rate > 0.7 and annual_return < 0.1:
            recommendations.append(
                "High win rate but low returns - consider increasing position sizes"
            )

        if annual_return > 0.15 and sharpe_ratio > 1.5:
            recommendations.append(
                "Excellent strategy performance - consider scaling up capital allocation"
            )

        return recommendations

    def create_performance_charts(
        self,
        model_results: Dict[str, Any] = None,
        backtest_results: Dict[str, Any] = None,
        portfolio_values: List[float] = None,
    ) -> Dict[str, str]:
        """Create comprehensive performance charts"""
        try:
            logger.info("Creating performance charts")

            chart_paths = {}

            # Model performance charts
            if model_results:
                chart_paths.update(self._create_model_charts(model_results))

            # Backtest performance charts
            if backtest_results:
                chart_paths.update(
                    self._create_backtest_charts(backtest_results, portfolio_values)
                )

            self.charts = chart_paths

            logger.info(f"Created {len(chart_paths)} performance charts")

            return chart_paths

        except Exception as e:
            logger.error(f"Error creating charts: {str(e)}")
            return {}

    def _create_model_charts(self, model_results: Dict[str, Any]) -> Dict[str, str]:
        """Create model performance charts"""
        chart_paths = {}

        try:
            # Model comparison chart
            fig, ax = plt.subplots(figsize=self.config.get("figure_size", (12, 8)))

            models = list(model_results.keys())
            r2_scores = [model_results[m]["metrics"]["test_r2"] for m in models]

            bars = ax.bar(models, r2_scores, color="skyblue", alpha=0.7)
            ax.set_title(
                "Model Performance Comparison (Test R²)", fontsize=16, fontweight="bold"
            )
            ax.set_ylabel("R² Score", fontsize=12)
            ax.set_xlabel("Models", fontsize=12)
            ax.set_ylim(0, 1)

            # Add value labels on bars
            for bar, score in zip(bars, r2_scores):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{score:.3f}",
                    ha="center",
                    va="bottom",
                )

            plt.xticks(rotation=45)
            plt.tight_layout()

            if self.config.get("save_charts", True):
                chart_path = f"{self.config.get('chart_dir', 'charts')}/model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(chart_path, dpi=300, bbox_inches="tight")
                chart_paths["model_comparison"] = chart_path

            plt.close()

        except Exception as e:
            logger.error(f"Error creating model charts: {str(e)}")

        return chart_paths

    def _create_backtest_charts(
        self, backtest_results: Dict[str, Any], portfolio_values: List[float] = None
    ) -> Dict[str, str]:
        """Create backtest performance charts"""
        chart_paths = {}

        try:
            if portfolio_values:
                # Portfolio value chart
                fig, ax = plt.subplots(figsize=self.config.get("figure_size", (12, 8)))

                dates = pd.date_range(
                    start="2020-01-01", periods=len(portfolio_values), freq="D"
                )
                ax.plot(
                    dates,
                    portfolio_values,
                    color="blue",
                    linewidth=2,
                    label="Portfolio Value",
                )

                # Add benchmark line
                initial_value = portfolio_values[0]
                benchmark_values = [
                    initial_value
                    * (1 + self.config.get("benchmark_return", 0.08) * i / 252)
                    for i in range(len(portfolio_values))
                ]
                ax.plot(
                    dates,
                    benchmark_values,
                    color="red",
                    linewidth=1,
                    linestyle="--",
                    label="Benchmark",
                )

                ax.set_title(
                    "Portfolio Performance vs Benchmark", fontsize=16, fontweight="bold"
                )
                ax.set_ylabel("Portfolio Value ($)", fontsize=12)
                ax.set_xlabel("Date", fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)

                plt.xticks(rotation=45)
                plt.tight_layout()

                if self.config.get("save_charts", True):
                    chart_path = f"{self.config.get('chart_dir', 'charts')}/portfolio_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
                    chart_paths["portfolio_performance"] = chart_path

                plt.close()

        except Exception as e:
            logger.error(f"Error creating backtest charts: {str(e)}")

        return chart_paths

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            logger.info("Generating comprehensive performance report")

            report = {
                "generated_at": datetime.now().isoformat(),
                "summary": {},
                "model_analysis": self.analysis_results.get("model_performance", {}),
                "backtest_analysis": self.analysis_results.get(
                    "backtest_performance", {}
                ),
                "charts": self.charts,
                "recommendations": [],
                "overall_grade": "N/A",
            }

            # Generate summary
            summary = self._generate_summary()
            report["summary"] = summary

            # Compile all recommendations
            all_recommendations = []
            if "model_performance" in self.analysis_results:
                all_recommendations.extend(
                    self.analysis_results["model_performance"].get(
                        "recommendations", []
                    )
                )
            if "backtest_performance" in self.analysis_results:
                all_recommendations.extend(
                    self.analysis_results["backtest_performance"].get(
                        "recommendations", []
                    )
                )

            report["recommendations"] = all_recommendations

            # Calculate overall grade
            overall_grade = self._calculate_overall_grade()
            report["overall_grade"] = overall_grade

            self.reports["comprehensive"] = report

            # Save report to file
            if self.config.get("save_reports", True):
                self._save_report(report)

            logger.info(
                f"Comprehensive report generated - Overall grade: {overall_grade}"
            )

            return report

        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return {}

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate performance summary"""
        summary = {
            "analysis_completed": len(self.analysis_results) > 0,
            "total_recommendations": 0,
            "key_metrics": {},
        }

        # Model performance summary
        if "model_performance" in self.analysis_results:
            model_perf = self.analysis_results["model_performance"]
            summary["key_metrics"]["best_model_r2"] = (
                model_perf.get("best_model_analysis", {})
                .get("metrics", {})
                .get("test_r2", 0)
            )
            summary["key_metrics"]["model_grade"] = model_perf.get(
                "best_model_analysis", {}
            ).get("performance_grade", "N/A")

        # Backtest performance summary
        if "backtest_performance" in self.analysis_results:
            backtest_perf = self.analysis_results["backtest_performance"]
            summary["key_metrics"]["annual_return"] = backtest_perf.get(
                "return_analysis", {}
            ).get("annualized_return", 0)
            summary["key_metrics"]["sharpe_ratio"] = backtest_perf.get(
                "risk_analysis", {}
            ).get("sharpe_ratio", 0)
            summary["key_metrics"]["max_drawdown"] = backtest_perf.get(
                "risk_analysis", {}
            ).get("max_drawdown", 0)
            summary["key_metrics"]["alpha"] = backtest_perf.get(
                "benchmark_comparison", {}
            ).get("alpha", 0)

        return summary

    def _calculate_overall_grade(self) -> str:
        """Calculate overall performance grade"""
        scores = []

        # Model score
        if "model_performance" in self.analysis_results:
            model_r2 = (
                self.analysis_results["model_performance"]
                .get("best_model_analysis", {})
                .get("metrics", {})
                .get("test_r2", 0)
            )
            scores.append(model_r2)

        # Backtest score (normalized)
        if "backtest_performance" in self.analysis_results:
            alpha = (
                self.analysis_results["backtest_performance"]
                .get("benchmark_comparison", {})
                .get("alpha", 0)
            )
            sharpe = (
                self.analysis_results["backtest_performance"]
                .get("risk_analysis", {})
                .get("sharpe_ratio", 0)
            )

            # Normalize and combine metrics
            alpha_score = min(1, max(0, (alpha + 0.1) / 0.2))  # -10% to +10% -> 0 to 1
            sharpe_score = min(1, max(0, sharpe / 3))  # 0 to 3 -> 0 to 1

            backtest_score = (alpha_score + sharpe_score) / 2
            scores.append(backtest_score)

        if not scores:
            return "N/A"

        overall_score = np.mean(scores)

        if overall_score >= 0.9:
            return "A+"
        elif overall_score >= 0.8:
            return "A"
        elif overall_score >= 0.7:
            return "B+"
        elif overall_score >= 0.6:
            return "B"
        elif overall_score >= 0.5:
            return "C+"
        elif overall_score >= 0.4:
            return "C"
        else:
            return "D"

    def _save_report(self, report: Dict[str, Any]):
        """Save report to file"""
        try:
            import os

            os.makedirs(self.config.get("report_dir", "reports"), exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.get('report_dir', 'reports')}/performance_report_{timestamp}.json"

            with open(filename, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Report saved to {filename}")

        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of performance analysis"""
        summary = {
            "analysis_completed": len(self.analysis_results) > 0,
            "charts_created": len(self.charts),
            "reports_generated": len(self.reports),
            "config": self.config,
        }

        if self.analysis_results:
            summary["latest_analysis"] = max(self.analysis_results.keys())

        return summary
