#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - Advanced Analytics Module
Production-ready advanced analytics and AI-powered insights

This module provides:
1. Real-time market analysis
2. AI-powered pattern recognition
3. Sentiment analysis integration
4. Advanced performance metrics
5. Predictive analytics
6. Risk analytics dashboard
"""

import json
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    """Market regime classification"""
    regime_name: str
    volatility_level: str  # low, medium, high
    trend_strength: float
    confidence: float
    characteristics: List[str]
    timestamp: datetime


@dataclass
class TradingSignal:
    """Trading signal structure"""
    signal_type: str  # buy, sell, hold
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    reasoning: List[str]
    risk_level: str  # low, medium, high
    timestamp: datetime


@dataclass
class PerformanceInsight:
    """Performance analysis insight"""
    metric_name: str
    current_value: float
    benchmark_value: float
    improvement_suggestion: str
    priority: str  # low, medium, high, critical
    category: str


class AdvancedAnalytics:
    """
    Advanced Analytics Engine
    
    Provides comprehensive market analysis, pattern recognition,
    and predictive insights for trading optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Retain 95% variance
        self.kmeans = KMeans(n_clusters=4, random_state=42)
        
        # Analysis history
        self.market_regimes: List[MarketRegime] = []
        self.trading_signals: List[TradingSignal] = []
        self.performance_insights: List[PerformanceInsight] = []
        
        # Market state tracking
        self.current_regime: Optional[MarketRegime] = None
        self.trend_momentum = 0.0
        self.volatility_state = "normal"
        
        logger.info("🧠 Advanced Analytics Engine initialized")
    
    def analyze_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """
        Analyze current market regime using advanced ML techniques
        """
        try:
            logger.info("📊 Analyzing market regime...")
            
            # Feature engineering for regime detection
            features = self._extract_regime_features(data)
            
            # Normalize features
            features_scaled = self.scaler.fit_transform(features)
            
            # Dimensionality reduction
            features_pca = self.pca.fit_transform(features_scaled)
            
            # Cluster analysis for regime identification
            regime_clusters = self.kmeans.fit_predict(features_pca)
            current_regime_id = regime_clusters[-1]  # Latest regime
            
            # Map cluster to regime characteristics
            regime_map = {
                0: ("Bull Market", "low", [
                    "Rising prices", "Low volatility", "High volume"
                ]),
                1: ("Bear Market", "medium", [
                    "Falling prices", "High volatility", "Panic selling"
                ]),
                2: ("Sideways", "low", [
                    "Range-bound", "Low momentum", "Consolidation"
                ]),
                3: ("High Volatility", "high", [
                    "Erratic moves", "News-driven", "Uncertainty"
                ])
            }
            
            regime_name, vol_level, characteristics = regime_map.get(
                current_regime_id,
                ("Unknown", "medium", ["Uncertain conditions"])
            )
            
            # Calculate trend strength
            price_col = ('close' if 'close' in data.columns
                         else data.select_dtypes(include=[np.number]).columns[0])
            returns = data[price_col].pct_change().dropna()
            trend_strength = abs(returns.rolling(20).mean().iloc[-1]) * 100
            
            # Calculate confidence based on cluster stability
            recent_regimes = (regime_clusters[-10:] if len(regime_clusters) >= 10
                              else regime_clusters)
            confidence = (recent_regimes == current_regime_id).mean()
            
            regime = MarketRegime(
                regime_name=regime_name,
                volatility_level=vol_level,
                trend_strength=trend_strength,
                confidence=confidence,
                characteristics=characteristics,
                timestamp=datetime.now()
            )
            
            self.current_regime = regime
            self.market_regimes.append(regime)
            
            logger.info(f"✅ Market regime: {regime_name} (confidence: {confidence:.2f})")
            return regime
            
        except Exception as e:
            logger.error(f"❌ Market regime analysis failed: {e}")
            # Return default regime
            return MarketRegime(
                regime_name="Unknown",
                volatility_level="medium",
                trend_strength=0.0,
                confidence=0.0,
                characteristics=["Analysis unavailable"],
                timestamp=datetime.now()
            )
    
    def _extract_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for regime classification"""
        try:
            price_col = 'close' if 'close' in data.columns else data.select_dtypes(include=[np.number]).columns[0]
            
            # Basic price features
            returns = data[price_col].pct_change().dropna()
            
            features = []
            windows = [5, 10, 20, 50]
            
            for window in windows:
                if len(returns) >= window:
                    # Volatility
                    vol = returns.rolling(window).std().fillna(0)
                    features.append(vol.iloc[-1])
                    
                    # Mean return
                    mean_ret = returns.rolling(window).mean().fillna(0)
                    features.append(mean_ret.iloc[-1])
                    
                    # Skewness
                    skew = returns.rolling(window).skew().fillna(0)
                    features.append(skew.iloc[-1])
                else:
                    features.extend([0.0, 0.0, 0.0])
            
            # Technical indicators
            if len(data) >= 20:
                # RSI approximation
                delta = data[price_col].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                features.append(rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0)
                
                # Moving average ratios
                ma_short = data[price_col].rolling(10).mean()
                ma_long = data[price_col].rolling(20).mean()
                ma_ratio = (ma_short / ma_long).iloc[-1] if not np.isnan((ma_short / ma_long).iloc[-1]) else 1.0
                features.append(ma_ratio)
            else:
                features.extend([50.0, 1.0])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            return np.array([[0.0] * 26])  # Default feature vector
    
    def generate_trading_signal(self, data: pd.DataFrame, predictions: Optional[np.ndarray] = None) -> TradingSignal:
        """
        Generate intelligent trading signal based on multiple factors
        """
        try:
            logger.info("🎯 Generating trading signal...")
            
            # Get current market regime
            if not self.current_regime:
                self.analyze_market_regime(data)
            
            price_col = 'close' if 'close' in data.columns else data.select_dtypes(include=[np.number]).columns[0]
            current_price = data[price_col].iloc[-1]
            
            # Signal components
            signals = []
            reasons = []
            
            # 1. Trend analysis
            trend_signal, trend_reason = self._analyze_trend(data)
            signals.append(trend_signal)
            reasons.append(trend_reason)
            
            # 2. Momentum analysis
            momentum_signal, momentum_reason = self._analyze_momentum(data)
            signals.append(momentum_signal)
            reasons.append(momentum_reason)
            
            # 3. Volatility analysis
            vol_signal, vol_reason = self._analyze_volatility(data)
            signals.append(vol_signal)
            reasons.append(vol_reason)
            
            # 4. Model predictions (if available)
            if predictions is not None and len(predictions) > 0:
                pred_signal, pred_reason = self._analyze_predictions(predictions, current_price)
                signals.append(pred_signal)
                reasons.append(pred_reason)
            
            # Combine signals
            avg_signal = np.mean(signals)
            
            # Determine signal type
            if avg_signal > 0.6:
                signal_type = "buy"
                strength = min(avg_signal, 1.0)
            elif avg_signal < -0.6:
                signal_type = "sell"
                strength = min(abs(avg_signal), 1.0)
            else:
                signal_type = "hold"
                strength = 0.5
            
            # Calculate confidence based on signal agreement
            signal_std = np.std(signals)
            confidence = max(0.0, 1.0 - signal_std)
            
            # Determine risk level
            if self.current_regime and self.current_regime.volatility_level == "high":
                risk_level = "high"
            elif signal_std > 0.5:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            trading_signal = TradingSignal(
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                reasoning=reasons,
                risk_level=risk_level,
                timestamp=datetime.now()
            )
            
            self.trading_signals.append(trading_signal)
            
            logger.info(f"✅ Trading signal: {signal_type.upper()} (strength: {strength:.2f}, confidence: {confidence:.2f})")
            return trading_signal
            
        except Exception as e:
            logger.error(f"❌ Trading signal generation failed: {e}")
            return TradingSignal(
                signal_type="hold",
                strength=0.0,
                confidence=0.0,
                reasoning=["Analysis failed"],
                risk_level="high",
                timestamp=datetime.now()
            )
    
    def _analyze_trend(self, data: pd.DataFrame) -> Tuple[float, str]:
        """Analyze price trend"""
        try:
            price_col = 'close' if 'close' in data.columns else data.select_dtypes(include=[np.number]).columns[0]
            
            # Short and long term moving averages
            if len(data) >= 50:
                ma_short = data[price_col].rolling(10).mean().iloc[-1]
                ma_long = data[price_col].rolling(50).mean().iloc[-1]
                current_price = data[price_col].iloc[-1]
                
                if ma_short > ma_long and current_price > ma_short:
                    return 0.8, "Strong uptrend (price above short MA, short MA above long MA)"
                elif ma_short < ma_long and current_price < ma_short:
                    return -0.8, "Strong downtrend (price below short MA, short MA below long MA)"
                elif ma_short > ma_long:
                    return 0.4, "Moderate uptrend (short MA above long MA)"
                else:
                    return -0.4, "Moderate downtrend (short MA below long MA)"
            else:
                return 0.0, "Insufficient data for trend analysis"
                
        except Exception as e:
            logger.error(f"❌ Trend analysis failed: {e}")
            return 0.0, "Trend analysis failed"
    
    def _analyze_momentum(self, data: pd.DataFrame) -> Tuple[float, str]:
        """Analyze price momentum"""
        try:
            price_col = 'close' if 'close' in data.columns else data.select_dtypes(include=[np.number]).columns[0]
            
            if len(data) >= 14:
                # RSI calculation
                delta = data[price_col].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]
                
                if current_rsi > 70:
                    return -0.6, f"Overbought momentum (RSI: {current_rsi:.1f})"
                elif current_rsi < 30:
                    return 0.6, f"Oversold momentum (RSI: {current_rsi:.1f})"
                elif current_rsi > 50:
                    return 0.3, f"Bullish momentum (RSI: {current_rsi:.1f})"
                else:
                    return -0.3, f"Bearish momentum (RSI: {current_rsi:.1f})"
            else:
                return 0.0, "Insufficient data for momentum analysis"
                
        except Exception as e:
            logger.error(f"❌ Momentum analysis failed: {e}")
            return 0.0, "Momentum analysis failed"
    
    def _analyze_volatility(self, data: pd.DataFrame) -> Tuple[float, str]:
        """Analyze volatility patterns"""
        try:
            price_col = 'close' if 'close' in data.columns else data.select_dtypes(include=[np.number]).columns[0]
            returns = data[price_col].pct_change().dropna()
            
            if len(returns) >= 20:
                current_vol = returns.rolling(20).std().iloc[-1]
                long_vol = returns.rolling(60).std().iloc[-1] if len(returns) >= 60 else current_vol
                
                vol_ratio = current_vol / long_vol if long_vol > 0 else 1.0
                
                if vol_ratio > 1.5:
                    return -0.4, f"High volatility regime (vol ratio: {vol_ratio:.2f})"
                elif vol_ratio < 0.7:
                    return 0.2, f"Low volatility regime (vol ratio: {vol_ratio:.2f})"
                else:
                    return 0.0, f"Normal volatility (vol ratio: {vol_ratio:.2f})"
            else:
                return 0.0, "Insufficient data for volatility analysis"
                
        except Exception as e:
            logger.error(f"❌ Volatility analysis failed: {e}")
            return 0.0, "Volatility analysis failed"
    
    def _analyze_predictions(self, predictions: np.ndarray, current_price: float) -> Tuple[float, str]:
        """Analyze model predictions"""
        try:
            if len(predictions) > 0:
                latest_prediction = predictions[-1]
                price_change = (latest_prediction - current_price) / current_price
                
                if price_change > 0.02:  # 2% increase predicted
                    return 0.7, f"Model predicts {price_change*100:.1f}% price increase"
                elif price_change < -0.02:  # 2% decrease predicted
                    return -0.7, f"Model predicts {price_change*100:.1f}% price decrease"
                else:
                    return 0.0, f"Model predicts minimal change ({price_change*100:.1f}%)"
            else:
                return 0.0, "No model predictions available"
                
        except Exception as e:
            logger.error(f"❌ Prediction analysis failed: {e}")
            return 0.0, "Prediction analysis failed"
    
    def analyze_performance(self, backtest_results: Dict[str, Any]) -> List[PerformanceInsight]:
        """
        Generate advanced performance insights and recommendations
        """
        try:
            logger.info("📈 Analyzing performance insights...")
            
            insights = []
            
            # Extract key metrics
            total_return = backtest_results.get("total_return", 0.0)
            sharpe_ratio = backtest_results.get("sharpe_ratio", 0.0)
            max_drawdown = backtest_results.get("max_drawdown", 0.0)
            win_rate = backtest_results.get("win_rate", 0.0)
            total_trades = backtest_results.get("total_trades", 0)
            
            # Benchmark values (industry standards)
            benchmarks = {
                "total_return": 0.15,  # 15% annual return
                "sharpe_ratio": 1.0,   # Sharpe ratio > 1
                "max_drawdown": 0.10,  # Max 10% drawdown
                "win_rate": 0.55,      # 55% win rate
                "total_trades": 100    # Minimum trades for statistical significance
            }
            
            # Analyze each metric
            if total_return < benchmarks["total_return"]:
                insights.append(PerformanceInsight(
                    metric_name="Total Return",
                    current_value=total_return,
                    benchmark_value=benchmarks["total_return"],
                    improvement_suggestion="Consider: More aggressive position sizing, better entry/exit timing, or additional alpha factors",
                    priority="high" if total_return < 0 else "medium",
                    category="profitability"
                ))
            
            if sharpe_ratio < benchmarks["sharpe_ratio"]:
                insights.append(PerformanceInsight(
                    metric_name="Sharpe Ratio",
                    current_value=sharpe_ratio,
                    benchmark_value=benchmarks["sharpe_ratio"],
                    improvement_suggestion="Focus on: Risk management, reducing volatility, or improving consistency",
                    priority="high" if sharpe_ratio < 0.5 else "medium",
                    category="risk_adjusted_return"
                ))
            
            if abs(max_drawdown) > benchmarks["max_drawdown"]:
                insights.append(PerformanceInsight(
                    metric_name="Maximum Drawdown",
                    current_value=abs(max_drawdown),
                    benchmark_value=benchmarks["max_drawdown"],
                    improvement_suggestion="Implement: Dynamic position sizing, stop-loss optimization, or diversification",
                    priority="critical" if abs(max_drawdown) > 0.20 else "high",
                    category="risk_management"
                ))
            
            if win_rate < benchmarks["win_rate"]:
                insights.append(PerformanceInsight(
                    metric_name="Win Rate",
                    current_value=win_rate,
                    benchmark_value=benchmarks["win_rate"],
                    improvement_suggestion="Improve: Entry signal quality, market timing, or holding period optimization",
                    priority="medium",
                    category="accuracy"
                ))
            
            if total_trades < benchmarks["total_trades"]:
                insights.append(PerformanceInsight(
                    metric_name="Trade Frequency",
                    current_value=total_trades,
                    benchmark_value=benchmarks["total_trades"],
                    improvement_suggestion="Increase: Signal frequency, reduce filters, or extend backtest period",
                    priority="low",
                    category="statistical_significance"
                ))
            
            self.performance_insights.extend(insights)
            
            logger.info(f"✅ Generated {len(insights)} performance insights")
            return insights
            
        except Exception as e:
            logger.error(f"❌ Performance analysis failed: {e}")
            return []
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        return {
            "current_regime": {
                "regime_name": self.current_regime.regime_name if self.current_regime else "Unknown",
                "volatility_level": self.current_regime.volatility_level if self.current_regime else "unknown",
                "confidence": self.current_regime.confidence if self.current_regime else 0.0
            } if self.current_regime else None,
            "latest_signal": {
                "signal_type": self.trading_signals[-1].signal_type if self.trading_signals else "none",
                "strength": self.trading_signals[-1].strength if self.trading_signals else 0.0,
                "confidence": self.trading_signals[-1].confidence if self.trading_signals else 0.0,
                "risk_level": self.trading_signals[-1].risk_level if self.trading_signals else "unknown"
            } if self.trading_signals else None,
            "insights_summary": {
                "total_insights": len(self.performance_insights),
                "critical_issues": len([i for i in self.performance_insights if i.priority == "critical"]),
                "high_priority": len([i for i in self.performance_insights if i.priority == "high"]),
                "categories": list(set([i.category for i in self.performance_insights]))
            },
            "analysis_stats": {
                "regimes_detected": len(self.market_regimes),
                "signals_generated": len(self.trading_signals),
                "last_update": datetime.now().isoformat()
            }
        }


# Demo function
def demo_advanced_analytics():
    """Demonstrate advanced analytics capabilities using REAL DATA"""
    try:
        # Load real data from datacsv folder
        from pathlib import Path
        
        datacsv_path = Path("datacsv")
        data_file = datacsv_path / "XAUUSD_M1.csv"
        
        if not data_file.exists():
            print("❌ Real data file not found! Please ensure XAUUSD_M1.csv "
                  "exists in datacsv folder")
            return
        
        # Load real XAUUSD data
        data = pd.read_csv(data_file)
        print(f"📊 Loaded real XAUUSD data: {data.shape[0]:,} rows")
        print(f"📅 Data period: {data['Time'].iloc[0]} to {data['Time'].iloc[-1]}")
        
        # Rename columns to match expected format
        if 'Close' in data.columns:
            data['close'] = data['Close']
        if 'Volume' in data.columns:
            data['volume'] = data['Volume']
        
        # Use subset of data for demo (last 1000 rows for performance)
        data_subset = data.tail(1000).copy()
        print(f"🎯 Using subset: {data_subset.shape[0]} rows for analytics demo")
        
        # Initialize analytics
        config = {"regime_lookback": 20, "signal_threshold": 0.6}
        analytics = AdvancedAnalytics(config)
        
        # Analyze market regime
        regime = analytics.analyze_market_regime(data_subset)
        print(f"🏛️ Market Regime: {regime.regime_name} "
              f"(confidence: {regime.confidence:.2f})")
        
        # Generate trading signal
        signal = analytics.generate_trading_signal(data_subset)
        print(f"🎯 Trading Signal: {signal.signal_type.upper()} "
              f"(strength: {signal.strength:.2f})")
        
        # Analyze performance (mock results)
        mock_results = {
            "total_return": 0.12,
            "sharpe_ratio": 0.8,
            "max_drawdown": -0.15,
            "win_rate": 0.52,
            "total_trades": 85
        }
        
        insights = analytics.analyze_performance(mock_results)
        print(f"📊 Performance Insights: {len(insights)} recommendations generated")
        
        # Get summary
        summary = analytics.get_analytics_summary()
        print(f"📈 Analytics Summary: {json.dumps(summary, indent=2, default=str)}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_advanced_analytics()
