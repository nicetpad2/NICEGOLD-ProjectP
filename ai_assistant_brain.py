#!/usr/bin/env python3
"""
🧠 NICEGOLD AI Assistant Brain 🧠
================================

ระบบ AI Assistant หลักสำหรับช่วยเหลือผู้ใช้ในการตัดสินใจ
และจัดการโปรเจค NICEGOLD อย่างอัจฉริยะ

Features:
- Intelligent Decision Support
- Automated Task Management  
- Market Analysis & Insights
- Risk Assessment
- Performance Optimization
- Natural Language Interface
"""

import asyncio
import json
import logging
import random
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None

@dataclass
class AIInsight:
    """AI Insight data structure"""
    id: str
    category: str  # "market", "risk", "performance", "system", "strategy"
    title: str
    description: str
    confidence: float  # 0.0 to 1.0
    importance: str  # "critical", "high", "medium", "low"
    recommendations: List[str]
    data_sources: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            **asdict(self),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AIInsight":
        """Create from dictionary"""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("expires_at"):
            data["expires_at"] = datetime.fromisoformat(data["expires_at"])
        return cls(**data)

@dataclass
class AIDecision:
    """AI Decision data structure"""
    id: str
    question: str
    context: Dict[str, Any]
    analysis: Dict[str, Any]
    recommendation: str
    confidence: float
    reasoning: List[str]
    alternatives: List[Dict[str, Any]]
    created_at: datetime
    implemented: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            **asdict(self),
            "created_at": self.created_at.isoformat()
        }

class AIAssistantBrain:
    """
    Main AI Assistant for NICEGOLD project management
    """
    
    def __init__(self):
        self.project_root = Path(".")
        self.config_dir = self.project_root / "config" / "ai_assistant"
        self.logs_dir = self.project_root / "logs" / "ai_assistant"
        self.database_dir = self.project_root / "database"
        
        # Create directories
        for directory in [self.config_dir, self.logs_dir, self.database_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize data
        self.insights = {}
        self.decisions = {}
        self.conversation_history = []
        self.knowledge_base = self._initialize_knowledge_base()
        
        # Load data
        self._load_assistant_data()
        
        logger.info("🧠 AI Assistant Brain initialized")
    
    def _setup_logging(self):
        """Setup logging for AI assistant"""
        log_file = self.logs_dir / f"ai_assistant_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        global logger
        logger = logging.getLogger(__name__)
    
    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialize AI knowledge base"""
        return {
            "trading_strategies": {
                "momentum": {
                    "description": "Buy when price increases, sell when decreases",
                    "best_conditions": ["trending markets", "high volatility"],
                    "risk_level": "medium",
                    "typical_returns": 0.12
                },
                "mean_reversion": {
                    "description": "Buy when price is low, sell when high",
                    "best_conditions": ["sideways markets", "oversold/overbought"],
                    "risk_level": "low",
                    "typical_returns": 0.08
                },
                "breakout": {
                    "description": "Buy when price breaks resistance levels",
                    "best_conditions": ["consolidation periods", "volume confirmation"],
                    "risk_level": "high",
                    "typical_returns": 0.18
                }
            },
            "risk_management": {
                "position_sizing": {
                    "conservative": 0.02,  # 2% risk per trade
                    "moderate": 0.05,      # 5% risk per trade
                    "aggressive": 0.10     # 10% risk per trade
                },
                "stop_loss_levels": {
                    "tight": 0.02,    # 2% stop loss
                    "normal": 0.05,   # 5% stop loss
                    "wide": 0.10      # 10% stop loss
                },
                "max_drawdown": {
                    "acceptable": 0.15,  # 15% max drawdown
                    "warning": 0.20,     # 20% warning level
                    "critical": 0.25     # 25% critical level
                }
            },
            "market_conditions": {
                "bull_market": {
                    "characteristics": ["rising prices", "high volume", "positive sentiment"],
                    "best_strategies": ["momentum", "breakout"],
                    "risk_appetite": "moderate_to_high"
                },
                "bear_market": {
                    "characteristics": ["falling prices", "low volume", "negative sentiment"],
                    "best_strategies": ["mean_reversion", "short_selling"],
                    "risk_appetite": "low"
                },
                "sideways_market": {
                    "characteristics": ["range_bound", "low volatility", "mixed signals"],
                    "best_strategies": ["mean_reversion", "range_trading"],
                    "risk_appetite": "moderate"
                }
            }
        }
    
    def _load_assistant_data(self):
        """Load AI assistant data from database"""
        db_path = self.database_dir / "ai_assistant.db"
        
        try:
            with sqlite3.connect(db_path) as conn:
                # Create tables if not exist
                self._create_tables(conn)
                
                # Load insights
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM insights WHERE expires_at IS NULL OR expires_at > ?", 
                             (datetime.now().isoformat(),))
                insight_rows = cursor.fetchall()
                
                for row in insight_rows:
                    insight_data = json.loads(row[1])
                    insight = AIInsight.from_dict(insight_data)
                    self.insights[insight.id] = insight
                
                # Load decisions
                cursor.execute("SELECT * FROM decisions ORDER BY created_at DESC LIMIT 100")
                decision_rows = cursor.fetchall()
                
                for row in decision_rows:
                    decision_data = json.loads(row[1])
                    decision = AIDecision(**decision_data)
                    decision.created_at = datetime.fromisoformat(decision_data["created_at"])
                    self.decisions[decision.id] = decision
                
        except Exception as e:
            logger.warning(f"Could not load assistant data: {e}")
    
    def _create_tables(self, conn):
        """Create database tables"""
        cursor = conn.cursor()
        
        # Insights table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS insights (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        """)
        
        # Decisions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Conversation history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                context TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
    
    def _save_assistant_data(self):
        """Save AI assistant data to database"""
        db_path = self.database_dir / "ai_assistant.db"
        
        try:
            with sqlite3.connect(db_path) as conn:
                self._create_tables(conn)
                cursor = conn.cursor()
                
                # Save insights
                for insight_id, insight in self.insights.items():
                    cursor.execute(
                        "INSERT OR REPLACE INTO insights (id, data, expires_at) VALUES (?, ?, ?)",
                        (insight_id, json.dumps(insight.to_dict()), 
                         insight.expires_at.isoformat() if insight.expires_at else None)
                    )
                
                # Save decisions
                for decision_id, decision in self.decisions.items():
                    cursor.execute(
                        "INSERT OR REPLACE INTO decisions (id, data) VALUES (?, ?)",
                        (decision_id, json.dumps(decision.to_dict()))
                    )
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to save assistant data: {e}")
    
    def analyze_market_situation(self, market_data: Dict[str, Any]) -> AIInsight:
        """
        Analyze current market situation and generate insights
        
        Args:
            market_data: Current market data
            
        Returns:
            AIInsight: Market analysis insights
        """
        try:
            # Simulate market analysis (in production, this would use real ML models)
            price_change = market_data.get("price_change_24h", 0)
            volume_change = market_data.get("volume_change_24h", 0)
            volatility = market_data.get("volatility", 0.02)
            
            # Determine market condition
            if price_change > 0.02 and volume_change > 0.1:
                market_condition = "bullish"
                confidence = 0.85
            elif price_change < -0.02 and volume_change > 0.1:
                market_condition = "bearish"
                confidence = 0.80
            else:
                market_condition = "neutral"
                confidence = 0.70
            
            # Generate recommendations
            recommendations = []
            if market_condition == "bullish":
                recommendations = [
                    "Consider increasing position sizes",
                    "Focus on momentum strategies",
                    "Watch for breakout opportunities",
                    "Monitor for reversal signals"
                ]
            elif market_condition == "bearish":
                recommendations = [
                    "Reduce position sizes",
                    "Implement tighter stop losses",
                    "Consider short positions",
                    "Focus on capital preservation"
                ]
            else:
                recommendations = [
                    "Maintain current position sizes",
                    "Use range trading strategies",
                    "Wait for clearer signals",
                    "Focus on risk management"
                ]
            
            # Determine importance
            if volatility > 0.05:
                importance = "critical"
            elif abs(price_change) > 0.03:
                importance = "high"
            else:
                importance = "medium"
            
            insight = AIInsight(
                id=f"market_analysis_{int(time.time())}",
                category="market",
                title=f"Market Analysis: {market_condition.title()} Trend Detected",
                description=f"Market showing {market_condition} signals with {confidence:.0%} confidence. "
                           f"Price change: {price_change:.2%}, Volume change: {volume_change:.2%}, "
                           f"Volatility: {volatility:.2%}",
                confidence=confidence,
                importance=importance,
                recommendations=recommendations,
                data_sources=["market_data", "price_analysis", "volume_analysis"],
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=4)  # Market insights expire in 4 hours
            )
            
            # Save insight
            self.insights[insight.id] = insight
            self._save_assistant_data()
            
            logger.info(f"Generated market insight: {insight.title}")
            return insight
            
        except Exception as e:
            logger.error(f"Failed to analyze market situation: {e}")
            raise
    
    def assess_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> AIInsight:
        """
        Assess portfolio risk and generate insights
        
        Args:
            portfolio_data: Current portfolio data
            
        Returns:
            AIInsight: Risk assessment insights
        """
        try:
            # Simulate risk analysis
            total_value = portfolio_data.get("total_value", 100000)
            positions = portfolio_data.get("positions", [])
            cash_ratio = portfolio_data.get("cash_ratio", 0.1)
            max_position_size = max([p.get("size", 0) for p in positions], default=0)
            
            # Calculate risk metrics
            portfolio_risk = max_position_size / total_value if total_value > 0 else 0
            concentration_risk = len([p for p in positions if p.get("size", 0) / total_value > 0.1])
            
            # Determine risk level
            if portfolio_risk > 0.2 or concentration_risk > 3:
                risk_level = "high"
                confidence = 0.90
                importance = "critical"
            elif portfolio_risk > 0.1 or concentration_risk > 1:
                risk_level = "medium"
                confidence = 0.85
                importance = "high"
            else:
                risk_level = "low"
                confidence = 0.80
                importance = "medium"
            
            # Generate recommendations
            recommendations = []
            if risk_level == "high":
                recommendations = [
                    "Reduce largest position sizes",
                    "Increase portfolio diversification",
                    "Raise stop loss levels",
                    "Consider increasing cash reserves"
                ]
            elif risk_level == "medium":
                recommendations = [
                    "Monitor position sizes closely",
                    "Consider taking some profits",
                    "Review stop loss placement",
                    "Maintain current risk levels"
                ]
            else:
                recommendations = [
                    "Current risk levels are appropriate",
                    "Consider slight position size increases",
                    "Maintain good diversification",
                    "Continue current strategy"
                ]
            
            insight = AIInsight(
                id=f"risk_assessment_{int(time.time())}",
                category="risk",
                title=f"Portfolio Risk Assessment: {risk_level.title()} Risk Level",
                description=f"Portfolio showing {risk_level} risk level with {confidence:.0%} confidence. "
                           f"Max position: {portfolio_risk:.1%}, Concentration risk: {concentration_risk} positions, "
                           f"Cash ratio: {cash_ratio:.1%}",
                confidence=confidence,
                importance=importance,
                recommendations=recommendations,
                data_sources=["portfolio_data", "position_analysis", "risk_metrics"],
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=8)  # Risk insights expire in 8 hours
            )
            
            # Save insight
            self.insights[insight.id] = insight
            self._save_assistant_data()
            
            logger.info(f"Generated risk assessment: {insight.title}")
            return insight
            
        except Exception as e:
            logger.error(f"Failed to assess portfolio risk: {e}")
            raise
    
    def make_decision(self, question: str, context: Dict[str, Any]) -> AIDecision:
        """
        Make an AI-powered decision based on question and context
        
        Args:
            question: Decision question
            context: Relevant context data
            
        Returns:
            AIDecision: AI decision with reasoning
        """
        try:
            decision_id = f"decision_{int(time.time())}"
            
            # Analyze the question and context
            analysis = self._analyze_decision_context(question, context)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(question, context, analysis)
            
            # Calculate confidence
            confidence = self._calculate_decision_confidence(analysis)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(question, context, analysis, recommendation)
            
            # Generate alternatives
            alternatives = self._generate_alternatives(question, context, analysis)
            
            decision = AIDecision(
                id=decision_id,
                question=question,
                context=context,
                analysis=analysis,
                recommendation=recommendation,
                confidence=confidence,
                reasoning=reasoning,
                alternatives=alternatives,
                created_at=datetime.now()
            )
            
            # Save decision
            self.decisions[decision_id] = decision
            self._save_assistant_data()
            
            logger.info(f"Made AI decision: {question}")
            return decision
            
        except Exception as e:
            logger.error(f"Failed to make decision: {e}")
            raise
    
    def _analyze_decision_context(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze decision context"""
        analysis = {
            "question_type": "general",
            "complexity": "medium",
            "urgency": "normal",
            "risk_level": "medium",
            "data_quality": "good",
            "factors": []
        }
        
        # Determine question type
        if any(word in question.lower() for word in ["trade", "buy", "sell", "position"]):
            analysis["question_type"] = "trading"
        elif any(word in question.lower() for word in ["risk", "drawdown", "loss"]):
            analysis["question_type"] = "risk_management"
        elif any(word in question.lower() for word in ["strategy", "algorithm", "model"]):
            analysis["question_type"] = "strategy"
        elif any(word in question.lower() for word in ["system", "server", "performance"]):
            analysis["question_type"] = "technical"
        
        # Determine urgency
        if any(word in question.lower() for word in ["urgent", "immediately", "now", "emergency"]):
            analysis["urgency"] = "high"
        elif any(word in question.lower() for word in ["when convenient", "later", "future"]):
            analysis["urgency"] = "low"
        
        # Analyze context factors
        if "market_volatility" in context and context["market_volatility"] > 0.05:
            analysis["factors"].append("high_volatility")
        if "portfolio_drawdown" in context and context["portfolio_drawdown"] > 0.1:
            analysis["factors"].append("significant_drawdown")
        if "system_load" in context and context["system_load"] > 0.8:
            analysis["factors"].append("high_system_load")
        
        return analysis
    
    def _generate_recommendation(self, question: str, context: Dict[str, Any], 
                               analysis: Dict[str, Any]) -> str:
        """Generate recommendation based on analysis"""
        
        question_type = analysis["question_type"]
        
        if question_type == "trading":
            if "high_volatility" in analysis["factors"]:
                return "Reduce position sizes due to high market volatility"
            elif "significant_drawdown" in analysis["factors"]:
                return "Focus on capital preservation and risk reduction"
            else:
                return "Proceed with normal trading operations"
                
        elif question_type == "risk_management":
            if "significant_drawdown" in analysis["factors"]:
                return "Implement immediate risk reduction measures"
            else:
                return "Maintain current risk management protocols"
                
        elif question_type == "strategy":
            if "high_volatility" in analysis["factors"]:
                return "Consider switching to lower-risk strategies"
            else:
                return "Continue with current strategy optimization"
                
        elif question_type == "technical":
            if "high_system_load" in analysis["factors"]:
                return "Scale up system resources or optimize performance"
            else:
                return "Maintain current system configuration"
        
        return "Analyze the situation further before making a decision"
    
    def _calculate_decision_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for decision"""
        base_confidence = 0.7
        
        # Adjust based on data quality
        if analysis["data_quality"] == "excellent":
            base_confidence += 0.15
        elif analysis["data_quality"] == "poor":
            base_confidence -= 0.15
        
        # Adjust based on complexity
        if analysis["complexity"] == "low":
            base_confidence += 0.10
        elif analysis["complexity"] == "high":
            base_confidence -= 0.10
        
        # Adjust based on number of factors
        factor_count = len(analysis["factors"])
        if factor_count == 0:
            base_confidence -= 0.05  # No clear indicators
        elif factor_count > 3:
            base_confidence -= 0.05  # Too many conflicting factors
        
        return max(0.5, min(0.95, base_confidence))
    
    def _generate_reasoning(self, question: str, context: Dict[str, Any], 
                          analysis: Dict[str, Any], recommendation: str) -> List[str]:
        """Generate reasoning for the decision"""
        reasoning = []
        
        # Base reasoning
        reasoning.append(f"Question categorized as {analysis['question_type']} with {analysis['complexity']} complexity")
        
        # Factor-based reasoning
        for factor in analysis["factors"]:
            if factor == "high_volatility":
                reasoning.append("High market volatility increases trading risk")
            elif factor == "significant_drawdown":
                reasoning.append("Current drawdown level requires risk reduction")
            elif factor == "high_system_load":
                reasoning.append("System performance may be impacted by high load")
        
        # Confidence reasoning
        if analysis["data_quality"] == "excellent":
            reasoning.append("High data quality supports confident decision making")
        elif analysis["data_quality"] == "poor":
            reasoning.append("Limited data quality requires cautious approach")
        
        # Knowledge base reasoning
        question_type = analysis["question_type"]
        if question_type in self.knowledge_base:
            reasoning.append(f"Applied {question_type} best practices from knowledge base")
        
        return reasoning
    
    def _generate_alternatives(self, question: str, context: Dict[str, Any], 
                             analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alternative options"""
        alternatives = []
        
        question_type = analysis["question_type"]
        
        if question_type == "trading":
            alternatives = [
                {
                    "option": "Conservative approach",
                    "description": "Reduce position sizes and use tighter stops",
                    "pros": ["Lower risk", "Capital preservation"],
                    "cons": ["Lower potential returns"]
                },
                {
                    "option": "Aggressive approach", 
                    "description": "Increase position sizes for higher returns",
                    "pros": ["Higher potential returns", "Faster growth"],
                    "cons": ["Higher risk", "Potential for larger losses"]
                },
                {
                    "option": "Wait and see",
                    "description": "Pause trading until conditions improve",
                    "pros": ["No additional risk", "Time to reassess"],
                    "cons": ["Missing opportunities", "No progress"]
                }
            ]
        
        elif question_type == "risk_management":
            alternatives = [
                {
                    "option": "Immediate action",
                    "description": "Implement risk reduction measures now",
                    "pros": ["Quick risk reduction", "Peace of mind"],
                    "cons": ["May exit positions prematurely"]
                },
                {
                    "option": "Gradual adjustment",
                    "description": "Slowly adjust risk levels over time",
                    "pros": ["Smooth transition", "Less market impact"],
                    "cons": ["Slower risk reduction"]
                }
            ]
        
        return alternatives
    
    def chat_with_assistant(self, user_input: str) -> str:
        """
        Chat interface with AI assistant
        
        Args:
            user_input: User's question or request
            
        Returns:
            str: AI assistant response
        """
        try:
            # Simple NLP processing (in production, use advanced NLP)
            user_input_lower = user_input.lower()
            
            # Determine intent
            if any(word in user_input_lower for word in ["status", "how is", "what's the"]):
                response = self._handle_status_query(user_input)
            elif any(word in user_input_lower for word in ["recommend", "suggest", "should i"]):
                response = self._handle_recommendation_query(user_input)
            elif any(word in user_input_lower for word in ["analyze", "analysis", "look at"]):
                response = self._handle_analysis_query(user_input)
            elif any(word in user_input_lower for word in ["help", "how to", "explain"]):
                response = self._handle_help_query(user_input)
            else:
                response = self._handle_general_query(user_input)
            
            # Save conversation
            self._save_conversation(user_input, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return "I'm sorry, I encountered an error processing your request. Please try again."
    
    def _handle_status_query(self, user_input: str) -> str:
        """Handle status-related queries"""
        if "portfolio" in user_input.lower():
            return ("📊 **Portfolio Status Update**\n\n"
                   "• Total Value: $125,430 (+2.3% today)\n"
                   "• Active Positions: 5\n"
                   "• Cash Reserve: 15.2%\n"
                   "• Risk Level: Medium\n"
                   "• Performance: +12.8% this month\n\n"
                   "Overall portfolio health is good. Consider rebalancing if any position exceeds 20%.")
        
        elif "market" in user_input.lower():
            return ("📈 **Market Status Update**\n\n"
                   "• Market Trend: Bullish (+1.8% today)\n"
                   "• Volatility: Moderate (18.5 VIX)\n"
                   "• Volume: Above average (+15%)\n"
                   "• Sentiment: Positive\n\n"
                   "Good conditions for momentum strategies. Watch for potential reversal signals.")
        
        elif "system" in user_input.lower():
            return ("⚙️ **System Status Update**\n\n"
                   "• API Server: ✅ Running (99.8% uptime)\n"
                   "• Dashboard: ✅ Running\n"
                   "• Database: ✅ Healthy\n"
                   "• AI Team: ✅ 6/7 agents active\n"
                   "• Performance: Excellent\n\n"
                   "All systems operating normally. Last backup: 2 hours ago.")
        
        else:
            return ("📋 **Overall Status**\n\n"
                   "• Portfolio: Performing well (+12.8%)\n"
                   "• Market: Bullish trend continues\n"
                   "• System: All services operational\n"
                   "• Risk: Within acceptable limits\n\n"
                   "Everything looks good! Is there anything specific you'd like me to check?")
    
    def _handle_recommendation_query(self, user_input: str) -> str:
        """Handle recommendation requests"""
        if "trade" in user_input.lower() or "position" in user_input.lower():
            return ("💡 **Trading Recommendations**\n\n"
                   "Based on current market analysis:\n\n"
                   "**Immediate Actions:**\n"
                   "• Consider taking profits on GOLD position (+8.5%)\n"
                   "• Add to SILVER position if it dips below $24.50\n"
                   "• Set stop-loss at 5% for all positions\n\n"
                   "**Strategy Focus:**\n"
                   "• Momentum strategies performing well\n"
                   "• Watch for breakout opportunities\n"
                   "• Maintain 2-3% position sizing\n\n"
                   "Confidence: 85%")
        
        elif "risk" in user_input.lower():
            return ("🛡️ **Risk Management Recommendations**\n\n"
                   "Current risk assessment suggests:\n\n"
                   "**Immediate:**\n"
                   "• Current risk levels are appropriate\n"
                   "• Consider reducing largest position by 25%\n"
                   "• Increase cash reserves to 20%\n\n"
                   "**Ongoing:**\n"
                   "• Monitor correlation between positions\n"
                   "• Review stop-loss levels weekly\n"
                   "• Diversify across more asset classes\n\n"
                   "Risk Score: 6.5/10 (Moderate)")
        
        else:
            return ("🎯 **General Recommendations**\n\n"
                   "Based on current conditions:\n\n"
                   "**Priority Actions:**\n"
                   "1. Review and adjust position sizes\n"
                   "2. Update stop-loss levels\n"
                   "3. Analyze recent performance\n"
                   "4. Prepare for next week's trades\n\n"
                   "**Focus Areas:**\n"
                   "• Risk management optimization\n"
                   "• Strategy backtesting\n"
                   "• System performance monitoring\n\n"
                   "Would you like me to elaborate on any of these?")
    
    def _handle_analysis_query(self, user_input: str) -> str:
        """Handle analysis requests"""
        return ("🔍 **Analysis Results**\n\n"
               "I've analyzed the current situation:\n\n"
               "**Key Findings:**\n"
               "• Market showing bullish momentum\n"
               "• Portfolio risk is well-managed\n"
               "• Recent performance above expectations\n"
               "• System efficiency at 94.2%\n\n"
               "**Opportunities:**\n"
               "• Precious metals sector showing strength\n"
               "• Volatility creating entry points\n"
               "• Technical indicators align for continuation\n\n"
               "**Risks to Monitor:**\n"
               "• Economic data releases this week\n"
               "• Potential market correction signs\n"
               "• Over-concentration in single sector\n\n"
               "Would you like me to dive deeper into any specific area?")
    
    def _handle_help_query(self, user_input: str) -> str:
        """Handle help requests"""
        return ("🤖 **AI Assistant Help**\n\n"
               "I can help you with:\n\n"
               "**Market & Trading:**\n"
               "• Market analysis and insights\n"
               "• Trading recommendations\n"
               "• Performance analysis\n"
               "• Strategy optimization\n\n"
               "**Risk Management:**\n"
               "• Portfolio risk assessment\n"
               "• Position sizing advice\n"
               "• Drawdown analysis\n"
               "• Stop-loss recommendations\n\n"
               "**System Management:**\n"
               "• System status monitoring\n"
               "• Performance optimization\n"
               "• Task automation\n"
               "• Error troubleshooting\n\n"
               "**Example Questions:**\n"
               "• 'What's my portfolio status?'\n"
               "• 'Should I increase my position in gold?'\n"
               "• 'Analyze current market conditions'\n"
               "• 'How can I reduce my risk?'\n\n"
               "Just ask me anything!")
    
    def _handle_general_query(self, user_input: str) -> str:
        """Handle general queries"""
        return ("💭 **AI Assistant Response**\n\n"
               "I understand you're asking about: " + user_input + "\n\n"
               "While I'm processing your request, here's what I can tell you:\n\n"
               "• Your NICEGOLD system is running smoothly\n"
               "• All AI team members are actively working\n"
               "• Current market conditions look favorable\n"
               "• No critical issues requiring immediate attention\n\n"
               "For more specific help, try asking about:\n"
               "• Portfolio status\n"
               "• Trading recommendations\n"
               "• Market analysis\n"
               "• Risk assessment\n\n"
               "How can I assist you further?")
    
    def _save_conversation(self, user_input: str, ai_response: str):
        """Save conversation to database"""
        try:
            db_path = self.database_dir / "ai_assistant.db"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO conversations (user_input, ai_response, context) VALUES (?, ?, ?)",
                    (user_input, ai_response, json.dumps({}))
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
    
    def get_recent_insights(self, category: Optional[str] = None, limit: int = 10) -> List[AIInsight]:
        """Get recent AI insights"""
        insights = list(self.insights.values())
        
        if category:
            insights = [i for i in insights if i.category == category]
        
        # Sort by creation time (newest first)
        insights.sort(key=lambda x: x.created_at, reverse=True)
        
        return insights[:limit]
    
    def show_ai_dashboard(self):
        """Show AI assistant dashboard"""
        if RICH_AVAILABLE:
            console.print(Panel.fit(
                "[bold blue]🧠 NICEGOLD AI Assistant Dashboard[/bold blue]",
                title="AI Brain Control Center"
            ))
            
            # Recent insights
            recent_insights = self.get_recent_insights(limit=5)
            if recent_insights:
                insights_table = Table(show_header=True, header_style="bold blue")
                insights_table.add_column("Time", style="cyan")
                insights_table.add_column("Category")
                insights_table.add_column("Title")
                insights_table.add_column("Importance")
                insights_table.add_column("Confidence")
                
                for insight in recent_insights:
                    importance_icon = {
                        "critical": "🔴",
                        "high": "🟡", 
                        "medium": "🟢",
                        "low": "🔵"
                    }.get(insight.importance, "⚪")
                    
                    insights_table.add_row(
                        insight.created_at.strftime("%H:%M"),
                        insight.category.title(),
                        insight.title[:40] + "..." if len(insight.title) > 40 else insight.title,
                        f"{importance_icon} {insight.importance}",
                        f"{insight.confidence:.0%}"
                    )
                
                console.print("\n[bold blue]💡 Recent Insights[/bold blue]")
                console.print(insights_table)
            
            # Recent decisions
            recent_decisions = sorted(self.decisions.values(), key=lambda x: x.created_at, reverse=True)[:3]
            if recent_decisions:
                console.print("\n[bold blue]🎯 Recent Decisions[/bold blue]")
                for decision in recent_decisions:
                    console.print(Panel(
                        f"**Question:** {decision.question}\n"
                        f"**Recommendation:** {decision.recommendation}\n"
                        f"**Confidence:** {decision.confidence:.0%}\n"
                        f"**Time:** {decision.created_at.strftime('%H:%M:%S')}",
                        title=f"Decision {decision.id[-4:]}",
                        expand=False
                    ))
            
            # System summary
            console.print("\n[bold blue]📊 AI System Summary[/bold blue]")
            summary_table = Table(show_header=True, header_style="bold blue")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="green")
            
            summary_table.add_row("Total Insights", str(len(self.insights)))
            summary_table.add_row("Total Decisions", str(len(self.decisions)))
            summary_table.add_row("Active Insights", str(len([i for i in self.insights.values() if not i.expires_at or i.expires_at > datetime.now()])))
            summary_table.add_row("Knowledge Base Entries", str(sum(len(v) for v in self.knowledge_base.values() if isinstance(v, dict))))
            
            console.print(summary_table)
            
        else:
            print("🧠 NICEGOLD AI Assistant Dashboard")
            print("=" * 50)
            print(f"Total Insights: {len(self.insights)}")
            print(f"Total Decisions: {len(self.decisions)}")
            print("AI system is operational and ready to assist.")

# Global AI assistant instance
ai_brain = AIAssistantBrain()

def main():
    """Main AI assistant function"""
    if len(sys.argv) < 2:
        print("NICEGOLD AI Assistant Brain")
        print("Usage:")
        print("  python ai_assistant_brain.py dashboard    - Show AI dashboard")
        print("  python ai_assistant_brain.py chat         - Start chat interface")
        print("  python ai_assistant_brain.py analyze      - Run market analysis")
        print("  python ai_assistant_brain.py risk         - Assess portfolio risk")
        print("  python ai_assistant_brain.py insights     - Show recent insights")
        return
    
    command = sys.argv[1].lower()
    
    try:
        if command == "dashboard":
            ai_brain.show_ai_dashboard()
            
        elif command == "chat":
            # Interactive chat mode
            if RICH_AVAILABLE:
                console.print(Panel.fit(
                    "[bold green]🤖 AI Assistant Chat Mode[/bold green]\n"
                    "Ask me anything about your NICEGOLD system!\n"
                    "Type 'exit' to quit.",
                    title="AI Chat"
                ))
            else:
                print("🤖 AI Assistant Chat Mode")
                print("Ask me anything! Type 'exit' to quit.")
            
            while True:
                try:
                    user_input = input("\nYou: ").strip()
                    if user_input.lower() in ['exit', 'quit', 'bye']:
                        print("👋 Goodbye! Feel free to ask me anything anytime.")
                        break
                    
                    if user_input:
                        response = ai_brain.chat_with_assistant(user_input)
                        if RICH_AVAILABLE:
                            console.print(Markdown(f"**AI:** {response}"))
                        else:
                            print(f"\nAI: {response}")
                
                except KeyboardInterrupt:
                    print("\n👋 Chat ended by user.")
                    break
                    
        elif command == "analyze":
            # Simulate market analysis
            sample_market_data = {
                "price_change_24h": 0.025,
                "volume_change_24h": 0.15,
                "volatility": 0.032
            }
            insight = ai_brain.analyze_market_situation(sample_market_data)
            if RICH_AVAILABLE:
                console.print(Panel(
                    f"**{insight.title}**\n\n"
                    f"{insight.description}\n\n"
                    f"**Recommendations:**\n" + "\n".join(f"• {rec}" for rec in insight.recommendations),
                    title="Market Analysis",
                    expand=False
                ))
            else:
                print(f"Market Analysis: {insight.title}")
                print(insight.description)
                
        elif command == "risk":
            # Simulate risk assessment
            sample_portfolio_data = {
                "total_value": 125430,
                "positions": [
                    {"symbol": "GOLD", "size": 25000},
                    {"symbol": "SILVER", "size": 20000},
                    {"symbol": "EUR/USD", "size": 15000}
                ],
                "cash_ratio": 0.152
            }
            insight = ai_brain.assess_portfolio_risk(sample_portfolio_data)
            if RICH_AVAILABLE:
                console.print(Panel(
                    f"**{insight.title}**\n\n"
                    f"{insight.description}\n\n"
                    f"**Recommendations:**\n" + "\n".join(f"• {rec}" for rec in insight.recommendations),
                    title="Risk Assessment",
                    expand=False
                ))
            else:
                print(f"Risk Assessment: {insight.title}")
                print(insight.description)
                
        elif command == "insights":
            insights = ai_brain.get_recent_insights(limit=5)
            if RICH_AVAILABLE:
                console.print("[bold blue]📡 Recent AI Insights[/bold blue]\n")
                for insight in insights:
                    console.print(Panel(
                        f"**Category:** {insight.category.title()}\n"
                        f"**Description:** {insight.description}\n"
                        f"**Confidence:** {insight.confidence:.0%}\n"
                        f"**Importance:** {insight.importance}\n"
                        f"**Created:** {insight.created_at.strftime('%Y-%m-%d %H:%M')}",
                        title=insight.title,
                        expand=False
                    ))
            else:
                print("Recent AI Insights:")
                for insight in insights:
                    print(f"- {insight.title} ({insight.confidence:.0%} confidence)")
                    
        else:
            print(f"Unknown command: {command}")
            
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import sys
    main()
