"""
Advanced Risk Management และ Position Management System
จัดการความเสี่ยงและตำแหน่งการลงทุนแบบ real-time
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import redis.asyncio as redis
from sqlalchemy import create_engine, Column, String, DateTime, Float, Integer, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from prometheus_client import Counter, Gauge, Histogram
import json

# Metrics
RISK_VIOLATIONS = Counter('risk_violations_total', 'Total risk violations', ['type'])
POSITION_COUNT = Gauge('active_positions_count', 'Number of active positions')
PORTFOLIO_VALUE = Gauge('portfolio_value_usd', 'Total portfolio value in USD')
DRAWDOWN = Gauge('portfolio_drawdown_percent', 'Portfolio drawdown percentage')
VAR_95 = Gauge('portfolio_var_95', 'Portfolio 95% Value at Risk')

Base = declarative_base()

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"
    CANCELLED = "cancelled"

@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_position_size: float = 0.05  # 5% of portfolio
    max_daily_loss: float = 0.02  # 2% daily loss limit
    max_portfolio_risk: float = 0.10  # 10% total portfolio risk
    max_correlation: float = 0.7  # Maximum correlation between positions
    max_leverage: float = 3.0  # Maximum leverage
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    var_limit_95: float = 0.05  # 5% VaR limit
    margin_call_level: float = 0.3  # 30% margin call level
    max_open_positions: int = 10  # Maximum open positions

@dataclass
class Position:
    """Trading position"""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN
    pnl: float = 0.0
    pnl_pct: float = 0.0
    margin_used: float = 0.0
    risk_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class PositionDB(Base):
    """Database model for positions"""
    __tablename__ = 'positions'
    
    id = Column(String, primary_key=True)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime)
    exit_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    status = Column(String, nullable=False)
    pnl = Column(Float, default=0.0)
    pnl_pct = Column(Float, default=0.0)
    margin_used = Column(Float, default=0.0)
    risk_score = Column(Float, default=0.0)
    metadata = Column(Text)  # JSON string

class RiskEvent(Base):
    """Database model for risk events"""
    __tablename__ = 'risk_events'
    
    id = Column(String, primary_key=True)
    event_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    position_id = Column(String)
    symbol = Column(String)
    action_taken = Column(Text)
    resolved = Column(Boolean, default=False)
    metadata = Column(Text)  # JSON string

class RiskManager:
    """Advanced Risk Management System"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.risk_limits = RiskLimits(**config.get('risk_limits', {}))
        
        # Database connection
        self.engine = create_engine(config['database_url'])
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.db_session = Session()
        
        # Redis for real-time data
        self.redis_client = None
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.portfolio_value = config.get('initial_balance', 100000.0)
        self.initial_portfolio_value = self.portfolio_value
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_value = self.portfolio_value
        
        # Risk monitoring
        self.risk_alerts: List[Dict] = []
        self.emergency_stop = False
        
    async def initialize(self):
        """Initialize risk manager"""
        try:
            # Connect to Redis
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                password=self.config.get('redis_password'),
                decode_responses=True
            )
            
            # Load existing positions
            await self._load_active_positions()
            
            # Start monitoring tasks
            asyncio.create_task(self._risk_monitoring_loop())
            asyncio.create_task(self._portfolio_update_loop())
            
            self.logger.info("Risk Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Risk Manager: {e}")
            raise
    
    async def _load_active_positions(self):
        """Load active positions from database"""
        try:
            active_positions = (self.db_session.query(PositionDB)
                              .filter(PositionDB.status == PositionStatus.OPEN.value)
                              .all())
            
            for pos_db in active_positions:
                position = Position(
                    id=pos_db.id,
                    symbol=pos_db.symbol,
                    side=pos_db.side,
                    quantity=pos_db.quantity,
                    entry_price=pos_db.entry_price,
                    current_price=pos_db.current_price,
                    entry_time=pos_db.entry_time,
                    stop_loss=pos_db.stop_loss,
                    take_profit=pos_db.take_profit,
                    status=PositionStatus(pos_db.status),
                    pnl=pos_db.pnl,
                    pnl_pct=pos_db.pnl_pct,
                    margin_used=pos_db.margin_used,
                    risk_score=pos_db.risk_score,
                    metadata=json.loads(pos_db.metadata) if pos_db.metadata else {}
                )
                self.positions[position.id] = position
            
            self.logger.info(f"Loaded {len(self.positions)} active positions")
            
        except Exception as e:
            self.logger.error(f"Failed to load active positions: {e}")
    
    async def validate_new_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ) -> Tuple[bool, List[str]]:
        """Validate if new position can be opened"""
        violations = []
        
        try:
            # Calculate position value
            position_value = quantity * price
            position_size_pct = position_value / self.portfolio_value
            
            # Check position size limit
            if position_size_pct > self.risk_limits.max_position_size:
                violations.append(
                    f"Position size {position_size_pct:.2%} exceeds limit "
                    f"{self.risk_limits.max_position_size:.2%}"
                )
            
            # Check maximum open positions
            if len(self.positions) >= self.risk_limits.max_open_positions:
                violations.append(
                    f"Maximum open positions {self.risk_limits.max_open_positions} reached"
                )
            
            # Check portfolio risk
            total_risk = await self._calculate_portfolio_risk(symbol, side, quantity, price)
            if total_risk > self.risk_limits.max_portfolio_risk:
                violations.append(
                    f"Total portfolio risk {total_risk:.2%} exceeds limit "
                    f"{self.risk_limits.max_portfolio_risk:.2%}"
                )
            
            # Check correlation with existing positions
            correlation = await self._calculate_correlation_risk(symbol)
            if correlation > self.risk_limits.max_correlation:
                violations.append(
                    f"Correlation {correlation:.2f} with existing positions exceeds limit "
                    f"{self.risk_limits.max_correlation:.2f}"
                )
            
            # Check margin requirements
            margin_required = await self._calculate_margin_required(quantity, price)
            available_margin = await self._calculate_available_margin()
            if margin_required > available_margin:
                violations.append(
                    f"Insufficient margin: required {margin_required:.2f}, "
                    f"available {available_margin:.2f}"
                )
            
            # Check emergency stop
            if self.emergency_stop:
                violations.append("Emergency stop activated - no new positions allowed")
            
            is_valid = len(violations) == 0
            
            if violations:
                RISK_VIOLATIONS.labels(type='position_validation').inc()
                await self._log_risk_event(
                    "POSITION_VALIDATION_FAILED",
                    RiskLevel.MEDIUM,
                    f"Position validation failed for {symbol}: {'; '.join(violations)}",
                    symbol=symbol
                )
            
            return is_valid, violations
            
        except Exception as e:
            self.logger.error(f"Error validating position: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    async def open_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Dict[str, Any] = None
    ) -> Optional[str]:
        """Open new position with risk checks"""
        try:
            # Validate position
            is_valid, violations = await self.validate_new_position(symbol, side, quantity, entry_price)
            if not is_valid:
                self.logger.warning(f"Position validation failed: {violations}")
                return None
            
            # Generate position ID
            position_id = f"{symbol}_{side}_{int(datetime.now().timestamp())}"
            
            # Calculate stop loss and take profit if not provided
            if stop_loss is None:
                if side == 'buy':
                    stop_loss = entry_price * (1 - self.risk_limits.stop_loss_pct)
                else:
                    stop_loss = entry_price * (1 + self.risk_limits.stop_loss_pct)
            
            if take_profit is None:
                if side == 'buy':
                    take_profit = entry_price * (1 + self.risk_limits.take_profit_pct)
                else:
                    take_profit = entry_price * (1 - self.risk_limits.take_profit_pct)
            
            # Create position
            position = Position(
                id=position_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                current_price=entry_price,
                entry_time=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                status=PositionStatus.OPEN,
                margin_used=await self._calculate_margin_required(quantity, entry_price),
                metadata=metadata or {}
            )
            
            # Calculate initial risk score
            position.risk_score = await self._calculate_position_risk(position)
            
            # Save to database
            pos_db = PositionDB(
                id=position.id,
                symbol=position.symbol,
                side=position.side,
                quantity=position.quantity,
                entry_price=position.entry_price,
                current_price=position.current_price,
                entry_time=position.entry_time,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                status=position.status.value,
                pnl=position.pnl,
                pnl_pct=position.pnl_pct,
                margin_used=position.margin_used,
                risk_score=position.risk_score,
                metadata=json.dumps(position.metadata)
            )
            
            self.db_session.add(pos_db)
            self.db_session.commit()
            
            # Add to active positions
            self.positions[position_id] = position
            
            # Update metrics
            POSITION_COUNT.set(len(self.positions))
            
            self.logger.info(f"Opened position: {position_id}")
            return position_id
            
        except Exception as e:
            self.logger.error(f"Failed to open position: {e}")
            self.db_session.rollback()
            return None
    
    async def close_position(
        self,
        position_id: str,
        exit_price: Optional[float] = None,
        reason: str = "manual"
    ) -> bool:
        """Close position"""
        try:
            if position_id not in self.positions:
                self.logger.warning(f"Position not found: {position_id}")
                return False
            
            position = self.positions[position_id]
            
            # Use current price if exit price not provided
            if exit_price is None:
                exit_price = position.current_price
            
            # Calculate final P&L
            if position.side == 'buy':
                pnl = (exit_price - position.entry_price) * position.quantity
            else:
                pnl = (position.entry_price - exit_price) * position.quantity
            
            pnl_pct = pnl / (position.entry_price * position.quantity) * 100
            
            # Update position
            position.status = PositionStatus.CLOSED
            position.pnl = pnl
            position.pnl_pct = pnl_pct
            
            # Update database
            pos_db = self.db_session.query(PositionDB).filter(PositionDB.id == position_id).first()
            if pos_db:
                pos_db.status = PositionStatus.CLOSED.value
                pos_db.exit_time = datetime.now()
                pos_db.exit_price = exit_price
                pos_db.pnl = pnl
                pos_db.pnl_pct = pnl_pct
                self.db_session.commit()
            
            # Update portfolio
            self.portfolio_value += pnl
            self.daily_pnl += pnl
            
            # Remove from active positions
            del self.positions[position_id]
            
            # Update metrics
            POSITION_COUNT.set(len(self.positions))
            PORTFOLIO_VALUE.set(self.portfolio_value)
            
            self.logger.info(f"Closed position: {position_id}, P&L: {pnl:.2f} ({pnl_pct:.2f}%)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to close position: {e}")
            self.db_session.rollback()
            return False
    
    async def update_position_prices(self, price_updates: Dict[str, float]):
        """Update current prices for all positions"""
        try:
            for position in self.positions.values():
                if position.symbol in price_updates:
                    old_price = position.current_price
                    new_price = price_updates[position.symbol]
                    position.current_price = new_price
                    
                    # Recalculate P&L
                    if position.side == 'buy':
                        position.pnl = (new_price - position.entry_price) * position.quantity
                    else:
                        position.pnl = (position.entry_price - new_price) * position.quantity
                    
                    position.pnl_pct = position.pnl / (position.entry_price * position.quantity) * 100
                    
                    # Check stop loss and take profit
                    await self._check_exit_conditions(position)
            
            # Update portfolio value
            total_pnl = sum(pos.pnl for pos in self.positions.values())
            self.portfolio_value = self.initial_portfolio_value + total_pnl
            
            # Update metrics
            PORTFOLIO_VALUE.set(self.portfolio_value)
            
        except Exception as e:
            self.logger.error(f"Error updating position prices: {e}")
    
    async def _check_exit_conditions(self, position: Position):
        """Check if position should be closed due to stop loss or take profit"""
        try:
            should_close = False
            reason = ""
            
            if position.side == 'buy':
                if position.current_price <= position.stop_loss:
                    should_close = True
                    reason = "stop_loss"
                elif position.current_price >= position.take_profit:
                    should_close = True
                    reason = "take_profit"
            else:  # sell
                if position.current_price >= position.stop_loss:
                    should_close = True
                    reason = "stop_loss"
                elif position.current_price <= position.take_profit:
                    should_close = True
                    reason = "take_profit"
            
            if should_close:
                await self.close_position(position.id, position.current_price, reason)
                await self._log_risk_event(
                    "POSITION_AUTO_CLOSED",
                    RiskLevel.LOW,
                    f"Position {position.id} auto-closed due to {reason}",
                    position_id=position.id,
                    symbol=position.symbol
                )
            
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
    
    async def _calculate_portfolio_risk(self, symbol: str, side: str, quantity: float, price: float) -> float:
        """Calculate total portfolio risk including new position"""
        try:
            # Current portfolio risk
            current_risk = sum(pos.risk_score for pos in self.positions.values())
            
            # New position risk (simplified)
            position_value = quantity * price
            position_risk = position_value / self.portfolio_value * 0.02  # 2% base risk
            
            return current_risk + position_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk: {e}")
            return 1.0  # Return high risk on error
    
    async def _calculate_correlation_risk(self, symbol: str) -> float:
        """Calculate correlation risk with existing positions"""
        try:
            # Simplified correlation calculation
            # In production, this would use historical price correlations
            symbols_in_portfolio = set(pos.symbol for pos in self.positions.values())
            
            if symbol in symbols_in_portfolio:
                return 1.0  # Perfect correlation with existing position
            
            # Check for similar instruments
            if any(symbol[:3] == existing[:3] for existing in symbols_in_portfolio):
                return 0.8  # High correlation with similar instruments
            
            return 0.2  # Low correlation assumed
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation risk: {e}")
            return 1.0
    
    async def _calculate_margin_required(self, quantity: float, price: float) -> float:
        """Calculate margin required for position"""
        position_value = quantity * price
        margin_rate = 1.0 / self.risk_limits.max_leverage  # e.g., 1/3 = 33.3% for 3:1 leverage
        return position_value * margin_rate
    
    async def _calculate_available_margin(self) -> float:
        """Calculate available margin"""
        used_margin = sum(pos.margin_used for pos in self.positions.values())
        return self.portfolio_value - used_margin
    
    async def _calculate_position_risk(self, position: Position) -> float:
        """Calculate risk score for position"""
        try:
            # Distance to stop loss as risk measure
            if position.side == 'buy':
                risk_distance = (position.entry_price - position.stop_loss) / position.entry_price
            else:
                risk_distance = (position.stop_loss - position.entry_price) / position.entry_price
            
            position_value = position.quantity * position.entry_price
            position_size = position_value / self.portfolio_value
            
            # Risk score combines position size and stop loss distance
            risk_score = position_size * risk_distance
            
            return min(risk_score, 1.0)  # Cap at 100%
            
        except Exception as e:
            self.logger.error(f"Error calculating position risk: {e}")
            return 0.5  # Default medium risk
    
    async def _risk_monitoring_loop(self):
        """Continuous risk monitoring"""
        while True:
            try:
                await self._check_daily_loss_limit()
                await self._check_drawdown_limit()
                await self._check_margin_levels()
                await self._calculate_var()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_daily_loss_limit(self):
        """Check daily loss limit"""
        try:
            daily_loss_pct = abs(self.daily_pnl) / self.initial_portfolio_value
            
            if daily_loss_pct > self.risk_limits.max_daily_loss:
                RISK_VIOLATIONS.labels(type='daily_loss').inc()
                
                # Emergency stop
                self.emergency_stop = True
                
                await self._log_risk_event(
                    "DAILY_LOSS_LIMIT_EXCEEDED",
                    RiskLevel.CRITICAL,
                    f"Daily loss {daily_loss_pct:.2%} exceeds limit {self.risk_limits.max_daily_loss:.2%}. Emergency stop activated."
                )
                
                # Close all positions
                for position_id in list(self.positions.keys()):
                    await self.close_position(position_id, reason="emergency_stop")
                
        except Exception as e:
            self.logger.error(f"Error checking daily loss limit: {e}")
    
    async def _check_drawdown_limit(self):
        """Check portfolio drawdown"""
        try:
            if self.portfolio_value > self.peak_value:
                self.peak_value = self.portfolio_value
            
            drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, drawdown)
            
            DRAWDOWN.set(drawdown * 100)
            
            if drawdown > 0.2:  # 20% drawdown warning
                await self._log_risk_event(
                    "HIGH_DRAWDOWN_WARNING",
                    RiskLevel.HIGH,
                    f"Portfolio drawdown {drawdown:.2%} is high"
                )
                
        except Exception as e:
            self.logger.error(f"Error checking drawdown: {e}")
    
    async def _check_margin_levels(self):
        """Check margin levels"""
        try:
            used_margin = sum(pos.margin_used for pos in self.positions.values())
            margin_level = used_margin / self.portfolio_value
            
            if margin_level > self.risk_limits.margin_call_level:
                RISK_VIOLATIONS.labels(type='margin_call').inc()
                
                await self._log_risk_event(
                    "MARGIN_CALL",
                    RiskLevel.CRITICAL,
                    f"Margin level {margin_level:.2%} exceeds limit {self.risk_limits.margin_call_level:.2%}"
                )
                
        except Exception as e:
            self.logger.error(f"Error checking margin levels: {e}")
    
    async def _calculate_var(self):
        """Calculate Value at Risk"""
        try:
            if not self.positions:
                VAR_95.set(0)
                return
            
            # Simplified VaR calculation
            # In production, this would use historical simulation or Monte Carlo
            total_position_value = sum(pos.quantity * pos.current_price for pos in self.positions.values())
            
            # Assume 2% daily volatility for simplification
            daily_volatility = 0.02
            var_95 = total_position_value * daily_volatility * 1.645  # 95% confidence
            
            VAR_95.set(var_95)
            
            var_limit = self.portfolio_value * self.risk_limits.var_limit_95
            if var_95 > var_limit:
                await self._log_risk_event(
                    "VAR_LIMIT_EXCEEDED",
                    RiskLevel.HIGH,
                    f"95% VaR {var_95:.2f} exceeds limit {var_limit:.2f}"
                )
                
        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
    
    async def _portfolio_update_loop(self):
        """Update portfolio metrics"""
        while True:
            try:
                # Update Redis with current state
                portfolio_state = {
                    'value': self.portfolio_value,
                    'daily_pnl': self.daily_pnl,
                    'positions_count': len(self.positions),
                    'max_drawdown': self.max_drawdown,
                    'emergency_stop': self.emergency_stop,
                    'timestamp': datetime.now().isoformat()
                }
                
                await self.redis_client.setex(
                    'portfolio:state',
                    300,  # 5 minutes TTL
                    json.dumps(portfolio_state)
                )
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in portfolio update loop: {e}")
                await asyncio.sleep(30)
    
    async def _log_risk_event(
        self,
        event_type: str,
        severity: RiskLevel,
        description: str,
        position_id: Optional[str] = None,
        symbol: Optional[str] = None,
        action_taken: Optional[str] = None
    ):
        """Log risk event"""
        try:
            event_id = f"{event_type}_{int(datetime.now().timestamp())}"
            
            risk_event = RiskEvent(
                id=event_id,
                event_type=event_type,
                severity=severity.value,
                description=description,
                timestamp=datetime.now(),
                position_id=position_id,
                symbol=symbol,
                action_taken=action_taken,
                resolved=False
            )
            
            self.db_session.add(risk_event)
            self.db_session.commit()
            
            # Add to alerts
            alert = {
                'id': event_id,
                'type': event_type,
                'severity': severity.value,
                'description': description,
                'timestamp': datetime.now().isoformat(),
                'position_id': position_id,
                'symbol': symbol
            }
            
            self.risk_alerts.append(alert)
            
            # Keep only last 100 alerts
            if len(self.risk_alerts) > 100:
                self.risk_alerts = self.risk_alerts[-100:]
            
            # Send to Redis for real-time notifications
            await self.redis_client.lpush('risk:alerts', json.dumps(alert))
            await self.redis_client.ltrim('risk:alerts', 0, 99)  # Keep last 100
            
            self.logger.warning(f"Risk event: {event_type} - {description}")
            
        except Exception as e:
            self.logger.error(f"Failed to log risk event: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        total_pnl = sum(pos.pnl for pos in self.positions.values())
        total_margin = sum(pos.margin_used for pos in self.positions.values())
        
        return {
            'portfolio_value': self.portfolio_value,
            'initial_value': self.initial_portfolio_value,
            'total_pnl': total_pnl,
            'daily_pnl': self.daily_pnl,
            'total_pnl_pct': (total_pnl / self.initial_portfolio_value) * 100,
            'open_positions': len(self.positions),
            'used_margin': total_margin,
            'available_margin': self.portfolio_value - total_margin,
            'margin_level': (total_margin / self.portfolio_value) * 100,
            'max_drawdown': self.max_drawdown * 100,
            'emergency_stop': self.emergency_stop,
            'risk_alerts_count': len(self.risk_alerts)
        }

# Example usage
if __name__ == "__main__":
    import asyncio
    
    config = {
        'database_url': 'postgresql://nicegold:password@localhost/nicegold',
        'redis_host': 'localhost',
        'redis_port': 6379,
        'initial_balance': 100000.0,
        'risk_limits': {
            'max_position_size': 0.05,
            'max_daily_loss': 0.02,
            'max_portfolio_risk': 0.10,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04
        }
    }
    
    async def test_risk_manager():
        risk_manager = RiskManager(config)
        await risk_manager.initialize()
        
        # Test opening a position
        position_id = await risk_manager.open_position(
            symbol="XAUUSD",
            side="buy",
            quantity=10.0,
            entry_price=2000.0
        )
        
        if position_id:
            print(f"Opened position: {position_id}")
            
            # Simulate price update
            await risk_manager.update_position_prices({"XAUUSD": 2010.0})
            
            # Get portfolio summary
            summary = risk_manager.get_portfolio_summary()
            print(f"Portfolio summary: {summary}")
    
    asyncio.run(test_risk_manager())
