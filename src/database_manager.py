from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import asyncio
import asyncpg
import json
import logging
import os
import pandas as pd
import redis.asyncio as redis
"""
Production Database Manager with PostgreSQL and Redis integration
"""

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Production Database Manager"""

    def __init__(self):
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None

    async def initialize(self):
        """Initialize database connections"""
        await self._init_postgresql()
        await self._init_redis()
        await self._create_tables()

    async def _init_postgresql(self):
        """Initialize PostgreSQL connection pool"""
        try:
            database_url = os.getenv("DATABASE_URL", "postgresql://nicegold:password@localhost:5432/nicegold")
            self.pg_pool = await asyncpg.create_pool(
                database_url, 
                min_size = 5, 
                max_size = 20, 
                command_timeout = 60
            )
            logger.info("PostgreSQL connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise

    async def _init_redis(self):
        """Initialize Redis connection"""
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url, decode_responses = True)
            await self.redis_client.ping()
            logger.info("Redis connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise

    async def _create_tables(self):
        """Create necessary database tables"""
        async with self.pg_pool.acquire() as conn:
            # Market data table (TimescaleDB hypertable)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id SERIAL PRIMARY KEY, 
                    symbol VARCHAR(10) NOT NULL, 
                    timeframe VARCHAR(5) NOT NULL, 
                    timestamp TIMESTAMPTZ NOT NULL, 
                    open DECIMAL(15, 5) NOT NULL, 
                    high DECIMAL(15, 5) NOT NULL, 
                    low DECIMAL(15, 5) NOT NULL, 
                    close DECIMAL(15, 5) NOT NULL, 
                    volume BIGINT DEFAULT 0, 
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)

            # Create hypertable for time series optimization
            try:
                await conn.execute("""
                    SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE);
                """)
            except Exception:
                # Fallback if TimescaleDB not available
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timeframe ON market_data(symbol, timeframe);
                """)

            # Trading signals table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id SERIAL PRIMARY KEY, 
                    symbol VARCHAR(10) NOT NULL, 
                    signal_type VARCHAR(10) NOT NULL, -- BUY, SELL, HOLD
                    confidence DECIMAL(5, 4) NOT NULL, 
                    features JSONB, 
                    model_version VARCHAR(50), 
                    timestamp TIMESTAMPTZ DEFAULT NOW(), 
                    is_executed BOOLEAN DEFAULT FALSE
                );
            """)

            # Trade execution table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_executions (
                    id SERIAL PRIMARY KEY, 
                    signal_id INTEGER REFERENCES trading_signals(id), 
                    symbol VARCHAR(10) NOT NULL, 
                    side VARCHAR(4) NOT NULL, -- BUY, SELL
                    quantity DECIMAL(15, 5) NOT NULL, 
                    entry_price DECIMAL(15, 5) NOT NULL, 
                    exit_price DECIMAL(15, 5), 
                    stop_loss DECIMAL(15, 5), 
                    take_profit DECIMAL(15, 5), 
                    pnl DECIMAL(15, 5), 
                    status VARCHAR(20) DEFAULT 'OPEN', -- OPEN, CLOSED, CANCELLED
                    entry_time TIMESTAMPTZ DEFAULT NOW(), 
                    exit_time TIMESTAMPTZ, 
                    metadata JSONB
                );
            """)

            # Model performance table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id SERIAL PRIMARY KEY, 
                    model_name VARCHAR(100) NOT NULL, 
                    model_version VARCHAR(50) NOT NULL, 
                    run_id VARCHAR(100), 
                    accuracy DECIMAL(8, 6), 
                    auc DECIMAL(8, 6), 
                    f1_score DECIMAL(8, 6), 
                    precision_score DECIMAL(8, 6), 
                    recall_score DECIMAL(8, 6), 
                    training_samples INTEGER, 
                    validation_samples INTEGER, 
                    hyperparameters JSONB, 
                    feature_importance JSONB, 
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)

            # System metrics table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id SERIAL PRIMARY KEY, 
                    metric_name VARCHAR(100) NOT NULL, 
                    metric_value DECIMAL(15, 5) NOT NULL, 
                    metric_type VARCHAR(20) NOT NULL, -- counter, gauge, histogram
                    labels JSONB, 
                    timestamp TIMESTAMPTZ DEFAULT NOW()
                );
            """)

            logger.info("Database tables created successfully")

    # Market Data Methods
    async def insert_market_data(self, data: List[Dict[str, Any]]) -> int:
        """Insert market data in batch"""
        async with self.pg_pool.acquire() as conn:
            query = """
                INSERT INTO market_data (symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """

            rows = [
                (
                    row['symbol'], row['timeframe'], row['timestamp'], 
                    row['open'], row['high'], row['low'], row['close'], 
                    row.get('volume', 0)
                )
                for row in data
            ]

            await conn.executemany(query, rows)
            return len(rows)

    async def get_market_data(self, 
                             symbol: str, 
                             timeframe: str, 
                             start_time: datetime, 
                             end_time: datetime, 
                             limit: int = 1000) -> List[Dict[str, Any]]:
        """Get market data for specified period"""
        async with self.pg_pool.acquire() as conn:
            query = """
                SELECT symbol, timeframe, timestamp, open, high, low, close, volume
                FROM market_data
                WHERE symbol = $1 AND timeframe = $2
                AND timestamp BETWEEN $3 AND $4
                ORDER BY timestamp ASC
                LIMIT $5
            """

            rows = await conn.fetch(query, symbol, timeframe, start_time, end_time, limit)
            return [dict(row) for row in rows]

    # Trading Signals Methods
    async def insert_trading_signal(self, signal_data: Dict[str, Any]) -> int:
        """Insert trading signal"""
        async with self.pg_pool.acquire() as conn:
            query = """
                INSERT INTO trading_signals
                (symbol, signal_type, confidence, features, model_version)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """

            signal_id = await conn.fetchval(
                query, 
                signal_data['symbol'], 
                signal_data['signal_type'], 
                signal_data['confidence'], 
                json.dumps(signal_data.get('features', {})), 
                signal_data.get('model_version', 'unknown')
            )

            return signal_id

    async def get_pending_signals(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get pending trading signals"""
        async with self.pg_pool.acquire() as conn:
            query = """
                SELECT * FROM trading_signals
                WHERE is_executed = FALSE
                ORDER BY timestamp ASC
                LIMIT $1
            """

            rows = await conn.fetch(query, limit)
            return [dict(row) for row in rows]

    # Trade Execution Methods
    async def insert_trade_execution(self, trade_data: Dict[str, Any]) -> int:
        """Insert trade execution"""
        async with self.pg_pool.acquire() as conn:
            query = """
                INSERT INTO trade_executions
                (signal_id, symbol, side, quantity, entry_price, stop_loss, take_profit, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
            """

            trade_id = await conn.fetchval(
                query, 
                trade_data.get('signal_id'), 
                trade_data['symbol'], 
                trade_data['side'], 
                trade_data['quantity'], 
                trade_data['entry_price'], 
                trade_data.get('stop_loss'), 
                trade_data.get('take_profit'), 
                json.dumps(trade_data.get('metadata', {}))
            )

            return trade_id

    async def update_trade_execution(self, trade_id: int, update_data: Dict[str, Any]):
        """Update trade execution"""
        async with self.pg_pool.acquire() as conn:
            # Build dynamic update query
            set_clauses = []
            values = []
            param_idx = 2  # $1 is trade_id

            for key, value in update_data.items():
                if key in ['exit_price', 'pnl', 'status', 'exit_time']:
                    set_clauses.append(f"{key} = ${param_idx}")
                    values.append(value)
                    param_idx += 1

            if set_clauses:
                query = f"""
                    UPDATE trade_executions
                    SET {', '.join(set_clauses)}
                    WHERE id = $1
                """
                await conn.execute(query, trade_id, *values)

    # Model Performance Methods
    async def insert_model_performance(self, performance_data: Dict[str, Any]) -> int:
        """Insert model performance metrics"""
        async with self.pg_pool.acquire() as conn:
            query = """
                INSERT INTO model_performance
                (model_name, model_version, run_id, accuracy, auc, f1_score, 
                 precision_score, recall_score, training_samples, validation_samples, 
                 hyperparameters, feature_importance)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                RETURNING id
            """

            perf_id = await conn.fetchval(
                query, 
                performance_data['model_name'], 
                performance_data['model_version'], 
                performance_data.get('run_id'), 
                performance_data.get('accuracy'), 
                performance_data.get('auc'), 
                performance_data.get('f1_score'), 
                performance_data.get('precision_score'), 
                performance_data.get('recall_score'), 
                performance_data.get('training_samples'), 
                performance_data.get('validation_samples'), 
                json.dumps(performance_data.get('hyperparameters', {})), 
                json.dumps(performance_data.get('feature_importance', {}))
            )

            return perf_id

    # Redis Cache Methods
    async def cache_set(self, key: str, value: Any, expire: int = 3600):
        """Set value in Redis cache"""
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        await self.redis_client.setex(key, expire, value)

    async def cache_get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        value = await self.redis_client.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return None

    async def cache_delete(self, key: str):
        """Delete value from Redis cache"""
        await self.redis_client.delete(key)

    # Analytics Methods
    async def get_trading_performance(self, 
                                    start_date: datetime, 
                                    end_date: datetime) -> Dict[str, Any]:
        """Get trading performance analytics"""
        async with self.pg_pool.acquire() as conn:
            # Total trades
            total_trades = await conn.fetchval("""
                SELECT COUNT(*) FROM trade_executions
                WHERE entry_time BETWEEN $1 AND $2
            """, start_date, end_date)

            # Closed trades stats
            closed_stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) as closed_trades, 
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades, 
                    SUM(pnl) as total_pnl, 
                    AVG(pnl) as avg_pnl, 
                    MAX(pnl) as max_profit, 
                    MIN(pnl) as max_loss
                FROM trade_executions
                WHERE status = 'CLOSED'
                AND entry_time BETWEEN $1 AND $2
            """, start_date, end_date)

            win_rate = (closed_stats['winning_trades'] / closed_stats['closed_trades']
                       if closed_stats['closed_trades'] > 0 else 0)

            return {
                'period_start': start_date.isoformat(), 
                'period_end': end_date.isoformat(), 
                'total_trades': total_trades, 
                'closed_trades': closed_stats['closed_trades'], 
                'winning_trades': closed_stats['winning_trades'], 
                'win_rate': win_rate, 
                'total_pnl': float(closed_stats['total_pnl'] or 0), 
                'average_pnl': float(closed_stats['avg_pnl'] or 0), 
                'max_profit': float(closed_stats['max_profit'] or 0), 
                'max_loss': float(closed_stats['max_loss'] or 0)
            }

    async def close(self):
        """Close database connections"""
        if self.pg_pool:
            await self.pg_pool.close()
        if self.redis_client:
            await self.redis_client.close()

# Global instance
db_manager = DatabaseManager()

@asynccontextmanager
async def get_db_connection():
    """Context manager for database connections"""
    await db_manager.initialize()
    try:
        yield db_manager
    finally:
        pass  # Keep connection pool alive

# Usage example
async def main():
    """Example usage"""
    await db_manager.initialize()

    # Insert sample market data
    sample_data = [{
        'symbol': 'XAUUSD', 
        'timeframe': 'M1', 
        'timestamp': datetime.utcnow(), 
        'open': 2000.50, 
        'high': 2001.00, 
        'low': 2000.00, 
        'close': 2000.75, 
        'volume': 1000
    }]

    await db_manager.insert_market_data(sample_data)
    logger.info("Sample data inserted")

if __name__ == "__main__":
    asyncio.run(main())