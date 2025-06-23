"""
Real-time Data Pipeline with Apache Kafka integration
"""
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import pandas as pd
from dataclasses import dataclass
import aiokafka
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import aioredis
import os

logger = logging.getLogger(__name__)

@dataclass
class MarketTick:
    """Market tick data structure"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    volume: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'bid': self.bid,
            'ask': self.ask,
            'volume': self.volume
        }

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    confidence: float
    features: Dict[str, Any]
    model_version: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'confidence': self.confidence,
            'features': self.features,
            'model_version': self.model_version,
            'timestamp': self.timestamp.isoformat()
        }

class DataPipeline:
    """Real-time data pipeline"""
    
    def __init__(self, 
                 kafka_servers: str = "localhost:9092",
                 redis_url: str = "redis://localhost:6379"):
        
        self.kafka_servers = kafka_servers
        self.redis_url = redis_url
        self.producer: Optional[AIOKafkaProducer] = None
        self.consumers: Dict[str, AIOKafkaConsumer] = {}
        self.redis_client: Optional[aioredis.Redis] = None
        self.running = False
        
        # Topics configuration
        self.topics = {
            'market_data': 'market_data_stream',
            'trading_signals': 'trading_signals_stream',
            'trade_executions': 'trade_executions_stream',
            'model_predictions': 'model_predictions_stream',
            'system_metrics': 'system_metrics_stream'
        }
        
    async def initialize(self):
        """Initialize Kafka producer and Redis client"""
        try:
            # Initialize Kafka producer
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.kafka_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                compression_type="gzip"
            )
            await self.producer.start()
            
            # Initialize Redis client
            self.redis_client = aioredis.from_url(self.redis_url)
            
            logger.info("Data pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize data pipeline: {e}")
            raise
    
    async def publish_market_data(self, tick: MarketTick):
        """Publish market tick data to Kafka"""
        try:
            await self.producer.send(
                self.topics['market_data'],
                value=tick.to_dict(),
                key=tick.symbol.encode('utf-8')
            )
            
            # Cache latest tick in Redis
            await self.redis_client.setex(
                f"latest_tick:{tick.symbol}",
                60,  # 1 minute TTL
                json.dumps(tick.to_dict())
            )
            
        except Exception as e:
            logger.error(f"Failed to publish market data: {e}")
    
    async def publish_trading_signal(self, signal: TradingSignal):
        """Publish trading signal to Kafka"""
        try:
            await self.producer.send(
                self.topics['trading_signals'],
                value=signal.to_dict(),
                key=signal.symbol.encode('utf-8')
            )
            
            # Cache latest signal in Redis
            await self.redis_client.setex(
                f"latest_signal:{signal.symbol}",
                300,  # 5 minutes TTL
                json.dumps(signal.to_dict())
            )
            
            logger.info(f"Published signal: {signal.signal_type} for {signal.symbol}")
            
        except Exception as e:
            logger.error(f"Failed to publish trading signal: {e}")
    
    async def publish_trade_execution(self, trade_data: Dict[str, Any]):
        """Publish trade execution to Kafka"""
        try:
            await self.producer.send(
                self.topics['trade_executions'],
                value=trade_data,
                key=trade_data['symbol'].encode('utf-8')
            )
            
        except Exception as e:
            logger.error(f"Failed to publish trade execution: {e}")
    
    async def publish_model_prediction(self, prediction_data: Dict[str, Any]):
        """Publish model prediction to Kafka"""
        try:
            await self.producer.send(
                self.topics['model_predictions'],
                value=prediction_data,
                key=prediction_data['symbol'].encode('utf-8')
            )
            
        except Exception as e:
            logger.error(f"Failed to publish model prediction: {e}")
    
    async def consume_market_data(self, callback: Callable[[MarketTick], None]):
        """Consume market data from Kafka"""
        consumer = AIOKafkaConsumer(
            self.topics['market_data'],
            bootstrap_servers=self.kafka_servers,
            group_id="market_data_processors",
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        self.consumers['market_data'] = consumer
        await consumer.start()
        
        try:
            async for message in consumer:
                tick_data = message.value
                tick = MarketTick(
                    symbol=tick_data['symbol'],
                    timestamp=datetime.fromisoformat(tick_data['timestamp']),
                    bid=tick_data['bid'],
                    ask=tick_data['ask'],
                    volume=tick_data.get('volume', 0)
                )
                
                # Process tick data
                await callback(tick)
                
        except Exception as e:
            logger.error(f"Error consuming market data: {e}")
        finally:
            await consumer.stop()
    
    async def consume_trading_signals(self, callback: Callable[[TradingSignal], None]):
        """Consume trading signals from Kafka"""
        consumer = AIOKafkaConsumer(
            self.topics['trading_signals'],
            bootstrap_servers=self.kafka_servers,
            group_id="signal_processors",
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        self.consumers['trading_signals'] = consumer
        await consumer.start()
        
        try:
            async for message in consumer:
                signal_data = message.value
                signal = TradingSignal(
                    symbol=signal_data['symbol'],
                    signal_type=signal_data['signal_type'],
                    confidence=signal_data['confidence'],
                    features=signal_data['features'],
                    model_version=signal_data['model_version'],
                    timestamp=datetime.fromisoformat(signal_data['timestamp'])
                )
                
                # Process trading signal
                await callback(signal)
                
        except Exception as e:
            logger.error(f"Error consuming trading signals: {e}")
        finally:
            await consumer.stop()
    
    async def get_latest_tick(self, symbol: str) -> Optional[MarketTick]:
        """Get latest market tick from Redis cache"""
        try:
            tick_data = await self.redis_client.get(f"latest_tick:{symbol}")
            if tick_data:
                data = json.loads(tick_data)
                return MarketTick(
                    symbol=data['symbol'],
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    bid=data['bid'],
                    ask=data['ask'],
                    volume=data.get('volume', 0)
                )
        except Exception as e:
            logger.error(f"Failed to get latest tick: {e}")
        
        return None
    
    async def get_latest_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Get latest trading signal from Redis cache"""
        try:
            signal_data = await self.redis_client.get(f"latest_signal:{symbol}")
            if signal_data:
                data = json.loads(signal_data)
                return TradingSignal(
                    symbol=data['symbol'],
                    signal_type=data['signal_type'],
                    confidence=data['confidence'],
                    features=data['features'],
                    model_version=data['model_version'],
                    timestamp=datetime.fromisoformat(data['timestamp'])
                )
        except Exception as e:
            logger.error(f"Failed to get latest signal: {e}")
        
        return None
    
    async def close(self):
        """Close all connections"""
        if self.producer:
            await self.producer.stop()
        
        for consumer in self.consumers.values():
            await consumer.stop()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Data pipeline closed")

class MarketDataSimulator:
    """Simulate market data for testing"""
    
    def __init__(self, pipeline: DataPipeline):
        self.pipeline = pipeline
        self.running = False
    
    async def start_simulation(self, symbol: str = "XAUUSD"):
        """Start market data simulation"""
        self.running = True
        import random
        
        base_price = 2000.0
        
        while self.running:
            # Generate random price movement
            change = random.uniform(-0.5, 0.5)
            base_price += change
            
            bid = base_price - random.uniform(0.1, 0.3)
            ask = base_price + random.uniform(0.1, 0.3)
            
            tick = MarketTick(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                bid=bid,
                ask=ask,
                volume=random.randint(1, 100)
            )
            
            await self.pipeline.publish_market_data(tick)
            await asyncio.sleep(1)  # 1 second interval
    
    def stop_simulation(self):
        """Stop market data simulation"""
        self.running = False

class RealTimeFeatureEngine:
    """Real-time feature engineering"""
    
    def __init__(self, pipeline: DataPipeline):
        self.pipeline = pipeline
        self.price_history: Dict[str, List[float]] = {}
        self.max_history = 100
    
    async def process_tick(self, tick: MarketTick):
        """Process incoming tick and generate features"""
        symbol = tick.symbol
        mid_price = (tick.bid + tick.ask) / 2
        
        # Update price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(mid_price)
        
        # Keep only recent history
        if len(self.price_history[symbol]) > self.max_history:
            self.price_history[symbol] = self.price_history[symbol][-self.max_history:]
        
        # Generate features if we have enough history
        if len(self.price_history[symbol]) >= 20:
            features = self._calculate_features(symbol)
            
            # Predict using model (placeholder)
            prediction = await self._predict_signal(symbol, features)
            
            if prediction:
                await self.pipeline.publish_trading_signal(prediction)
    
    def _calculate_features(self, symbol: str) -> Dict[str, Any]:
        """Calculate technical features from price history"""
        prices = pd.Series(self.price_history[symbol])
        
        # Simple features
        features = {
            'price': prices.iloc[-1],
            'sma_5': prices.rolling(5).mean().iloc[-1],
            'sma_20': prices.rolling(20).mean().iloc[-1],
            'volatility': prices.rolling(20).std().iloc[-1],
            'price_change': prices.pct_change().iloc[-1],
            'volume_sma': 50.0  # Placeholder
        }
        
        # Remove NaN values
        features = {k: v for k, v in features.items() if pd.notna(v)}
        
        return features
    
    async def _predict_signal(self, symbol: str, features: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate trading signal from features (placeholder)"""
        # Simple rule-based signal generation for demo
        if features.get('price', 0) > features.get('sma_20', 0):
            return TradingSignal(
                symbol=symbol,
                signal_type="BUY",
                confidence=0.75,
                features=features,
                model_version="demo_v1.0",
                timestamp=datetime.utcnow()
            )
        elif features.get('price', 0) < features.get('sma_20', 0):
            return TradingSignal(
                symbol=symbol,
                signal_type="SELL",
                confidence=0.70,
                features=features,
                model_version="demo_v1.0",
                timestamp=datetime.utcnow()
            )
        
        return None

# Global pipeline instance
data_pipeline = DataPipeline(
    kafka_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379")
)

async def main():
    """Example usage of real-time data pipeline"""
    # Initialize pipeline
    await data_pipeline.initialize()
    
    # Create feature engine
    feature_engine = RealTimeFeatureEngine(data_pipeline)
    
    # Create market data simulator
    simulator = MarketDataSimulator(data_pipeline)
    
    # Start market data consumer
    async def process_market_tick(tick: MarketTick):
        logger.info(f"Received tick: {tick.symbol} @ {tick.bid}/{tick.ask}")
        await feature_engine.process_tick(tick)
    
    # Start signal consumer
    async def process_trading_signal(signal: TradingSignal):
        logger.info(f"Received signal: {signal.signal_type} for {signal.symbol} (confidence: {signal.confidence})")
    
    # Start consumers
    consumer_tasks = [
        asyncio.create_task(data_pipeline.consume_market_data(process_market_tick)),
        asyncio.create_task(data_pipeline.consume_trading_signals(process_trading_signal))
    ]
    
    # Start simulator
    simulator_task = asyncio.create_task(simulator.start_simulation("XAUUSD"))
    
    try:
        # Run for demo
        await asyncio.sleep(30)
    finally:
        # Cleanup
        simulator.stop_simulation()
        for task in consumer_tasks + [simulator_task]:
            task.cancel()
        
        await data_pipeline.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
