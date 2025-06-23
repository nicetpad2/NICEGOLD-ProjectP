#!/usr/bin/env python3
"""
Production Integration Tests
ทดสอบระบบแบบ end-to-end ใน production environment
"""

import pytest
import asyncio
import aiohttp
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionIntegrationTest:
    """Test suite for production integration"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.test_data = self._generate_test_data()
    
    async def setup(self):
        """Setup test environment"""
        self.session = aiohttp.ClientSession()
        logger.info("Test environment setup complete")
    
    async def teardown(self):
        """Cleanup test environment"""
        if self.session:
            await self.session.close()
        logger.info("Test environment cleanup complete")
    
    def _generate_test_data(self):
        """Generate realistic test data"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=100),
            end=datetime.now(),
            freq='1min'
        )
        
        # Generate OHLCV data with realistic patterns
        np.random.seed(42)
        base_price = 2000.0
        prices = []
        
        for i in range(len(dates)):
            if i == 0:
                price = base_price
            else:
                # Add trend and volatility
                trend = 0.0001 * np.sin(i / 1000)  # Long-term trend
                volatility = 0.002 * np.random.randn()  # Random volatility
                price = prices[-1] * (1 + trend + volatility)
            prices.append(price)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.randn() * 0.001)) for p in prices],
            'low': [p * (1 - abs(np.random.randn() * 0.001)) for p in prices],
            'close': prices,
            'volume': np.random.randint(100, 10000, len(dates))
        })
        
        return df
    
    async def test_health_endpoint(self):
        """Test API health endpoint"""
        try:
            async with self.session.get(f"{self.base_url}/health") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "healthy"
                logger.info("✅ Health endpoint test passed")
                return True
        except Exception as e:
            logger.error(f"❌ Health endpoint test failed: {e}")
            return False
    
    async def test_data_ingestion(self):
        """Test real-time data ingestion"""
        try:
            # Test data upload
            test_data = self.test_data.tail(100).to_dict('records')
            
            async with self.session.post(
                f"{self.base_url}/api/v1/data/ingest",
                json={
                    "symbol": "XAUUSD",
                    "timeframe": "M1",
                    "data": test_data
                }
            ) as resp:
                assert resp.status == 200
                result = await resp.json()
                assert result["status"] == "success"
                logger.info("✅ Data ingestion test passed")
                return True
        except Exception as e:
            logger.error(f"❌ Data ingestion test failed: {e}")
            return False
    
    async def test_signal_generation(self):
        """Test trading signal generation"""
        try:
            # Prepare features
            features = {
                "close": 2050.0,
                "sma_20": 2048.0,
                "rsi": 65.5,
                "macd": 1.2,
                "bb_upper": 2055.0,
                "bb_lower": 2040.0,
                "volume": 5000
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/signals/generate",
                json={
                    "symbol": "XAUUSD",
                    "features": features,
                    "model_version": "latest"
                }
            ) as resp:
                assert resp.status == 200
                result = await resp.json()
                assert "signal" in result
                assert "confidence" in result
                assert "timestamp" in result
                logger.info("✅ Signal generation test passed")
                return True
        except Exception as e:
            logger.error(f"❌ Signal generation test failed: {e}")
            return False
    
    async def test_backtest_execution(self):
        """Test backtest execution"""
        try:
            backtest_request = {
                "symbol": "XAUUSD",
                "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
                "end_date": datetime.now().isoformat(),
                "initial_balance": 10000.0,
                "strategy_params": {
                    "risk_per_trade": 0.02,
                    "stop_loss": 0.01,
                    "take_profit": 0.02
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/backtest/run",
                json=backtest_request
            ) as resp:
                assert resp.status == 200
                result = await resp.json()
                assert "backtest_id" in result
                assert "status" in result
                logger.info("✅ Backtest execution test passed")
                return True
        except Exception as e:
            logger.error(f"❌ Backtest execution test failed: {e}")
            return False
    
    async def test_model_management(self):
        """Test model management endpoints"""
        try:
            # List models
            async with self.session.get(f"{self.base_url}/api/v1/models") as resp:
                assert resp.status == 200
                models = await resp.json()
                assert isinstance(models, list)
                logger.info("✅ Model listing test passed")
            
            # Get model info
            if models:
                model_id = models[0]["id"]
                async with self.session.get(
                    f"{self.base_url}/api/v1/models/{model_id}"
                ) as resp:
                    assert resp.status == 200
                    model_info = await resp.json()
                    assert "version" in model_info
                    logger.info("✅ Model info test passed")
            
            return True
        except Exception as e:
            logger.error(f"❌ Model management test failed: {e}")
            return False
    
    async def test_performance_monitoring(self):
        """Test performance monitoring"""
        try:
            # Check metrics endpoint
            async with self.session.get(f"{self.base_url}/metrics") as resp:
                assert resp.status == 200
                metrics = await resp.text()
                assert "http_requests_total" in metrics
                logger.info("✅ Performance monitoring test passed")
                return True
        except Exception as e:
            logger.error(f"❌ Performance monitoring test failed: {e}")
            return False
    
    async def test_database_connectivity(self):
        """Test database connectivity"""
        try:
            async with self.session.get(f"{self.base_url}/api/v1/system/database") as resp:
                assert resp.status == 200
                db_status = await resp.json()
                assert db_status["status"] == "connected"
                logger.info("✅ Database connectivity test passed")
                return True
        except Exception as e:
            logger.error(f"❌ Database connectivity test failed: {e}")
            return False
    
    async def test_redis_connectivity(self):
        """Test Redis connectivity"""
        try:
            async with self.session.get(f"{self.base_url}/api/v1/system/cache") as resp:
                assert resp.status == 200
                cache_status = await resp.json()
                assert cache_status["status"] == "connected"
                logger.info("✅ Redis connectivity test passed")
                return True
        except Exception as e:
            logger.error(f"❌ Redis connectivity test failed: {e}")
            return False
    
    async def test_load_handling(self):
        """Test system load handling"""
        try:
            # Send multiple concurrent requests
            tasks = []
            for i in range(10):
                task = self.session.get(f"{self.base_url}/health")
                tasks.append(task)
            
            start_time = time.time()
            responses = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # Check all responses are successful
            for resp in responses:
                assert resp.status == 200
                resp.close()
            
            # Check response time is reasonable
            avg_response_time = (end_time - start_time) / len(tasks)
            assert avg_response_time < 1.0  # Less than 1 second average
            
            logger.info(f"✅ Load handling test passed (avg response time: {avg_response_time:.3f}s)")
            return True
        except Exception as e:
            logger.error(f"❌ Load handling test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all integration tests"""
        logger.info("🚀 Starting Production Integration Tests")
        
        await self.setup()
        
        tests = [
            ("Health Endpoint", self.test_health_endpoint),
            ("Data Ingestion", self.test_data_ingestion),
            ("Signal Generation", self.test_signal_generation),
            ("Backtest Execution", self.test_backtest_execution),
            ("Model Management", self.test_model_management),
            ("Performance Monitoring", self.test_performance_monitoring),
            ("Database Connectivity", self.test_database_connectivity),
            ("Redis Connectivity", self.test_redis_connectivity),
            ("Load Handling", self.test_load_handling),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"Running {test_name} test...")
            try:
                result = await test_func()
                results[test_name] = result
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                results[test_name] = False
        
        await self.teardown()
        
        # Summary
        passed = sum(results.values())
        total = len(results)
        
        logger.info(f"\n📊 Test Results Summary:")
        logger.info(f"Passed: {passed}/{total}")
        logger.info(f"Success Rate: {passed/total*100:.1f}%")
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {test_name}: {status}")
        
        return results

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Integration Tests")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--test", help="Run specific test")
    
    args = parser.parse_args()
    
    tester = ProductionIntegrationTest(args.url)
    
    async def main():
        if args.test:
            # Run specific test
            test_method = getattr(tester, f"test_{args.test}", None)
            if test_method:
                await tester.setup()
                result = await test_method()
                await tester.teardown()
                logger.info(f"Test result: {'PASS' if result else 'FAIL'}")
            else:
                logger.error(f"Test '{args.test}' not found")
        else:
            # Run all tests
            await tester.run_all_tests()
    
    asyncio.run(main())
