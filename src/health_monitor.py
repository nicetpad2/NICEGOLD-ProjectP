            from aiokafka import AIOKafkaProducer
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
            from pathlib import Path
            from src.database_manager import DatabaseManager
from typing import Dict, Any, List, Optional
import asyncio
            import httpx
import json
import logging
            import os
import psutil
            import redis.asyncio as redis
import time
"""
Comprehensive health check and monitoring system
"""

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DOWN = "down"

@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None

class HealthMonitor:
    """Comprehensive system health monitoring"""

    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.thresholds = {
            'cpu_usage': 80.0, 
            'memory_usage': 85.0, 
            'disk_usage': 90.0, 
            'response_time_ms': 1000.0, 
            'error_rate': 5.0
        }

    async def check_system_health(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {
            'overall_status': HealthStatus.HEALTHY, 
            'timestamp': datetime.utcnow().isoformat(), 
            'checks': {}, 
            'summary': {
                'healthy': 0, 
                'warning': 0, 
                'critical': 0, 
                'down': 0
            }
        }

        # Run individual checks
        checks = [
            self._check_cpu(), 
            self._check_memory(), 
            self._check_disk(), 
            self._check_database(), 
            self._check_redis(), 
            self._check_kafka(), 
            self._check_api_endpoints(), 
            self._check_model_availability()
        ]

        check_results = await asyncio.gather(*checks, return_exceptions = True)

        # Process results
        for check in check_results:
            if isinstance(check, Exception):
                logger.error(f"Health check failed: {check}")
                continue

            results['checks'][check.name] = {
                'status': check.status.value, 
                'message': check.message, 
                'timestamp': check.timestamp.isoformat(), 
                'response_time_ms': check.response_time_ms, 
                'details': check.details
            }

            # Update summary
            results['summary'][check.status.value] += 1

            # Update overall status
            if check.status == HealthStatus.CRITICAL or check.status == HealthStatus.DOWN:
                results['overall_status'] = HealthStatus.CRITICAL
            elif check.status == HealthStatus.WARNING and results['overall_status'] == HealthStatus.HEALTHY:
                results['overall_status'] = HealthStatus.WARNING

        return results

    async def _check_cpu(self) -> HealthCheck:
        """Check CPU usage"""
        start_time = time.time()

        try:
            cpu_percent = psutil.cpu_percent(interval = 1)
            response_time = (time.time() - start_time) * 1000

            if cpu_percent > self.thresholds['cpu_usage']:
                status = HealthStatus.CRITICAL
                message = f"High CPU usage: {cpu_percent:.1f}%"
            elif cpu_percent > self.thresholds['cpu_usage'] * 0.8:
                status = HealthStatus.WARNING
                message = f"Elevated CPU usage: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"

            return HealthCheck(
                name = "cpu", 
                status = status, 
                message = message, 
                timestamp = datetime.utcnow(), 
                response_time_ms = response_time, 
                details = {
                    'cpu_percent': cpu_percent, 
                    'cpu_count': psutil.cpu_count(), 
                    'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                }
            )

        except Exception as e:
            return HealthCheck(
                name = "cpu", 
                status = HealthStatus.DOWN, 
                message = f"CPU check failed: {e}", 
                timestamp = datetime.utcnow()
            )

    async def _check_memory(self) -> HealthCheck:
        """Check memory usage"""
        start_time = time.time()

        try:
            memory = psutil.virtual_memory()
            response_time = (time.time() - start_time) * 1000

            if memory.percent > self.thresholds['memory_usage']:
                status = HealthStatus.CRITICAL
                message = f"High memory usage: {memory.percent:.1f}%"
            elif memory.percent > self.thresholds['memory_usage'] * 0.8:
                status = HealthStatus.WARNING
                message = f"Elevated memory usage: {memory.percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory.percent:.1f}%"

            return HealthCheck(
                name = "memory", 
                status = status, 
                message = message, 
                timestamp = datetime.utcnow(), 
                response_time_ms = response_time, 
                details = {
                    'total_gb': memory.total / (1024**3), 
                    'available_gb': memory.available / (1024**3), 
                    'used_gb': memory.used / (1024**3), 
                    'percent': memory.percent
                }
            )

        except Exception as e:
            return HealthCheck(
                name = "memory", 
                status = HealthStatus.DOWN, 
                message = f"Memory check failed: {e}", 
                timestamp = datetime.utcnow()
            )

    async def _check_disk(self) -> HealthCheck:
        """Check disk usage"""
        start_time = time.time()

        try:
            disk = psutil.disk_usage('/')
            response_time = (time.time() - start_time) * 1000

            disk_percent = (disk.used / disk.total) * 100

            if disk_percent > self.thresholds['disk_usage']:
                status = HealthStatus.CRITICAL
                message = f"High disk usage: {disk_percent:.1f}%"
            elif disk_percent > self.thresholds['disk_usage'] * 0.8:
                status = HealthStatus.WARNING
                message = f"Elevated disk usage: {disk_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {disk_percent:.1f}%"

            return HealthCheck(
                name = "disk", 
                status = status, 
                message = message, 
                timestamp = datetime.utcnow(), 
                response_time_ms = response_time, 
                details = {
                    'total_gb': disk.total / (1024**3), 
                    'free_gb': disk.free / (1024**3), 
                    'used_gb': disk.used / (1024**3), 
                    'percent': disk_percent
                }
            )

        except Exception as e:
            return HealthCheck(
                name = "disk", 
                status = HealthStatus.DOWN, 
                message = f"Disk check failed: {e}", 
                timestamp = datetime.utcnow()
            )

    async def _check_database(self) -> HealthCheck:
        """Check database connectivity"""
        start_time = time.time()

        try:

            db = DatabaseManager()
            await db.initialize()

            # Test query
            async with db.pg_pool.acquire() as conn:
                result = await conn.fetchval('SELECT version()')

            response_time = (time.time() - start_time) * 1000

            if response_time > self.thresholds['response_time_ms']:
                status = HealthStatus.WARNING
                message = f"Database slow response: {response_time:.0f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = f"Database healthy: {response_time:.0f}ms"

            return HealthCheck(
                name = "database", 
                status = status, 
                message = message, 
                timestamp = datetime.utcnow(), 
                response_time_ms = response_time, 
                details = {
                    'version': result, 
                    'pool_size': db.pg_pool.get_size() if db.pg_pool else 0, 
                    'pool_available': db.pg_pool.get_idle_size() if db.pg_pool else 0
                }
            )

        except Exception as e:
            return HealthCheck(
                name = "database", 
                status = HealthStatus.DOWN, 
                message = f"Database connection failed: {e}", 
                timestamp = datetime.utcnow()
            )

    async def _check_redis(self) -> HealthCheck:
        """Check Redis connectivity"""
        start_time = time.time()

        try:

            client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
            await client.ping()

            # Test set/get
            test_key = f"health_check_{int(time.time())}"
            await client.setex(test_key, 10, "test")
            result = await client.get(test_key)
            await client.delete(test_key)

            response_time = (time.time() - start_time) * 1000

            if response_time > self.thresholds['response_time_ms']:
                status = HealthStatus.WARNING
                message = f"Redis slow response: {response_time:.0f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = f"Redis healthy: {response_time:.0f}ms"

            await client.close()

            return HealthCheck(
                name = "redis", 
                status = status, 
                message = message, 
                timestamp = datetime.utcnow(), 
                response_time_ms = response_time, 
                details = {
                    'test_successful': result == "test"
                }
            )

        except Exception as e:
            return HealthCheck(
                name = "redis", 
                status = HealthStatus.DOWN, 
                message = f"Redis connection failed: {e}", 
                timestamp = datetime.utcnow()
            )

    async def _check_kafka(self) -> HealthCheck:
        """Check Kafka connectivity"""
        start_time = time.time()

        try:

            producer = AIOKafkaProducer(
                bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
            )
            await producer.start()

            # Test message
            test_topic = "health_check"
            await producer.send(test_topic, b"health_check_message")

            await producer.stop()

            response_time = (time.time() - start_time) * 1000

            if response_time > self.thresholds['response_time_ms']:
                status = HealthStatus.WARNING
                message = f"Kafka slow response: {response_time:.0f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = f"Kafka healthy: {response_time:.0f}ms"

            return HealthCheck(
                name = "kafka", 
                status = status, 
                message = message, 
                timestamp = datetime.utcnow(), 
                response_time_ms = response_time
            )

        except Exception as e:
            return HealthCheck(
                name = "kafka", 
                status = HealthStatus.WARNING,  # Kafka is optional for basic functionality
                message = f"Kafka connection failed: {e}", 
                timestamp = datetime.utcnow()
            )

    async def _check_api_endpoints(self) -> HealthCheck:
        """Check critical API endpoints"""
        start_time = time.time()

        try:

            # Test internal health endpoint
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8000/health")
                response.raise_for_status()

            response_time = (time.time() - start_time) * 1000

            if response_time > self.thresholds['response_time_ms']:
                status = HealthStatus.WARNING
                message = f"API slow response: {response_time:.0f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = f"API healthy: {response_time:.0f}ms"

            return HealthCheck(
                name = "api_endpoints", 
                status = status, 
                message = message, 
                timestamp = datetime.utcnow(), 
                response_time_ms = response_time, 
                details = {
                    'status_code': response.status_code, 
                    'response_size': len(response.content)
                }
            )

        except Exception as e:
            return HealthCheck(
                name = "api_endpoints", 
                status = HealthStatus.DOWN, 
                message = f"API endpoint check failed: {e}", 
                timestamp = datetime.utcnow()
            )

    async def _check_model_availability(self) -> HealthCheck:
        """Check ML model availability"""
        start_time = time.time()

        try:

            model_dir = Path("models")

            # Check if models directory exists and has models
            if not model_dir.exists():
                status = HealthStatus.WARNING
                message = "Models directory not found"
            else:
                model_files = list(model_dir.glob("*.pkl")) + list(model_dir.glob("*.joblib"))

                if len(model_files) == 0:
                    status = HealthStatus.WARNING
                    message = "No model files found"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Found {len(model_files)} model files"

            response_time = (time.time() - start_time) * 1000

            return HealthCheck(
                name = "models", 
                status = status, 
                message = message, 
                timestamp = datetime.utcnow(), 
                response_time_ms = response_time, 
                details = {
                    'model_count': len(model_files) if 'model_files' in locals() else 0, 
                    'model_dir_exists': model_dir.exists()
                }
            )

        except Exception as e:
            return HealthCheck(
                name = "models", 
                status = HealthStatus.WARNING, 
                message = f"Model check failed: {e}", 
                timestamp = datetime.utcnow()
            )

# Global health monitor instance
health_monitor = HealthMonitor()

async def get_health_status() -> Dict[str, Any]:
    """Get current system health status"""
    return await health_monitor.check_system_health()

# Background health monitoring
class BackgroundHealthMonitor:
    """Background health monitoring service"""

    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.running = False
        self.last_status: Optional[Dict[str, Any]] = None

    async def start(self):
        """Start background monitoring"""
        self.running = True
        while self.running:
            try:
                self.last_status = await health_monitor.check_system_health()

                # Log critical issues
                if self.last_status['overall_status'] == HealthStatus.CRITICAL.value:
                    logger.critical(f"System health critical: {self.last_status}")
                elif self.last_status['overall_status'] == HealthStatus.WARNING.value:
                    logger.warning(f"System health warning: {self.last_status}")

                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Background health check failed: {e}")
                await asyncio.sleep(10)  # Short retry interval

    def stop(self):
        """Stop background monitoring"""
        self.running = False

    def get_last_status(self) -> Optional[Dict[str, Any]]:
        """Get last health check result"""
        return self.last_status

# Global background monitor
background_monitor = BackgroundHealthMonitor()

if __name__ == "__main__":
    async def main():
        """Test health monitoring"""
        logging.basicConfig(level = logging.INFO)

        health_status = await get_health_status()
        print(json.dumps(health_status, indent = 2))

    asyncio.run(main())