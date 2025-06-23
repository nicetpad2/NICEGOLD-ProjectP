"""
Production Configuration Management System
จัดการ configuration สำหรับ environments ต่างๆ
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime

@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str
    pool_size: int = 20
    max_overflow: int = 0
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False

@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 100
    socket_timeout: int = 30
    socket_connect_timeout: int = 30

@dataclass
class KafkaConfig:
    """Kafka configuration"""
    bootstrap_servers: str = "localhost:9092"
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    topics: Dict[str, str] = field(default_factory=lambda: {
        'market_data': 'market_data',
        'signals': 'trading_signals',
        'orders': 'orders',
        'risk_events': 'risk_events'
    })

@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 60
    max_request_size: int = 16 * 1024 * 1024  # 16MB
    cors_origins: list = field(default_factory=lambda: ["*"])
    rate_limit: str = "100/minute"
    jwt_secret: str = "your-secret-key-change-this"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

@dataclass
class TradingConfig:
    """Trading configuration"""
    initial_balance: float = 100000.0
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
class MLConfig:
    """ML configuration"""
    model_registry_url: str = "http://localhost:5000"
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    model_bucket: str = "nicegold-models"
    feature_store_url: Optional[str] = None
    auto_retrain_enabled: bool = True
    retrain_interval_hours: int = 24
    min_samples_for_retrain: int = 1000
    model_performance_threshold: float = 0.6

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    prometheus_url: str = "http://localhost:9090"
    grafana_url: str = "http://localhost:3000"
    alertmanager_url: str = "http://localhost:9093"
    log_level: str = "INFO"
    metrics_enabled: bool = True
    health_check_interval: int = 30
    alert_email: Optional[str] = None
    slack_webhook: Optional[str] = None

@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_https: bool = True
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    enable_api_key_auth: bool = True
    enable_rate_limiting: bool = True
    enable_request_logging: bool = True
    allowed_hosts: list = field(default_factory=lambda: ["localhost", "127.0.0.1"])
    cors_enabled: bool = True
    csrf_enabled: bool = False

@dataclass
class ProductionConfig:
    """Complete production configuration"""
    environment: str = "production"
    debug: bool = False
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

class ConfigManager:
    """Configuration manager for different environments"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Environment variable mapping
        self.env_mapping = {
            'DATABASE_URL': 'database.url',
            'REDIS_HOST': 'redis.host',
            'REDIS_PORT': 'redis.port',
            'REDIS_PASSWORD': 'redis.password',
            'KAFKA_BOOTSTRAP_SERVERS': 'kafka.bootstrap_servers',
            'API_HOST': 'api.host',
            'API_PORT': 'api.port',
            'JWT_SECRET': 'api.jwt_secret',
            'INITIAL_BALANCE': 'trading.initial_balance',
            'MAX_POSITION_SIZE': 'trading.max_position_size',
            'S3_ENDPOINT': 'ml.s3_endpoint',
            'S3_ACCESS_KEY': 'ml.s3_access_key',
            'S3_SECRET_KEY': 'ml.s3_secret_key',
            'LOG_LEVEL': 'monitoring.log_level',
            'ENABLE_HTTPS': 'security.enable_https',
        }
    
    def create_default_configs(self):
        """Create default configuration files for all environments"""
        environments = ['development', 'staging', 'production']
        
        for env in environments:
            config = self._get_default_config(env)
            self.save_config(config, env)
            self.logger.info(f"Created default config for {env}")
    
    def _get_default_config(self, environment: str) -> ProductionConfig:
        """Get default configuration for environment"""
        base_config = ProductionConfig(environment=environment)
        
        if environment == 'development':
            base_config.debug = True
            base_config.database.url = "sqlite:///nicegold_dev.db"
            base_config.database.echo = True
            base_config.api.workers = 1
            base_config.security.enable_https = False
            base_config.monitoring.log_level = "DEBUG"
            
        elif environment == 'staging':
            base_config.debug = False
            base_config.database.url = "postgresql://nicegold:password@localhost/nicegold_staging"
            base_config.redis.host = "redis-staging"
            base_config.kafka.bootstrap_servers = "kafka-staging:9092"
            base_config.security.enable_https = True
            
        else:  # production
            base_config.debug = False
            base_config.database.url = "postgresql://nicegold:password@postgres:5432/nicegold"
            base_config.redis.host = "redis"
            base_config.kafka.bootstrap_servers = "kafka:9092"
            base_config.api.workers = 4
            base_config.security.enable_https = True
            base_config.security.enable_api_key_auth = True
            base_config.security.enable_rate_limiting = True
            base_config.monitoring.metrics_enabled = True
        
        return base_config
    
    def load_config(self, environment: str) -> ProductionConfig:
        """Load configuration for environment"""
        config_file = self.config_dir / f"{environment}.yaml"
        
        if not config_file.exists():
            self.logger.warning(f"Config file not found: {config_file}")
            return self._get_default_config(environment)
        
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Apply environment variable overrides
            config_data = self._apply_env_overrides(config_data)
            
            # Convert to config object
            config = self._dict_to_config(config_data)
            
            self.logger.info(f"Loaded config for {environment}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self._get_default_config(environment)

# Global config manager instance
config_manager = ConfigManager()

def get_config(environment: str = None) -> ProductionConfig:
    """Get configuration for environment"""
    if environment is None:
        environment = os.getenv('ENVIRONMENT', 'development')
    
    return config_manager.load_config(environment)
