#!/bin/bash

# =============================================================================
# NICEGOLD Enterprise Production Deployment Script
# =============================================================================

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_ENV=${DEPLOYMENT_ENV:-production}
PROJECT_NAME="nicegold-enterprise"
NAMESPACE="nicegold"
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"localhost:5000"}
VERSION=${VERSION:-$(date +%Y%m%d-%H%M%S)}

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}🚀 NICEGOLD Enterprise Deployment${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}Environment: ${DEPLOYMENT_ENV}${NC}"
echo -e "${GREEN}Version: ${VERSION}${NC}"
echo -e "${GREEN}Registry: ${DOCKER_REGISTRY}${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${BLUE}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="nicegold-enterprise"
DOCKER_COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env.production"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    local missing_deps=()
    
    # Check Docker
    if ! command_exists docker; then
        missing_deps+=("docker")
    fi
    
    # Check Docker Compose
    if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
        missing_deps+=("docker-compose")
    fi
    
    # Check Python
    if ! command_exists python3; then
        missing_deps+=("python3")
    fi
    
    # Check Git
    if ! command_exists git; then
        missing_deps+=("git")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        error "Missing dependencies: ${missing_deps[*]}"
        error "Please install the missing dependencies and try again."
        exit 1
    fi
    
    log "All prerequisites satisfied ✅"
}

# Setup environment
setup_environment() {
    log "Setting up environment..."
    
    # Create .env.production if it doesn't exist
    if [ ! -f "$ENV_FILE" ]; then
        info "Creating $ENV_FILE..."
        cat > "$ENV_FILE" << EOF
# Production Environment Configuration
ENVIRONMENT=production

# Database
POSTGRES_DB=nicegold
POSTGRES_USER=nicegold
POSTGRES_PASSWORD=$(openssl rand -base64 32)
DATABASE_URL=postgresql://nicegold:\${POSTGRES_PASSWORD}@postgres:5432/nicegold

# Redis
REDIS_PASSWORD=$(openssl rand -base64 32)
REDIS_URL=redis://:\${REDIS_PASSWORD}@redis:6379

# JWT
JWT_SECRET_KEY=$(openssl rand -base64 64)
JWT_EXPIRE_MINUTES=60

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=4

# MLflow
MLFLOW_URI=http://mlflow:5000

# S3/MinIO
S3_ENDPOINT=http://minio:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=$(openssl rand -base64 32)
MODEL_BUCKET=nicegold-models

# Risk Management
INITIAL_BALANCE=100000.0
MAX_POSITION_SIZE=0.05
MAX_DAILY_LOSS=0.02

# Monitoring
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_URL=http://grafana:3000

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
EOF
        log "Environment file created ✅"
    else
        info "Environment file already exists"
    fi
    
    # Create required directories
    mkdir -p logs
    mkdir -p models
    mkdir -p data
    mkdir -p tmp_logs
    mkdir -p monitoring/grafana/data
    mkdir -p monitoring/prometheus/data
    
    log "Environment setup complete ✅"
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    # Build main application image
    docker build -t ${PROJECT_NAME}:latest .
    
    log "Docker images built ✅"
}

# Initialize database
init_database() {
    log "Initializing database..."
    
    # Create init SQL script
    cat > scripts/init.sql << EOF
-- Create database extensions
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create tables for model registry
CREATE TABLE IF NOT EXISTS model_registry (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    algorithm VARCHAR(100) NOT NULL,
    framework VARCHAR(100) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    created_by VARCHAR(100) NOT NULL,
    description TEXT,
    hyperparameters TEXT,
    metrics TEXT,
    dataset_hash VARCHAR(255) NOT NULL,
    features TEXT,
    target VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size INTEGER NOT NULL,
    checksum VARCHAR(255) NOT NULL
);

-- Create tables for model deployments
CREATE TABLE IF NOT EXISTS model_deployments (
    id VARCHAR(255) PRIMARY KEY,
    model_id VARCHAR(255) NOT NULL,
    environment VARCHAR(50) NOT NULL,
    deployed_at TIMESTAMP NOT NULL,
    deployed_by VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    endpoint_url VARCHAR(500),
    health_check_url VARCHAR(500),
    last_health_check TIMESTAMP,
    performance_metrics TEXT
);

-- Create tables for positions
CREATE TABLE IF NOT EXISTS positions (
    id VARCHAR(255) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity FLOAT NOT NULL,
    entry_price FLOAT NOT NULL,
    current_price FLOAT NOT NULL,
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP,
    exit_price FLOAT,
    stop_loss FLOAT,
    take_profit FLOAT,
    status VARCHAR(20) NOT NULL,
    pnl FLOAT DEFAULT 0.0,
    pnl_pct FLOAT DEFAULT 0.0,
    margin_used FLOAT DEFAULT 0.0,
    risk_score FLOAT DEFAULT 0.0,
    metadata TEXT
);

-- Create tables for risk events
CREATE TABLE IF NOT EXISTS risk_events (
    id VARCHAR(255) PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    description TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    position_id VARCHAR(255),
    symbol VARCHAR(20),
    action_taken TEXT,
    resolved BOOLEAN DEFAULT FALSE,
    metadata TEXT
);

-- Create tables for market data (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS market_data (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    open FLOAT NOT NULL,
    high FLOAT NOT NULL,
    low FLOAT NOT NULL,
    close FLOAT NOT NULL,
    volume INTEGER NOT NULL,
    PRIMARY KEY (timestamp, symbol, timeframe)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_model_registry_status ON model_registry(status);
CREATE INDEX IF NOT EXISTS idx_model_registry_created_at ON model_registry(created_at);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_risk_events_timestamp ON risk_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timeframe ON market_data(symbol, timeframe);

-- Insert initial data
INSERT INTO model_registry (
    id, name, version, algorithm, framework, created_at, created_by,
    description, hyperparameters, metrics, dataset_hash, features, target,
    status, file_path, file_size, checksum
) VALUES (
    'xauusd_predictor_v1',
    'XAUUSD Predictor',
    '1.0.0',
    'CatBoost',
    'catboost',
    NOW(),
    'system',
    'Initial XAUUSD prediction model',
    '{"iterations": 1000, "depth": 6}',
    '{"auc": 0.75, "accuracy": 0.68}',
    'initial_dataset',
    '["close", "sma_20", "rsi", "macd"]',
    'signal',
    'production',
    'models/xauusd_predictor_v1/1.0.0/model.pkl',
    1024000,
    'initial_checksum'
) ON CONFLICT (id) DO NOTHING;

EOF

    mkdir -p scripts
    log "Database initialization script created ✅"
}

# Start services
start_services() {
    log "Starting services with Docker Compose..."
    
    # Stop any existing services
    docker-compose -f "$DOCKER_COMPOSE_FILE" down || true
    
    # Start services
    docker-compose -f "$DOCKER_COMPOSE_FILE" --env-file "$ENV_FILE" up -d
    
    log "Services started ✅"
}

# Health check
health_check() {
    log "Performing health checks..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        info "Health check attempt $attempt/$max_attempts..."
        
        # Check API health
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            log "API is healthy ✅"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            error "Health check failed after $max_attempts attempts"
            return 1
        fi
        
        sleep 10
        ((attempt++))
    done
    
    # Additional service checks
    info "Checking individual services..."
    
    # Check PostgreSQL
    if docker-compose exec -T postgres pg_isready -U nicegold >/dev/null 2>&1; then
        log "PostgreSQL is ready ✅"
    else
        warn "PostgreSQL health check failed"
    fi
    
    # Check Redis
    if docker-compose exec -T redis redis-cli ping >/dev/null 2>&1; then
        log "Redis is ready ✅"
    else
        warn "Redis health check failed"
    fi
    
    log "Health checks completed ✅"
}

# Run tests
run_tests() {
    log "Running production validation tests..."
    
    # Wait a bit for services to stabilize
    sleep 30
    
    # Run validation
    if python3 tests/production_validator.py --api-url http://localhost:8000 --output validation_report.json; then
        log "Production validation passed ✅"
    else
        error "Production validation failed ❌"
        warn "Check validation_report.json for details"
        return 1
    fi
}

# Show status
show_status() {
    log "System Status:"
    echo ""
    
    # Service status
    docker-compose ps
    echo ""
    
    # API endpoints
    info "Available endpoints:"
    echo "  • API Documentation: http://localhost:8000/docs"
    echo "  • Health Check: http://localhost:8000/health"
    echo "  • Metrics: http://localhost:8000/metrics"
    echo "  • Dashboard: http://localhost:8501 (if Streamlit is running)"
    echo ""
    
    # Database info
    info "Database connection:"
    echo "  • Host: localhost:5432"
    echo "  • Database: nicegold"
    echo "  • User: nicegold"
    echo ""
    
    # Logs
    info "To view logs:"
    echo "  • API logs: docker-compose logs -f nicegold-app"
    echo "  • All logs: docker-compose logs -f"
    echo ""
    
    # Monitoring
    info "Monitoring:"
    echo "  • Prometheus: http://localhost:9090"
    echo "  • Grafana: http://localhost:3000"
    echo ""
}

# Stop services
stop_services() {
    log "Stopping services..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" down
    log "Services stopped ✅"
}

# Cleanup
cleanup() {
    log "Cleaning up..."
    
    # Stop services
    docker-compose -f "$DOCKER_COMPOSE_FILE" down -v --remove-orphans
    
    # Remove images
    docker rmi ${PROJECT_NAME}:latest || true
    
    # Clean volumes
    docker volume prune -f
    
    log "Cleanup completed ✅"
}

# Update system
update_system() {
    log "Updating system..."
    
    # Pull latest changes
    git pull origin main || warn "Failed to pull latest changes"
    
    # Rebuild images
    build_images
    
    # Restart services
    docker-compose -f "$DOCKER_COMPOSE_FILE" --env-file "$ENV_FILE" up -d --force-recreate
    
    # Health check
    health_check
    
    log "System updated ✅"
}

# Backup system
backup_system() {
    log "Creating system backup..."
    
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup database
    docker-compose exec -T postgres pg_dump -U nicegold nicegold > "$backup_dir/database.sql"
    
    # Backup environment
    cp "$ENV_FILE" "$backup_dir/"
    
    # Backup models (if using local storage)
    if [ -d "models" ]; then
        cp -r models "$backup_dir/"
    fi
    
    # Create backup archive
    tar -czf "${backup_dir}.tar.gz" -C backups "$(basename "$backup_dir")"
    rm -rf "$backup_dir"
    
    log "Backup created: ${backup_dir}.tar.gz ✅"
}

# Main function
main() {
    case "${1:-}" in
        "setup")
            log "Starting production setup..."
            check_prerequisites
            setup_environment
            init_database
            build_images
            start_services
            health_check
            run_tests
            show_status
            log "Production setup completed successfully! 🚀"
            ;;
        "start")
            log "Starting services..."
            start_services
            health_check
            show_status
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            log "Restarting services..."
            stop_services
            start_services
            health_check
            ;;
        "status")
            show_status
            ;;
        "test")
            run_tests
            ;;
        "update")
            update_system
            ;;
        "backup")
            backup_system
            ;;
        "cleanup")
            cleanup
            ;;
        "logs")
            docker-compose logs -f "${2:-}"
            ;;
        *)
            echo "NICEGOLD Enterprise Production Management"
            echo ""
            echo "Usage: $0 <command>"
            echo ""
            echo "Commands:"
            echo "  setup     - Full production setup (first time)"
            echo "  start     - Start all services"
            echo "  stop      - Stop all services"
            echo "  restart   - Restart all services"
            echo "  status    - Show system status"
            echo "  test      - Run production validation tests"
            echo "  update    - Update system with latest changes"
            echo "  backup    - Create system backup"
            echo "  cleanup   - Clean up all resources"
            echo "  logs      - Show logs (optionally specify service)"
            echo ""
            echo "Examples:"
            echo "  $0 setup                    # First time setup"
            echo "  $0 start                    # Start services"
            echo "  $0 logs nicegold-app        # Show API logs"
            echo "  $0 test                     # Run validation"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
