#!/bin/bash
# Production startup script for NICEGOLD Enterprise

set -e  # Exit on any error

echo "🚀 Starting NICEGOLD Enterprise Production Setup..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root (not recommended for production)
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root is not recommended for production!"
fi

# Check environment
ENVIRONMENT=${ENVIRONMENT:-production}
print_status "Environment: $ENVIRONMENT"

# Check required environment variables
required_vars=(
    "DATABASE_URL"
    "REDIS_URL"
    "JWT_SECRET_KEY"
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        print_error "Required environment variable $var is not set!"
        exit 1
    fi
done

print_status "All required environment variables are set"

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3.10 -m venv venv
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install production dependencies
print_status "Installing production dependencies..."
pip install -r requirements.txt

# Install development dependencies if not in production
if [ "$ENVIRONMENT" != "production" ]; then
    print_status "Installing development dependencies..."
    pip install -r dev-requirements.txt
fi

# Check database connectivity
print_status "Checking database connectivity..."
python -c "
import asyncio
import asyncpg
import os
import sys

async def test_db():
    try:
        conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
        await conn.fetchval('SELECT version()')
        await conn.close()
        print('✅ Database connection successful')
        return True
    except Exception as e:
        print(f'❌ Database connection failed: {e}')
        return False

if not asyncio.run(test_db()):
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    print_error "Database connectivity check failed!"
    exit 1
fi

# Check Redis connectivity
print_status "Checking Redis connectivity..."
python -c "
import redis.asyncio as redis
import asyncio
import os
import sys

async def test_redis():
    try:
        r = redis.from_url(os.getenv('REDIS_URL'))
        await r.ping()
        await r.close()
        print('✅ Redis connection successful')
        return True
    except Exception as e:
        print(f'❌ Redis connection failed: {e}')
        return False

if not asyncio.run(test_redis()):
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    print_error "Redis connectivity check failed!"
    exit 1
fi

# Run database migrations
print_status "Running database migrations..."
python -c "
import asyncio
from src.database_manager import DatabaseManager

async def migrate():
    db = DatabaseManager()
    await db.initialize()
    print('✅ Database migrations completed')
    await db.close()

asyncio.run(migrate())
"

# Validate configuration
print_status "Validating configuration..."
python -c "
from src.config import *
print('✅ Configuration validation passed')
"

# Run tests if not in production
if [ "$ENVIRONMENT" != "production" ]; then
    print_status "Running tests..."
    pytest tests/ -v --tb=short
    
    if [ $? -ne 0 ]; then
        print_warning "Some tests failed, but continuing..."
    fi
fi

# Check for security issues
print_status "Running security checks..."
if command -v bandit &> /dev/null; then
    bandit -r src/ -f json -o security-report.json || print_warning "Security issues detected"
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p logs/$(date +%Y-%m-%d)/fold0
mkdir -p models
mkdir -p data
mkdir -p output_default

# Set proper permissions
chmod 755 logs models data output_default
chmod -R 644 logs/* 2>/dev/null || true

# Start the application
print_status "🎯 Starting NICEGOLD Enterprise API..."

# Production startup
if [ "$ENVIRONMENT" = "production" ]; then
    # Use gunicorn for production
    if command -v gunicorn &> /dev/null; then
        print_status "Starting with Gunicorn..."
        exec gunicorn src.api:app \
            --workers 4 \
            --worker-class uvicorn.workers.UvicornWorker \
            --bind 0.0.0.0:8000 \
            --access-logfile logs/access.log \
            --error-logfile logs/error.log \
            --log-level info \
            --timeout 30 \
            --keepalive 2 \
            --max-requests 1000 \
            --max-requests-jitter 50 \
            --worker-connections 1000
    else
        print_status "Starting with Uvicorn..."
        exec uvicorn src.api:app \
            --host 0.0.0.0 \
            --port 8000 \
            --workers 4 \
            --log-config logging.yaml \
            --access-log \
            --no-use-colors
    fi
else
    # Development startup
    print_status "Starting in development mode..."
    exec uvicorn src.api:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload \
        --log-level debug
fi
