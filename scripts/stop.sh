#!/bin/bash
# NICEGOLD Enterprise Stop Script

echo "🛑 Stopping NICEGOLD Enterprise..."

# Stop API server
if [ -f run/api.pid ]; then
    PID=$(cat run/api.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "API server stopped"
    fi
    rm -f run/api.pid
fi

# Stop Dashboard
if [ -f run/dashboard.pid ]; then
    PID=$(cat run/dashboard.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "Dashboard stopped"
    fi
    rm -f run/dashboard.pid
fi

echo "✅ NICEGOLD Enterprise stopped successfully!"
