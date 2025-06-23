#!/bin/bash
# NICEGOLD Enterprise Status Script

echo "📊 NICEGOLD Enterprise Status"
echo "================================"

# Check API server
if [ -f run/api.pid ]; then
    PID=$(cat run/api.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "✅ API Server: Running (PID: $PID)"
    else
        echo "❌ API Server: Not running"
        rm -f run/api.pid
    fi
else
    echo "❌ API Server: Not running"
fi

# Check Dashboard
if [ -f run/dashboard.pid ]; then
    PID=$(cat run/dashboard.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "✅ Dashboard: Running (PID: $PID)"
    else
        echo "❌ Dashboard: Not running"
        rm -f run/dashboard.pid
    fi
else
    echo "❌ Dashboard: Not running"
fi

# Check database
if [ -f database/production.db ]; then
    echo "✅ Database: Available"
else
    echo "❌ Database: Not found"
fi

echo ""
echo "System Information:"
echo "==================="
df -h . | tail -1 | awk '{print "Disk Usage: " $5 " (" $3 " used, " $4 " available)"}'
free -h | grep Mem | awk '{print "Memory Usage: " $3 "/" $2}'
