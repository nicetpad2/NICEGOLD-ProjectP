#!/bin/bash
# NICEGOLD Enterprise Start Script

echo "🚀 Starting NICEGOLD Enterprise..."

# Load environment
if [ -f .env.production ]; then
    export $(cat .env.production | grep -v '^#' | xargs)
fi

# Start API server
echo "Starting API server..."
nohup python -m uvicorn src.api:app --host $API_HOST --port $API_PORT --workers 4 > logs/api.log 2>&1 &
echo $! > run/api.pid

# Start Dashboard
echo "Starting Dashboard..."
nohup streamlit run single_user_dashboard.py --server.address $DASHBOARD_HOST --server.port $DASHBOARD_PORT > logs/dashboard.log 2>&1 &
echo $! > run/dashboard.pid

echo "✅ NICEGOLD Enterprise started successfully!"
echo "API: http://$API_HOST:$API_PORT"
echo "Dashboard: http://$DASHBOARD_HOST:$DASHBOARD_PORT"
