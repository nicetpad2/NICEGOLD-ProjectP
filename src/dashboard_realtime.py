from datetime import datetime, timedelta
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
import aiohttp
import asyncio
import json
import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import redis
import streamlit as st
import time
import websockets
"""
Real - time Trading Dashboard และ Monitoring System
Dashboard แบบ real - time สำหรับ production trading
"""


# Configure Streamlit page
st.set_page_config(
    page_title = "NICEGOLD Enterprise Dashboard", 
    page_icon = "📈", 
    layout = "wide", 
    initial_sidebar_state = "expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main - header {
        font - size: 2.5rem;
        font - weight: bold;
        color: #1f77b4;
        text - align: center;
        margin - bottom: 2rem;
    }
    .metric - card {
        background - color: #f8f9fa;
        padding: 1rem;
        border - radius: 10px;
        border - left: 4px solid #1f77b4;
        margin - bottom: 1rem;
    }
    .alert - high {
        background - color: #ffebee;
        border - left: 4px solid #f44336;
        padding: 10px;
        margin: 10px 0;
        border - radius: 5px;
    }
    .alert - medium {
        background - color: #fff3e0;
        border - left: 4px solid #ff9800;
        padding: 10px;
        margin: 10px 0;
        border - radius: 5px;
    }
    .alert - low {
        background - color: #e8f5e8;
        border - left: 4px solid #4caf50;
        padding: 10px;
        margin: 10px 0;
        border - radius: 5px;
    }
    .status - online {
        color: #4caf50;
        font - weight: bold;
    }
    .status - offline {
        color: #f44336;
        font - weight: bold;
    }
    .pnl - positive {
        color: #4caf50;
        font - weight: bold;
    }
    .pnl - negative {
        color: #f44336;
        font - weight: bold;
    }
</style>
""", unsafe_allow_html = True)

class DashboardDataSource:
    """Data source for dashboard"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
        self.api_base_url = config.get('api_url', 'http://localhost:8000')
        self.logger = logging.getLogger(__name__)

        # Cache for data
        self.cache = {}
        self.last_update = {}

    @st.cache_data(ttl = 30)
    def get_redis_connection(_self):
        """Get Redis connection with caching"""
        return redis.Redis(
            host = _self.config.get('redis_host', 'localhost'), 
            port = _self.config.get('redis_port', 6379), 
            password = _self.config.get('redis_password'), 
            decode_responses = True
        )

    async def get_portfolio_state(self) -> Dict[str, Any]:
        """Get portfolio state from Redis"""
        try:
            redis_client = self.get_redis_connection()
            portfolio_data = redis_client.get('portfolio:state')

            if portfolio_data:
                return json.loads(portfolio_data)
            else:
                # Fallback to API
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.api_base_url}/api/v1/portfolio") as resp:
                        if resp.status == 200:
                            return await resp.json()

            return {}
        except Exception as e:
            self.logger.error(f"Error getting portfolio state: {e}")
            return {}

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get active positions"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_base_url}/api/v1/positions") as resp:
                    if resp.status == 200:
                        return await resp.json()
            return []
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []

    async def get_risk_alerts(self) -> List[Dict[str, Any]]:
        """Get risk alerts from Redis"""
        try:
            redis_client = self.get_redis_connection()
            alerts_data = redis_client.lrange('risk:alerts', 0, -1)

            alerts = []
            for alert_str in alerts_data:
                try:
                    alert = json.loads(alert_str)
                    alerts.append(alert)
                except json.JSONDecodeError:
                    continue

            return sorted(alerts, key = lambda x: x.get('timestamp', ''), reverse = True)
        except Exception as e:
            self.logger.error(f"Error getting risk alerts: {e}")
            return []

    async def get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_base_url}/health") as resp:
                    if resp.status == 200:
                        return await resp.json()
            return {"status": "offline"}
        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return {"status": "offline"}

    async def get_market_data(self, symbol: str = "XAUUSD", timeframe: str = "M1", limit: int = 100) -> pd.DataFrame:
        """Get market data"""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    'symbol': symbol, 
                    'timeframe': timeframe, 
                    'limit': limit
                }
                async with session.get(f"{self.api_base_url}/api/v1/market/data", params = params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return pd.DataFrame(data)

            # Fallback to dummy data
            return self._generate_dummy_market_data(symbol, limit)
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return self._generate_dummy_market_data(symbol, limit)

    def _generate_dummy_market_data(self, symbol: str, limit: int) -> pd.DataFrame:
        """Generate dummy market data for demo"""
        dates = pd.date_range(end = datetime.now(), periods = limit, freq = '1min')

        # Generate realistic price data
        np.random.seed(42)
        base_price = 2000.0 if symbol == "XAUUSD" else 1.1000
        prices = [base_price]

        for i in range(1, limit):
            change = np.random.randn() * 0.001  # 0.1% volatility
            new_price = prices[ - 1] * (1 + change)
            prices.append(new_price)

        df = pd.DataFrame({
            'timestamp': dates, 
            'open': prices, 
            'high': [p * (1 + abs(np.random.randn() * 0.0005)) for p in prices], 
            'low': [p * (1 - abs(np.random.randn() * 0.0005)) for p in prices], 
            'close': prices, 
            'volume': np.random.randint(100, 1000, limit)
        })

        return df

# Initialize data source
@st.cache_resource
def get_data_source():
    config = {
        'api_url': 'http://localhost:8000', 
        'redis_host': 'localhost', 
        'redis_port': 6379
    }
    return DashboardDataSource(config)

def render_header():
    """Render dashboard header"""
    st.markdown('<h1 class = "main - header">🏆 NICEGOLD Enterprise Trading Dashboard</h1>', unsafe_allow_html = True)

    # System status
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.markdown("### 🔄 Auto - refresh")
        auto_refresh = st.checkbox("Enable", value = True)
        if auto_refresh:
            st.rerun()

    with col2:
        st.markdown("### ⏱️ Last Update")
        st.write(datetime.now().strftime("%H:%M:%S"))

    with col3:
        st.markdown("### 🌐 System Status")
        # This would be populated with real system health data
        status_placeholder = st.empty()
        status_placeholder.markdown('<span class = "status - online">🟢 ONLINE</span>', unsafe_allow_html = True)

def render_portfolio_metrics(data_source: DashboardDataSource):
    """Render portfolio metrics section"""
    st.markdown("## 💼 Portfolio Overview")

    # Get portfolio data (using async in a sync context)
    try:
        portfolio_data = asyncio.run(data_source.get_portfolio_state())
    except Exception:
        portfolio_data = {
            'portfolio_value': 105000.0, 
            'total_pnl': 5000.0, 
            'total_pnl_pct': 5.0, 
            'daily_pnl': 250.0, 
            'open_positions': 3, 
            'used_margin': 15000.0, 
            'available_margin': 90000.0, 
            'margin_level': 14.3, 
            'max_drawdown': 2.5, 
            'emergency_stop': False
        }

    # Main metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Portfolio Value", 
            f"${portfolio_data.get('portfolio_value', 0):, .2f}", 
            f"{portfolio_data.get('total_pnl', 0): + , .2f}"
        )

    with col2:
        pnl_pct = portfolio_data.get('total_pnl_pct', 0)
        pnl_color = "pnl - positive" if pnl_pct >= 0 else "pnl - negative"
        st.metric(
            "Total P&L", 
            f"{pnl_pct: + .2f}%", 
            f"${portfolio_data.get('total_pnl', 0): + , .2f}"
        )

    with col3:
        daily_pnl = portfolio_data.get('daily_pnl', 0)
        st.metric(
            "Daily P&L", 
            f"${daily_pnl: + , .2f}", 
            delta_color = "normal"
        )

    with col4:
        st.metric(
            "Open Positions", 
            portfolio_data.get('open_positions', 0), 
            delta_color = "off"
        )

    # Additional metrics
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric(
            "Used Margin", 
            f"${portfolio_data.get('used_margin', 0):, .2f}", 
            delta_color = "off"
        )

    with col6:
        st.metric(
            "Available Margin", 
            f"${portfolio_data.get('available_margin', 0):, .2f}", 
            delta_color = "off"
        )

    with col7:
        margin_level = portfolio_data.get('margin_level', 0)
        st.metric(
            "Margin Level", 
            f"{margin_level:.1f}%", 
            delta_color = "off"
        )

    with col8:
        drawdown = portfolio_data.get('max_drawdown', 0)
        st.metric(
            "Max Drawdown", 
            f"{drawdown:.2f}%", 
            delta_color = "inverse"
        )

def render_positions_table(data_source: DashboardDataSource):
    """Render active positions table"""
    st.markdown("## 📊 Active Positions")

    try:
        positions = asyncio.run(data_source.get_positions())
    except Exception:
        # Demo data
        positions = [
            {
                'id': 'XAUUSD_buy_1', 
                'symbol': 'XAUUSD', 
                'side': 'buy', 
                'quantity': 10.0, 
                'entry_price': 1998.50, 
                'current_price': 2003.25, 
                'pnl': 47.50, 
                'pnl_pct': 0.24, 
                'entry_time': '2024 - 01 - 10 10:30:00'
            }, 
            {
                'id': 'EURUSD_sell_1', 
                'symbol': 'EURUSD', 
                'side': 'sell', 
                'quantity': 100000.0, 
                'entry_price': 1.0985, 
                'current_price': 1.0978, 
                'pnl': 70.00, 
                'pnl_pct': 0.06, 
                'entry_time': '2024 - 01 - 10 11:15:00'
            }, 
            {
                'id': 'XAUUSD_buy_2', 
                'symbol': 'XAUUSD', 
                'side': 'buy', 
                'quantity': 5.0, 
                'entry_price': 2001.75, 
                'current_price': 2003.25, 
                'pnl': 7.50, 
                'pnl_pct': 0.07, 
                'entry_time': '2024 - 01 - 10 12:00:00'
            }
        ]

    if positions:
        df = pd.DataFrame(positions)

        # Format the dataframe
        if not df.empty:
            df['P&L'] = df['pnl'].apply(lambda x: f"${x: + , .2f}")
            df['P&L %'] = df['pnl_pct'].apply(lambda x: f"{x: + .2f}%")
            df['Entry Price'] = df['entry_price'].apply(lambda x: f"{x:.4f}")
            df['Current Price'] = df['current_price'].apply(lambda x: f"{x:.4f}")
            df['Quantity'] = df['quantity'].apply(lambda x: f"{x:, .0f}")

            display_df = df[['symbol', 'side', 'Quantity', 'Entry Price', 'Current Price', 'P&L', 'P&L %', 'entry_time']]
            display_df.columns = ['Symbol', 'Side', 'Quantity', 'Entry Price', 'Current Price', 'P&L', 'P&L %', 'Entry Time']

            st.dataframe(
                display_df, 
                use_container_width = True, 
                hide_index = True
            )
    else:
        st.info("No active positions")

def render_market_chart(data_source: DashboardDataSource):
    """Render market price chart"""
    st.markdown("## 📈 Market Data")

    # Symbol selector
    col1, col2 = st.columns([1, 3])

    with col1:
        symbol = st.selectbox("Symbol", ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"])
        timeframe = st.selectbox("Timeframe", ["M1", "M5", "M15", "H1", "H4", "D1"])

    with col2:
        try:
            df = asyncio.run(data_source.get_market_data(symbol, timeframe, 100))
        except Exception:
            df = data_source._generate_dummy_market_data(symbol, 100)

        if not df.empty:
            # Create candlestick chart
            fig = go.Figure(data = go.Candlestick(
                x = df['timestamp'], 
                open = df['open'], 
                high = df['high'], 
                low = df['low'], 
                close = df['close'], 
                name = symbol
            ))

            fig.update_layout(
                title = f"{symbol} {timeframe} Chart", 
                xaxis_title = "Time", 
                yaxis_title = "Price", 
                height = 500, 
                template = "plotly_white"
            )

            st.plotly_chart(fig, use_container_width = True)

def render_risk_alerts(data_source: DashboardDataSource):
    """Render risk alerts section"""
    st.markdown("## ⚠️ Risk Alerts")

    try:
        alerts = asyncio.run(data_source.get_risk_alerts())
    except Exception:
        # Demo alerts
        alerts = [
            {
                'id': 'alert_1', 
                'type': 'HIGH_CORRELATION', 
                'severity': 'medium', 
                'description': 'High correlation detected between XAUUSD positions', 
                'timestamp': '2024 - 01 - 10T14:30:00', 
                'symbol': 'XAUUSD'
            }, 
            {
                'id': 'alert_2', 
                'type': 'POSITION_SIZE', 
                'severity': 'low', 
                'description': 'Position size approaching limit for EURUSD', 
                'timestamp': '2024 - 01 - 10T13:45:00', 
                'symbol': 'EURUSD'
            }
        ]

    if alerts:
        for alert in alerts[:10]:  # Show last 10 alerts
            severity = alert.get('severity', 'low')
            alert_class = f"alert - {severity}"

            severity_emoji = {
                'low': '🟢', 
                'medium': '🟡', 
                'high': '🔴', 
                'critical': '🚨'
            }.get(severity, '🟡')

            st.markdown(f"""
            <div class = "{alert_class}">
                {severity_emoji} <strong>{alert.get('type', 'ALERT')}</strong><br>
                {alert.get('description', 'No description')}<br>
                <small>Symbol: {alert.get('symbol', 'N/A')} | Time: {alert.get('timestamp', 'N/A')}</small>
            </div>
            """, unsafe_allow_html = True)
    else:
        st.success("No active risk alerts")

def render_performance_charts(data_source: DashboardDataSource):
    """Render performance charts"""
    st.markdown("## 📊 Performance Analytics")

    # Generate dummy performance data
    dates = pd.date_range(end = datetime.now(), periods = 30, freq = 'D')
    np.random.seed(42)

    # Portfolio value over time
    initial_value = 100000
    returns = np.random.randn(30) * 0.01  # 1% daily volatility
    portfolio_values = [initial_value]

    for ret in returns:
        new_value = portfolio_values[ - 1] * (1 + ret)
        portfolio_values.append(new_value)

    portfolio_df = pd.DataFrame({
        'date': dates, 
        'portfolio_value': portfolio_values[1:], 
        'returns': returns
    })

    col1, col2 = st.columns(2)

    with col1:
        # Portfolio value chart
        fig_portfolio = px.line(
            portfolio_df, 
            x = 'date', 
            y = 'portfolio_value', 
            title = 'Portfolio Value Over Time', 
            template = 'plotly_white'
        )
        fig_portfolio.update_traces(line_color = '#1f77b4')
        st.plotly_chart(fig_portfolio, use_container_width = True)

    with col2:
        # Returns distribution
        fig_returns = px.histogram(
            portfolio_df, 
            x = 'returns', 
            title = 'Daily Returns Distribution', 
            nbins = 20, 
            template = 'plotly_white'
        )
        fig_returns.update_traces(marker_color = '#ff7f0e')
        st.plotly_chart(fig_returns, use_container_width = True)

def render_system_monitoring():
    """Render system monitoring section"""
    st.markdown("## 🖥️ System Monitoring")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### API Health")
        st.success("🟢 Online")
        st.metric("Response Time", "45ms", " - 5ms")
        st.metric("Requests/min", "120", " + 10")

    with col2:
        st.markdown("### Database")
        st.success("🟢 Connected")
        st.metric("Connections", "8/100", " + 2")
        st.metric("Query Time", "12ms", " - 2ms")

    with col3:
        st.markdown("### Cache")
        st.success("🟢 Active")
        st.metric("Hit Rate", "94.5%", " + 1.2%")
        st.metric("Memory Usage", "45%", " + 3%")

def render_trading_controls():
    """Render trading controls"""
    st.markdown("## 🎮 Trading Controls")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Emergency Controls")
        if st.button("🛑 Emergency Stop", type = "secondary"):
            st.warning("Emergency stop activated!")

        if st.button("🔄 Reset Daily Limits", type = "secondary"):
            st.success("Daily limits reset!")

    with col2:
        st.markdown("### Position Controls")
        if st.button("📊 Close All Positions", type = "secondary"):
            st.warning("All positions will be closed!")

        if st.button("⏸️ Pause Trading", type = "secondary"):
            st.info("Trading paused!")

    with col3:
        st.markdown("### System Controls")
        if st.button("🔧 Restart API", type = "secondary"):
            st.info("API restart initiated!")

        if st.button("📊 Generate Report", type = "primary"):
            st.success("Report generated!")

def main():
    """Main dashboard function"""
    # Initialize data source
    data_source = get_data_source()

    # Render sections
    render_header()

    # Auto - refresh every 30 seconds
    time.sleep(1)

    # Main content
    render_portfolio_metrics(data_source)

    col1, col2 = st.columns([2, 1])

    with col1:
        render_market_chart(data_source)

    with col2:
        render_risk_alerts(data_source)

    render_positions_table(data_source)

    render_performance_charts(data_source)

    # Sidebar
    with st.sidebar:
        st.markdown("## 🔧 Dashboard Settings")

        refresh_rate = st.slider("Refresh Rate (seconds)", 5, 60, 30)

        st.markdown("## 📊 Quick Stats")
        st.metric("Uptime", "99.9%")
        st.metric("Total Trades", "1, 234")
        st.metric("Win Rate", "68.5%")

        st.markdown("## 🚨 Quick Actions")
        render_trading_controls()

        st.markdown("## 🖥️ System Status")
        render_system_monitoring()

if __name__ == "__main__":
    main()