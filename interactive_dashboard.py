#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 INTERACTIVE DASHBOARD SYSTEM
Real-time trading dashboard with advanced visualization
for NICEGOLD ProjectP v2.1

Author: NICEGOLD Enterprise
Date: June 25, 2025
"""

import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Dashboard imports with fallback
PLOTLY_AVAILABLE = False
DASH_AVAILABLE = False
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    pass

try:
    import dash
    from dash import Input, Output, callback, dcc, html
    DASH_AVAILABLE = True
except ImportError:
    pass

# Rich imports with fallback
RICH_AVAILABLE = False
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    pass


class InteractiveDashboard:
    """📊 Advanced Interactive Dashboard for Trading Analytics"""
    
    def __init__(self, console_output: bool = True):
        self.console = Console() if RICH_AVAILABLE else None
        self.console_output = console_output
        self.data_cache = {}
        self.performance_metrics = {}
        self.live_data = []
        
    def create_plotly_charts(self, data: pd.DataFrame, 
                           predictions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        📈 สร้างกราฟแบบ interactive ด้วย Plotly
        """
        if not PLOTLY_AVAILABLE:
            if self.console_output:
                print("⚠️ Plotly not available. Install with: pip install plotly")
            return {}
        
        if self.console_output and RICH_AVAILABLE:
            self._show_chart_creation_header()
        
        charts = {}
        
        # 1. Price and Volume Chart
        charts['price_volume'] = self._create_price_volume_chart(data)
        
        # 2. Technical Indicators Chart
        charts['technical'] = self._create_technical_indicators_chart(data)
        
        # 3. Prediction vs Actual Chart (if predictions available)
        if predictions is not None:
            charts['predictions'] = self._create_prediction_chart(data, predictions)
        
        # 4. Performance Metrics Chart
        charts['performance'] = self._create_performance_chart(data)
        
        # 5. Risk Analysis Chart
        charts['risk'] = self._create_risk_analysis_chart(data)
        
        if self.console_output:
            if RICH_AVAILABLE:
                self._display_chart_summary(charts)
            else:
                print(f"✅ Created {len(charts)} interactive charts")
        
        return charts
    
    def _create_price_volume_chart(self, data: pd.DataFrame) -> go.Figure:
        """สร้างกราฟราคาและปริมาณ"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('GOLD Price Movement', 'Volume'),
            row_width=[0.2, 0.7]
        )
        
        # Candlestick chart
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name="GOLD Price"
                ),
                row=1, col=1
            )
        else:
            # Fallback to line chart
            price_col = 'close' if 'close' in data.columns else data.columns[0]
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[price_col],
                    mode='lines',
                    name="Price",
                    line=dict(color='gold')
                ),
                row=1, col=1
            )
        
        # Volume chart
        if 'volume' in data.columns:
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['volume'],
                    name="Volume",
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title="🏆 NICEGOLD ProjectP - Price Analysis",
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        return fig
    
    def _create_technical_indicators_chart(self, data: pd.DataFrame) -> go.Figure:
        """สร้างกราฟตัวชี้วัดทางเทคนิค"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Moving Averages', 'RSI', 'MACD')
        )
        
        price_col = 'close' if 'close' in data.columns else data.columns[0]
        
        # Moving Averages
        data_ma = data.copy()
        data_ma['SMA_20'] = data_ma[price_col].rolling(20).mean()
        data_ma['SMA_50'] = data_ma[price_col].rolling(50).mean()
        data_ma['EMA_12'] = data_ma[price_col].ewm(span=12).mean()
        
        fig.add_trace(
            go.Scatter(x=data.index, y=data_ma[price_col], 
                      name="Price", line=dict(color='gold')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data_ma['SMA_20'], 
                      name="SMA 20", line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data_ma['SMA_50'], 
                      name="SMA 50", line=dict(color='red')),
            row=1, col=1
        )
        
        # RSI
        rsi = self._calculate_rsi(data_ma[price_col])
        fig.add_trace(
            go.Scatter(x=data.index, y=rsi, name="RSI", 
                      line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        macd_line, macd_signal = self._calculate_macd(data_ma[price_col])
        fig.add_trace(
            go.Scatter(x=data.index, y=macd_line, name="MACD", 
                      line=dict(color='blue')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=macd_signal, name="Signal", 
                      line=dict(color='red')),
            row=3, col=1
        )
        
        fig.update_layout(
            title="📊 Technical Indicators Analysis",
            height=800
        )
        
        return fig
    
    def _create_prediction_chart(self, data: pd.DataFrame, 
                               predictions: np.ndarray) -> go.Figure:
        """สร้างกราฟการทำนาย"""
        fig = go.Figure()
        
        price_col = 'close' if 'close' in data.columns else data.columns[0]
        
        # Actual prices
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[price_col],
                mode='lines',
                name='Actual Price',
                line=dict(color='blue')
            )
        )
        
        # Predictions
        if len(predictions) == len(data):
            # Classification predictions to price signals
            price_changes = data[price_col].pct_change()
            predicted_direction = predictions > 0.5
            prediction_signals = np.where(predicted_direction, 1, -1)
            
            # Create predicted price movements
            predicted_prices = data[price_col].copy()
            for i in range(1, len(predicted_prices)):
                predicted_prices.iloc[i] = (predicted_prices.iloc[i-1] * 
                                          (1 + prediction_signals[i] * 0.001))
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=predicted_prices,
                    mode='lines',
                    name='AI Prediction',
                    line=dict(color='red', dash='dot')
                )
            )
            
            # Buy/Sell signals
            buy_signals = data.index[predictions > 0.7]
            sell_signals = data.index[predictions < 0.3]
            
            if len(buy_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals,
                        y=data.loc[buy_signals, price_col],
                        mode='markers',
                        name='Buy Signal',
                        marker=dict(color='green', size=10, symbol='triangle-up')
                    )
                )
            
            if len(sell_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals,
                        y=data.loc[sell_signals, price_col],
                        mode='markers',
                        name='Sell Signal',
                        marker=dict(color='red', size=10, symbol='triangle-down')
                    )
                )
        
        fig.update_layout(
            title="🎯 AI Predictions vs Actual Performance",
            xaxis_title="Time",
            yaxis_title="Price",
            height=500
        )
        
        return fig
    
    def _create_performance_chart(self, data: pd.DataFrame) -> go.Figure:
        """สร้างกราฟประสิทธิภาพ"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Returns Distribution', 'Cumulative Returns', 
                          'Volatility', 'Sharpe Ratio'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        price_col = 'close' if 'close' in data.columns else data.columns[0]
        returns = data[price_col].pct_change().dropna()
        cumulative_returns = (1 + returns).cumprod()
        
        # Returns distribution
        fig.add_trace(
            go.Histogram(x=returns, name="Returns", nbinsx=50),
            row=1, col=1
        )
        
        # Cumulative returns
        fig.add_trace(
            go.Scatter(x=data.index[1:], y=cumulative_returns, 
                      name="Cumulative Returns", line=dict(color='green')),
            row=1, col=2
        )
        
        # Rolling volatility
        volatility = returns.rolling(window=30).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(x=data.index[1:], y=volatility, 
                      name="30-day Volatility", line=dict(color='orange')),
            row=2, col=1
        )
        
        # Rolling Sharpe ratio
        rolling_sharpe = (returns.rolling(window=30).mean() / 
                         returns.rolling(window=30).std()) * np.sqrt(252)
        fig.add_trace(
            go.Scatter(x=data.index[1:], y=rolling_sharpe, 
                      name="30-day Sharpe", line=dict(color='purple')),
            row=2, col=2
        )
        
        fig.update_layout(
            title="📈 Performance Analytics Dashboard",
            height=600
        )
        
        return fig
    
    def _create_risk_analysis_chart(self, data: pd.DataFrame) -> go.Figure:
        """สร้างกราฟการวิเคราะห์ความเสี่ยง"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Value at Risk (VaR)', 'Drawdown Analysis',
                          'Risk-Return Scatter', 'Correlation Matrix')
        )
        
        price_col = 'close' if 'close' in data.columns else data.columns[0]
        returns = data[price_col].pct_change().dropna()
        
        # Value at Risk
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        fig.add_trace(
            go.Histogram(x=returns, name="Returns Distribution", nbinsx=50),
            row=1, col=1
        )
        fig.add_vline(x=var_95, line_dash="dash", line_color="orange", 
                     row=1, col=1)
        fig.add_vline(x=var_99, line_dash="dash", line_color="red", 
                     row=1, col=1)
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig.add_trace(
            go.Scatter(x=data.index[1:], y=drawdown * 100, 
                      name="Drawdown %", fill='tonexty', 
                      line=dict(color='red')),
            row=1, col=2
        )
        
        # Risk-Return scatter (rolling windows)
        window_size = 30
        rolling_return = returns.rolling(window_size).mean() * 252
        rolling_vol = returns.rolling(window_size).std() * np.sqrt(252)
        
        fig.add_trace(
            go.Scatter(x=rolling_vol, y=rolling_return, mode='markers',
                      name="Risk-Return", marker=dict(color='blue')),
            row=2, col=1
        )
        
        # Simple correlation visualization
        if len(data.columns) > 1:
            corr_data = data.select_dtypes(include=[np.number]).corr()
            fig.add_trace(
                go.Heatmap(z=corr_data.values, x=corr_data.columns, 
                          y=corr_data.columns, colorscale='RdBu'),
                row=2, col=2
            )
        
        fig.update_layout(
            title="⚠️ Risk Analysis Dashboard",
            height=600
        )
        
        return fig
    
    def create_live_dashboard(self, data: pd.DataFrame) -> str:
        """
        📱 สร้าง Live Dashboard แบบ Real-time
        """
        if not DASH_AVAILABLE:
            return self._create_static_dashboard(data)
        
        # Create Dash app
        app = dash.Dash(__name__)
        
        # Layout
        app.layout = html.Div([
            html.H1("🏆 NICEGOLD ProjectP Live Dashboard", 
                   style={'textAlign': 'center', 'color': 'gold'}),
            
            dcc.Interval(
                id='interval-component',
                interval=5000,  # Update every 5 seconds
                n_intervals=0
            ),
            
            html.Div([
                dcc.Graph(id='live-price-chart'),
                dcc.Graph(id='live-indicators-chart')
            ]),
            
            html.Div([
                dcc.Graph(id='live-performance-chart'),
                dcc.Graph(id='live-risk-chart')
            ])
        ])
        
        # Callbacks for live updates
        @app.callback(
            [Output('live-price-chart', 'figure'),
             Output('live-indicators-chart', 'figure'),
             Output('live-performance-chart', 'figure'),
             Output('live-risk-chart', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_charts(n):
            # In real implementation, this would fetch live data
            charts = self.create_plotly_charts(data)
            return (charts.get('price_volume', {}),
                   charts.get('technical', {}),
                   charts.get('performance', {}),
                   charts.get('risk', {}))
        
        return app
    
    def _create_static_dashboard(self, data: pd.DataFrame) -> str:
        """สร้าง Static Dashboard สำหรับกรณีที่ไม่มี Dash"""
        if not PLOTLY_AVAILABLE:
            if self.console_output:
                print("⚠️ Plotly not available for dashboard creation")
            return ""
        
        charts = self.create_plotly_charts(data)
        
        # Create HTML dashboard
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>NICEGOLD ProjectP Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { text-align: center; color: gold; }
                .chart-container { margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1 class="header">🏆 NICEGOLD ProjectP Dashboard</h1>
        """
        
        for chart_name, chart in charts.items():
            if chart:
                chart_html = chart.to_html(include_plotlyjs=False, div_id=f"div_{chart_name}")
                html_content += f'<div class="chart-container">{chart_html}</div>'
        
        html_content += """
        </body>
        </html>
        """
        
        # Save to file
        dashboard_path = "nicegold_dashboard.html"
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        if self.console_output:
            print(f"✅ Dashboard saved to: {dashboard_path}")
        
        return dashboard_path
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """คำนวณ RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, 
                       fast: int = 12, slow: int = 26, signal: int = 9):
        """คำนวณ MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        return macd_line, macd_signal
    
    def _show_chart_creation_header(self):
        """แสดงหัวข้อการสร้างกราฟ"""
        if RICH_AVAILABLE:
            header = Panel(
                "[bold cyan]📊 INTERACTIVE CHART CREATION[/bold cyan]\n"
                "[yellow]Generating advanced trading visualizations[/yellow]",
                border_style="cyan"
            )
            self.console.print(header)
    
    def _display_chart_summary(self, charts: Dict[str, Any]):
        """แสดงสรุปกราฟที่สร้าง"""
        if RICH_AVAILABLE:
            table = Table(title="📈 Generated Charts", border_style="green")
            table.add_column("Chart Type", style="cyan")
            table.add_column("Status", style="yellow")
            table.add_column("Description", style="green")
            
            chart_descriptions = {
                'price_volume': 'Price movement and trading volume',
                'technical': 'Moving averages, RSI, MACD indicators',
                'predictions': 'AI predictions vs actual performance',
                'performance': 'Returns, volatility, Sharpe ratio',
                'risk': 'VaR, drawdown, risk-return analysis'
            }
            
            for chart_name, chart in charts.items():
                status = "✅ Created" if chart else "❌ Failed"
                description = chart_descriptions.get(chart_name, "Custom chart")
                table.add_row(chart_name, status, description)
            
            self.console.print(table)


if __name__ == "__main__":
    # Demo/Test the Interactive Dashboard
    print("🚀 NICEGOLD ProjectP - Interactive Dashboard Demo")
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1H')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'open': 2000 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'high': 2000 + np.cumsum(np.random.randn(len(dates)) * 0.5) + np.random.rand(len(dates)) * 5,
        'low': 2000 + np.cumsum(np.random.randn(len(dates)) * 0.5) - np.random.rand(len(dates)) * 5,
        'close': 2000 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Sample predictions
    sample_predictions = np.random.random(len(sample_data))
    
    # Test the dashboard
    dashboard = InteractiveDashboard()
    
    # 1. Test chart creation
    print("\n1. Creating Interactive Charts...")
    charts = dashboard.create_plotly_charts(sample_data, sample_predictions)
    
    # 2. Test dashboard creation
    print("\n2. Creating Dashboard...")
    dashboard_path = dashboard.create_live_dashboard(sample_data)
    
    print(f"\n✅ Demo completed successfully!")
    print(f"📊 Created {len(charts)} interactive charts")
    if isinstance(dashboard_path, str):
        print(f"📱 Dashboard available at: {dashboard_path}")
    else:
        print(f"📱 Live dashboard created (run with app.run_server())")
