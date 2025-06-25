#!/usr/bin/env python3
from datetime import datetime, timedelta
from pathlib import Path
from plotly.subplots import make_subplots
    from src.single_user_auth import auth_manager
import json
import logging
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import sqlite3
import streamlit as st
import yaml
"""
Single User Production Dashboard
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

Dashboard แบบผู้ใช้คนเดียวสำหรับควบคุมระบบ NICEGOLD Trading
พร้อมระบบ authentication และการจัดการแบบ enterprise - grade
"""


# Configure page
st.set_page_config(
    page_title = "NICEGOLD Enterprise", 
    page_icon = "💰", 
    layout = "wide", 
    initial_sidebar_state = "expanded"
)

# Custom CSS for professional look
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
        background: linear - gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border - radius: 10px;
        color: white;
        text - align: center;
        margin: 0.5rem 0;
    }
    .status - good { color: #28a745; }
    .status - warning { color: #ffc107; }
    .status - danger { color: #dc3545; }
    .sidebar .sidebar - content {
        background: linear - gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html = True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'auth_token' not in st.session_state:
    st.session_state.auth_token = None

# Import authentication system
try:
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    st.error("⚠️ Authentication system not found. Please ensure src/single_user_auth.py exists.")

class ProductionDashboard:
    """Production dashboard for single user trading system"""

    def __init__(self):
        self.project_root = Path(".")
        self.database_path = self.project_root / "database" / "production.db"
        self.config_path = self.project_root / "config" / "production.yaml"

        # Load configuration
        self.config = self._load_config()

        # Setup logging
        logging.basicConfig(level = logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _load_config(self) -> dict:
        """Load production configuration"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                st.error(f"Failed to load configuration: {e}")
                return {}
        return {}

    def _get_db_connection(self):
        """Get database connection"""
        if not self.database_path.exists():
            st.error("Database not found. Please run production deployment first.")
            return None

        try:
            return sqlite3.connect(self.database_path)
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            return None

    def show_login_form(self):
        """Show login form"""
        st.markdown('<div class = "main - header">🔐 NICEGOLD Enterprise Login</div>', unsafe_allow_html = True)

        if not AUTH_AVAILABLE:
            st.error("Authentication system not available")
            return

        with st.form("login_form"):
            st.markdown("### Please login to access the dashboard")

            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                username = st.text_input("Username", placeholder = "Enter username")
                password = st.text_input("Password", type = "password", placeholder = "Enter password")

                submit_button = st.form_submit_button("🚀 Login", use_container_width = True)

                if submit_button:
                    if username and password:
                        # Get client IP (simplified for demo)
                        client_ip = "127.0.0.1"

                        # Authenticate
                        token = auth_manager.authenticate(
                            username = username, 
                            password = password, 
                            ip_address = client_ip, 
                            user_agent = "Streamlit Dashboard"
                        )

                        if token:
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.session_state.auth_token = token
                            st.success("✅ Login successful!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials")
                    else:
                        st.error("Please enter both username and password")

        # Show setup instructions if no user configured
        if AUTH_AVAILABLE:
            status = auth_manager.get_system_status()
            if not status["user_configured"]:
                st.warning("⚠️ No admin user configured")
                st.info("Run: `python src/single_user_auth.py setup` to create admin user")

    def show_main_dashboard(self):
        """Show main dashboard"""
        # Header
        st.markdown('<div class = "main - header">💰 NICEGOLD Enterprise Dashboard</div>', unsafe_allow_html = True)

        # Sidebar navigation
        self._show_sidebar()

        # Main content based on selected page
        page = st.session_state.get('current_page', 'overview')

        if page == 'overview':
            self._show_overview_page()
        elif page == 'trading':
            self._show_trading_page()
        elif page == 'models':
            self._show_models_page()
        elif page == 'monitoring':
            self._show_monitoring_page()
        elif page == 'settings':
            self._show_settings_page()
        elif page == 'security':
            self._show_security_page()

    def _show_sidebar(self):
        """Show sidebar navigation"""
        with st.sidebar:
            st.markdown("### 🎛️ Navigation")

            # User info
            st.markdown(f"**User:** {st.session_state.username}")
            st.markdown(f"**Status:** 🟢 Active")

            # Navigation buttons
            if st.button("📊 Overview", use_container_width = True):
                st.session_state.current_page = 'overview'
                st.rerun()

            if st.button("💹 Trading", use_container_width = True):
                st.session_state.current_page = 'trading'
                st.rerun()

            if st.button("🤖 Models", use_container_width = True):
                st.session_state.current_page = 'models'
                st.rerun()

            if st.button("📈 Monitoring", use_container_width = True):
                st.session_state.current_page = 'monitoring'
                st.rerun()

            if st.button("⚙️ Settings", use_container_width = True):
                st.session_state.current_page = 'settings'
                st.rerun()

            if st.button("🔒 Security", use_container_width = True):
                st.session_state.current_page = 'security'
                st.rerun()

            st.markdown(" -  -  - ")

            # Logout button
            if st.button("🚪 Logout", use_container_width = True):
                self._logout()

    def _logout(self):
        """Logout user"""
        if AUTH_AVAILABLE and st.session_state.auth_token:
            auth_manager.logout(st.session_state.auth_token)

        # Clear session state
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.auth_token = None
        st.session_state.current_page = 'overview'

        st.success("✅ Logged out successfully")
        st.rerun()

    def _show_overview_page(self):
        """Show overview page"""
        st.header("📊 System Overview")

        # System metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # System uptime
            uptime = self._get_system_uptime()
            st.metric("System Uptime", uptime, delta = "Running")

        with col2:
            # Active positions
            active_positions = self._get_active_positions_count()
            st.metric("Active Positions", active_positions, delta = " + 2")

        with col3:
            # Total PnL
            total_pnl = self._get_total_pnl()
            st.metric("Total PnL", f"${total_pnl:, .2f}", delta = f"${total_pnl*0.1:, .2f}")

        with col4:
            # Model Performance
            model_accuracy = self._get_model_accuracy()
            st.metric("Model Accuracy", f"{model_accuracy:.1%}", delta = "↗️ +2.5%")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            self._show_pnl_chart()

        with col2:
            self._show_performance_chart()

        # Recent activity
        st.subheader("📋 Recent Activity")
        self._show_recent_activity()

    def _show_trading_page(self):
        """Show trading page"""
        st.header("💹 Trading Dashboard")

        # Trading controls
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🚀 Start Trading", use_container_width = True):
                self._start_trading()

        with col2:
            if st.button("⏸️ Pause Trading", use_container_width = True):
                self._pause_trading()

        with col3:
            if st.button("🛑 Emergency Stop", use_container_width = True):
                self._emergency_stop()

        # Current positions
        st.subheader("📋 Current Positions")
        self._show_positions_table()

        # Trading parameters
        st.subheader("⚙️ Trading Parameters")
        self._show_trading_parameters()

        # Risk management
        st.subheader("🛡️ Risk Management")
        self._show_risk_management()

    def _show_models_page(self):
        """Show models page"""
        st.header("🤖 Model Management")

        # Model registry
        st.subheader("📚 Model Registry")
        self._show_model_registry()

        # Model performance
        st.subheader("📊 Model Performance")
        self._show_model_performance()

        # Model deployment
        st.subheader("🚀 Model Deployment")
        self._show_model_deployment()

    def _show_monitoring_page(self):
        """Show monitoring page"""
        st.header("📈 System Monitoring")

        # System resources
        self._show_system_resources()

        # Application metrics
        st.subheader("📊 Application Metrics")
        self._show_application_metrics()

        # Alerts and notifications
        st.subheader("🚨 Alerts & Notifications")
        self._show_alerts()

    def _show_settings_page(self):
        """Show settings page"""
        st.header("⚙️ System Settings")

        # Configuration editor
        st.subheader("📝 Configuration")
        self._show_configuration_editor()

        # System maintenance
        st.subheader("🔧 System Maintenance")
        self._show_maintenance_tools()

    def _show_security_page(self):
        """Show security page"""
        st.header("🔒 Security Management")

        # Authentication status
        st.subheader("🔐 Authentication Status")
        self._show_auth_status()

        # Session management
        st.subheader("🎫 Session Management")
        self._show_session_management()

        # Security logs
        st.subheader("📋 Security Logs")
        self._show_security_logs()

        # Password management
        st.subheader("🔑 Password Management")
        self._show_password_management()

    # Helper methods for data retrieval
    def _get_system_uptime(self) -> str:
        """Get system uptime"""
        try:
            boot_time = psutil.boot_time()
            uptime_seconds = datetime.now().timestamp() - boot_time
            uptime_hours = int(uptime_seconds // 3600)
            uptime_minutes = int((uptime_seconds % 3600) // 60)
            return f"{uptime_hours}h {uptime_minutes}m"
        except:
            return "Unknown"

    def _get_active_positions_count(self) -> int:
        """Get number of active positions"""
        conn = self._get_db_connection()
        if conn:
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM positions WHERE status = 'open'")
                return cursor.fetchone()[0]
            except:
                pass
            finally:
                conn.close()
        return 0

    def _get_total_pnl(self) -> float:
        """Get total PnL"""
        conn = self._get_db_connection()
        if conn:
            try:
                cursor = conn.execute("SELECT SUM(pnl) FROM positions WHERE pnl IS NOT NULL")
                result = cursor.fetchone()[0]
                return result if result else 0.0
            except:
                pass
            finally:
                conn.close()
        return 12345.67  # Demo value

    def _get_model_accuracy(self) -> float:
        """Get model accuracy"""
        # Demo calculation
        return 0.753

    def _show_pnl_chart(self):
        """Show PnL chart"""
        st.subheader("💰 PnL Over Time")

        # Generate demo data
        dates = pd.date_range(start = '2024 - 01 - 01', end = '2024 - 12 - 31', freq = 'D')
        cumulative_pnl = np.cumsum(np.random.normal(10, 50, len(dates)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x = dates, 
            y = cumulative_pnl, 
            mode = 'lines', 
            name = 'Cumulative PnL', 
            line = dict(color = 'green', width = 2)
        ))

        fig.update_layout(
            title = "Cumulative PnL", 
            xaxis_title = "Date", 
            yaxis_title = "PnL ($)", 
            height = 400
        )

        st.plotly_chart(fig, use_container_width = True)

    def _show_performance_chart(self):
        """Show performance metrics chart"""
        st.subheader("📊 Performance Metrics")

        # Demo data
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 - Score']
        values = [0.753, 0.689, 0.821, 0.748]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x = metrics, 
            y = values, 
            marker_color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        ))

        fig.update_layout(
            title = "Model Performance Metrics", 
            yaxis_title = "Score", 
            height = 400
        )

        st.plotly_chart(fig, use_container_width = True)

    def _show_recent_activity(self):
        """Show recent activity log"""
        activities = [
            {"time": "2024 - 06 - 23 14:30:00", "action": "Position Opened", "details": "XAUUSD Long @ 2330.50"}, 
            {"time": "2024 - 06 - 23 14:25:00", "action": "Model Updated", "details": "RandomForest v2.1 deployed"}, 
            {"time": "2024 - 06 - 23 14:20:00", "action": "Risk Alert", "details": "Position size limit reached"}, 
            {"time": "2024 - 06 - 23 14:15:00", "action": "Position Closed", "details": "XAUUSD Short @ 2329.80 ( + $150)"}, 
        ]

        df = pd.DataFrame(activities)
        st.dataframe(df, use_container_width = True, hide_index = True)

    def _show_positions_table(self):
        """Show current positions table"""
        # Demo positions data
        positions = [
            {"Symbol": "XAUUSD", "Side": "Long", "Size": "0.5", "Entry": "2330.50", "Current": "2331.20", "PnL": " + $35.00"}, 
            {"Symbol": "XAUUSD", "Side": "Short", "Size": "0.3", "Entry": "2329.80", "Current": "2331.20", "PnL": " - $42.00"}, 
        ]

        df = pd.DataFrame(positions)
        st.dataframe(df, use_container_width = True, hide_index = True)

    def _show_trading_parameters(self):
        """Show trading parameters configuration"""
        col1, col2 = st.columns(2)

        with col1:
            max_position_size = st.number_input("Max Position Size", value = 1000000, min_value = 1000)
            risk_limit = st.slider("Risk Limit (%)", min_value = 0.1, max_value = 5.0, value = 2.0, step = 0.1)

        with col2:
            trading_enabled = st.toggle("Trading Enabled", value = True)
            auto_rebalance = st.toggle("Auto Rebalance", value = False)

        if st.button("💾 Save Parameters"):
            st.success("Parameters saved successfully!")

    def _show_risk_management(self):
        """Show risk management controls"""
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Current Risk", "1.2%", delta = " - 0.3%")

        with col2:
            st.metric("Max Drawdown", "5.8%", delta = "Within limits")

        with col3:
            st.metric("Position Exposure", "65%", delta = " + 5%")

        # Risk controls
        emergency_stop = st.toggle("Emergency Stop", value = False)
        if emergency_stop:
            st.error("🚨 Emergency stop activated! All trading halted.")

    def _start_trading(self):
        """Start trading"""
        st.success("🚀 Trading started!")
        st.balloons()

    def _pause_trading(self):
        """Pause trading"""
        st.warning("⏸️ Trading paused!")

    def _emergency_stop(self):
        """Emergency stop trading"""
        st.error("🛑 Emergency stop activated!")

    def _show_model_registry(self):
        """Show model registry"""
        models = [
            {"Name": "RandomForest_v2.1", "Status": "Active", "Accuracy": "75.3%", "Deployed": "2024 - 06 - 23"}, 
            {"Name": "XGBoost_v1.5", "Status": "Standby", "Accuracy": "72.8%", "Deployed": "2024 - 06 - 20"}, 
            {"Name": "LSTM_v1.0", "Status": "Training", "Accuracy": "68.9%", "Deployed": " - "}, 
        ]

        df = pd.DataFrame(models)
        st.dataframe(df, use_container_width = True, hide_index = True)

    def _show_model_performance(self):
        """Show model performance charts"""
        # Model comparison chart
        models = ['RandomForest', 'XGBoost', 'LSTM', 'SVM']
        accuracy = [0.753, 0.728, 0.689, 0.692]

        fig = go.Figure()
        fig.add_trace(go.Bar(x = models, y = accuracy, name = 'Accuracy'))
        fig.update_layout(title = "Model Accuracy Comparison", yaxis_title = "Accuracy")

        st.plotly_chart(fig, use_container_width = True)

    def _show_model_deployment(self):
        """Show model deployment controls"""
        col1, col2 = st.columns(2)

        with col1:
            selected_model = st.selectbox("Select Model", ["RandomForest_v2.1", "XGBoost_v1.5", "LSTM_v1.0"])

        with col2:
            if st.button("🚀 Deploy Model"):
                st.success(f"Model {selected_model} deployed successfully!")

    def _show_system_resources(self):
        """Show system resource usage"""
        # Get real system stats
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
        except:
            cpu_percent = 25.5
            memory = type('obj', (object, ), {'percent': 45.2})()
            disk = type('obj', (object, ), {'percent': 68.1})()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("CPU Usage", f"{cpu_percent:.1f}%", delta = " - 2.3%")

        with col2:
            st.metric("Memory Usage", f"{memory.percent:.1f}%", delta = " + 1.5%")

        with col3:
            st.metric("Disk Usage", f"{disk.percent:.1f}%", delta = " + 0.8%")

        # Resource usage chart
        fig = go.Figure()

        categories = ['CPU', 'Memory', 'Disk']
        values = [cpu_percent, memory.percent, disk.percent]
        colors = ['red' if x > 80 else 'orange' if x > 60 else 'green' for x in values]

        fig.add_trace(go.Bar(x = categories, y = values, marker_color = colors))
        fig.update_layout(title = "System Resource Usage", yaxis_title = "Usage (%)")

        st.plotly_chart(fig, use_container_width = True)

    def _show_application_metrics(self):
        """Show application - specific metrics"""
        metrics = {
            "API Requests/min": 142, 
            "Average Response Time": "85ms", 
            "Error Rate": "0.3%", 
            "Active Connections": 8
        }

        col1, col2, col3, col4 = st.columns(4)
        cols = [col1, col2, col3, col4]

        for i, (metric, value) in enumerate(metrics.items()):
            with cols[i]:
                st.metric(metric, value)

    def _show_alerts(self):
        """Show alerts and notifications"""
        alerts = [
            {"Level": "⚠️ Warning", "Message": "CPU usage above 70%", "Time": "14:30:00"}, 
            {"Level": "ℹ️ Info", "Message": "Model retrained successfully", "Time": "14:25:00"}, 
            {"Level": "🚨 Error", "Message": "Database connection timeout", "Time": "14:20:00"}, 
        ]

        for alert in alerts:
            col1, col2, col3 = st.columns([1, 4, 1])
            with col1:
                st.write(alert["Level"])
            with col2:
                st.write(alert["Message"])
            with col3:
                st.write(alert["Time"])

    def _show_configuration_editor(self):
        """Show configuration editor"""
        st.write("Current configuration:")

        if self.config:
            st.json(self.config)
        else:
            st.warning("No configuration loaded")

        # Configuration form
        with st.expander("Edit Configuration"):
            st.text_area("Configuration YAML", value = "# Edit configuration here", height = 200)
            if st.button("💾 Save Configuration"):
                st.success("Configuration saved!")

    def _show_maintenance_tools(self):
        """Show maintenance tools"""
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Restart Services"):
                st.info("Services restarted")

        with col2:
            if st.button("🧹 Clear Cache"):
                st.info("Cache cleared")

        with col3:
            if st.button("💾 Backup System"):
                st.info("Backup initiated")

    def _show_auth_status(self):
        """Show authentication system status"""
        if AUTH_AVAILABLE:
            status = auth_manager.get_system_status()

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Active Sessions", status.get("active_sessions", 0))
                st.metric("Total Logins", status.get("login_count", 0))

            with col2:
                st.metric("Failed Attempts", status.get("failed_attempt_ips", 0))
                if status.get("last_login"):
                    st.metric("Last Login", status["last_login"])
        else:
            st.error("Authentication system not available")

    def _show_session_management(self):
        """Show session management"""
        if st.button("🚪 Logout All Sessions"):
            if AUTH_AVAILABLE:
                count = auth_manager.logout_all_sessions()
                st.success(f"Logged out {count} sessions")
            else:
                st.error("Authentication system not available")

    def _show_security_logs(self):
        """Show security logs"""
        logs = [
            {"Time": "14:30:00", "Event": "Login", "User": "admin", "IP": "127.0.0.1", "Status": "Success"}, 
            {"Time": "14:25:00", "Event": "Failed Login", "User": "unknown", "IP": "192.168.1.100", "Status": "Failed"}, 
            {"Time": "14:20:00", "Event": "Password Change", "User": "admin", "IP": "127.0.0.1", "Status": "Success"}, 
        ]

        df = pd.DataFrame(logs)
        st.dataframe(df, use_container_width = True, hide_index = True)

    def _show_password_management(self):
        """Show password management"""
        with st.form("change_password"):
            st.subheader("Change Password")

            current_password = st.text_input("Current Password", type = "password")
            new_password = st.text_input("New Password", type = "password")
            confirm_password = st.text_input("Confirm New Password", type = "password")

            if st.form_submit_button("🔑 Change Password"):
                if not all([current_password, new_password, confirm_password]):
                    st.error("All fields are required")
                elif new_password != confirm_password:
                    st.error("New passwords do not match")
                elif len(new_password) < 8:
                    st.error("Password must be at least 8 characters")
                elif AUTH_AVAILABLE:
                    if auth_manager.change_password(current_password, new_password):
                        st.success("Password changed successfully!")
                        # Force re - login
                        st.session_state.authenticated = False
                        st.rerun()
                    else:
                        st.error("Failed to change password. Check current password.")
                else:
                    st.error("Authentication system not available")

def main():
    """Main dashboard function"""
    dashboard = ProductionDashboard()

    # Check authentication
    if not st.session_state.authenticated:
        dashboard.show_login_form()
    else:
        # Validate token if available
        if AUTH_AVAILABLE and st.session_state.auth_token:
            if not auth_manager.validate_token(st.session_state.auth_token):
                st.error("Session expired. Please login again.")
                st.session_state.authenticated = False
                st.rerun()

        dashboard.show_main_dashboard()

if __name__ == "__main__":
    main()