# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - Dashboard Server
════════════════════════════════════════════════════════════════════════════════

Streamlit dashboard server for monitoring and visualization.

Author: NICEGOLD Team
Version: 3.0
Created: 2025-01-05
"""

import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional

# Import from parent modules
sys.path.append(str(Path(__file__).parent.parent))
from core.colors import Colors, colorize


class DashboardServer:
    """Streamlit dashboard server for monitoring and visualization"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8501,
        project_root: Optional[Path] = None,
    ):
        self.host = host
        self.port = port
        self.project_root = project_root or Path.cwd()
        self.process = None
        self.is_running = False
        self.start_time = None

    def start_dashboard(
        self, background: bool = True, open_browser: bool = True
    ) -> bool:
        """Start the Streamlit dashboard"""
        try:
            # Check if dashboard app exists
            dashboard_files = [
                "dashboard_app.py",
                "ai_agents_web.py",
                "ai_agents_web_enhanced.py",
            ]

            dashboard_app = None
            for app_file in dashboard_files:
                app_path = self.project_root / app_file
                if app_path.exists():
                    dashboard_app = app_file
                    break

            if not dashboard_app:
                print(f"{colorize('❌ No dashboard app found', Colors.BRIGHT_RED)}")
                print(
                    f"{colorize('📁 Looking for:', Colors.BRIGHT_YELLOW)} {', '.join(dashboard_files)}"
                )
                return False

            print(
                f"{colorize('🌐 Starting Streamlit Dashboard...', Colors.BRIGHT_GREEN)}"
            )
            print(
                f"{colorize('📁 Using dashboard app:', Colors.BRIGHT_CYAN)} {dashboard_app}"
            )
            print(
                f"{colorize('📍 Dashboard will be available at:', Colors.BRIGHT_CYAN)} http://{self.host}:{self.port}"
            )

            # Prepare streamlit command
            cmd = [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                dashboard_app,
                "--server.address",
                self.host,
                "--server.port",
                str(self.port),
                "--server.headless",
                "true" if not open_browser else "false",
                "--theme.primaryColor",
                "#00D4AA",
                "--theme.backgroundColor",
                "#0E1117",
                "--theme.secondaryBackgroundColor",
                "#262730",
                "--theme.textColor",
                "#FAFAFA",
            ]

            # Start the process
            if background:
                self.process = subprocess.Popen(
                    cmd,
                    cwd=self.project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                # Give it time to start
                time.sleep(3)

                # Check if process is running
                if self.process.poll() is None:
                    self.is_running = True
                    self.start_time = time.time()

                    print(
                        f"{colorize('✅ Dashboard started successfully in background', Colors.BRIGHT_GREEN)}"
                    )

                    if open_browser:
                        time.sleep(2)  # Give more time for full startup
                        try:
                            webbrowser.open(f"http://{self.host}:{self.port}")
                            print(
                                f"{colorize('🌐 Browser opened to dashboard', Colors.BRIGHT_CYAN)}"
                            )
                        except Exception as e:
                            print(
                                f"{colorize('⚠️ Could not open browser:', Colors.BRIGHT_YELLOW)} {e}"
                            )

                    return True
                else:
                    stdout, stderr = self.process.communicate()
                    print(
                        f"{colorize('❌ Dashboard failed to start', Colors.BRIGHT_RED)}"
                    )
                    if stderr:
                        print(f"{colorize('Error:', Colors.BRIGHT_RED)} {stderr}")
                    return False
            else:
                # Run in foreground
                self.is_running = True
                self.start_time = time.time()
                result = subprocess.run(cmd, cwd=self.project_root)
                self.is_running = False
                return result.returncode == 0

        except Exception as e:
            print(f"{colorize('❌ Error starting dashboard:', Colors.BRIGHT_RED)} {e}")
            return False

    def stop_dashboard(self) -> None:
        """Stop the Streamlit dashboard"""
        if self.is_running and self.process:
            print(
                f"{colorize('🛑 Stopping Streamlit dashboard...', Colors.BRIGHT_YELLOW)}"
            )
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
                print(
                    f"{colorize('✅ Dashboard stopped successfully', Colors.BRIGHT_GREEN)}"
                )
            except subprocess.TimeoutExpired:
                print(
                    f"{colorize('⚠️ Forcing dashboard shutdown...', Colors.BRIGHT_YELLOW)}"
                )
                self.process.kill()
                self.process.wait()
                print(
                    f"{colorize('✅ Dashboard forced shutdown complete', Colors.BRIGHT_GREEN)}"
                )
            except Exception as e:
                print(
                    f"{colorize('❌ Error stopping dashboard:', Colors.BRIGHT_RED)} {e}"
                )
            finally:
                self.is_running = False
                self.process = None

    def get_dashboard_status(self) -> Dict[str, Any]:
        """Get dashboard status"""
        status = {
            "is_running": self.is_running,
            "host": self.host,
            "port": self.port,
            "url": f"http://{self.host}:{self.port}",
            "uptime": 0,
            "process_id": None,
        }

        if self.is_running and self.start_time:
            status["uptime"] = time.time() - self.start_time

        if self.process:
            status["process_id"] = self.process.pid

        return status

    def check_streamlit_available(self) -> bool:
        """Check if Streamlit is available"""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "streamlit", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def install_streamlit(self) -> bool:
        """Install Streamlit if not available"""
        try:
            print(f"{colorize('📦 Installing Streamlit...', Colors.BRIGHT_BLUE)}")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "streamlit"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print(
                    f"{colorize('✅ Streamlit installed successfully', Colors.BRIGHT_GREEN)}"
                )
                return True
            else:
                print(
                    f"{colorize('❌ Failed to install Streamlit:', Colors.BRIGHT_RED)} {result.stderr}"
                )
                return False
        except Exception as e:
            print(
                f"{colorize('❌ Error installing Streamlit:', Colors.BRIGHT_RED)} {e}"
            )
            return False

    def start_dashboard_with_fallback(
        self, background: bool = True, open_browser: bool = True
    ) -> bool:
        """Start dashboard with automatic Streamlit installation if needed"""
        # Check if Streamlit is available
        if not self.check_streamlit_available():
            print(
                f"{colorize('⚠️ Streamlit not found, attempting installation...', Colors.BRIGHT_YELLOW)}"
            )
            if not self.install_streamlit():
                print(
                    f"{colorize('❌ Could not install Streamlit', Colors.BRIGHT_RED)}"
                )
                return False

        # Try to start dashboard
        return self.start_dashboard(background, open_browser)

    def create_simple_dashboard(self) -> bool:
        """Create a simple dashboard file if none exists"""
        dashboard_content = """import streamlit as st
import pandas as pd
import time
from datetime import datetime

st.set_page_config(
    page_title="NICEGOLD ProjectP Dashboard",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 NICEGOLD ProjectP Dashboard")
st.markdown("### Advanced AI Trading Pipeline Monitoring")

# Sidebar
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox("Choose a page", ["Overview", "Models", "Performance", "System"])

if page == "Overview":
    st.header("📈 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Status", "🟢 Online", "✅ Healthy")
    
    with col2:
        st.metric("Models Loaded", "3", "+1")
    
    with col3:
        st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))
    
    with col4:
        st.metric("Uptime", "Running", "Active")

elif page == "Models":
    st.header("🤖 Model Information")
    st.info("Model management features will be available here")

elif page == "Performance":
    st.header("📊 Performance Metrics")
    st.info("Performance analytics will be displayed here")

elif page == "System":
    st.header("⚙️ System Information")
    st.info("System diagnostics and logs will be shown here")

# Auto-refresh
time.sleep(1)
st.rerun()
"""

        try:
            dashboard_path = self.project_root / "simple_dashboard.py"
            with open(dashboard_path, "w", encoding="utf-8") as f:
                f.write(dashboard_content)

            print(
                f"{colorize('✅ Created simple dashboard:', Colors.BRIGHT_GREEN)} {dashboard_path}"
            )
            return True
        except Exception as e:
            print(
                f"{colorize('❌ Failed to create dashboard:', Colors.BRIGHT_RED)} {e}"
            )
            return False
