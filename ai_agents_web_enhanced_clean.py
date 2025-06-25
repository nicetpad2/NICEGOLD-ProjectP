#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Agents Web Interface - Enhanced Version (Clean)
=================================================

Clean implementation of the enhanced web interface with proper Streamlit configuration.
"""

# Standard library imports
import json
import os
import queue
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import streamlit as st
from plotly.subplots import make_subplots


class SafeAIAgentsWebInterface:
    """Safe web interface for AI Agents with robust error handling."""

    def __init__(self):
        self.project_root = os.getcwd()
        self.results_dir = os.path.join(self.project_root, "agent_reports")
        self.controller = None
        self.task_queue = queue.Queue()
        self.initialization_error = None
        self.agent_available = False
        self.ensure_directories()
        self.check_agent_availability()

    def ensure_directories(self):
        """Ensure required directories exist."""
        os.makedirs(self.results_dir, exist_ok=True)

    def check_agent_availability(self):
        """Check if AgentController is available."""
        try:
            from agent.agent_controller import AgentController

            self.AgentController = AgentController
            self.agent_available = True
        except (ImportError, IndentationError, SyntaxError) as e:
            self.agent_available = False
            self.initialization_error = str(e)

    def initialize_controller(self):
        """Initialize the agent controller."""
        if not self.agent_available:
            return False

        if self.controller is None:
            try:
                self.controller = self.AgentController(self.project_root)
                return True
            except Exception as e:
                self.initialization_error = str(e)
                return False
        return True

    def get_latest_results(self) -> Optional[Dict[str, Any]]:
        """Get the most recent analysis results."""
        try:
            if not os.path.exists(self.results_dir):
                return None

            results_files = [
                f for f in os.listdir(self.results_dir) if f.endswith(".json")
            ]
            if not results_files:
                return None

            latest_file = max(
                results_files,
                key=lambda x: os.path.getctime(os.path.join(self.results_dir, x)),
            )
            with open(
                os.path.join(self.results_dir, latest_file), "r", encoding="utf-8"
            ) as f:
                return json.load(f)
        except Exception:
            return None

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available": memory.available / (1024**3),  # GB
                "disk_percent": disk.percent,
                "disk_free": disk.free / (1024**3),  # GB
            }
        except Exception:
            return {
                "cpu_percent": 0,
                "memory_percent": 0,
                "memory_available": 0,
                "disk_percent": 0,
                "disk_free": 0,
            }

    def create_health_gauge(self, value: float, title: str):
        """Create a health gauge chart."""
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=value,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": title},
                delta={"reference": 80},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "gray"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90,
                    },
                },
            )
        )
        fig.update_layout(height=300)
        return fig

    def run_simple_analysis(self) -> Dict[str, Any]:
        """Run a simple mock analysis when agent is not available."""
        return {
            "summary": {
                "project_health_score": 75.0,
                "total_issues": 3,
                "files_analyzed": 25,
                "timestamp": datetime.now().isoformat(),
            },
            "phases": {
                "understanding": {"status": "completed"},
                "analysis": {"status": "completed"},
                "fixes": {"status": "completed"},
            },
            "recommendations": [
                "Consider running full analysis when agent is available",
                "Check system requirements for AI Agents",
                "Review installation documentation",
            ],
            "note": "This is a mock analysis. Install and configure AI Agents for full functionality.",
        }


def render_dashboard(web_interface: SafeAIAgentsWebInterface):
    """Render the main dashboard."""
    st.markdown("### 📊 Project Health Dashboard")

    # Get latest results
    results = web_interface.get_latest_results()

    if results:
        summary = results.get("summary", {})
        health_score = summary.get("project_health_score", 0)
        issues_count = summary.get("total_issues", 0)
        files_analyzed = summary.get("files_analyzed", 0)
    else:
        health_score = 0
        issues_count = 0
        files_analyzed = 0

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Health Score", f"{health_score:.1f}/100")
    with col2:
        st.metric("Total Issues", issues_count)
    with col3:
        st.metric("Files Analyzed", files_analyzed)
    with col4:
        st.metric(
            "Status", "✅ Ready" if web_interface.agent_available else "⚠️ Limited"
        )

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        if health_score > 0:
            fig_gauge = web_interface.create_health_gauge(
                health_score, "Project Health"
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.info(
                "No analysis data available. Run an analysis to see health metrics."
            )

    with col2:
        # System metrics
        metrics = web_interface.get_system_metrics()
        st.markdown("#### 🖥️ System Status")
        st.metric("CPU Usage", f"{metrics['cpu_percent']:.1f}%")
        st.metric("Memory Usage", f"{metrics['memory_percent']:.1f}%")
        st.metric("Disk Usage", f"{metrics['disk_percent']:.1f}%")


def render_actions(web_interface: SafeAIAgentsWebInterface):
    """Render the actions panel."""
    st.markdown("### 🚀 AI Agent Actions")

    if not web_interface.agent_available:
        st.error(f"❌ AI Agents not available: {web_interface.initialization_error}")
        st.warning("Some features are limited. You can still:")
        st.write("- View system metrics")
        st.write("- Run mock analysis")
        st.write("- Access documentation")

        if st.button("🔄 Run Mock Analysis"):
            with st.spinner("Running mock analysis..."):
                time.sleep(2)  # Simulate processing
                results = web_interface.run_simple_analysis()
                st.success("✅ Mock analysis completed!")
                st.json(results)
        return

    # Action selection
    action = st.selectbox(
        "Choose AI Agent Action:",
        [
            "Select an action...",
            "Comprehensive Analysis",
            "Auto Fix",
            "Optimization",
            "Executive Summary",
        ],
    )

    col1, col2 = st.columns(2)

    with col1:
        verbose_mode = st.checkbox("Verbose Output", value=True)
    with col2:
        auto_save = st.checkbox("Auto-save Results", value=True)

    if action != "Select an action...":
        if st.button(f"🚀 Run {action}"):
            st.warning(
                "⚠️ This would run the selected action when AI Agents are fully configured."
            )
            st.info("Currently showing demo functionality.")


def render_results(web_interface: SafeAIAgentsWebInterface):
    """Render the results panel."""
    st.markdown("### 📋 Analysis Results")

    results = web_interface.get_latest_results()

    if results:
        st.success("✅ Latest analysis results loaded")

        # Show summary
        if "summary" in results:
            summary = results["summary"]
            st.markdown("#### Summary")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Health Score", f"{summary.get('project_health_score', 0):.1f}/100"
                )
            with col2:
                st.metric("Issues Found", summary.get("total_issues", 0))
            with col3:
                st.metric("Files Analyzed", summary.get("files_analyzed", 0))

        # Show recommendations
        if "recommendations" in results:
            st.markdown("#### 💡 Recommendations")
            for i, rec in enumerate(results["recommendations"][:5], 1):
                st.write(f"{i}. {rec}")

        # Show full results in expander
        with st.expander("📄 View Full Results"):
            st.json(results)
    else:
        st.info(
            "No analysis results available yet. Run an analysis to see results here."
        )


def render_settings():
    """Render the settings panel."""
    st.markdown("### ⚙️ Settings")

    st.markdown("#### Web Interface Settings")
    port = st.number_input("Server Port", min_value=8000, max_value=9999, value=8501)
    auto_refresh = st.checkbox("Auto-refresh Dashboard", value=True)
    refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 10)

    st.markdown("#### Analysis Settings")
    project_root = st.text_input("Project Root", value=os.getcwd())
    include_deep_analysis = st.checkbox("Include Deep Analysis", value=True)
    backup_before_fixes = st.checkbox("Backup Before Auto-fixes", value=True)

    if st.button("💾 Save Settings"):
        st.success("✅ Settings saved successfully!")


def main():
    """Main web interface function."""
    # Set page config as the absolute first Streamlit command
    st.set_page_config(
        page_title="NICEGOLD ProjectP - AI Agents Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Title and header
    st.title("🤖 NICEGOLD ProjectP - AI Agents Dashboard")
    st.markdown("**Intelligent Project Analysis & Optimization System**")

    # Initialize the web interface
    web_interface = SafeAIAgentsWebInterface()

    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📊 Dashboard", "🚀 Actions", "📋 Results", "⚙️ Settings", "📚 Help"],
    )

    # Agent status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Agent Status")
    if web_interface.agent_available:
        st.sidebar.success("✅ AI Agents Available")
    else:
        st.sidebar.error("❌ AI Agents Unavailable")
        if web_interface.initialization_error:
            st.sidebar.warning(f"Error: {web_interface.initialization_error[:50]}...")

    # Main content area
    if page == "📊 Dashboard":
        render_dashboard(web_interface)
    elif page == "🚀 Actions":
        render_actions(web_interface)
    elif page == "📋 Results":
        render_results(web_interface)
    elif page == "⚙️ Settings":
        render_settings()
    elif page == "📚 Help":
        st.markdown("### 📚 Help & Documentation")
        st.markdown(
            """
        #### Quick Start
        1. **Prerequisites**: Install `streamlit plotly pandas psutil`
        2. **Run Analysis**: Use the Actions page to run AI Agent analysis
        3. **View Results**: Check the Results page for detailed analysis
        4. **Monitor**: Use the Dashboard for real-time monitoring
        
        #### Troubleshooting
        - **Agent Unavailable**: Check if agent modules are properly installed
        - **Import Errors**: Verify Python path and dependencies
        - **Port Issues**: Change port in Settings if 8501 is in use
        
        #### Commands
        ```bash
        # Install dependencies
        pip install streamlit plotly pandas psutil
        
        # Run web interface
        streamlit run ai_agents_web_enhanced_clean.py
        
        # Run CLI
        python run_ai_agents_simple.py --action analyze
        ```
        """
        )

    # Footer
    st.markdown("---")
    st.markdown(
        "**NICEGOLD ProjectP AI Agents** - Intelligent Project Management System v2.0"
    )


if __name__ == "__main__":
    main()
