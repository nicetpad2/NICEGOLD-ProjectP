import json
import os
import queue
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import streamlit as st

# MUST be the very first Streamlit command - configure page BEFORE any other st commands
st.set_page_config(
    page_title="NICEGOLD ProjectP - AI Agents Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

from plotly.subplots import make_subplots

# Try to import AgentController with error handling
try:
    from agent.agent_controller import AgentController

    AGENT_AVAILABLE = True
except (ImportError, IndentationError, SyntaxError) as e:
    AGENT_AVAILABLE = False
    AgentController = None
    print(f"Warning: Could not import AgentController: {e}")

"""
AI Agents Web Interface - Enhanced Version
===============================================

Advanced web-based interface for controlling and monitoring AI Agents with real-time results display.
Provides comprehensive control over all AI Agent operations through an intuitive web dashboard.
Features include real-time monitoring, advanced visualizations, and result export capabilities.
"""


class EnhancedAIAgentsWebInterface:
    """Enhanced web interface for AI Agents control and monitoring."""

    def __init__(self):
        self.project_root = os.getcwd()
        self.results_dir = os.path.join(self.project_root, "agent_reports")
        self.web_interface_url = "http://localhost:8501"
        self.controller = None
        self.task_queue = queue.Queue()
        self.initialization_error = None
        self.ensure_directories()

    def ensure_directories(self):
        """Ensure required directories exist."""
        os.makedirs(self.results_dir, exist_ok=True)

    def initialize_controller(self):
        """Initialize the agent controller."""
        if self.controller is None:
            try:
                self.controller = AgentController(self.project_root)
                return True
            except Exception as e:
                # Store error for later display in main UI
                self.initialization_error = str(e)
                return False
        return True

    def run_agent_action_async(
        self, action: str, options: Dict[str, Any] = None
    ) -> str:
        """Run agent action asynchronously and return task ID."""
        if not self.initialize_controller():
            return None

        task_id = f"task_{int(datetime.now().timestamp())}"

        def run_task():
            try:
                if action == "comprehensive_analysis":
                    result = self.controller.run_comprehensive_analysis()
                elif action == "deep_analysis":
                    result = self.controller.run_comprehensive_deep_analysis()
                elif action == "auto_fix":
                    result = self.controller.auto_fixer.run_comprehensive_fixes()
                elif action == "optimization":
                    result = self.controller.optimizer.run_comprehensive_optimization()
                elif action == "executive_summary":
                    result = self.controller.generate_executive_summary()
                else:
                    result = {"error": f"Unknown action: {action}"}

                # Save result with task ID
                self.save_task_result(task_id, result, action)

            except Exception as e:
                error_result = {"error": str(e), "action": action}
                self.save_task_result(task_id, error_result, action)

        # Start task in background
        thread = threading.Thread(target=run_task)
        thread.daemon = True
        thread.start()

        return task_id

    def save_task_result(self, task_id: str, result: Dict[str, Any], action: str):
        """Save task result to file."""
        result["task_metadata"] = {
            "task_id": task_id,
            "action": action,
            "completed_at": datetime.now().isoformat(),
            "status": "completed" if "error" not in result else "failed",
        }

        filename = f"{task_id}_{action}.json"
        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, "w", encoding="utf - 8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task result by ID."""
        try:
            for filename in os.listdir(self.results_dir):
                if filename.startswith(task_id) and filename.endswith(".json"):
                    filepath = os.path.join(self.results_dir, filename)
                    with open(filepath, "r", encoding="utf - 8") as f:
                        return json.load(f)
            return None
        except Exception:
            return None

    def get_latest_results(self) -> Optional[Dict[str, Any]]:
        """Get the latest analysis results."""
        try:
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
                os.path.join(self.results_dir, latest_file), "r", encoding="utf - 8"
            ) as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading results: {e}")
            return None

    def get_all_results(self) -> List[Dict[str, Any]]:
        """Get all historical results."""
        try:
            results = []
            for filename in os.listdir(self.results_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(self.results_dir, filename)
                    with open(filepath, "r", encoding="utf - 8") as f:
                        data = json.load(f)
                        data["filename"] = filename
                        results.append(data)

            # Sort by timestamp
            results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return results
        except Exception:
            return []

    def display_enhanced_dashboard(self, results: Dict[str, Any]):
        """Display enhanced project health dashboard."""
        st.markdown("### 📊 Enhanced Project Health Dashboard")

        # Create metrics in columns
        col1, col2, col3, col4, col5 = st.columns(5)

        summary = results.get("summary", {})

        with col1:
            health_score = summary.get("project_health_score", 0)
            delta_color = "normal" if health_score >= 80 else "inverse"
            st.metric("Health Score", f"{health_score:.1f}/100", delta=None)

        with col2:
            issues_count = summary.get("total_issues", 0)
            st.metric("Total Issues", issues_count, delta=None)

        with col3:
            phases = results.get("phases", {})
            fixes_applied = phases.get("auto_fixes", {}).get("fixes_applied", 0)
            st.metric("Fixes Applied", fixes_applied)

        with col4:
            optimizations = phases.get("optimization", {}).get("optimizations_count", 0)
            st.metric("Optimizations", optimizations)

        with col5:
            files_analyzed = summary.get("files_analyzed", 0)
            st.metric("Files Analyzed", files_analyzed)

        # Enhanced visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Health score gauge
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge + number + delta",
                    value=health_score,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Project Health Score"},
                    delta={"reference": 80},
                    gauge={
                        "axis": {"range": [None, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 50], "color": "lightgray"},
                            {"range": [50, 80], "color": "yellow"},
                            {"range": [80, 100], "color": "green"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 90,
                        },
                    },
                )
            )
            fig_gauge.update_layout(height=300, title="Health Score")
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            # Analysis phases pie chart
            phases_data = []
            phases_labels = []
            for phase_name, phase_data in phases.items():
                if isinstance(phase_data, dict) and phase_data:
                    phases_data.append(1)
                    phases_labels.append(phase_name.replace("_", " ").title())

            if phases_data:
                fig_pie = px.pie(
                    values=phases_data,
                    names=phases_labels,
                    title="Analysis Phases Completed",
                )
                fig_pie.update_layout(height=300)
                st.plotly_chart(fig_pie, use_container_width=True)

    def display_real_time_monitoring(self):
        """Display real - time monitoring section."""
        st.markdown("### 👁️ Real - time System Monitoring")

        # Create monitoring placeholders
        try:

            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # Display real - time metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "CPU Usage",
                    f"{cpu_percent:.1f}%",
                    delta=f"{cpu_percent - 50:.1f}%" if cpu_percent > 50 else None,
                )

            with col2:
                st.metric(
                    "Memory Usage",
                    f"{memory.percent:.1f}%",
                    delta=(
                        f"{memory.percent - 70:.1f}%" if memory.percent > 70 else None
                    ),
                )

            with col3:
                disk = psutil.disk_usage("/")
                st.metric("Disk Usage", f"{disk.percent:.1f}%")

            # Alert system
            alerts = []
            if cpu_percent > 80:
                alerts.append("🔴 High CPU usage detected")
            if memory.percent > 85:
                alerts.append("🟡 High memory usage detected")

            if alerts:
                st.warning("⚠️ System Alerts:")
                for alert in alerts:
                    st.write(f"• {alert}")
            else:
                st.success("✅ System running normally")

        except Exception as e:
            st.error(f"Error getting system metrics: {e}")

    def display_historical_trends(self):
        """Display historical analysis trends."""
        st.markdown("### 📈 Historical Analysis Trends")

        all_results = self.get_all_results()

        if not all_results:
            st.info("No historical data available yet. Run some analysis first!")
            return

        # Prepare data for visualization
        dates = []
        health_scores = []
        issues_counts = []
        fixes_applied = []

        for result in all_results:
            timestamp = result.get("timestamp")
            if timestamp:
                try:
                    date = datetime.fromisoformat(timestamp.replace("Z", " + 00:00"))
                    dates.append(date)

                    summary = result.get("summary", {})
                    health_scores.append(summary.get("project_health_score", 0))
                    issues_counts.append(summary.get("total_issues", 0))

                    phases = result.get("phases", {})
                    fixes = phases.get("auto_fixes", {}).get("fixes_applied", 0)
                    fixes_applied.append(fixes)
                except:
                    continue

        if dates:
            # Create trend charts
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Health Score Trend",
                    "Issues Count Trend",
                    "Fixes Applied Trend",
                    "Performance Summary",
                ),
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                ],
            )

            # Health score trend
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=health_scores,
                    name="Health Score",
                    line=dict(color="green"),
                ),
                row=1,
                col=1,
            )

            # Issues trend
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=issues_counts,
                    name="Issues Count",
                    line=dict(color="red"),
                ),
                row=1,
                col=2,
            )

            # Fixes trend
            fig.add_trace(
                go.Bar(x=dates, y=fixes_applied, name="Fixes Applied"), row=2, col=1
            )

            # Performance summary
            avg_health = sum(health_scores) / len(health_scores) if health_scores else 0

            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=avg_health,
                    title={"text": "Avg Health Score"},
                    number={"suffix": "/100"},
                ),
                row=2,
                col=2,
            )

            fig.update_layout(height=600, title="Project Analysis Trends")
            st.plotly_chart(fig, use_container_width=True)


def main():
    """Main web interface function."""
    st.title("🤖 NICEGOLD ProjectP - AI Agents Dashboard")
    st.markdown("**Intelligent Project Analysis & Optimization System**")

    # Initialize enhanced web interface
    web_interface = EnhancedAIAgentsWebInterface()

    # Check for initialization errors
    if web_interface.initialization_error:
        st.error(
            f"❌ Failed to initialize AgentController: {web_interface.initialization_error}"
        )
        st.warning(
            "⚠️ Some features may not be available. Web interface functionality will still work."
        )

    # Sidebar controls
    st.sidebar.header("🎛️ Agent Controls")

    # Action selection
    action = st.sidebar.selectbox(
        "Select Action:",
        [
            "comprehensive_analysis",
            "deep_analysis",
            "auto_fix",
            "optimization",
            "executive_summary",
        ],
        format_func=lambda x: {
            "comprehensive_analysis": "🔍 Comprehensive Analysis",
            "deep_analysis": "🧠 Deep Analysis",
            "auto_fix": "🔧 Auto Fix Issues",
            "optimization": "⚡ Optimize Performance",
            "executive_summary": "📋 Executive Summary",
        }.get(x, x),
    )

    # Advanced options
    with st.sidebar.expander("⚙️ Advanced Options"):
        verbose_mode = st.checkbox("Verbose Output", value=True)
        save_results = st.checkbox("Auto - save Results", value=True)
        real_time_monitor = st.checkbox("Enable Real - time Monitoring", value=False)

    # Run button
    if st.sidebar.button("🚀 Run Agent Action", type="primary"):
        with st.spinner(f"Running {action}..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Start async task
            task_id = web_interface.run_agent_action_async(action)

            if task_id:
                # Poll for completion
                for i in range(100):
                    progress_bar.progress(i + 1)
                    status_text.text(f"Running {action}... {i + 1}%")

                    # Check if task completed
                    result = web_interface.get_task_result(task_id)
                    if result:
                        break

                    time.sleep(0.1)

                # Get final result
                results = web_interface.get_task_result(task_id)

                if results:
                    st.session_state["latest_results"] = results
                    st.session_state["action_completed"] = True

                    if "error" in results:
                        st.error(f"❌ Error: {results['error']}")
                    else:
                        st.success("✅ Action completed successfully!")
                else:
                    st.error("❌ Failed to get results")
            else:
                st.error("❌ Failed to start task")

    # Real - time monitoring section
    if real_time_monitor:
        with st.container():
            web_interface.display_real_time_monitoring()

    # Display results section
    st.markdown(" -  -  - ")

    # Create main tabs
    main_tabs = st.tabs(["📊 Dashboard", "📈 Trends", "📋 Results", "💾 Export"])

    with main_tabs[0]:
        # Dashboard tab
        if "latest_results" in st.session_state and st.session_state.get(
            "action_completed"
        ):
            results = st.session_state["latest_results"]

            if "error" not in results:
                web_interface.display_enhanced_dashboard(results)
            else:
                st.error(f"Error in latest results: {results['error']}")
        else:
            latest_results = web_interface.get_latest_results()
            if latest_results:
                st.info("📋 Showing latest available results:")
                web_interface.display_enhanced_dashboard(latest_results)
            else:
                st.info(
                    "👆 Select an action from the sidebar to start AI Agent analysis"
                )

    with main_tabs[1]:
        # Trends tab
        web_interface.display_historical_trends()

    with main_tabs[2]:
        # Results tab
        st.markdown("### 📋 Detailed Results")

        all_results = web_interface.get_all_results()
        if all_results:
            selected_result = st.selectbox(
                "Select result to view:",
                options=range(len(all_results)),
                format_func=lambda i: f"{all_results[i].get('task_metadata', {}).get('action', 'Unknown')} - {all_results[i].get('timestamp', 'Unknown time')}",
            )

            if selected_result is not None:
                result_data = all_results[selected_result]

                # Display result details
                col1, col2 = st.columns(2)

                with col1:
                    st.json(result_data.get("summary", {}))

                with col2:
                    st.json(result_data.get("phases", {}))
        else:
            st.info("No results available yet.")

    with main_tabs[3]:
        # Export tab
        st.markdown("### 💾 Export & Download")

        if "latest_results" in st.session_state:
            results = st.session_state["latest_results"]

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📥 Download JSON"):
                    st.download_button(
                        label="Download Latest Results",
                        data=json.dumps(results, indent=2, ensure_ascii=False),
                        file_name=f"ai_agent_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                    )

            with col2:
                if st.button("📊 Export CSV"):
                    # Convert results to DataFrame for CSV export
                    summary_data = results.get("summary", {})
                    df = pd.DataFrame([summary_data])
                    csv = df.to_csv(index=False)

                    st.download_button(
                        label="Download Summary CSV",
                        data=csv,
                        file_name=f"ai_agent_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

            with col3:
                if st.button("📋 Generate Report"):
                    # Generate comprehensive report
                    report = f"""
NICEGOLD ProjectP - AI Agent Analysis Report
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

Generated: {datetime.now().strftime('%Y - %m - %d %H:%M:%S')}
Action: {results.get('task_metadata', {}).get('action', 'Unknown')}

SUMMARY:
- Health Score: {results.get('summary', {}).get('project_health_score', 0):.1f}/100
- Total Issues: {results.get('summary', {}).get('total_issues', 0)}
- Files Analyzed: {results.get('summary', {}).get('files_analyzed', 0)}

RECOMMENDATIONS:
{chr(10).join(f"- {rec}" for rec in results.get('recommendations', []))}

DETAILED RESULTS:
{json.dumps(results, indent = 2, ensure_ascii = False)}
                    """

                    st.download_button(
                        label="Download Full Report",
                        data=report,
                        file_name=f"ai_agent_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                    )
        else:
            st.info("No results to export. Run an analysis first.")

    # Footer
    st.markdown(" -  -  - ")
    st.markdown(
        "**NICEGOLD ProjectP AI Agents** - Intelligent Project Management System v2.0"
    )
    st.markdown("Real - time monitoring • Advanced analytics • Automated optimization")


if __name__ == "__main__":
    main()
