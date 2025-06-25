from agent.agent_controller import AgentController
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import threading
import time
"""
AI Agents Web Interface
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

Web - based interface for controlling and monitoring AI Agents with real - time results display.
Provides comprehensive control over all AI Agent operations through an intuitive web dashboard.
"""


# Import agent controller
try:
except ImportError:
    st.error(
        "❌ Could not import AgentController. Please ensure agent modules are available."
    )
    st.stop()


class AIAgentsWebInterface:
    """Web interface for AI Agents control and monitoring."""

    def __init__(self):
        self.project_root = os.getcwd()
        self.results_dir = os.path.join(self.project_root, "agent_reports")
        self.controller = None
        self.ensure_directories()

    def ensure_directories(self):
        """Ensure required directories exist."""
        os.makedirs(self.results_dir, exist_ok = True)

    def initialize_controller(self):
        """Initialize the agent controller."""
        if self.controller is None:
            try:
                self.controller = AgentController(self.project_root)
                return True
            except Exception as e:
                st.error(f"❌ Failed to initialize AgentController: {e}")
                return False
        return True

    def run_agent_action(
        self, action: str, options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Run specified agent action."""
        if not self.initialize_controller():
            return {"error": "Failed to initialize controller"}

        try:
            if action == "comprehensive_analysis":
                return self.controller.run_comprehensive_analysis()
            elif action == "deep_analysis":
                return self.controller.run_comprehensive_deep_analysis()
            elif action == "auto_fix":
                return self.controller.auto_fixer.run_comprehensive_fixes()
            elif action == "optimization":
                return self.controller.optimizer.run_comprehensive_optimization()
            elif action == "executive_summary":
                return self.controller.generate_executive_summary()
            else:
                return {"error": f"Unknown action: {action}"}
        except Exception as e:
            return {"error": str(e)}

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
                key = lambda x: os.path.getctime(os.path.join(self.results_dir, x)), 
            )
            with open(
                os.path.join(self.results_dir, latest_file), "r", encoding = "utf - 8"
            ) as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading results: {e}")
            return None

    def display_project_health_dashboard(self, results: Dict[str, Any]):
        """Display project health dashboard."""
        st.subheader("📊 Project Health Dashboard")

        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            health_score = results.get("summary", {}).get("project_health_score", 0)
            st.metric("Health Score", f"{health_score:.1f}/100", delta = None)

        with col2:
            issues_count = results.get("summary", {}).get("total_issues", 0)
            st.metric("Total Issues", issues_count)

        with col3:
            fixes_applied = (
                results.get("phases", {}).get("auto_fixes", {}).get("fixes_applied", 0)
            )
            st.metric("Fixes Applied", fixes_applied)

        with col4:
            optimizations = (
                results.get("phases", {})
                .get("optimization", {})
                .get("optimizations_count", 0)
            )
            st.metric("Optimizations", optimizations)

        # Health score gauge
        fig = go.Figure(
            go.Indicator(
                mode = "gauge + number + delta", 
                value = health_score, 
                domain = {"x": [0, 1], "y": [0, 1]}, 
                title = {"text": "Project Health Score"}, 
                delta = {"reference": 80}, 
                gauge = {
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
        fig.update_layout(height = 300)
        st.plotly_chart(fig, use_container_width = True)

    def display_analysis_results(self, results: Dict[str, Any]):
        """Display comprehensive analysis results."""
        st.subheader("🔍 Analysis Results")

        # Create tabs for different analysis phases
        tabs = st.tabs(
            [
                "Understanding", 
                "Code Analysis", 
                "Auto Fixes", 
                "Optimization", 
                "Recommendations", 
            ]
        )

        with tabs[0]:
            understanding = results.get("phases", {}).get("understanding", {})
            if understanding:
                st.write("**Project Structure Analysis:**")
                st.json(understanding.get("structure_analysis", {}))

                if "files_analyzed" in understanding:
                    st.write(f"**Files Analyzed:** {understanding['files_analyzed']}")

        with tabs[1]:
            code_analysis = results.get("phases", {}).get("code_analysis", {})
            if code_analysis:
                st.write("**Code Quality Metrics:**")
                quality_data = code_analysis.get("quality_metrics", {})
                if quality_data:
                    df = pd.DataFrame([quality_data])
                    st.dataframe(df)

        with tabs[2]:
            auto_fixes = results.get("phases", {}).get("auto_fixes", {})
            if auto_fixes:
                st.write("**Applied Fixes:**")
                fixes = auto_fixes.get("applied_fixes", [])
                if fixes:
                    for fix in fixes:
                        st.success(f"✅ {fix}")
                else:
                    st.info("No fixes were applied")

        with tabs[3]:
            optimization = results.get("phases", {}).get("optimization", {})
            if optimization:
                st.write("**Optimization Results:**")
                optimizations = optimization.get("optimizations", [])
                if optimizations:
                    for opt in optimizations:
                        st.info(f"⚡ {opt}")
                else:
                    st.info("No optimizations found")

        with tabs[4]:
            recommendations = results.get("recommendations", [])
            if recommendations:
                st.write("**AI Recommendations:**")
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
            else:
                st.info("No recommendations available")


def main():
    """Main web interface function."""
    st.set_page_config(
        page_title = "NICEGOLD ProjectP - AI Agents Dashboard", 
        page_icon = "🤖", 
        layout = "wide", 
        initial_sidebar_state = "expanded", 
    )

    st.title("🤖 NICEGOLD ProjectP - AI Agents Dashboard")
    st.markdown("**Intelligent Project Analysis & Optimization System**")

    # Initialize web interface
    web_interface = AIAgentsWebInterface()

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
        format_func = lambda x: {
            "comprehensive_analysis": "🔍 Comprehensive Analysis", 
            "deep_analysis": "🧠 Deep Analysis", 
            "auto_fix": "🔧 Auto Fix Issues", 
            "optimization": "⚡ Optimize Performance", 
            "executive_summary": "📋 Executive Summary", 
        }.get(x, x), 
    )

    # Run button
    if st.sidebar.button("🚀 Run Agent Action", type = "primary"):
        with st.spinner(f"Running {action}..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Simulate progress
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"Running {action}... {i + 1}%")
                time.sleep(0.01)  # Simulated delay

            # Run actual action
            results = web_interface.run_agent_action(action)

            # Store results in session state
            st.session_state["latest_results"] = results
            st.session_state["action_completed"] = True

            if "error" in results:
                st.error(f"❌ Error: {results['error']}")
            else:
                st.success("✅ Action completed successfully!")

    # Display results section
    st.markdown(" -  -  - ")

    # Check if we have results to display
    if "latest_results" in st.session_state and st.session_state.get(
        "action_completed"
    ):
        results = st.session_state["latest_results"]

        if "error" not in results:
            # Display dashboard
            web_interface.display_project_health_dashboard(results)

            # Display detailed results
            web_interface.display_analysis_results(results)

            # Download results
            st.markdown(" -  -  - ")
            st.subheader("💾 Export Results")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Download JSON Report"):
                    st.download_button(
                        label = "Download Report", 
                        data = json.dumps(results, indent = 2, ensure_ascii = False), 
                        file_name = f"agent_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 
                        mime = "application/json", 
                    )

            with col2:
                if st.button("📊 View Historical Results"):
                    st.info("Historical results viewer - Coming soon!")

    else:
        # Show latest available results
        latest_results = web_interface.get_latest_results()
        if latest_results:
            st.info("📋 Showing latest available results:")
            web_interface.display_project_health_dashboard(latest_results)
            web_interface.display_analysis_results(latest_results)
        else:
            st.info("👆 Select an action from the sidebar to start AI Agent analysis")

    # Footer
    st.markdown(" -  -  - ")
    st.markdown(
        "**NICEGOLD ProjectP AI Agents** - Intelligent Project Management System"
    )


if __name__ == "__main__":
    main()