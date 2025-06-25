import json
import os
import subprocess
import sys
import time
import webbrowser
from datetime import datetime
from typing import Any, Dict, Optional

import streamlit

"""
AI Agents Menu Integration
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

Integration functions for connecting AI Agents with the main ProjectP menu system.
Provides seamless access to all AI Agent capabilities through the command - line interface.
"""


# Add agent path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
agent_path = os.path.join(current_dir, "agent")
if agent_path not in sys.path:
    sys.path.append(agent_path)


def _get_agent_controller():
    """Lazy import to avoid circular import"""
    try:
        from agent.agent_controller import AgentController

        return AgentController
    except ImportError as e:
        print(f"Warning: Could not import AgentController: {e}")
        return None


class AIAgentsMenuIntegration:
    """Integration class for AI Agents menu operations."""

    def __init__(self, project_root: str = None):
        self.project_root = project_root or os.getcwd()
        self.reports_dir = os.path.join(self.project_root, "agent_reports")
        self.web_interface_url = "http://localhost:8501"
        self.ensure_directories()

    def ensure_directories(self):
        """Ensure required directories exist."""
        os.makedirs(self.reports_dir, exist_ok=True)

    def run_project_analysis(self) -> bool:
        """Run comprehensive project analysis."""
        try:
            print("🔍 Starting comprehensive project analysis...")
            print(" = " * 60)

            # Import and run agent controller
            AgentController = _get_agent_controller()
            if AgentController is None:
                print("❌ AgentController not available")
                return False

            controller = AgentController(self.project_root)
            results = controller.run_comprehensive_analysis()

            if "error" in results:
                print(f"❌ Error during analysis: {results['error']}")
                return False

            # Display summary
            self._display_analysis_summary(results)

            # Save results
            results_file = self._save_results(results, "comprehensive_analysis")
            print(f"\n📁 Full results saved to: {results_file}")

            print("✅ Project analysis completed successfully!")
            return True

        except Exception as e:
            print(f"❌ Error running project analysis: {e}")
            return False

    def run_auto_fix(self) -> bool:
        """Run automatic issue fixing."""
        try:
            print("🔧 Starting automatic issue fixing...")
            print(" = " * 60)

            AgentController = _get_agent_controller()
            if AgentController is None:
                print("❌ AgentController not available")
                return False

            controller = AgentController(self.project_root)
            results = controller.auto_fixer.run_comprehensive_fixes()

            if "error" in results:
                print(f"❌ Error during auto - fix: {results['error']}")
                return False

            # Display fix summary
            self._display_fix_summary(results)

            # Save results
            results_file = self._save_results(results, "auto_fix")
            print(f"\n📁 Fix results saved to: {results_file}")

            print("✅ Auto - fix completed successfully!")
            return True

        except Exception as e:
            print(f"❌ Error running auto - fix: {e}")
            return False

    def run_optimization(self) -> bool:
        """Run project optimization."""
        try:
            print("⚡ Starting project optimization...")
            print(" = " * 60)

            AgentController = _get_agent_controller()
            if AgentController is None:
                print("❌ AgentController not available")
                return False

            controller = AgentController(self.project_root)
            results = controller.optimizer.run_comprehensive_optimization()

            if "error" in results:
                print(f"❌ Error during optimization: {results['error']}")
                return False

            # Display optimization summary
            self._display_optimization_summary(results)

            # Save results
            results_file = self._save_results(results, "optimization")
            print(f"\n📁 Optimization results saved to: {results_file}")

            print("✅ Optimization completed successfully!")
            return True

        except Exception as e:
            print(f"❌ Error running optimization: {e}")
            return False

    def generate_executive_summary(self) -> bool:
        """Generate executive summary."""
        try:
            print("📋 Generating executive summary...")
            print(" = " * 60)

            AgentController = _get_agent_controller()
            if AgentController is None:
                print("❌ AgentController not available")
                return False

            controller = AgentController(self.project_root)

            # Run comprehensive analysis first if no recent results exist
            if not self._has_recent_results():
                print("🔍 Running analysis first...")
                analysis_results = controller.run_comprehensive_analysis()

            # Generate executive summary
            summary_results = controller.generate_executive_summary()

            if "error" in summary_results:
                print(f"❌ Error generating summary: {summary_results['error']}")
                return False

            # Display executive summary
            self._display_executive_summary(summary_results)

            # Save results
            results_file = self._save_results(summary_results, "executive_summary")
            print(f"\n📁 Executive summary saved to: {results_file}")

            print("✅ Executive summary generated successfully!")
            return True

        except Exception as e:
            print(f"❌ Error generating executive summary: {e}")
            return False

    def launch_web_dashboard(self) -> bool:
        """Launch the AI Agents web dashboard."""
        try:
            print("🌐 Launching AI Agents Web Dashboard...")
            print(" = " * 60)

            # Check if streamlit is available
            try:
                import plotly
                import streamlit
            except ImportError:
                print("❌ Streamlit not found. Installing...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "streamlit", "plotly"],
                    check=True,
                )

            # Launch streamlit app
            web_app_path = os.path.join(self.project_root, "ai_agents_web.py")

            if not os.path.exists(web_app_path):
                print(f"❌ Web interface file not found: {web_app_path}")
                return False

            print(f"🚀 Starting web dashboard at {self.web_interface_url}")
            print("📱 Opening browser...")

            # Start streamlit in background
            process = subprocess.Popen(
                [
                    sys.executable,
                    " - m",
                    "streamlit",
                    "run",
                    web_app_path,
                    " -  - server.port",
                    "8501",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait a moment for server to start
            time.sleep(3)

            # Open browser
            webbrowser.open(self.web_interface_url)

            print("✅ Web dashboard launched successfully!")
            print("🔗 Access URL: http://localhost:8501")
            print("💡 Press Ctrl + C in terminal to stop the web server")

            return True

        except Exception as e:
            print(f"❌ Error launching web dashboard: {e}")
            return False

    def _display_analysis_summary(self, results: Dict[str, Any]):
        """Display analysis summary."""
        summary = results.get("summary", {})

        print("📊 ANALYSIS SUMMARY")
        print(" - " * 40)

        health_score = summary.get("project_health_score", 0)
        print(f"🏥 Project Health Score: {health_score:.1f}/100")

        total_issues = summary.get("total_issues", 0)
        print(f"⚠️  Total Issues Found: {total_issues}")

        files_analyzed = summary.get("files_analyzed", 0)
        print(f"📁 Files Analyzed: {files_analyzed}")

        # Display phase summaries
        phases = results.get("phases", {})
        if phases.get("understanding"):
            print(f"📋 Project Understanding: ✅ Complete")
        if phases.get("code_analysis"):
            print(f"🔍 Code Analysis: ✅ Complete")
        if phases.get("auto_fixes"):
            fixes_count = phases["auto_fixes"].get("fixes_applied", 0)
            print(f"🔧 Auto Fixes Applied: {fixes_count}")
        if phases.get("optimization"):
            opts_count = phases["optimization"].get("optimizations_count", 0)
            print(f"⚡ Optimizations: {opts_count}")

    def _display_fix_summary(self, results: Dict[str, Any]):
        """Display fix summary."""
        print("🔧 AUTO - FIX SUMMARY")
        print(" - " * 40)

        fixes_applied = results.get("fixes_applied", 0)
        print(f"✅ Fixes Applied: {fixes_applied}")

        applied_fixes = results.get("applied_fixes", [])
        if applied_fixes:
            print("\n🔧 Applied Fixes:")
            for fix in applied_fixes[:5]:  # Show first 5
                print(f"  • {fix}")
            if len(applied_fixes) > 5:
                print(f"  ... and {len(applied_fixes) - 5} more")

    def _display_optimization_summary(self, results: Dict[str, Any]):
        """Display optimization summary."""
        print("⚡ OPTIMIZATION SUMMARY")
        print(" - " * 40)

        optimizations = results.get("optimizations", [])
        print(f"🚀 Optimizations Applied: {len(optimizations)}")

        if optimizations:
            print("\n⚡ Optimizations:")
            for opt in optimizations[:5]:  # Show first 5
                print(f"  • {opt}")
            if len(optimizations) > 5:
                print(f"  ... and {len(optimizations) - 5} more")

    def _display_executive_summary(self, results: Dict[str, Any]):
        """Display executive summary."""
        print("📋 EXECUTIVE SUMMARY")
        print(" - " * 40)

        summary = results.get("executive_summary", {})

        if "key_findings" in summary:
            print("🔍 Key Findings:")
            for finding in summary["key_findings"][:3]:
                print(f"  • {finding}")

        if "recommendations" in summary:
            print("\n💡 Top Recommendations:")
            for rec in summary["recommendations"][:3]:
                print(f"  • {rec}")

        if "next_steps" in summary:
            print("\n🎯 Next Steps:")
            for step in summary["next_steps"][:3]:
                print(f"  • {step}")

    def _save_results(self, results: Dict[str, Any], action_type: str) -> str:
        """Save results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent_{action_type}_{timestamp}.json"
        filepath = os.path.join(self.reports_dir, filename)

        # Add metadata
        results["metadata"] = {
            "action_type": action_type,
            "timestamp": datetime.now().isoformat(),
            "project_root": self.project_root,
        }

        with open(filepath, "w", encoding="utf - 8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return filepath

    def _has_recent_results(self) -> bool:
        """Check if there are recent analysis results."""
        try:
            results_files = [
                f for f in os.listdir(self.reports_dir) if f.endswith(".json")
            ]
            if not results_files:
                return False

            # Check if latest file is less than 1 hour old
            latest_file = max(
                results_files,
                key=lambda x: os.path.getctime(os.path.join(self.reports_dir, x)),
            )
            latest_time = os.path.getctime(os.path.join(self.reports_dir, latest_file))
            current_time = time.time()

            return (current_time - latest_time) < 3600  # 1 hour
        except:
            return False


# Global instance for easy access
ai_agents_integration = AIAgentsMenuIntegration()


# Menu handler functions
def handle_project_analysis():
    """Handle project analysis menu option."""
    return ai_agents_integration.run_project_analysis()


def handle_auto_fix():
    """Handle auto - fix menu option."""
    return ai_agents_integration.run_auto_fix()


def handle_optimization():
    """Handle optimization menu option."""
    return ai_agents_integration.run_optimization()


def handle_executive_summary():
    """Handle executive summary menu option."""
    return ai_agents_integration.generate_executive_summary()


def handle_web_dashboard():
    """Handle web dashboard launch menu option."""
    return ai_agents_integration.launch_web_dashboard()


# Functions required by ProjectP.py
def run_ai_analysis():
    """Run AI analysis - wrapper function for ProjectP compatibility"""
    try:
        integration = AIAgentsMenuIntegration()
        return integration.run_project_analysis()
    except Exception as e:
        print(f"❌ Error in AI analysis: {e}")
        return False


def run_ai_autofix():
    """Run AI auto-fix - wrapper function for ProjectP compatibility"""
    try:
        integration = AIAgentsMenuIntegration()
        return integration.run_auto_fix()
    except Exception as e:
        print(f"❌ Error in AI auto-fix: {e}")
        return False


def run_ai_optimization():
    """Run AI optimization - wrapper function for ProjectP compatibility"""
    try:
        integration = AIAgentsMenuIntegration()
        return integration.run_optimization()
    except Exception as e:
        print(f"❌ Error in AI optimization: {e}")
        return False


def show_ai_agents_menu():
    """Show AI agents menu - wrapper function for ProjectP compatibility"""
    try:
        print("🤖 AI Agents Menu")
        print("=" * 50)
        print("1. Project Analysis")
        print("2. Auto Fix")
        print("3. Optimization")
        print("4. Executive Summary")
        print("5. Web Dashboard")
        choice = input("Select option (1-5): ").strip()

        integration = AIAgentsMenuIntegration()
        if choice == "1":
            return integration.run_project_analysis()
        elif choice == "2":
            return integration.run_auto_fix()
        elif choice == "3":
            return integration.run_optimization()
        elif choice == "4":
            return integration.generate_executive_summary()
        elif choice == "5":
            return integration.launch_web_dashboard()
        else:
            print("Invalid choice")
            return False
    except Exception as e:
        print(f"❌ Error in AI agents menu: {e}")
        return False


def show_ai_dashboard():
    """Show AI dashboard - wrapper function for ProjectP compatibility"""
    try:
        integration = AIAgentsMenuIntegration()
        return integration.launch_web_dashboard()
    except Exception as e:
        print(f"❌ Error launching AI dashboard: {e}")
        return False
