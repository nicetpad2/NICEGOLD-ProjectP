#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Agents Standalone Runner
===========================

Standalone script for running AI Agents independently from the main ProjectP system.
Provides command-line interface for all AI Agent capabilities.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import AgentController
try:
    from agent.agent_controller import AgentController

    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: AgentController not available ({e}). Some features may not work.")
    AgentController = None
    AGENT_AVAILABLE = False


def setup_args():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(
        description="NICEGOLD ProjectP AI Agents - Intelligent Project Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ai_agents.py --action analyze
  python run_ai_agents.py --action fix --output results.json
  python run_ai_agents.py --action optimize --verbose
  python run_ai_agents.py --action summary
  python run_ai_agents.py --action web --port 8502
        """,
    )

    parser.add_argument(
        "--action",
        "-a",
        choices=["analyze", "fix", "optimize", "summary", "web"],
        required=True,
        help="AI Agent action to perform",
    )

    parser.add_argument(
        "--output", "-o", type=str, help="Output file for results (JSON format)"
    )

    parser.add_argument(
        "--project-root",
        "-p",
        type=str,
        default=None,
        help="Project root directory (default: current directory)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--port", type=int, default=8501, help="Port for web interface (default: 8501)"
    )

    parser.add_argument("--config", type=str, help="Configuration file path")

    return parser.parse_args()


def print_banner():
    """Print AI Agents banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    🤖 NICEGOLD ProjectP - AI Agents System 🤖                               ║
║                                                                              ║
║    Intelligent Project Analysis & Optimization Platform                     ║
║    ─────────────────────────────────────────────────────                   ║
║                                                                              ║
║    🔍 Analysis   🔧 Auto - Fix   ⚡ Optimization   📋 Summaries              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_analyze_action(project_root: str, verbose: bool = False) -> dict:
    """Run comprehensive analysis action."""
    try:
        if not AGENT_AVAILABLE:
            return {
                "error": "AgentController not available. Please check your installation.",
                "action": "analyze",
            }

        if verbose:
            print("🔍 Initializing AgentController...")

        controller = AgentController(project_root)

        if verbose:
            print("🚀 Running comprehensive analysis...")

        results = controller.run_comprehensive_analysis()

        if verbose:
            print("✅ Analysis completed successfully!")

        return results
    except Exception as e:
        return {"error": str(e), "action": "analyze"}


def run_fix_action(project_root: str, verbose: bool = False) -> dict:
    """Run auto-fix action."""
    try:
        if not AGENT_AVAILABLE:
            return {
                "error": "AgentController not available. Please check your installation.",
                "action": "fix",
            }

        if verbose:
            print("🔧 Initializing AgentController...")

        controller = AgentController(project_root)

        if verbose:
            print("🚀 Running auto-fix system...")

        results = controller.auto_fixer.run_comprehensive_fixes()

        if verbose:
            print("✅ Auto-fix completed successfully!")

        return results
    except Exception as e:
        return {"error": str(e), "action": "fix"}


def run_optimize_action(project_root: str, verbose: bool = False) -> dict:
    """Run optimization action."""
    try:
        if not AGENT_AVAILABLE:
            return {
                "error": "AgentController not available. Please check your installation.",
                "action": "optimize",
            }

        if verbose:
            print("⚡ Initializing AgentController...")

        controller = AgentController(project_root)

        if verbose:
            print("🚀 Running optimization system...")

        results = controller.optimizer.run_comprehensive_optimization()

        if verbose:
            print("✅ Optimization completed successfully!")

        return results
    except Exception as e:
        return {"error": str(e), "action": "optimize"}


def run_summary_action(project_root: str, verbose: bool = False) -> dict:
    """Run executive summary action."""
    try:
        if not AGENT_AVAILABLE:
            return {
                "error": "AgentController not available. Please check your installation.",
                "action": "summary",
            }

        if verbose:
            print("📋 Initializing AgentController...")

        controller = AgentController(project_root)

        if verbose:
            print("🚀 Generating executive summary...")

        results = controller.generate_executive_summary()

        if verbose:
            print("✅ Executive summary generated successfully!")

        return results
    except Exception as e:
        return {"error": str(e), "action": "summary"}


def run_web_action(port: int = 8501, verbose: bool = False) -> dict:
    """Run web interface action."""
    try:
        if verbose:
            print(f"🌐 Starting web interface on port {port}...")

        # Check if streamlit is available
        try:
            import streamlit
        except ImportError:
            if verbose:
                print("📦 Installing Streamlit...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "streamlit", "plotly"],
                check=True,
            )

        web_app_path = os.path.join(PROJECT_ROOT, "ai_agents_web.py")

        if not os.path.exists(web_app_path):
            return {
                "error": f"Web interface file not found: {web_app_path}",
                "action": "web",
            }

        if verbose:
            print(f"🚀 Launching web dashboard at http://localhost:{port}")

        # Start streamlit in background
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                web_app_path,
                "--server.port",
                str(port),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to start
        time.sleep(3)

        # Open browser
        webbrowser.open(f"http://localhost:{port}")

        if verbose:
            print("✅ Web dashboard launched successfully!")
            print(f"🔗 Access URL: http://localhost:{port}")
            print("💡 Press Ctrl+C to stop the web server")

        return {
            "action": "web",
            "status": "success",
            "url": f"http://localhost:{port}",
            "process_id": process.pid,
        }

    except Exception as e:
        return {"error": str(e), "action": "web"}


def display_results(results: dict, action: str, verbose: bool = False):
    """Display results in a formatted way."""
    if "error" in results:
        print(f"❌ Error in {action}: {results['error']}")
        return

    print(f"\n📊 {action.upper()} RESULTS")
    print(" = " * 60)

    if action == "analyze":
        summary = results.get("summary", {})
        health_score = summary.get("project_health_score", 0)
        total_issues = summary.get("total_issues", 0)

        print(f"🏥 Project Health Score: {health_score:.1f}/100")
        print(f"⚠️  Total Issues: {total_issues}")

        if verbose:
            print(f"📁 Files Analyzed: {summary.get('files_analyzed', 0)}")
            print(f"🔍 Analysis Phases: {len(results.get('phases', {}))}")

    elif action == "fix":
        fixes_applied = results.get("fixes_applied", 0)
        print(f"🔧 Fixes Applied: {fixes_applied}")

        if verbose and "applied_fixes" in results:
            applied_fixes = results["applied_fixes"]
            print("\n🔧 Applied Fixes:")
            for fix in applied_fixes[:5]:
                print(f"  • {fix}")
            if len(applied_fixes) > 5:
                print(f"  ... and {len(applied_fixes) - 5} more")

    elif action == "optimize":
        optimizations = results.get("optimizations", [])
        print(f"⚡ Optimizations Applied: {len(optimizations)}")

        if verbose and optimizations:
            print("\n⚡ Optimizations:")
            for opt in optimizations[:5]:
                print(f"  • {opt}")
            if len(optimizations) > 5:
                print(f"  ... and {len(optimizations) - 5} more")

    elif action == "summary":
        executive_summary = results.get("executive_summary", {})

        if "key_findings" in executive_summary:
            key_findings = executive_summary["key_findings"]
            print(f"🔍 Key Findings: {len(key_findings)}")
            if verbose:
                for finding in key_findings[:3]:
                    print(f"  • {finding}")

        if "recommendations" in executive_summary:
            recommendations = executive_summary["recommendations"]
            print(f"💡 Recommendations: {len(recommendations)}")
            if verbose:
                for rec in recommendations[:3]:
                    print(f"  • {rec}")

    elif action == "web":
        if results.get("status") == "success":
            print(f"🌐 Web dashboard launched at: {results.get('url')}")
            print(f"🆔 Process ID: {results.get('process_id')}")
        else:
            print("❌ Failed to launch web dashboard")


def save_results(results: dict, output_file: str):
    """Save results to file."""
    try:
        # Add metadata
        results["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "generated_by": "AI Agents Standalone Runner",
            "version": "1.0",
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"💾 Results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")


def main():
    """Main function."""
    args = setup_args()

    print_banner()

    project_root = getattr(args, "project_root", None) or os.getcwd()

    print(f"📁 Project Root: {project_root}")
    print(f"🎯 Action: {args.action}")
    if args.verbose:
        print("🔊 Verbose mode enabled")
    print("─" * 60)

    # Execute the requested action
    results = {}

    if args.action == "analyze":
        results = run_analyze_action(project_root, args.verbose)
    elif args.action == "fix":
        results = run_fix_action(project_root, args.verbose)
    elif args.action == "optimize":
        results = run_optimize_action(project_root, args.verbose)
    elif args.action == "summary":
        results = run_summary_action(project_root, args.verbose)
    elif args.action == "web":
        results = run_web_action(args.port, args.verbose)

    # Display results
    display_results(results, args.action, args.verbose)

    # Save results if requested
    if args.output and args.action != "web":
        save_results(results, args.output)

    print("\n✅ AI Agents operation completed!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
