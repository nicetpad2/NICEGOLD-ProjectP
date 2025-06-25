# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - AI Commands
════════════════════════════════════════════════════════════════════════════════

AI agents and intelligent automation commands.

Author: NICEGOLD Team
Version: 3.0
Created: 2025-01-05
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import from parent modules
sys.path.append(str(Path(__file__).parent.parent))
from core.colors import Colors, colorize


class AICommands:
    """Handler for AI-related commands"""

    def __init__(self, project_root: Path, csv_manager=None, logger=None):
        self.project_root = project_root
        self.csv_manager = csv_manager
        self.logger = logger
        self.python_cmd = [sys.executable]

    def run_command(
        self, command: List[str], description: str, capture_output: bool = False
    ) -> bool:
        """Execute a command with proper error handling"""
        try:
            print(f"\n{colorize('⚡ ' + description, Colors.BRIGHT_CYAN)}")
            print(f"{colorize('═' * 60, Colors.DIM)}")

            # Ensure we're in the right directory
            os.chdir(self.project_root)

            if capture_output:
                result = subprocess.run(
                    command, capture_output=True, text=True, cwd=self.project_root
                )
                if result.returncode == 0:
                    print(
                        f"{colorize('✅ Command completed successfully', Colors.BRIGHT_GREEN)}"
                    )
                    if result.stdout:
                        print(result.stdout)
                    return True
                else:
                    print(f"{colorize('❌ Command failed', Colors.BRIGHT_RED)}")
                    if result.stderr:
                        print(result.stderr)
                    return False
            else:
                result = subprocess.run(command, cwd=self.project_root)
                return result.returncode == 0

        except Exception as e:
            print(f"{colorize('❌ Error executing command:', Colors.BRIGHT_RED)} {e}")
            if self.logger:
                self.logger.error(
                    f"Command execution failed: {description}", "AI", str(e), e
                )
            return False

    def show_ai_agents_menu(self) -> bool:
        """Show AI agents menu"""
        print(f"{colorize('🤖 Launching AI Agents Menu...', Colors.BRIGHT_MAGENTA)}")

        try:
            # Import and run AI agents menu
            from ai_agents_menu import show_ai_agents_menu

            return show_ai_agents_menu()
        except ImportError:
            print(f"{colorize('❌ AI Agents menu not available', Colors.BRIGHT_RED)}")
            return False

    def run_ai_analysis(self) -> bool:
        """Run AI analysis"""
        print(f"{colorize('🧠 Starting AI Analysis...', Colors.BRIGHT_BLUE)}")

        try:
            from ai_agents_menu import run_ai_analysis

            return run_ai_analysis()
        except ImportError:
            return self.run_command(
                ["python", "ai_agents_menu.py", "--mode", "analysis"], "AI Analysis"
            )

    def run_ai_optimization(self) -> bool:
        """Run AI optimization"""
        print(f"{colorize('⚙️ Starting AI Optimization...', Colors.BRIGHT_YELLOW)}")

        try:
            from ai_agents_menu import run_ai_optimization

            return run_ai_optimization()
        except ImportError:
            return self.run_command(
                ["python", "ai_agents_menu.py", "--mode", "optimization"],
                "AI Optimization",
            )

    def run_ai_autofix(self) -> bool:
        """Run AI autofix"""
        print(f"{colorize('🔧 Starting AI AutoFix...', Colors.BRIGHT_GREEN)}")

        try:
            from ai_agents_menu import run_ai_autofix

            return run_ai_autofix()
        except ImportError:
            return self.run_command(
                ["python", "ai_agents_menu.py", "--mode", "autofix"], "AI AutoFix"
            )

    def show_ai_dashboard(self) -> bool:
        """Show AI dashboard"""
        print(f"{colorize('📊 Launching AI Dashboard...', Colors.BRIGHT_CYAN)}")

        try:
            from ai_agents_menu import show_ai_dashboard

            return show_ai_dashboard()
        except ImportError:
            return self.run_command(
                ["python", "ai_agents_menu.py", "--mode", "dashboard"], "AI Dashboard"
            )

    def web_dashboard(self) -> bool:
        """Launch web dashboard"""
        print(f"{colorize('🌐 Launching Web Dashboard...', Colors.BRIGHT_MAGENTA)}")
        print(
            f"{colorize('Dashboard will open in your default browser...', Colors.DIM)}"
        )

        try:
            # Try to import and run web dashboard
            from ai_agents_web import main as run_web_dashboard

            return run_web_dashboard()
        except ImportError:
            return self.run_command(["python", "ai_agents_web.py"], "Web Dashboard")

    def executive_summary(self) -> bool:
        """Generate executive summary"""
        print(f"{colorize('📋 Generating Executive Summary...', Colors.BRIGHT_BLUE)}")

        try:
            from ai_agents_menu import handle_executive_summary

            return handle_executive_summary()
        except ImportError:
            return self.run_command(
                ["python", "ai_agents_menu.py", "--mode", "summary"],
                "Executive Summary Generation",
            )

    def project_analysis(self) -> bool:
        """Run comprehensive project analysis"""
        print(f"{colorize('🔍 Starting Project Analysis...', Colors.BRIGHT_CYAN)}")

        try:
            from ai_agents_menu import handle_project_analysis

            return handle_project_analysis()
        except ImportError:
            return self.run_command(
                ["python", "ai_agents_menu.py", "--mode", "project_analysis"],
                "Project Analysis",
            )

    def ai_team_coordination(self) -> bool:
        """Run AI team coordination"""
        print(
            f"{colorize('👥 Starting AI Team Coordination...', Colors.BRIGHT_MAGENTA)}"
        )

        # Create AI team coordination script
        coordination_code = f"""
import sys
import os
sys.path.append('.')
os.chdir('{self.project_root}')

print('👥 NICEGOLD AI Team Coordination')
print('=' * 50)

try:
    # Import AI modules
    from ai_team_manager import AITeamManager
    from ai_orchestrator import AIOrchestrator
    
    print('🤖 Initializing AI team...')
    
    # Create team manager
    team_manager = AITeamManager()
    orchestrator = AIOrchestrator()
    
    print('🔄 Starting coordination session...')
    
    # Run coordination
    results = team_manager.coordinate_team()
    orchestrator.orchestrate_pipeline()
    
    print('✅ AI team coordination completed!')
    print('📊 Results available in AI coordination logs')

except ImportError as e:
    print(f'❌ Import error: {{e}}')
    print('💡 Please ensure AI modules are available')
except Exception as e:
    print(f'❌ Coordination error: {{e}}')
    import traceback
    traceback.print_exc()
"""

        return self.run_command(
            self.python_cmd + ["-c", coordination_code], "AI Team Coordination"
        )

    def intelligent_automation(self) -> bool:
        """Run intelligent automation suite"""
        print(
            f"{colorize('🧠 Starting Intelligent Automation...', Colors.BOLD + Colors.BRIGHT_BLUE)}"
        )
        print(f"{colorize('Running full AI automation suite...', Colors.DIM)}")

        success = True

        # Run AI automation components
        automations = [
            ("AI Analysis", self.run_ai_analysis),
            ("AI Optimization", self.run_ai_optimization),
            ("Project Analysis", self.project_analysis),
            ("AI Team Coordination", self.ai_team_coordination),
        ]

        for name, func in automations:
            print(f"\n{colorize(f'Running {name}...', Colors.BRIGHT_CYAN)}")
            if not func():
                print(f"{colorize(f'❌ {name} failed', Colors.BRIGHT_RED)}")
                success = False
            else:
                print(f"{colorize(f'✅ {name} completed', Colors.BRIGHT_GREEN)}")

        if success:
            print(
                f"\n{colorize('🎉 Intelligent automation completed successfully!', Colors.BOLD + Colors.BRIGHT_GREEN)}"
            )
        else:
            print(
                f"\n{colorize('⚠️ Some automations failed - check logs for details', Colors.BRIGHT_YELLOW)}"
            )

        return success
