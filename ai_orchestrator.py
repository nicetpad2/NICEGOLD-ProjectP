#!/usr/bin/env python3
    from ai_assistant_brain import AIAssistantBrain, ai_brain
    from ai_team_manager import AITeamManager, ai_team
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
    from rich.columns import Columns
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, IntPrompt, Prompt
    from rich.table import Table
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import json
import logging
import sqlite3
    import sys
import threading
import time
import yaml
"""
🎯 NICEGOLD AI Orchestrator 🎯
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

ระบบประสานงานและจัดการ AI ทีมสำหรับผู้ใช้คนเดียว
รวมการทำงานของ AI Team Manager และ AI Assistant Brain
เพื่อให้ผู้ใช้สามารถบริหารโปรเจคได้อย่างมีประสิทธิภาพ

Features:
- Unified AI Command Center
- Intelligent Task Distribution
- Automated Decision Making
- Performance Monitoring
- Resource Optimization
- User - Friendly Interface
"""


try:
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Import our AI systems
try:
    AI_SYSTEMS_AVAILABLE = True
except ImportError:
    AI_SYSTEMS_AVAILABLE = False
    logging.warning("AI systems not available")

console = Console() if RICH_AVAILABLE else None

@dataclass
class AIWorkflow:
    """AI Workflow data structure"""
    id: str
    name: str
    description: str
    steps: List[Dict[str, Any]]
    status: str  # "pending", "running", "completed", "failed"
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            **asdict(self), 
            "created_at": self.created_at.isoformat(), 
            "started_at": self.started_at.isoformat() if self.started_at else None, 
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }

class AIOrchestrator:
    """
    AI Orchestrator for managing all AI systems and workflows
    """

    def __init__(self):
        self.project_root = Path(".")
        self.config_dir = self.project_root / "config" / "ai_orchestrator"
        self.logs_dir = self.project_root / "logs" / "ai_orchestrator"
        self.database_dir = self.project_root / "database"

        # Create directories
        for directory in [self.config_dir, self.logs_dir, self.database_dir]:
            directory.mkdir(parents = True, exist_ok = True)

        # Setup logging
        self._setup_logging()

        # Initialize systems
        self.ai_team = None
        self.ai_brain = None
        self.workflows = {}
        self.workflow_counter = 0
        self.is_running = False
        self.monitoring_thread = None

        # Load AI systems
        self._initialize_ai_systems()

        # Load data
        self._load_orchestrator_data()

        # Start monitoring
        self.start_monitoring()

        logger.info("🎯 AI Orchestrator initialized")

    def _setup_logging(self):
        """Setup logging for AI orchestrator"""
        log_file = self.logs_dir / f"ai_orchestrator_{datetime.now().strftime('%Y%m%d')}.log"

        logging.basicConfig(
            level = logging.INFO, 
            format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
            handlers = [
                logging.FileHandler(log_file), 
                logging.StreamHandler()
            ]
        )
        global logger
        logger = logging.getLogger(__name__)

    def _initialize_ai_systems(self):
        """Initialize AI team and brain systems"""
        try:
            if AI_SYSTEMS_AVAILABLE:
                self.ai_team = ai_team
                self.ai_brain = ai_brain
                logger.info("✅ AI systems connected successfully")
            else:
                # Fallback: create mock systems
                self.ai_team = self._create_mock_team()
                self.ai_brain = self._create_mock_brain()
                logger.warning("⚠️ Using mock AI systems")
        except Exception as e:
            logger.error(f"Failed to initialize AI systems: {e}")
            self.ai_team = self._create_mock_team()
            self.ai_brain = self._create_mock_brain()

    def _create_mock_team(self):
        """Create mock AI team for fallback"""
        class MockTeam:
            def assign_task(self, title, description, agent, priority = "medium"):
                return f"mock_task_{int(time.time())}"
            def start_task(self, task_id):
                return True
            def get_team_status(self):
                return {"team_size": 7, "total_tasks": 0, "agents": {}}
            def auto_assign_daily_tasks(self):
                return 5
        return MockTeam()

    def _create_mock_brain(self):
        """Create mock AI brain for fallback"""
        class MockBrain:
            def analyze_market_situation(self, data):
                class MockInsight:
                    def __init__(self):
                        self.title = "Mock Market Analysis"
                        self.description = "Mock analysis result"
                        self.confidence = 0.75
                return MockInsight()
            def assess_portfolio_risk(self, data):
                class MockInsight:
                    def __init__(self):
                        self.title = "Mock Risk Assessment"
                        self.description = "Mock risk analysis"
                        self.confidence = 0.80
                return MockInsight()
            def chat_with_assistant(self, message):
                return f"Mock response to: {message}"
            def get_recent_insights(self, **kwargs):
                return []
        return MockBrain()

    def _load_orchestrator_data(self):
        """Load orchestrator data from database"""
        db_path = self.database_dir / "ai_orchestrator.db"

        try:
            with sqlite3.connect(db_path) as conn:
                # Create tables if not exist
                self._create_tables(conn)

                # Load workflows
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM workflows ORDER BY created_at DESC LIMIT 100")
                workflow_rows = cursor.fetchall()

                for row in workflow_rows:
                    workflow_data = json.loads(row[1])
                    workflow = AIWorkflow(**workflow_data)
                    workflow.created_at = datetime.fromisoformat(workflow_data["created_at"])
                    if workflow_data.get("started_at"):
                        workflow.started_at = datetime.fromisoformat(workflow_data["started_at"])
                    if workflow_data.get("completed_at"):
                        workflow.completed_at = datetime.fromisoformat(workflow_data["completed_at"])
                    self.workflows[workflow.id] = workflow

                # Get workflow counter
                cursor.execute("SELECT MAX(CAST(SUBSTR(id, 10) AS INTEGER)) FROM workflows WHERE id LIKE 'workflow_%'")
                result = cursor.fetchone()
                self.workflow_counter = result[0] if result[0] else 0

        except Exception as e:
            logger.warning(f"Could not load orchestrator data: {e}")

    def _create_tables(self, conn):
        """Create database tables"""
        cursor = conn.cursor()

        # Workflows table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflows (
                id TEXT PRIMARY KEY, 
                data TEXT NOT NULL, 
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Automation rules table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS automation_rules (
                id TEXT PRIMARY KEY, 
                name TEXT NOT NULL, 
                condition_type TEXT NOT NULL, 
                condition_params TEXT NOT NULL, 
                action_type TEXT NOT NULL, 
                action_params TEXT NOT NULL, 
                enabled BOOLEAN DEFAULT 1, 
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # System metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                metric_name TEXT NOT NULL, 
                metric_value REAL NOT NULL, 
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()

    def _save_orchestrator_data(self):
        """Save orchestrator data to database"""
        db_path = self.database_dir / "ai_orchestrator.db"

        try:
            with sqlite3.connect(db_path) as conn:
                self._create_tables(conn)
                cursor = conn.cursor()

                # Save workflows
                for workflow_id, workflow in self.workflows.items():
                    cursor.execute(
                        "INSERT OR REPLACE INTO workflows (id, data) VALUES (?, ?)", 
                        (workflow_id, json.dumps(workflow.to_dict()))
                    )

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to save orchestrator data: {e}")

    def create_workflow(self, name: str, description: str, steps: List[Dict[str, Any]]) -> str:
        """
        Create a new AI workflow

        Args:
            name: Workflow name
            description: Workflow description
            steps: List of workflow steps

        Returns:
            str: Workflow ID
        """
        try:
            # Generate workflow ID
            self.workflow_counter += 1
            workflow_id = f"workflow_{self.workflow_counter:04d}"

            # Create workflow
            workflow = AIWorkflow(
                id = workflow_id, 
                name = name, 
                description = description, 
                steps = steps, 
                status = "pending", 
                created_at = datetime.now()
            )

            self.workflows[workflow_id] = workflow

            # Save data
            self._save_orchestrator_data()

            if RICH_AVAILABLE:
                console.print(f"[green]✅ Workflow '{name}' created with ID: {workflow_id}[/green]")

            logger.info(f"Workflow {workflow_id} created: {name}")
            return workflow_id

        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            raise

    def execute_workflow(self, workflow_id: str) -> bool:
        """
        Execute a workflow

        Args:
            workflow_id: Workflow ID to execute

        Returns:
            bool: True if workflow started successfully
        """
        try:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow {workflow_id} not found")

            workflow = self.workflows[workflow_id]

            # Update workflow status
            workflow.status = "running"
            workflow.started_at = datetime.now()

            # Save data
            self._save_orchestrator_data()

            if RICH_AVAILABLE:
                console.print(f"[blue]🚀 Starting workflow '{workflow.name}'[/blue]")

            logger.info(f"Starting workflow {workflow_id}: {workflow.name}")

            # Execute workflow in background
            asyncio.create_task(self._execute_workflow_steps(workflow_id))

            return True

        except Exception as e:
            logger.error(f"Failed to execute workflow: {e}")
            return False

    async def _execute_workflow_steps(self, workflow_id: str):
        """Execute workflow steps asynchronously"""
        try:
            workflow = self.workflows[workflow_id]
            results = {}

            for i, step in enumerate(workflow.steps):
                step_name = step.get("name", f"Step {i + 1}")
                step_type = step.get("type", "unknown")
                step_params = step.get("params", {})

                if RICH_AVAILABLE:
                    console.print(f"[yellow]⚡ Executing: {step_name}[/yellow]")

                logger.info(f"Executing workflow step: {step_name}")

                # Execute step based on type
                step_result = await self._execute_workflow_step(step_type, step_params)
                results[step_name] = step_result

                # Wait between steps
                await asyncio.sleep(1)

            # Complete workflow
            workflow.status = "completed"
            workflow.completed_at = datetime.now()
            workflow.results = results

            # Save data
            self._save_orchestrator_data()

            if RICH_AVAILABLE:
                console.print(f"[green]✅ Workflow '{workflow.name}' completed successfully[/green]")

            logger.info(f"Workflow {workflow_id} completed successfully")

        except Exception as e:
            # Mark workflow as failed
            workflow = self.workflows[workflow_id]
            workflow.status = "failed"
            workflow.completed_at = datetime.now()
            workflow.results = {"error": str(e)}

            self._save_orchestrator_data()

            if RICH_AVAILABLE:
                console.print(f"[red]❌ Workflow '{workflow.name}' failed: {e}[/red]")

            logger.error(f"Workflow {workflow_id} failed: {e}")

    async def _execute_workflow_step(self, step_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step"""

        if step_type == "ai_team_task":
            # Assign task to AI team
            task_id = self.ai_team.assign_task(
                params.get("title", "Workflow Task"), 
                params.get("description", "Automated workflow task"), 
                params.get("agent", "data_analyst"), 
                params.get("priority", "medium")
            )
            self.ai_team.start_task(task_id)
            return {"task_id": task_id, "status": "assigned"}

        elif step_type == "market_analysis":
            # Run market analysis
            market_data = params.get("market_data", {
                "price_change_24h": 0.02, 
                "volume_change_24h": 0.1, 
                "volatility": 0.03
            })
            insight = self.ai_brain.analyze_market_situation(market_data)
            return {
                "title": insight.title, 
                "confidence": insight.confidence, 
                "description": insight.description
            }

        elif step_type == "risk_assessment":
            # Run risk assessment
            portfolio_data = params.get("portfolio_data", {
                "total_value": 100000, 
                "positions": [], 
                "cash_ratio": 0.1
            })
            insight = self.ai_brain.assess_portfolio_risk(portfolio_data)
            return {
                "title": insight.title, 
                "confidence": insight.confidence, 
                "description": insight.description
            }

        elif step_type == "ai_chat":
            # Chat with AI assistant
            message = params.get("message", "System status check")
            response = self.ai_brain.chat_with_assistant(message)
            return {"message": message, "response": response}

        elif step_type == "wait":
            # Wait for specified time
            wait_time = params.get("seconds", 5)
            await asyncio.sleep(wait_time)
            return {"waited_seconds": wait_time}

        else:
            return {"error": f"Unknown step type: {step_type}"}

    def create_daily_automation_workflow(self) -> str:
        """Create daily automation workflow"""
        steps = [
            {
                "name": "Daily Market Analysis", 
                "type": "market_analysis", 
                "params": {
                    "market_data": {
                        "price_change_24h": 0.015, 
                        "volume_change_24h": 0.08, 
                        "volatility": 0.025
                    }
                }
            }, 
            {
                "name": "Portfolio Risk Check", 
                "type": "risk_assessment", 
                "params": {
                    "portfolio_data": {
                        "total_value": 125000, 
                        "positions": [
                            {"symbol": "GOLD", "size": 25000}, 
                            {"symbol": "SILVER", "size": 20000}
                        ], 
                        "cash_ratio": 0.15
                    }
                }
            }, 
            {
                "name": "AI Team Daily Tasks", 
                "type": "ai_team_task", 
                "params": {
                    "title": "Daily System Health Check", 
                    "description": "Automated daily health check and monitoring", 
                    "agent": "sysadmin", 
                    "priority": "medium"
                }
            }, 
            {
                "name": "System Status Query", 
                "type": "ai_chat", 
                "params": {
                    "message": "Provide a comprehensive system status update"
                }
            }
        ]

        return self.create_workflow(
            "Daily Automation", 
            "Automated daily tasks for system monitoring and analysis", 
            steps
        )

    def create_trading_decision_workflow(self, symbol: str, action: str) -> str:
        """Create trading decision workflow"""
        steps = [
            {
                "name": "Market Analysis", 
                "type": "market_analysis", 
                "params": {
                    "market_data": {
                        "symbol": symbol, 
                        "price_change_24h": 0.02, 
                        "volume_change_24h": 0.12, 
                        "volatility": 0.035
                    }
                }
            }, 
            {
                "name": "Risk Assessment", 
                "type": "risk_assessment", 
                "params": {
                    "portfolio_data": {
                        "total_value": 125000, 
                        "action": action, 
                        "symbol": symbol
                    }
                }
            }, 
            {
                "name": "Strategy Analysis", 
                "type": "ai_team_task", 
                "params": {
                    "title": f"Analyze {action} strategy for {symbol}", 
                    "description": f"Evaluate {action} decision for {symbol} based on current market conditions", 
                    "agent": "strategy_ai", 
                    "priority": "high"
                }
            }, 
            {
                "name": "Decision Consultation", 
                "type": "ai_chat", 
                "params": {
                    "message": f"Should I {action} {symbol} based on current analysis?"
                }
            }
        ]

        return self.create_workflow(
            f"Trading Decision: {action} {symbol}", 
            f"Comprehensive analysis for {action} decision on {symbol}", 
            steps
        )

    def start_monitoring(self):
        """Start AI system monitoring"""
        if not self.is_running:
            self.is_running = True
            self.monitoring_thread = threading.Thread(target = self._monitoring_loop, daemon = True)
            self.monitoring_thread.start()
            logger.info("🔍 AI monitoring started")

    def stop_monitoring(self):
        """Stop AI system monitoring"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout = 5)
        logger.info("🛑 AI monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Monitor AI team status
                if self.ai_team:
                    team_status = self.ai_team.get_team_status()
                    self._record_metric("ai_team_active_tasks", team_status.get("task_summary", {}).get("in_progress", 0))

                # Monitor workflows
                active_workflows = len([w for w in self.workflows.values() if w.status == "running"])
                self._record_metric("active_workflows", active_workflows)

                # Monitor system health
                self._record_metric("orchestrator_uptime", 1)

                # Sleep for monitoring interval
                time.sleep(60)  # Monitor every minute

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(60)

    def _record_metric(self, metric_name: str, metric_value: float):
        """Record system metric"""
        try:
            db_path = self.database_dir / "ai_orchestrator.db"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO system_metrics (metric_name, metric_value) VALUES (?, ?)", 
                    (metric_name, metric_value)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to record metric: {e}")

    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        overview = {
            "ai_orchestrator": {
                "status": "running" if self.is_running else "stopped", 
                "total_workflows": len(self.workflows), 
                "active_workflows": len([w for w in self.workflows.values() if w.status == "running"]), 
                "completed_workflows": len([w for w in self.workflows.values() if w.status == "completed"])
            }, 
            "ai_team": {}, 
            "ai_brain": {}, 
            "recent_activity": []
        }

        # Get AI team status
        if self.ai_team:
            try:
                team_status = self.ai_team.get_team_status()
                overview["ai_team"] = team_status
            except Exception as e:
                overview["ai_team"] = {"error": str(e)}

        # Get AI brain status
        if self.ai_brain:
            try:
                recent_insights = self.ai_brain.get_recent_insights(limit = 3)
                overview["ai_brain"] = {
                    "recent_insights_count": len(recent_insights), 
                    "insights": [
                        {
                            "title": insight.title, 
                            "category": insight.category, 
                            "confidence": insight.confidence
                        } for insight in recent_insights
                    ]
                }
            except Exception as e:
                overview["ai_brain"] = {"error": str(e)}

        # Get recent workflows
        recent_workflows = sorted(self.workflows.values(), key = lambda w: w.created_at, reverse = True)[:5]
        overview["recent_activity"] = [
            {
                "type": "workflow", 
                "name": w.name, 
                "status": w.status, 
                "created_at": w.created_at.isoformat()
            } for w in recent_workflows
        ]

        return overview

    def show_control_center(self):
        """Show AI control center dashboard"""
        if RICH_AVAILABLE:
            overview = self.get_system_overview()

            console.print(Panel.fit(
                "[bold blue]🎯 NICEGOLD AI Control Center[/bold blue]\n"
                "[yellow]Unified AI Management Dashboard[/yellow]", 
                title = "AI Orchestrator"
            ))

            # Create layout with multiple panels
            layout = Layout()
            layout.split_column(
                Layout(name = "top", size = 10), 
                Layout(name = "middle", size = 15), 
                Layout(name = "bottom")
            )

            layout["top"].split_row(
                Layout(name = "orchestrator"), 
                Layout(name = "team"), 
                Layout(name = "brain")
            )

            # Orchestrator status
            orch_status = overview["ai_orchestrator"]
            orchestrator_table = Table(show_header = False, box = None)
            orchestrator_table.add_column("Metric", style = "cyan")
            orchestrator_table.add_column("Value", style = "green")
            orchestrator_table.add_row("Status", "🟢 Running" if orch_status["status"] == "running" else "🔴 Stopped")
            orchestrator_table.add_row("Workflows", str(orch_status["total_workflows"]))
            orchestrator_table.add_row("Active", str(orch_status["active_workflows"]))
            orchestrator_table.add_row("Completed", str(orch_status["completed_workflows"]))

            layout["orchestrator"] = Panel(
                orchestrator_table, 
                title = "🎯 Orchestrator", 
                border_style = "blue"
            )

            # AI Team status
            team_status = overview["ai_team"]
            team_table = Table(show_header = False, box = None)
            team_table.add_column("Metric", style = "cyan")
            team_table.add_column("Value", style = "green")

            if "error" not in team_status:
                team_table.add_row("Team Size", str(team_status.get("team_size", 0)))
                team_table.add_row("Total Tasks", str(team_status.get("total_tasks", 0)))
                task_summary = team_status.get("task_summary", {})
                team_table.add_row("Active", str(task_summary.get("in_progress", 0)))
                team_table.add_row("Completed", str(task_summary.get("completed", 0)))
            else:
                team_table.add_row("Status", "❌ Error")

            layout["team"] = Panel(
                team_table, 
                title = "👥 AI Team", 
                border_style = "green"
            )

            # AI Brain status
            brain_status = overview["ai_brain"]
            brain_table = Table(show_header = False, box = None)
            brain_table.add_column("Metric", style = "cyan")
            brain_table.add_column("Value", style = "green")

            if "error" not in brain_status:
                brain_table.add_row("Insights", str(brain_status.get("recent_insights_count", 0)))
                insights = brain_status.get("insights", [])
                for insight in insights[:3]:
                    brain_table.add_row("", f"• {insight['title'][:20]}...")
            else:
                brain_table.add_row("Status", "❌ Error")

            layout["brain"] = Panel(
                brain_table, 
                title = "🧠 AI Brain", 
                border_style = "magenta"
            )

            # Recent workflows
            workflows_table = Table(show_header = True, header_style = "bold blue")
            workflows_table.add_column("Workflow", style = "cyan")
            workflows_table.add_column("Status")
            workflows_table.add_column("Created")

            for activity in overview["recent_activity"]:
                if activity["type"] == "workflow":
                    status_icon = {
                        "pending": "⏳", 
                        "running": "🔄", 
                        "completed": "✅", 
                        "failed": "❌"
                    }.get(activity["status"], "❓")

                    workflows_table.add_row(
                        activity["name"][:30] + "..." if len(activity["name"]) > 30 else activity["name"], 
                        f"{status_icon} {activity['status']}", 
                        datetime.fromisoformat(activity["created_at"]).strftime("%H:%M")
                    )

            layout["middle"] = Panel(
                workflows_table, 
                title = "📋 Recent Workflows", 
                border_style = "yellow"
            )

            # Quick actions
            actions_text = """
🚀 Quick Actions:
  1. Run Daily Automation
  2. Create Trading Decision Workflow
  3. Ask AI Assistant
  4. Monitor AI Team
  5. Generate Market Analysis
  6. Assess Portfolio Risk

💡 AI Status: All systems operational
🔍 Monitoring: Active
⚡ Ready for commands!
            """

            layout["bottom"] = Panel(
                actions_text, 
                title = "🎮 Control Panel", 
                border_style = "green"
            )

            console.print(layout)

        else:
            print("🎯 NICEGOLD AI Control Center")
            print(" = " * 50)

            overview = self.get_system_overview()

            print(f"Orchestrator Status: {overview['ai_orchestrator']['status']}")
            print(f"Total Workflows: {overview['ai_orchestrator']['total_workflows']}")
            print(f"Active Workflows: {overview['ai_orchestrator']['active_workflows']}")

            if "error" not in overview["ai_team"]:
                print(f"AI Team Size: {overview['ai_team'].get('team_size', 0)}")
                print(f"Active Tasks: {overview['ai_team'].get('task_summary', {}).get('in_progress', 0)}")

            print("\nAll AI systems operational and ready for commands!")

    def interactive_mode(self):
        """Interactive command mode"""
        if RICH_AVAILABLE:
            console.print(Panel.fit(
                "[bold green]🎮 AI Orchestrator Interactive Mode[/bold green]\n"
                "Control your entire AI team from here!\n"
                "Type 'help' for available commands or 'exit' to quit.", 
                title = "Interactive Control"
            ))
        else:
            print("🎮 AI Orchestrator Interactive Mode")
            print("Type 'help' for commands or 'exit' to quit.")

        while True:
            try:
                if RICH_AVAILABLE:
                    command = Prompt.ask("\n[cyan]AI Control[/cyan]").strip().lower()
                else:
                    command = input("\nAI Control> ").strip().lower()

                if command in ['exit', 'quit', 'bye']:
                    if RICH_AVAILABLE:
                        console.print("[yellow]👋 Goodbye! Your AI team will continue working.[/yellow]")
                    else:
                        print("👋 Goodbye! Your AI team will continue working.")
                    break

                elif command == 'help':
                    self._show_help()

                elif command == 'status' or command == 'dashboard':
                    self.show_control_center()

                elif command == 'daily':
                    workflow_id = self.create_daily_automation_workflow()
                    self.execute_workflow(workflow_id)

                elif command.startswith('chat '):
                    message = command[5:]
                    if self.ai_brain:
                        response = self.ai_brain.chat_with_assistant(message)
                        if RICH_AVAILABLE:
                            console.print(f"[blue]🤖 AI:[/blue] {response}")
                        else:
                            print(f"🤖 AI: {response}")

                elif command == 'analyze':
                    if self.ai_brain:
                        market_data = {
                            "price_change_24h": 0.025, 
                            "volume_change_24h": 0.15, 
                            "volatility": 0.032
                        }
                        insight = self.ai_brain.analyze_market_situation(market_data)
                        if RICH_AVAILABLE:
                            console.print(Panel(
                                f"**{insight.title}**\n\n{insight.description}", 
                                title = "Market Analysis", 
                                border_style = "blue"
                            ))
                        else:
                            print(f"📊 {insight.title}: {insight.description}")

                elif command == 'risk':
                    if self.ai_brain:
                        portfolio_data = {
                            "total_value": 125430, 
                            "positions": [{"symbol": "GOLD", "size": 25000}], 
                            "cash_ratio": 0.152
                        }
                        insight = self.ai_brain.assess_portfolio_risk(portfolio_data)
                        if RICH_AVAILABLE:
                            console.print(Panel(
                                f"**{insight.title}**\n\n{insight.description}", 
                                title = "Risk Assessment", 
                                border_style = "red"
                            ))
                        else:
                            print(f"🛡️ {insight.title}: {insight.description}")

                elif command == 'team':
                    if self.ai_team:
                        if hasattr(self.ai_team, 'show_team_dashboard'):
                            self.ai_team.show_team_dashboard()
                        else:
                            print("AI team dashboard not available")

                elif command == 'workflows':
                    self._show_workflows()

                else:
                    if RICH_AVAILABLE:
                        console.print(f"[red]❓ Unknown command: {command}[/red]")
                        console.print("[yellow]Type 'help' for available commands[/yellow]")
                    else:
                        print(f"❓ Unknown command: {command}")
                        print("Type 'help' for available commands")

            except KeyboardInterrupt:
                if RICH_AVAILABLE:
                    console.print("\n[yellow]⚠️ Use 'exit' to quit properly[/yellow]")
                else:
                    print("\n⚠️ Use 'exit' to quit properly")
            except Exception as e:
                if RICH_AVAILABLE:
                    console.print(f"[red]❌ Error: {e}[/red]")
                else:
                    print(f"❌ Error: {e}")

    def _show_help(self):
        """Show help information"""
        if RICH_AVAILABLE:
            help_text = """
🎮 **AI Orchestrator Commands**

**System Control:**
• `status` / `dashboard` - Show AI control center
• `team` - Show AI team dashboard
• `workflows` - Show workflow history

**AI Interaction:**
• `chat <message>` - Chat with AI assistant
• `analyze` - Run market analysis
• `risk` - Assess portfolio risk

**Automation:**
• `daily` - Run daily automation workflow

**General:**
• `help` - Show this help
• `exit` - Quit interactive mode

**Example Usage:**
• `chat What's my portfolio status?`
• `analyze`
• `daily`
            """
            console.print(Panel(help_text, title = "Help", border_style = "cyan"))
        else:
            print("""
🎮 AI Orchestrator Commands:
- status/dashboard: Show control center
- team: Show AI team dashboard
- chat <message>: Chat with AI
- analyze: Market analysis
- risk: Risk assessment
- daily: Run daily automation
- workflows: Show workflow history
- help: Show this help
- exit: Quit
            """)

    def _show_workflows(self):
        """Show workflow history"""
        recent_workflows = sorted(self.workflows.values(), key = lambda w: w.created_at, reverse = True)[:10]

        if RICH_AVAILABLE:
            workflows_table = Table(show_header = True, header_style = "bold blue")
            workflows_table.add_column("ID", style = "cyan")
            workflows_table.add_column("Name")
            workflows_table.add_column("Status")
            workflows_table.add_column("Created")
            workflows_table.add_column("Duration")

            for workflow in recent_workflows:
                duration = "N/A"
                if workflow.started_at and workflow.completed_at:
                    duration = str(workflow.completed_at - workflow.started_at)
                elif workflow.started_at:
                    duration = str(datetime.now() - workflow.started_at)

                status_icon = {
                    "pending": "⏳", 
                    "running": "🔄", 
                    "completed": "✅", 
                    "failed": "❌"
                }.get(workflow.status, "❓")

                workflows_table.add_row(
                    workflow.id[ - 4:], 
                    workflow.name[:25] + "..." if len(workflow.name) > 25 else workflow.name, 
                    f"{status_icon} {workflow.status}", 
                    workflow.created_at.strftime("%m - %d %H:%M"), 
                    duration
                )

            console.print(Panel(workflows_table, title = "📋 Workflow History", border_style = "yellow"))
        else:
            print("📋 Recent Workflows:")
            for workflow in recent_workflows:
                print(f"- {workflow.name} ({workflow.status}) - {workflow.created_at.strftime('%H:%M')}")

# Global AI orchestrator instance
ai_orchestrator = AIOrchestrator()

def main():
    """Main AI orchestrator function"""
    if len(sys.argv) < 2:
        print("NICEGOLD AI Orchestrator")
        print("Usage:")
        print("  python ai_orchestrator.py control      - Interactive control mode")
        print("  python ai_orchestrator.py dashboard    - Show control center")
        print("  python ai_orchestrator.py daily        - Run daily automation")
        print("  python ai_orchestrator.py status       - Show system status")
        print("  python ai_orchestrator.py workflow     - Create custom workflow")
        return

    command = sys.argv[1].lower()

    try:
        if command == "control":
            ai_orchestrator.interactive_mode()

        elif command == "dashboard":
            ai_orchestrator.show_control_center()

        elif command == "daily":
            workflow_id = ai_orchestrator.create_daily_automation_workflow()
            ai_orchestrator.execute_workflow(workflow_id)
            if RICH_AVAILABLE:
                console.print(f"[green]✅ Daily automation workflow started: {workflow_id}[/green]")
            else:
                print(f"✅ Daily automation started: {workflow_id}")

        elif command == "status":
            overview = ai_orchestrator.get_system_overview()
            print(json.dumps(overview, indent = 2, default = str))

        elif command == "workflow":
            # Interactive workflow creation
            if RICH_AVAILABLE:
                name = Prompt.ask("Workflow name")
                description = Prompt.ask("Workflow description")
                # For simplicity, create a basic workflow
                steps = [
                    {
                        "name": "Market Analysis", 
                        "type": "market_analysis", 
                        "params": {}
                    }, 
                    {
                        "name": "AI Team Task", 
                        "type": "ai_team_task", 
                        "params": {
                            "title": "Custom Task", 
                            "description": description, 
                            "agent": "data_analyst"
                        }
                    }
                ]
                workflow_id = ai_orchestrator.create_workflow(name, description, steps)

                if Confirm.ask("Execute workflow now?"):
                    ai_orchestrator.execute_workflow(workflow_id)
            else:
                print("Interactive workflow creation requires rich library")

        else:
            print(f"Unknown command: {command}")

    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        ai_orchestrator.stop_monitoring()

if __name__ == "__main__":
    main()