#!/usr/bin/env python3
"""
🤖 NICEGOLD AI Team Manager 🤖
=============================

ระบบจัดการทีม AI สำหรับช่วยเหลือผู้ใช้คนเดียวในการบริหารโปรเจค NICEGOLD
AI จะทำหน้าที่เป็นทีมงานเสมือนจริงในด้านต่างๆ เช่น:
- Data Analyst AI
- Trading Strategy AI  
- Risk Management AI
- System Administrator AI
- Quality Assurance AI
- DevOps AI
"""

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None

@dataclass
class AIAgent:
    """AI Agent data structure"""
    name: str
    role: str
    description: str
    capabilities: List[str]
    status: str = "idle"
    current_task: Optional[str] = None
    last_activity: Optional[datetime] = None
    performance_score: float = 0.0
    tasks_completed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            **asdict(self),
            "last_activity": self.last_activity.isoformat() if self.last_activity else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AIAgent":
        """Create from dictionary"""
        if data.get("last_activity"):
            data["last_activity"] = datetime.fromisoformat(data["last_activity"])
        return cls(**data)

@dataclass
class AITask:
    """AI Task data structure"""
    id: str
    title: str
    description: str
    assigned_agent: str
    priority: str  # "high", "medium", "low"
    status: str  # "pending", "in_progress", "completed", "failed"
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            **asdict(self),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AITask":
        """Create from dictionary"""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("started_at"):
            data["started_at"] = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            data["completed_at"] = datetime.fromisoformat(data["completed_at"])
        return cls(**data)

class AITeamManager:
    """
    AI Team Manager for single-user project management
    """
    
    def __init__(self):
        self.project_root = Path(".")
        self.config_dir = self.project_root / "config" / "ai_team"
        self.logs_dir = self.project_root / "logs" / "ai_team"
        self.database_dir = self.project_root / "database"
        
        # Create directories
        for directory in [self.config_dir, self.logs_dir, self.database_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize AI team
        self.agents = self._initialize_ai_team()
        self.tasks = {}
        self.task_counter = 0
        
        # Load data
        self._load_team_data()
        
        logger.info("🤖 AI Team Manager initialized")
    
    def _setup_logging(self):
        """Setup logging for AI team"""
        log_file = self.logs_dir / f"ai_team_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        global logger
        logger = logging.getLogger(__name__)
    
    def _initialize_ai_team(self) -> Dict[str, AIAgent]:
        """Initialize AI team members"""
        agents = {}
        
        # Data Analyst AI
        agents["data_analyst"] = AIAgent(
            name="DataBot Alpha",
            role="Data Analyst",
            description="ผู้เชี่ยวชาญด้านการวิเคราะห์ข้อมูลการเทรด",
            capabilities=[
                "analyze_market_data",
                "generate_reports",
                "data_quality_check",
                "statistical_analysis",
                "trend_detection",
                "data_visualization"
            ]
        )
        
        # Trading Strategy AI
        agents["strategy_ai"] = AIAgent(
            name="StrategyBot Pro",
            role="Trading Strategist",
            description="ผู้เชี่ยวชาญด้านกลยุทธ์การเทรดและการวิเคราะห์ตลาด",
            capabilities=[
                "strategy_development",
                "backtesting",
                "signal_generation",
                "market_analysis",
                "optimization",
                "risk_assessment"
            ]
        )
        
        # Risk Management AI
        agents["risk_manager"] = AIAgent(
            name="RiskBot Guardian",
            role="Risk Manager",
            description="ผู้เชี่ยวชาญด้านการจัดการความเสี่ยง",
            capabilities=[
                "risk_analysis",
                "portfolio_monitoring",
                "position_sizing",
                "stop_loss_management",
                "drawdown_control",
                "compliance_check"
            ]
        )
        
        # System Administrator AI
        agents["sysadmin"] = AIAgent(
            name="SysBot Admin",
            role="System Administrator",
            description="ผู้เชี่ยวชาญด้านการจัดการระบบและโครงสร้างพื้นฐาน",
            capabilities=[
                "system_monitoring",
                "performance_optimization",
                "backup_management",
                "security_audit",
                "update_management",
                "troubleshooting"
            ]
        )
        
        # Quality Assurance AI
        agents["qa_specialist"] = AIAgent(
            name="QABot Tester",
            role="Quality Assurance",
            description="ผู้เชี่ยวชาญด้านการทดสอบและประกันคุณภาพ",
            capabilities=[
                "code_testing",
                "bug_detection",
                "performance_testing",
                "security_testing",
                "regression_testing",
                "quality_reporting"
            ]
        )
        
        # DevOps AI
        agents["devops"] = AIAgent(
            name="DevBot Ops",
            role="DevOps Engineer",
            description="ผู้เชี่ยวชาญด้าน DevOps และการปรับใช้ระบบ",
            capabilities=[
                "deployment_automation",
                "ci_cd_management",
                "infrastructure_monitoring",
                "scaling_management",
                "log_analysis",
                "incident_response"
            ]
        )
        
        # Research AI
        agents["researcher"] = AIAgent(
            name="ResearchBot Scholar",
            role="Research Analyst",
            description="ผู้เชี่ยวชาญด้านการวิจัยและพัฒนาเทคนิคใหม่",
            capabilities=[
                "market_research",
                "technology_evaluation",
                "algorithm_research",
                "competitive_analysis",
                "innovation_tracking",
                "documentation"
            ]
        )
        
        return agents
    
    def _load_team_data(self):
        """Load team data from database"""
        db_path = self.database_dir / "ai_team.db"
        
        try:
            with sqlite3.connect(db_path) as conn:
                # Create tables if not exist
                self._create_tables(conn)
                
                # Load agents
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM agents")
                agent_rows = cursor.fetchall()
                
                for row in agent_rows:
                    agent_data = json.loads(row[1])
                    agent = AIAgent.from_dict(agent_data)
                    self.agents[agent.name.lower().replace(" ", "_")] = agent
                
                # Load tasks
                cursor.execute("SELECT * FROM tasks")
                task_rows = cursor.fetchall()
                
                for row in task_rows:
                    task_data = json.loads(row[1])
                    task = AITask.from_dict(task_data)
                    self.tasks[task.id] = task
                
                # Get task counter
                cursor.execute("SELECT MAX(CAST(SUBSTR(id, 6) AS INTEGER)) FROM tasks WHERE id LIKE 'task_%'")
                result = cursor.fetchone()
                self.task_counter = result[0] if result[0] else 0
                
        except Exception as e:
            logger.warning(f"Could not load team data: {e}")
            self._save_team_data()  # Create initial data
    
    def _create_tables(self, conn):
        """Create database tables"""
        cursor = conn.cursor()
        
        # Agents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tasks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Task history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                action TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                details TEXT
            )
        """)
        
        conn.commit()
    
    def _save_team_data(self):
        """Save team data to database"""
        db_path = self.database_dir / "ai_team.db"
        
        try:
            with sqlite3.connect(db_path) as conn:
                self._create_tables(conn)
                cursor = conn.cursor()
                
                # Save agents
                for agent_id, agent in self.agents.items():
                    cursor.execute(
                        "INSERT OR REPLACE INTO agents (id, data) VALUES (?, ?)",
                        (agent_id, json.dumps(agent.to_dict()))
                    )
                
                # Save tasks
                for task_id, task in self.tasks.items():
                    cursor.execute(
                        "INSERT OR REPLACE INTO tasks (id, data) VALUES (?, ?)",
                        (task_id, json.dumps(task.to_dict()))
                    )
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to save team data: {e}")
    
    def assign_task(self, title: str, description: str, agent_role: str, 
                   priority: str = "medium") -> str:
        """
        Assign task to AI agent
        
        Args:
            title: Task title
            description: Task description
            agent_role: Role of agent to assign (data_analyst, strategy_ai, etc.)
            priority: Task priority (high, medium, low)
            
        Returns:
            str: Task ID
        """
        try:
            # Generate task ID
            self.task_counter += 1
            task_id = f"task_{self.task_counter:04d}"
            
            # Find agent
            if agent_role not in self.agents:
                raise ValueError(f"Unknown agent role: {agent_role}")
            
            agent = self.agents[agent_role]
            
            # Create task
            task = AITask(
                id=task_id,
                title=title,
                description=description,
                assigned_agent=agent_role,
                priority=priority,
                status="pending",
                created_at=datetime.now()
            )
            
            self.tasks[task_id] = task
            
            # Update agent status
            agent.status = "assigned"
            agent.current_task = task_id
            
            # Save data
            self._save_team_data()
            
            # Log assignment
            self._log_task_history(task_id, agent_role, "assigned", 
                                 {"priority": priority, "title": title})
            
            if RICH_AVAILABLE:
                console.print(f"[green]✅ Task '{title}' assigned to {agent.name}[/green]")
            
            logger.info(f"Task {task_id} assigned to {agent.name}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to assign task: {e}")
            raise
    
    def start_task(self, task_id: str) -> bool:
        """
        Start executing a task
        
        Args:
            task_id: Task ID to start
            
        Returns:
            bool: True if task started successfully
        """
        try:
            if task_id not in self.tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self.tasks[task_id]
            agent = self.agents[task.assigned_agent]
            
            # Update task status
            task.status = "in_progress"
            task.started_at = datetime.now()
            
            # Update agent status
            agent.status = "working"
            agent.last_activity = datetime.now()
            
            # Save data
            self._save_team_data()
            
            # Log start
            self._log_task_history(task_id, task.assigned_agent, "started")
            
            if RICH_AVAILABLE:
                console.print(f"[blue]🚀 {agent.name} started working on '{task.title}'[/blue]")
            
            logger.info(f"Task {task_id} started by {agent.name}")
            
            # Simulate AI work (in real implementation, this would be actual AI processing)
            asyncio.create_task(self._simulate_ai_work(task_id))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start task: {e}")
            return False
    
    async def _simulate_ai_work(self, task_id: str):
        """
        Simulate AI agent working on task
        In production, this would be replaced with actual AI processing
        """
        try:
            task = self.tasks[task_id]
            agent = self.agents[task.assigned_agent]
            
            # Simulate work time based on task complexity
            work_time = {
                "high": 30,    # 30 seconds for demo
                "medium": 20,  # 20 seconds
                "low": 10      # 10 seconds
            }.get(task.priority, 20)
            
            await asyncio.sleep(work_time)
            
            # Generate simulated result based on agent capabilities
            result = self._generate_task_result(task, agent)
            
            # Complete task
            self.complete_task(task_id, result)
            
        except Exception as e:
            logger.error(f"AI work simulation failed for task {task_id}: {e}")
            self.fail_task(task_id, str(e))
    
    def _generate_task_result(self, task: AITask, agent: AIAgent) -> Dict[str, Any]:
        """Generate simulated task result based on agent capabilities"""
        
        # Base result structure
        result = {
            "status": "completed",
            "agent": agent.name,
            "completed_at": datetime.now().isoformat(),
            "execution_time": "N/A",
            "confidence": 0.85,
            "recommendations": []
        }
        
        # Generate specific results based on agent role
        if agent.role == "Data Analyst":
            result.update({
                "analysis_type": "market_data_analysis",
                "data_points_analyzed": 1000,
                "trends_identified": ["upward_trend", "high_volatility"],
                "quality_score": 0.92,
                "recommendations": [
                    "Consider increasing position size during upward trend",
                    "Monitor volatility levels for risk management"
                ]
            })
            
        elif agent.role == "Trading Strategist":
            result.update({
                "strategy_evaluated": "momentum_strategy",
                "backtest_results": {
                    "total_return": 15.2,
                    "sharpe_ratio": 1.45,
                    "max_drawdown": -8.5,
                    "win_rate": 0.62
                },
                "recommendations": [
                    "Strategy shows good performance",
                    "Consider implementing with 2% position sizing"
                ]
            })
            
        elif agent.role == "Risk Manager":
            result.update({
                "risk_assessment": "moderate",
                "portfolio_risk": 0.12,
                "var_95": -2.5,
                "stress_test_results": "passed",
                "recommendations": [
                    "Current risk levels are acceptable",
                    "Implement stop-loss at 5% below entry"
                ]
            })
            
        elif agent.role == "System Administrator":
            result.update({
                "system_health": "excellent",
                "cpu_usage": 25.6,
                "memory_usage": 42.1,
                "disk_space": 78.3,
                "recommendations": [
                    "System performance is optimal",
                    "Schedule maintenance for next week"
                ]
            })
            
        elif agent.role == "Quality Assurance":
            result.update({
                "tests_run": 156,
                "tests_passed": 152,
                "tests_failed": 4,
                "coverage": 94.2,
                "critical_issues": 0,
                "recommendations": [
                    "Fix 4 minor test failures",
                    "Increase test coverage to 95%"
                ]
            })
            
        elif agent.role == "DevOps Engineer":
            result.update({
                "deployment_status": "ready",
                "infrastructure_health": "green",
                "performance_metrics": {
                    "response_time": "150ms",
                    "throughput": "1000 req/sec",
                    "error_rate": "0.01%"
                },
                "recommendations": [
                    "Deploy to production environment",
                    "Monitor performance for first 24 hours"
                ]
            })
            
        elif agent.role == "Research Analyst":
            result.update({
                "research_topic": "new_ml_algorithms",
                "papers_reviewed": 25,
                "promising_techniques": ["Transformer models", "Reinforcement Learning"],
                "implementation_feasibility": "high",
                "recommendations": [
                    "Consider implementing transformer-based models",
                    "Start with proof-of-concept development"
                ]
            })
        
        return result
    
    def complete_task(self, task_id: str, result: Dict[str, Any]) -> bool:
        """
        Complete a task with results
        
        Args:
            task_id: Task ID to complete
            result: Task result data
            
        Returns:
            bool: True if task completed successfully
        """
        try:
            if task_id not in self.tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self.tasks[task_id]
            agent = self.agents[task.assigned_agent]
            
            # Update task
            task.status = "completed"
            task.completed_at = datetime.now()
            task.result = result
            
            # Update agent
            agent.status = "idle"
            agent.current_task = None
            agent.last_activity = datetime.now()
            agent.tasks_completed += 1
            agent.performance_score = min(1.0, agent.performance_score + 0.1)
            
            # Save data
            self._save_team_data()
            
            # Log completion
            self._log_task_history(task_id, task.assigned_agent, "completed", result)
            
            if RICH_AVAILABLE:
                console.print(f"[green]✅ {agent.name} completed '{task.title}'[/green]")
            
            logger.info(f"Task {task_id} completed by {agent.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete task: {e}")
            return False
    
    def fail_task(self, task_id: str, error_message: str) -> bool:
        """
        Mark task as failed
        
        Args:
            task_id: Task ID that failed
            error_message: Error message
            
        Returns:
            bool: True if task marked as failed
        """
        try:
            if task_id not in self.tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self.tasks[task_id]
            agent = self.agents[task.assigned_agent]
            
            # Update task
            task.status = "failed"
            task.completed_at = datetime.now()
            task.error_message = error_message
            
            # Update agent
            agent.status = "idle"
            agent.current_task = None
            agent.last_activity = datetime.now()
            agent.performance_score = max(0.0, agent.performance_score - 0.1)
            
            # Save data
            self._save_team_data()
            
            # Log failure
            self._log_task_history(task_id, task.assigned_agent, "failed", 
                                 {"error": error_message})
            
            if RICH_AVAILABLE:
                console.print(f"[red]❌ {agent.name} failed on '{task.title}': {error_message}[/red]")
            
            logger.error(f"Task {task_id} failed: {error_message}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark task as failed: {e}")
            return False
    
    def _log_task_history(self, task_id: str, agent_id: str, action: str, 
                         details: Optional[Dict[str, Any]] = None):
        """Log task history for auditing"""
        try:
            db_path = self.database_dir / "ai_team.db"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO task_history (task_id, agent_id, action, details) VALUES (?, ?, ?, ?)",
                    (task_id, agent_id, action, json.dumps(details) if details else None)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log task history: {e}")
    
    def get_team_status(self) -> Dict[str, Any]:
        """Get current team status"""
        status = {
            "team_size": len(self.agents),
            "total_tasks": len(self.tasks),
            "agents": {},
            "task_summary": {
                "pending": 0,
                "in_progress": 0,
                "completed": 0,
                "failed": 0
            }
        }
        
        # Agent status
        for agent_id, agent in self.agents.items():
            status["agents"][agent_id] = {
                "name": agent.name,
                "role": agent.role,
                "status": agent.status,
                "current_task": agent.current_task,
                "tasks_completed": agent.tasks_completed,
                "performance_score": agent.performance_score,
                "last_activity": agent.last_activity.isoformat() if agent.last_activity else None
            }
        
        # Task summary
        for task in self.tasks.values():
            status["task_summary"][task.status] += 1
        
        return status
    
    def show_team_dashboard(self):
        """Show team dashboard"""
        if RICH_AVAILABLE:
            console.print(Panel.fit(
                "[bold blue]🤖 NICEGOLD AI Team Dashboard[/bold blue]",
                title="AI Team Manager"
            ))
            
            status = self.get_team_status()
            
            # Team overview
            overview_table = Table(show_header=True, header_style="bold blue")
            overview_table.add_column("Metric", style="cyan")
            overview_table.add_column("Value", style="green")
            
            overview_table.add_row("Team Size", str(status["team_size"]))
            overview_table.add_row("Total Tasks", str(status["total_tasks"]))
            overview_table.add_row("Pending Tasks", str(status["task_summary"]["pending"]))
            overview_table.add_row("Active Tasks", str(status["task_summary"]["in_progress"]))
            overview_table.add_row("Completed Tasks", str(status["task_summary"]["completed"]))
            overview_table.add_row("Failed Tasks", str(status["task_summary"]["failed"]))
            
            console.print("\n[bold blue]📊 Team Overview[/bold blue]")
            console.print(overview_table)
            
            # Agent status
            agents_table = Table(show_header=True, header_style="bold blue")
            agents_table.add_column("Agent", style="cyan")
            agents_table.add_column("Role")
            agents_table.add_column("Status")
            agents_table.add_column("Current Task")
            agents_table.add_column("Completed")
            agents_table.add_column("Performance", style="green")
            
            for agent_id, agent_info in status["agents"].items():
                status_icon = {
                    "idle": "💤",
                    "assigned": "📋",
                    "working": "⚡",
                }.get(agent_info["status"], "❓")
                
                agents_table.add_row(
                    agent_info["name"],
                    agent_info["role"],
                    f"{status_icon} {agent_info['status']}",
                    agent_info["current_task"] or "-",
                    str(agent_info["tasks_completed"]),
                    f"{agent_info['performance_score']:.2f}"
                )
            
            console.print("\n[bold blue]👥 AI Team Members[/bold blue]")
            console.print(agents_table)
            
            # Recent tasks
            recent_tasks = sorted(
                self.tasks.values(), 
                key=lambda t: t.created_at, 
                reverse=True
            )[:5]
            
            if recent_tasks:
                tasks_table = Table(show_header=True, header_style="bold blue")
                tasks_table.add_column("Task ID", style="cyan")
                tasks_table.add_column("Title")
                tasks_table.add_column("Agent")
                tasks_table.add_column("Status")
                tasks_table.add_column("Priority")
                tasks_table.add_column("Created")
                
                for task in recent_tasks:
                    agent = self.agents[task.assigned_agent]
                    status_icon = {
                        "pending": "⏳",
                        "in_progress": "🔄",
                        "completed": "✅",
                        "failed": "❌"
                    }.get(task.status, "❓")
                    
                    priority_icon = {
                        "high": "🔴",
                        "medium": "🟡",
                        "low": "🟢"
                    }.get(task.priority, "⚪")
                    
                    tasks_table.add_row(
                        task.id,
                        task.title[:30] + "..." if len(task.title) > 30 else task.title,
                        agent.name,
                        f"{status_icon} {task.status}",
                        f"{priority_icon} {task.priority}",
                        task.created_at.strftime("%H:%M")
                    )
                
                console.print("\n[bold blue]📋 Recent Tasks[/bold blue]")
                console.print(tasks_table)
        else:
            print("🤖 NICEGOLD AI Team Dashboard")
            print("=" * 50)
            
            status = self.get_team_status()
            print(f"Team Size: {status['team_size']}")
            print(f"Total Tasks: {status['total_tasks']}")
            print(f"Active Tasks: {status['task_summary']['in_progress']}")
            print(f"Completed Tasks: {status['task_summary']['completed']}")
    
    def auto_assign_daily_tasks(self):
        """Automatically assign daily tasks to AI team"""
        daily_tasks = [
            {
                "title": "Daily Market Analysis",
                "description": "Analyze current market conditions and identify trends",
                "agent": "data_analyst",
                "priority": "high"
            },
            {
                "title": "Portfolio Risk Assessment",
                "description": "Evaluate current portfolio risk and suggest adjustments",
                "agent": "risk_manager",
                "priority": "high"
            },
            {
                "title": "System Health Check",
                "description": "Monitor system performance and identify issues",
                "agent": "sysadmin",
                "priority": "medium"
            },
            {
                "title": "Strategy Performance Review",
                "description": "Review trading strategy performance and optimization opportunities",
                "agent": "strategy_ai",
                "priority": "medium"
            },
            {
                "title": "Quality Assurance Scan",
                "description": "Run automated tests and check system quality",
                "agent": "qa_specialist",
                "priority": "low"
            }
        ]
        
        if RICH_AVAILABLE:
            console.print("[blue]🔄 Assigning daily tasks to AI team...[/blue]")
        
        assigned_count = 0
        for task_info in daily_tasks:
            try:
                task_id = self.assign_task(
                    task_info["title"],
                    task_info["description"],
                    task_info["agent"],
                    task_info["priority"]
                )
                
                # Auto-start high priority tasks
                if task_info["priority"] == "high":
                    self.start_task(task_id)
                
                assigned_count += 1
                
            except Exception as e:
                logger.error(f"Failed to assign daily task: {e}")
        
        if RICH_AVAILABLE:
            console.print(f"[green]✅ Assigned {assigned_count} daily tasks[/green]")
        
        logger.info(f"Auto-assigned {assigned_count} daily tasks")
        return assigned_count

# Global AI team manager instance
ai_team = AITeamManager()

def main():
    """Main AI team management function"""
    if len(sys.argv) < 2:
        print("NICEGOLD AI Team Manager")
        print("Usage:")
        print("  python ai_team_manager.py dashboard      - Show team dashboard")
        print("  python ai_team_manager.py assign         - Assign new task (interactive)")
        print("  python ai_team_manager.py auto_daily     - Auto-assign daily tasks")
        print("  python ai_team_manager.py status         - Show team status")
        print("  python ai_team_manager.py start <task_id> - Start specific task")
        return
    
    command = sys.argv[1].lower()
    
    try:
        if command == "dashboard":
            ai_team.show_team_dashboard()
            
        elif command == "assign":
            # Interactive task assignment
            if RICH_AVAILABLE:
                title = Prompt.ask("Task title")
                description = Prompt.ask("Task description")
                
                # Show available agents
                console.print("\n[bold blue]Available AI Agents:[/bold blue]")
                for agent_id, agent in ai_team.agents.items():
                    console.print(f"  {agent_id}: {agent.name} ({agent.role})")
                
                agent_role = Prompt.ask("Agent role", choices=list(ai_team.agents.keys()))
                priority = Prompt.ask("Priority", choices=["high", "medium", "low"], default="medium")
                
                task_id = ai_team.assign_task(title, description, agent_role, priority)
                
                if Confirm.ask("Start task immediately?"):
                    ai_team.start_task(task_id)
            else:
                print("Interactive mode requires rich library")
                
        elif command == "auto_daily":
            ai_team.auto_assign_daily_tasks()
            
        elif command == "status":
            status = ai_team.get_team_status()
            print(json.dumps(status, indent=2, default=str))
            
        elif command == "start" and len(sys.argv) > 2:
            task_id = sys.argv[2]
            ai_team.start_task(task_id)
            
        else:
            print(f"Unknown command: {command}")
            
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import sys
    main()
