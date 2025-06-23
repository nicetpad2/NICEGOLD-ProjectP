"""
Training Performance Monitor
===========================
Real-time monitoring and performance tracking for training pipelines
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.text import Text

from projectp.pro_log import pro_log

console = Console()

@dataclass
class TrainingMetrics:
    """Training metrics tracking"""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_duration: Optional[timedelta] = None
    current_iteration: int = 0
    total_iterations: int = 0
    current_auc: float = 0.0
    best_auc: float = 0.0
    target_auc: float = 70.0
    target_achieved: bool = False
    
    # Performance metrics
    cpu_usage: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    disk_io: List[Dict[str, float]] = field(default_factory=list)
    
    # Training progress
    iterations_history: List[Dict[str, Any]] = field(default_factory=list)
    model_performance: Dict[str, Any] = field(default_factory=dict)
    
    def update_iteration(self, iteration: int, auc: float, additional_metrics: Optional[Dict] = None):
        """Update iteration metrics"""
        self.current_iteration = iteration
        self.current_auc = auc
        if auc > self.best_auc:
            self.best_auc = auc
        
        if auc >= self.target_auc and not self.target_achieved:
            self.target_achieved = True
        
        iteration_data = {
            'iteration': iteration,
            'auc': auc,
            'timestamp': datetime.now(),
            'is_best': auc == self.best_auc
        }
        
        if additional_metrics:
            iteration_data.update(additional_metrics)
        
        self.iterations_history.append(iteration_data)
    
    def finish_training(self):
        """Mark training as finished"""
        self.end_time = datetime.now()
        self.total_duration = self.end_time - self.start_time
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_duration_seconds': self.total_duration.total_seconds() if self.total_duration else None,
            'total_iterations': len(self.iterations_history),
            'best_auc': self.best_auc,
            'target_auc': self.target_auc,
            'target_achieved': self.target_achieved,
            'avg_cpu_usage': sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
            'peak_memory_usage': max(self.memory_usage) if self.memory_usage else 0,
            'iterations_history': self.iterations_history
        }

class PerformanceMonitor:
    """Real-time performance monitoring for training"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics = TrainingMetrics()
        self.callbacks: List[Callable] = []
        
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add performance callback"""
        self.callbacks.append(callback)
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        pro_log.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.metrics.finish_training()
        pro_log.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                
                # Update metrics
                self.metrics.cpu_usage.append(cpu_percent)
                self.metrics.memory_usage.append(memory.percent)
                
                # Disk I/O (if available)
                try:
                    disk_io = psutil.disk_io_counters()
                    if disk_io:
                        self.metrics.disk_io.append({
                            'read_bytes': disk_io.read_bytes,
                            'write_bytes': disk_io.write_bytes,
                            'timestamp': datetime.now().isoformat()
                        })
                except:
                    pass
                
                # Call callbacks
                current_stats = {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'current_iteration': self.metrics.current_iteration,
                    'current_auc': self.metrics.current_auc,
                    'best_auc': self.metrics.best_auc
                }
                
                for callback in self.callbacks:
                    try:
                        callback(current_stats)
                    except Exception as e:
                        pro_log.warning(f"Performance callback failed: {e}")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                pro_log.error(f"Performance monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def update_training_progress(self, iteration: int, auc: float, **kwargs):
        """Update training progress"""
        self.metrics.update_iteration(iteration, auc, kwargs)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        if not self.metrics.cpu_usage:
            return {}
        
        return {
            'current_cpu': self.metrics.cpu_usage[-1] if self.metrics.cpu_usage else 0,
            'current_memory': self.metrics.memory_usage[-1] if self.metrics.memory_usage else 0,
            'avg_cpu': sum(self.metrics.cpu_usage) / len(self.metrics.cpu_usage),
            'avg_memory': sum(self.metrics.memory_usage) / len(self.metrics.memory_usage),
            'training_progress': {
                'current_iteration': self.metrics.current_iteration,
                'current_auc': self.metrics.current_auc,
                'best_auc': self.metrics.best_auc,
                'target_achieved': self.metrics.target_achieved
            }
        }

class TrainingDashboard:
    """Real-time training dashboard"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.layout = Layout()
        self.is_running = False
        
        # Setup dashboard layout
        self._setup_layout()
        
        # Add monitor callback
        self.monitor.add_callback(self._update_dashboard_data)
        
        # Dashboard data
        self.current_stats = {}
    
    def _setup_layout(self):
        """Setup dashboard layout"""
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        self.layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        self.layout["left"].split_column(
            Layout(name="progress", size=8),
            Layout(name="metrics")
        )
        
        self.layout["right"].split_column(
            Layout(name="performance", size=8),
            Layout(name="history")
        )
    
    def _update_dashboard_data(self, stats: Dict[str, Any]):
        """Update dashboard data from monitor callback"""
        self.current_stats = stats
    
    def _generate_header(self) -> Panel:
        """Generate dashboard header"""
        title = Text("🤖 ML Training Dashboard", style="bold blue")
        timestamp = Text(f"Last Update: {datetime.now().strftime('%H:%M:%S')}", style="dim")
        return Panel(title + " " + timestamp, style="blue")
    
    def _generate_progress_panel(self) -> Panel:
        """Generate training progress panel"""
        metrics = self.monitor.metrics
        
        progress_table = Table(title="Training Progress", show_header=False)
        progress_table.add_column("Metric", style="cyan", width=20)
        progress_table.add_column("Value", style="magenta")
        
        progress_table.add_row("Current Iteration", f"{metrics.current_iteration}")
        progress_table.add_row("Current AUC", f"{metrics.current_auc:.2f}%")
        progress_table.add_row("Best AUC", f"{metrics.best_auc:.2f}%")
        progress_table.add_row("Target AUC", f"{metrics.target_auc:.2f}%")
        
        target_status = "✅ Achieved" if metrics.target_achieved else "🎯 In Progress"
        progress_table.add_row("Target Status", target_status)
        
        if metrics.total_duration:
            progress_table.add_row("Total Time", str(metrics.total_duration).split('.')[0])
        
        return Panel(progress_table, title="📊 Training Progress", border_style="green")
    
    def _generate_performance_panel(self) -> Panel:
        """Generate system performance panel"""
        stats = self.current_stats
        
        perf_table = Table(title="System Performance", show_header=False)
        perf_table.add_column("Metric", style="cyan", width=20)
        perf_table.add_column("Value", style="magenta")
        
        cpu = stats.get('cpu_percent', 0)
        memory = stats.get('memory_percent', 0)
        memory_gb = stats.get('memory_available_gb', 0)
        
        perf_table.add_row("CPU Usage", f"{cpu:.1f}%")
        perf_table.add_row("Memory Usage", f"{memory:.1f}%")
        perf_table.add_row("Available RAM", f"{memory_gb:.1f} GB")
        
        # Performance warnings
        warnings = []
        if cpu > 90:
            warnings.append("🔥 High CPU")
        if memory > 90:
            warnings.append("⚠️ Low Memory")
        
        if warnings:
            perf_table.add_row("Warnings", " ".join(warnings))
        
        return Panel(perf_table, title="💻 System Performance", border_style="yellow")
    
    def _generate_metrics_panel(self) -> Panel:
        """Generate additional metrics panel"""
        metrics = self.monitor.metrics
        
        metrics_table = Table(title="Training Metrics", show_header=False)
        metrics_table.add_column("Metric", style="cyan", width=20)
        metrics_table.add_column("Value", style="magenta")
        
        if metrics.iterations_history:
            total_iterations = len(metrics.iterations_history)
            best_iteration = max(metrics.iterations_history, key=lambda x: x['auc'])
            
            metrics_table.add_row("Total Iterations", str(total_iterations))
            metrics_table.add_row("Best Iteration", f"#{best_iteration['iteration']}")
            
            # AUC improvement rate
            if total_iterations > 1:
                first_auc = metrics.iterations_history[0]['auc']
                improvement = metrics.best_auc - first_auc
                metrics_table.add_row("AUC Improvement", f"+{improvement:.2f}%")
        
        return Panel(metrics_table, title="📈 Training Metrics", border_style="blue")
    
    def _generate_history_panel(self) -> Panel:
        """Generate training history panel"""
        metrics = self.monitor.metrics
        
        history_table = Table(title="Recent Iterations", show_header=True, header_style="bold")
        history_table.add_column("#", width=4)
        history_table.add_column("AUC", width=8)
        history_table.add_column("Status", width=12)
        
        # Show last 5 iterations
        recent_iterations = metrics.iterations_history[-5:]
        for iteration in recent_iterations:
            auc = iteration['auc']
            status = "🏆 Best" if iteration.get('is_best', False) else "📊 Progress"
            if auc >= metrics.target_auc:
                status = "🎯 Target"
            
            history_table.add_row(
                str(iteration['iteration']),
                f"{auc:.2f}%",
                status
            )
        
        return Panel(history_table, title="📋 Recent History", border_style="cyan")
    
    def _generate_footer(self) -> Panel:
        """Generate dashboard footer"""
        status_text = "🟢 Training Active" if self.monitor.is_monitoring else "🔴 Training Stopped"
        return Panel(status_text, style="dim")
    
    def start_live_dashboard(self):
        """Start live dashboard"""
        def generate_layout():
            self.layout["header"].update(self._generate_header())
            self.layout["progress"].update(self._generate_progress_panel())
            self.layout["performance"].update(self._generate_performance_panel())
            self.layout["metrics"].update(self._generate_metrics_panel())
            self.layout["history"].update(self._generate_history_panel())
            self.layout["footer"].update(self._generate_footer())
            return self.layout
        
        self.is_running = True
        with Live(generate_layout(), refresh_per_second=1, console=console) as live:
            while self.is_running and self.monitor.is_monitoring:
                time.sleep(1)
                live.update(generate_layout())
    
    def stop_dashboard(self):
        """Stop live dashboard"""
        self.is_running = False

class TrainingReporter:
    """Generate comprehensive training reports"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
    
    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        report = {
            'report_generated': datetime.now().isoformat(),
            'training_summary': self.monitor.metrics.get_summary(),
            'performance_analysis': self._analyze_performance(),
            'recommendations': self._generate_recommendations()
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            pro_log.info(f"Training report saved to: {output_path}")
        
        return report
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze training performance"""
        metrics = self.monitor.metrics
        
        analysis = {
            'system_performance': {},
            'training_efficiency': {},
            'resource_utilization': {}
        }
        
        # System performance analysis
        if metrics.cpu_usage:
            analysis['system_performance'] = {
                'avg_cpu_usage': sum(metrics.cpu_usage) / len(metrics.cpu_usage),
                'max_cpu_usage': max(metrics.cpu_usage),
                'cpu_stability': max(metrics.cpu_usage) - min(metrics.cpu_usage)
            }
        
        if metrics.memory_usage:
            analysis['system_performance']['memory'] = {
                'avg_memory_usage': sum(metrics.memory_usage) / len(metrics.memory_usage),
                'max_memory_usage': max(metrics.memory_usage),
                'memory_stability': max(metrics.memory_usage) - min(metrics.memory_usage)
            }
        
        # Training efficiency
        if metrics.iterations_history:
            aucs = [iter_data['auc'] for iter_data in metrics.iterations_history]
            analysis['training_efficiency'] = {
                'auc_progression': aucs,
                'convergence_rate': self._calculate_convergence_rate(aucs),
                'best_iteration': max(range(len(aucs)), key=lambda i: aucs[i]) + 1,
                'early_stopping_recommended': self._should_early_stop(aucs)
            }
        
        return analysis
    
    def _calculate_convergence_rate(self, aucs: List[float]) -> float:
        """Calculate AUC convergence rate"""
        if len(aucs) < 2:
            return 0.0
        
        improvements = []
        for i in range(1, len(aucs)):
            improvement = aucs[i] - aucs[i-1]
            improvements.append(improvement)
        
        return sum(improvements) / len(improvements) if improvements else 0.0
    
    def _should_early_stop(self, aucs: List[float], patience: int = 3) -> bool:
        """Determine if early stopping should be recommended"""
        if len(aucs) < patience + 1:
            return False
        
        recent_aucs = aucs[-patience:]
        max_recent = max(recent_aucs)
        
        # Check if no improvement in recent iterations
        return all(auc <= max_recent for auc in recent_aucs[:-1])
    
    def _generate_recommendations(self) -> List[str]:
        """Generate training recommendations"""
        recommendations = []
        metrics = self.monitor.metrics
        
        # AUC-based recommendations
        if metrics.best_auc < metrics.target_auc:
            gap = metrics.target_auc - metrics.best_auc
            if gap > 10:
                recommendations.append("Consider different model architectures or feature engineering")
            elif gap > 5:
                recommendations.append("Try hyperparameter optimization or longer training")
            else:
                recommendations.append("Fine-tune hyperparameters for final improvement")
        
        # Performance-based recommendations
        if metrics.cpu_usage and max(metrics.cpu_usage) > 90:
            recommendations.append("Consider reducing model complexity or batch size for CPU optimization")
        
        if metrics.memory_usage and max(metrics.memory_usage) > 90:
            recommendations.append("Monitor memory usage - consider reducing data batch size")
        
        # Training efficiency recommendations
        if len(metrics.iterations_history) > 5:
            recent_improvements = [
                metrics.iterations_history[i]['auc'] - metrics.iterations_history[i-1]['auc']
                for i in range(-3, 0)
            ]
            if all(imp <= 0 for imp in recent_improvements):
                recommendations.append("Training appears to have converged - consider early stopping")
        
        return recommendations
    
    def display_final_report(self):
        """Display final training report in console"""
        report = self.generate_report()
        
        console.print("\n")
        console.print(Panel(
            "📋 Final Training Report",
            style="bold blue"
        ))
        
        # Training Summary
        summary = report['training_summary']
        summary_table = Table(title="Training Summary", show_header=True, header_style="bold")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="magenta")
        
        summary_table.add_row("Best AUC", f"{summary['best_auc']:.2f}%")
        summary_table.add_row("Target Achieved", "✅ Yes" if summary['target_achieved'] else "❌ No")
        summary_table.add_row("Total Iterations", str(summary['total_iterations']))
        
        if summary['total_duration_seconds']:
            summary_table.add_row("Total Duration", f"{summary['total_duration_seconds']:.1f}s")
        
        console.print(summary_table)
        
        # Recommendations
        recommendations = report['recommendations']
        if recommendations:
            console.print("\n[bold yellow]💡 Recommendations:[/bold yellow]")
            for i, rec in enumerate(recommendations, 1):
                console.print(f"  {i}. {rec}")
        
        console.print("\n")
