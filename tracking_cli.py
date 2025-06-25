# Enterprise Tracking CLI
# tracking_cli.py
from datetime import datetime, timedelta
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.table import Table
from tracking import ExperimentTracker, tracker
from tracking_integration import production_tracker, pipeline_tracker
    from tracking_integration import start_production_monitoring
from typing import Dict, Any, List, Optional
import click
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
"""
Command - line interface for enterprise tracking system management
"""


console = Console()

@click.group()
@click.version_option(version = "1.0.0", prog_name = "Enterprise ML Tracking")
def cli():
    """Enterprise ML Tracking System CLI"""
    pass

@cli.command()
@click.option(' -  - config', ' - c', help = 'Configuration file path')
@click.option(' -  - experiment', ' - e', default = 'cli_experiment', help = 'Experiment name')
@click.option(' -  - run - name', ' - r', help = 'Run name')
@click.option(' -  - tags', ' - t', multiple = True, help = 'Tags (format: key = value)')
def start(config, experiment, run_name, tags):
    """Start a new experiment run"""

    # Parse tags
    tag_dict = {}
    for tag in tags:
        if ' = ' in tag:
            key, value = tag.split(' = ', 1)
            tag_dict[key] = value

    # Initialize tracker
    if config:
        exp_tracker = ExperimentTracker(config)
    else:
        exp_tracker = tracker

    # Start run
    with exp_tracker.start_run(experiment, run_name, tag_dict) as run:
        console.print(f"✅ Started experiment run: {run.run_id}")
        console.print("💡 Use 'tracking_cli log' commands to add data")

        # Wait for user input to keep run active
        click.pause("Press any key to end run...")

@cli.command()
@click.option(' -  - param', ' - p', multiple = True, help = 'Parameters (format: key = value)')
@click.option(' -  - file', ' - f', help = 'JSON/YAML file with parameters')
def log_params(param, file):
    """Log parameters to active run"""

    params = {}

    # Parse individual parameters
    for p in param:
        if ' = ' in p:
            key, value = p.split(' = ', 1)
            try:
                # Try to parse as number
                if '.' in value:
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except ValueError:
                params[key] = value

    # Load from file
    if file:
        file_path = Path(file)
        if file_path.exists():
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    file_params = yaml.safe_load(f)
                else:
                    file_params = json.load(f)
                params.update(file_params)

    if params:
        tracker.log_params(params)
        console.print(f"✅ Logged {len(params)} parameters")
    else:
        console.print("❌ No parameters provided")

@cli.command()
@click.option(' -  - metric', ' - m', multiple = True, help = 'Metrics (format: key = value)')
@click.option(' -  - step', ' - s', type = int, help = 'Step number')
@click.option(' -  - file', ' - f', help = 'JSON file with metrics')
def log_metrics(metric, step, file):
    """Log metrics to active run"""

    metrics = {}

    # Parse individual metrics
    for m in metric:
        if ' = ' in m:
            key, value = m.split(' = ', 1)
            try:
                metrics[key] = float(value)
            except ValueError:
                console.print(f"❌ Invalid metric value: {value}")
                return

    # Load from file
    if file:
        file_path = Path(file)
        if file_path.exists():
            with open(file_path, 'r') as f:
                file_metrics = json.load(f)
                metrics.update(file_metrics)

    if metrics:
        tracker.log_metrics(metrics, step)
        console.print(f"✅ Logged {len(metrics)} metrics")
    else:
        console.print("❌ No metrics provided")

@cli.command()
@click.option(' -  - format', 'output_format', default = 'table', 
              type = click.Choice(['table', 'json', 'csv']), 
              help = 'Output format')
@click.option(' -  - limit', ' - l', default = 10, help = 'Number of experiments to show')
@click.option(' -  - experiment', ' - e', help = 'Filter by experiment name')
def list_experiments(output_format, limit, experiment):
    """List recent experiments"""

    experiments = tracker.list_experiments()

    # Filter by experiment name
    if experiment:
        experiments = [exp for exp in experiments if experiment in exp.get('run_id', '')]

    # Limit results
    experiments = experiments[:limit]

    if not experiments:
        console.print("No experiments found")
        return

    if output_format == 'table':
        table = Table(title = f"Recent Experiments (Top {len(experiments)})")
        table.add_column("Run ID", style = "cyan", no_wrap = True)
        table.add_column("Start Time", style = "green")
        table.add_column("Duration", style = "yellow")
        table.add_column("Metrics", style = "magenta")
        table.add_column("Status", style = "blue")

        for exp in experiments:
            start_time = exp.get('start_time', 'Unknown')
            if start_time != 'Unknown':
                start_time = datetime.fromisoformat(start_time).strftime('%Y - %m - %d %H:%M')

            duration = exp.get('duration', 0)
            duration_str = f"{duration:.1f}s" if duration > 0 else "Unknown"

            table.add_row(
                exp.get('run_id', 'Unknown')[:20], 
                start_time, 
                duration_str, 
                str(exp.get('metrics_count', 0)), 
                exp.get('status', 'Unknown')
            )

        console.print(table)

    elif output_format == 'json':
        console.print(json.dumps(experiments, indent = 2, default = str))

    elif output_format == 'csv':
        df = pd.DataFrame(experiments)
        console.print(df.to_csv(index = False))

@cli.command()
@click.argument('run_id')
@click.option(' -  - format', 'output_format', default = 'table', 
              type = click.Choice(['table', 'json']), 
              help = 'Output format')
def show_run(run_id, output_format):
    """Show details of a specific run"""

    if run_id not in tracker.metadata:
        console.print(f"❌ Run not found: {run_id}")
        return

    run_data = tracker.metadata[run_id]

    if output_format == 'table':
        # Run information
        info_table = Table(title = f"Run Details: {run_id}")
        info_table.add_column("Property", style = "cyan")
        info_table.add_column("Value", style = "green")

        info_table.add_row("Run ID", run_id)
        info_table.add_row("Start Time", run_data.get('start_time', 'Unknown'))
        info_table.add_row("End Time", run_data.get('end_time', 'Unknown'))
        info_table.add_row("Duration", f"{run_data.get('duration_seconds', 0):.2f}s")

        console.print(info_table)

        # Parameters
        params = run_data.get('parameters', {})
        if params:
            param_table = Table(title = "Parameters")
            param_table.add_column("Parameter", style = "cyan")
            param_table.add_column("Value", style = "yellow")

            for key, value in params.items():
                param_table.add_row(key, str(value))

            console.print(param_table)

        # Metrics
        metrics = run_data.get('metrics', {})
        if metrics:
            metric_table = Table(title = "Metrics")
            metric_table.add_column("Metric", style = "cyan")
            metric_table.add_column("Value", style = "magenta")

            for key, value in metrics.items():
                metric_table.add_row(key, f"{value:.4f}" if isinstance(value, float) else str(value))

            console.print(metric_table)

    elif output_format == 'json':
        console.print(json.dumps(run_data, indent = 2, default = str))

@cli.command()
@click.option(' -  - metric', ' - m', required = True, help = 'Metric to optimize')
@click.option(' -  - mode', default = 'max', type = click.Choice(['max', 'min']), 
              help = 'Optimization mode')
@click.option(' -  - top - k', ' - k', default = 5, help = 'Number of top runs to show')
def best_runs(metric, mode, top_k):
    """Find best runs based on a metric"""

    runs_with_metric = []

    for run_id, run_data in tracker.metadata.items():
        metrics = run_data.get('metrics', {})
        if metric in metrics:
            runs_with_metric.append({
                'run_id': run_id, 
                'metric_value': metrics[metric], 
                'start_time': run_data.get('start_time'), 
                'parameters': run_data.get('parameters', {})
            })

    if not runs_with_metric:
        console.print(f"❌ No runs found with metric: {metric}")
        return

    # Sort runs
    reverse = (mode == 'max')
    sorted_runs = sorted(runs_with_metric, 
                        key = lambda x: x['metric_value'], 
                        reverse = reverse)[:top_k]

    # Display results
    table = Table(title = f"Top {len(sorted_runs)} Runs by {metric} ({mode})")
    table.add_column("Rank", style = "cyan")
    table.add_column("Run ID", style = "green")
    table.add_column(metric.title(), style = "magenta")
    table.add_column("Start Time", style = "yellow")

    for i, run in enumerate(sorted_runs, 1):
        start_time = run['start_time']
        if start_time:
            start_time = datetime.fromisoformat(start_time).strftime('%Y - %m - %d %H:%M')

        table.add_row(
            str(i), 
            run['run_id'][:20], 
            f"{run['metric_value']:.4f}", 
            start_time or 'Unknown'
        )

    console.print(table)

@cli.command()
@click.option(' -  - output', ' - o', default = 'tracking_report.html', help = 'Output file')
@click.option(' -  - experiment', ' - e', help = 'Filter by experiment name')
@click.option(' -  - days', ' - d', default = 7, help = 'Number of days to include')
def generate_report(output, experiment, days):
    """Generate HTML report of experiments"""

    # Filter experiments by date
    cutoff_date = datetime.now() - timedelta(days = days)
    recent_experiments = []

    for run_id, run_data in tracker.metadata.items():
        start_time_str = run_data.get('start_time')
        if start_time_str:
            start_time = datetime.fromisoformat(start_time_str)
            if start_time >= cutoff_date:
                recent_experiments.append((run_id, run_data))

    if not recent_experiments:
        console.print(f"❌ No experiments found in the last {days} days")
        return

    # Generate report
    html_content = _generate_html_report(recent_experiments, days)

    # Save report
    with open(output, 'w', encoding = 'utf - 8') as f:
        f.write(html_content)

    console.print(f"✅ Report generated: {output}")
    console.print(f"📊 Included {len(recent_experiments)} experiments from last {days} days")

def _generate_html_report(experiments: List, days: int) -> str:
    """Generate HTML report content"""

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Tracking Report - Last {days} Days</title>
        <style>
            body {{ font - family: Arial, sans - serif; margin: 20px; }}
            .header {{ background: #f0f0f0; padding: 20px; border - radius: 5px; }}
            .experiment {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border - radius: 5px; }}
            .metrics {{ display: flex; flex - wrap: wrap; gap: 10px; }}
            .metric {{ background: #e7f3ff; padding: 5px 10px; border - radius: 3px; }}
            table {{ width: 100%; border - collapse: collapse; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text - align: left; }}
            th {{ background - color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class = "header">
            <h1>🧪 ML Tracking Report</h1>
            <p>Generated: {datetime.now().strftime('%Y - %m - %d %H:%M:%S')}</p>
            <p>Period: Last {days} days ({len(experiments)} experiments)</p>
        </div>
    """

    # Summary statistics
    total_duration = sum(exp[1].get('duration_seconds', 0) for exp in experiments)
    avg_duration = total_duration / len(experiments) if experiments else 0

    html += f"""
        <h2>📊 Summary Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Experiments</td><td>{len(experiments)}</td></tr>
            <tr><td>Total Duration</td><td>{total_duration:.1f} seconds</td></tr>
            <tr><td>Average Duration</td><td>{avg_duration:.1f} seconds</td></tr>
        </table>
    """

    # Individual experiments
    html += "<h2>🔬 Individual Experiments</h2>"

    for run_id, exp_data in experiments:
        start_time = exp_data.get('start_time', 'Unknown')
        duration = exp_data.get('duration_seconds', 0)
        metrics = exp_data.get('metrics', {})
        parameters = exp_data.get('parameters', {})

        html += f"""
        <div class = "experiment">
            <h3>🧪 {run_id}</h3>
            <p><strong>Start Time:</strong> {start_time}</p>
            <p><strong>Duration:</strong> {duration:.2f} seconds</p>

            <h4>Parameters</h4>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
        """

        for key, value in parameters.items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>"

        html += "</table><h4>Metrics</h4><table><tr><th>Metric</th><th>Value</th></tr>"

        for key, value in metrics.items():
            html += f"<tr><td>{key}</td><td>{value:.4f if isinstance(value, float) else value}</td></tr>"

        html += "</table></div>"

    html += "</body></html>"
    return html

@cli.command()
@click.option(' -  - port', ' - p', default = 8080, help = 'Port for web interface')
def serve(port):
    """Start web interface for tracking system"""
    console.print(f"🌐 Starting web interface on port {port}")
    console.print("💡 This is a placeholder - implement with Flask/FastAPI")
    console.print(f"   URL would be: http://localhost:{port}")

@cli.group()
def production():
    """Production monitoring commands"""
    pass

@production.command()
@click.argument('model_name')
@click.argument('deployment_id')
def start_monitoring(model_name, deployment_id):
    """Start production monitoring"""
    start_production_monitoring(model_name, deployment_id)
    console.print(f"✅ Started monitoring for {model_name} (ID: {deployment_id})")

@production.command()
@click.argument('deployment_id')
def status(deployment_id):
    """Check production monitoring status"""
    summary = production_tracker.get_production_summary(deployment_id)

    if not summary:
        console.print(f"❌ No monitoring found for deployment: {deployment_id}")
        return

    table = Table(title = f"Production Status: {deployment_id}")
    table.add_column("Property", style = "cyan")
    table.add_column("Value", style = "green")

    for key, value in summary.items():
        table.add_row(key.replace('_', ' ').title(), str(value))

    console.print(table)

if __name__ == '__main__':
    cli()