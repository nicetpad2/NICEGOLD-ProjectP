# versioning.py
from rich.console import Console
from rich.panel import Panel
        import mlflow
    import subprocess
"""
Data/model versioning utilities (DVC, MLflow)
"""
console = Console()

# Example: DVC add/commit

def dvc_add_commit(file_path):
    subprocess.run(["dvc", "add", file_path])
    subprocess.run(["git", "add", file_path + ".dvc"])
    subprocess.run(["git", "commit", " - m", f"Add {file_path} to DVC"])
    console.print(Panel(f"[green]DVC tracked: {file_path}", title = "DVC", border_style = "green"))

# Example: MLflow tracking stub
def mlflow_log_metric(metric, value):
    try:
        mlflow.log_metric(metric, value)
        console.print(Panel(f"[green]MLflow logged: {metric} = {value}", title = "MLflow", border_style = "green"))
    except ImportError:
        console.print(Panel("[red]MLflow not installed", title = "MLflow", border_style = "red"))

# TODO: Add model versioning, experiment tracking, rollback, etc.