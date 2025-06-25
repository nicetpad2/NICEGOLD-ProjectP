# serving.py
from rich.console import Console
from rich.panel import Panel
import os
    import subprocess
    import torch
"""
Production/Serving utilities
- Export model as ONNX, TorchScript, PMML
- REST API (FastAPI/Flask)
- Auto versioning, model registry, monitoring, rollback
"""
console = Console()

# Example: Export PyTorch model to ONNX
def export_pytorch_onnx(model, dummy_input, out_path = 'output_default/model.onnx'):
    torch.onnx.export(model, dummy_input, out_path)
    console.print(Panel(f"[green]Model exported to ONNX: {out_path}", title = "ONNX Export", border_style = "green"))

# Example: FastAPI serving stub
def launch_fastapi_server(script_path = 'serve_app.py'):
    console.print(Panel(f"[green]Launching FastAPI server: {script_path}", title = "Serving", border_style = "green"))
    subprocess.Popen(["uvicorn", script_path.replace('.py', '') + ":app", " -  - reload"])

def launch_fastapi_server_on_port(script_path = 'serve_app.py', port = 8001):
    console.print(Panel(f"[green]Launching FastAPI server: {script_path} on port {port}", title = "Serving", border_style = "green"))
    subprocess.Popen(["uvicorn", script_path.replace('.py', '') + ":app", " -  - reload", " -  - port", str(port)])

# TODO: Add TorchScript, PMML, Flask, model registry, monitoring, rollback