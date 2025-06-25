# visualization_dashboard.py
from rich.console import Console
from rich.panel import Panel
    import plotly.express as px
    import subprocess
"""
Visualization/Explainability utilities
- Interactive dashboard (Plotly Dash, Streamlit, Gradio)
- EDA, drift, fairness, diagnostics report
- Export HTML/Markdown/Excel/PNG
"""
console = Console()

# Example: Streamlit dashboard launcher
def launch_streamlit_dashboard(script_path = 'dashboard_app.py'):
    console.print(Panel(f"[green]Launching Streamlit dashboard: {script_path}", title = "Dashboard", border_style = "green"))
    subprocess.Popen(["streamlit", "run", script_path])

# Example: Plotly EDA
def plot_eda(df, out_html = 'output_default/eda_report.html'):
    fig = px.histogram(df, x = df.columns[0])
    fig.write_html(out_html)
    console.print(Panel(f"[green]EDA report saved: {out_html}", title = "EDA", border_style = "green"))

# TODO: Add SHAP, drift, fairness, diagnostics, Gradio, Dash, export options