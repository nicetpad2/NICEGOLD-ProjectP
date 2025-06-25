# notify.py
from rich.console import Console
from rich.panel import Panel
    import requests
"""
Notification utilities (LINE, Slack, Email)
"""
console = Console()

# Example: LINE Notify
def line_notify(token, message):
    url = "https://notify - api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": message}
    r = requests.post(url, headers = headers, data = data)
    if r.status_code == 200:
        console.print(Panel("[green]LINE notify sent!", title = "LINE Notify", border_style = "green"))
    else:
        console.print(Panel(f"[red]LINE notify failed: {r.text}", title = "LINE Notify", border_style = "red"))

# TODO: Add Slack, Email, webhook, etc.