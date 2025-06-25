
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme
from src.utils.log_utils import pro_log, pro_log_json
import logging
custom_theme = Theme({
    "info": "bold cyan", 
    "success": "bold green", 
    "warning": "bold yellow", 
    "error": "bold red", 
    "step": "bold magenta", 
    "tag": "bold blue", 
})
console = Console(theme = custom_theme)

def pro_log(msg, tag = None, level = "info"):
    now = datetime.now().strftime("%H:%M:%S")
    tag_str = f"[{tag}]" if tag else ""
    level_str = level.upper()
    color = {
        "info": "info", 
        "success": "success", 
        "warn": "warning", 
        "warning": "warning", 
        "error": "error", 
        "step": "step", 
    }.get(level, "info")
    text = Text(f"{now} | {level_str} {tag_str} {msg}", style = color)
    if level in ("error", "warn", "warning"):
        console.print(Panel(text, border_style = color))
    else:
        console.print(text)
    # Log to file as well
    logging.log({
        "info": logging.INFO, 
        "success": logging.INFO, 
        "warn": logging.WARNING, 
        "warning": logging.WARNING, 
        "error": logging.ERROR, 
        "step": logging.INFO, 
    }.get(level, logging.INFO), f"{tag_str} {msg}")