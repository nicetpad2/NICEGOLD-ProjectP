import os
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console

orders = [
    {"id": "80cedc2e-21a4-461f-b28d-639d55819784", "symbol": "XAUUSD", "status": "FILLED"},
    {"id": "f733867e-3987-4ccf-8195-88094168f1d6", "symbol": "XAUUSD", "status": "FILLED"},
    {"id": "4e644039-2b64-4fb8-81e9-9a837f672820", "symbol": "XAUUSD", "status": "FILLED"},
    {"id": "0dc0e827-166f-40c5-ad53-04b2b32c2494", "symbol": "XAUUSD", "status": "FILLED"},
    {"id": "d71793a5-681c-40aa-b624-6328306e76af", "symbol": "XAUUSD", "status": "FILLED"},
    {"id": "1f412a6e-a98e-44fd-bda4-5b55c40e1e41", "symbol": "XAUUSD", "status": "FILLED"},
    {"id": "d1cfedd5-801d-4527-96e0-610abff84028", "symbol": "XAUUSD", "status": "FILLED"},
    {"id": "640c9508-11ca-49b5-8490-528be5ad9301", "symbol": "XAUUSD", "status": "FILLED"},
    {"id": "d396492c-a4eb-41d1-9235-acaf3c04bdf3", "symbol": "XAUUSD", "status": "FILLED"},
    {"id": "92eafb13-7671-4f72-9182-abc36608706c", "symbol": "XAUUSD", "status": "FILLED"},
    {"id": "70bdc385-53da-44b0-b433-d5f86a7272fc", "symbol": "XAUUSD", "status": "FILLED"},
    {"id": "8641a105-6c7d-4806-a42d-46ffc3120750", "symbol": "XAUUSD", "status": "FILLED"},
    {"id": "c50a96ae-f02c-4c4a-bc7c-cff5d68517a5", "symbol": "XAUUSD", "status": "FILLED"},
    {"id": "7161af88-34fc-4ec8-8075-fd6c9265d2b4", "symbol": "XAUUSD", "status": "FILLED"},
    {"id": "727f6955-8277-4892-bfe4-6dd015a34342", "symbol": "XAUUSD", "status": "FILLED"},
]

def show_order_progress(orders):
    console = Console()
    total = len(orders)
    with Progress(
        TextColumn("[bold blue]สถานะออเดอร์[/]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[green]กำลังดำเนินการ...", total=total)
        for i, order in enumerate(orders, 1):
            console.print(f"Order ID: {order['id']} ({order['status']})")
            progress.update(task, advance=1)

if __name__ == "__main__":
    show_order_progress(orders)
