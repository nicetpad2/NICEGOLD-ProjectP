# deep_learning.py
"""
Deep Learning utilities for train/predict with PyTorch, Keras, TabNet, TabTransformer
- Auto GPU allocation
- Early stopping
- Rich progress bar
- Model/learning curve/attention visualization
"""
import os
import torch
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.console import Console
from torch.utils.data import DataLoader
import multiprocessing

console = Console()

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def print_dl_resource():
    device = get_device()
    if device == 'cuda':
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        console.print(Panel(f"[bold green]GPU: {mem:.2f} GB available", title="[green]DL Resource", border_style="green"))
    else:
        console.print(Panel("[yellow]CPU only", title="DL Resource", border_style="yellow"))

def get_maximum_dataloader(dataset, batch_size=None):
    """
    สร้าง DataLoader ที่เทพที่สุด: batch_size ใหญ่สุด, num_workers สูงสุด, pin_memory True
    """
    if batch_size is None:
        # กำหนด batch_size ใหญ่สุดเท่าที่ไม่ OOM (เช่น 4096 หรือมากกว่านั้น)
        batch_size = 4096
    num_workers = multiprocessing.cpu_count()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

# Example: PyTorch training loop with early stopping and rich progress

def train_pytorch(model, train_loader, val_loader, criterion, optimizer, epochs=20, patience=3):
    device = get_device()
    model.to(device)
    best_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    with Progress(BarColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("[cyan]Training DL model...", total=epochs)
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    loss = criterion(out, yb)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            progress.update(task, advance=1, description=f"[green]Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'output_default/best_dl_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    console.print(Panel(f"[bold yellow]Early stopping at epoch {epoch+1}", title="Early Stopping", border_style="yellow"))
                    break
    return history

# Visualization utilities (learning curve, model graph, attention map)
def plot_learning_curve(history, out_path='output_default/dl_learning_curve.png'):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Learning Curve')
    plt.savefig(out_path)
    plt.close()
    console.print(Panel(f"[bold green]Learning curve saved: {out_path}", title="DL Visualization", border_style="green"))

# TODO: Add Keras/TabNet/TabTransformer, attention viz, ONNX export, etc.
