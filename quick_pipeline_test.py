#!/usr/bin/env python3
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import os
import pandas as pd
import sys
"""
🎯 Quick Pipeline Test - ทดสอบความพร้อมของระบบหลังการเตรียมข้อมูล
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
"""


console = Console()

def test_data_readiness():
    """ทดสอบความพร้อมของข้อมูล"""
    console.print(Panel("🔍 [bold blue]ทดสอบความพร้อมของข้อมูล Multi - timeframe[/bold blue]", 
                        border_style = "cyan"))

    # ตรวจสอบไฟล์ข้อมูล
    data_file = "output_default/real_data.csv"
    if not os.path.exists(data_file):
        console.print("[red]❌ ไม่พบไฟล์ข้อมูล real_data.csv[/red]")
        return False

    # โหลดข้อมูล
    try:
        df = pd.read_csv(data_file)
        console.print(f"[green]✅ โหลดข้อมูลสำเร็จ: {df.shape}[/green]")

        # แสดงสถิติข้อมูล
        table = Table(title = "📊 Data Summary", show_header = True, header_style = "bold magenta")
        table.add_column("Metric", style = "cyan")
        table.add_column("Value", style = "white")
        table.add_row("Total Rows", f"{len(df):, }")
        table.add_row("Total Columns", f"{len(df.columns)}")
        table.add_row("Memory Usage", f"{df.memory_usage(deep = True).sum() / 1024**2:.1f} MB")

        # Target distribution
        if 'target' in df.columns:
            target_dist = df['target'].value_counts().sort_index()
            for i, count in target_dist.items():
                table.add_row(f"Target {i}", f"{count:, } ({count/len(df)*100:.1f}%)")

        console.print(table)
        return True

    except Exception as e:
        console.print(f"[red]❌ Error loading data: {e}[/red]")
        return False

def test_pipeline_modules():
    """ทดสอบความพร้อมของโมดูล Pipeline"""
    console.print(Panel("🔧 [bold blue]ทดสอบโมดูล Pipeline[/bold blue]", border_style = "green"))

    modules_to_test = [
        ("projectp.steps.train", "run_train"), 
        ("projectp.steps.sweep", "run_sweep"), 
        ("projectp.steps.threshold", "run_threshold"), 
        ("projectp.steps.walkforward", "run_walkforward"), 
        ("projectp.steps.backtest", "run_backtest"), 
        ("projectp.steps.report", "run_report"), 
    ]

    results = []
    for module_name, func_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist = [func_name])
            func = getattr(module, func_name)
            results.append((module_name, "✅", "Ready"))
        except Exception as e:
            results.append((module_name, "❌", str(e)[:50]))

    table = Table(title = "🔧 Pipeline Modules Status", show_header = True, header_style = "bold green")
    table.add_column("Module", style = "cyan")
    table.add_column("Status", style = "white")
    table.add_column("Details", style = "yellow")

    for module, status, details in results:
        table.add_row(module.split('.')[ - 1], status, details)

    console.print(table)

    # Count ready modules
    ready_count = sum(1 for _, status, _ in results if status == "✅")
    total_count = len(results)

    if ready_count == total_count:
        console.print(f"[green]🎉 ทุกโมดูลพร้อมใช้งาน ({ready_count}/{total_count})[/green]")
        return True
    else:
        console.print(f"[yellow]⚠️ โมดูลพร้อมใช้งาน: {ready_count}/{total_count}[/yellow]")
        return False

def test_output_directories():
    """ทดสอบและสร้างโฟลเดอร์เอาต์พุต"""
    console.print(Panel("📁 [bold blue]ตรวจสอบและสร้างโฟลเดอร์[/bold blue]", border_style = "yellow"))

    required_dirs = [
        "output_default", 
        "output_default/models", 
        "output_default/logs", 
        "output_default/reports", 
        "logs"
    ]

    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents = True, exist_ok = True)
        if os.path.exists(dir_path):
            console.print(f"[green]✅ {dir_path}[/green]")
        else:
            console.print(f"[red]❌ {dir_path}[/red]")

    return True

def show_next_steps():
    """แสดงขั้นตอนถัดไป"""
    console.print(Panel("""
🚀 [bold green]ระบบพร้อมใช้งาน! ขั้นตอนที่แนะนำ:[/bold green]

1. [cyan]python ProjectP.py - - run_full_pipeline[/cyan]
   รันไปป์ไลน์แบบเต็ม (All steps)

2. [cyan]python ProjectP.py - - debug_full_pipeline[/cyan]
   รันในโหมด Debug พร้อม Logging รายละเอียด

3. [cyan]รันทีละขั้นตอน:[/cyan]
   • Train: เทรนโมเดล ML
   • Sweep: หา Hyperparameters ที่ดีที่สุด
   • Threshold: หา Threshold ที่เหมาะสม
   • WalkForward: ทดสอบแบบ Walk - Forward
   • Backtest: ทดสอบกลับ
   • Report: สร้างรายงาน

4. [yellow]ไฟล์ข้อมูลหลัก:[/yellow]
   📁 output_default/real_data.csv (22, 474 samples, 56 features)

[bold blue]🔥 ข้อมูล Multi - timeframe พร้อมสำหรับการวิเคราะห์ระดับ Production![/bold blue]
    """, title = "[bold magenta]🎯 Next Steps", border_style = "bright_green"))

def main():
    """เมนหลัก"""
    console.print(Panel("""
🎯 [bold blue]Quick Pipeline Test[/bold blue]
ตรวจสอบความพร้อมของระบบหลังการเตรียมข้อมูล Multi - timeframe
    """, title = "[bold green]Pipeline Readiness Check", border_style = "blue"))

    # รันการทดสอบ
    data_ready = test_data_readiness()
    modules_ready = test_pipeline_modules()
    dirs_ready = test_output_directories()

    # สรุปผล
    console.print("\n" + " = "*60)
    if data_ready and modules_ready and dirs_ready:
        console.print("[bold green]🎉 ระบบพร้อมใช้งานครบถ้วน![/bold green]")
        show_next_steps()
    else:
        console.print("[bold yellow]⚠️ พบปัญหาบางส่วน กรุณาตรวจสอบข้างต้น[/bold yellow]")

if __name__ == "__main__":
    main()