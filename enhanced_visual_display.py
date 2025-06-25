#!/usr/bin/env python3
from datetime import datetime, timedelta
from rich.align import Align
from rich.box import ROUNDED
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
from rich.table import Table
from rich.text import Text
from typing import Any, Dict, List, Optional
import json
import os
import time
"""
🎨 ENHANCED VISUAL DISPLAY SYSTEM
สวยงามและทันสมัยสำหรับ NICEGOLD Pipeline
"""


    BarColumn, 
    MofNCompleteColumn, 
    Progress, 
    SpinnerColumn, 
    TaskProgressColumn, 
    TextColumn, 
    TimeElapsedColumn, 
    TimeRemainingColumn, 
)

console = Console()


class ThaiVisualDisplay:
    """ระบบแสดงผลภาษาไทยที่สวยงาม"""

    def __init__(self):
        self.console = Console()
        self.pipeline_start_time = None
        self.current_stage = None
        self.stage_metrics = {}

    def show_welcome_screen(
        self, title: str = "🏆 NICEGOLD ProjectP - ระบบเทรดทองคำอัจฉริยะ"
    ):
        """แสดงหน้าจอต้อนรับที่สวยงาม"""
        self.console.clear()
        self.show_pipeline_banner(title)

    def show_pipeline_banner(self, title: str):
        """แสดงแบนเนอร์หลักของ Pipeline"""
        banner_text = Text(title, style = "bold gold1")
        subtitle = Text("🚀 ระบบ AI ทำนายราคาทองคำที่ทันสมัย", style = "italic blue")

        banner = Panel(
            Align.center(
                Text.assemble(
                    banner_text, 
                    "\n\n", 
                    subtitle, 
                    "\n", 
                    Text("⭐ พร้อมใช้งานสำหรับการเทรดอัจฉริยะ ⭐", style = "green"), 
                )
            ), 
            box = ROUNDED, 
            style = "gold1", 
            padding = (1, 2), 
        )

        self.console.print(banner)
        self.console.print()

    def create_progress_tracker(self) -> Progress:
        """สร้าง Progress Tracker ที่สวยงาม"""
        return Progress(
            SpinnerColumn(), 
            TextColumn("[progress.description]{task.description}"), 
            BarColumn(), 
            MofNCompleteColumn(), 
            TextColumn("•"), 
            TimeElapsedColumn(), 
            TextColumn("•"), 
            TimeRemainingColumn(), 
            console = self.console, 
        )

    def update_stage_status(self, stage: str, status: str, details: str = ""):
        """อัปเดตสถานะของแต่ละขั้นตอน"""
        self.current_stage = stage

        status_colors = {
            "เริ่มต้น": "yellow", 
            "กำลังดำเนินการ": "blue", 
            "สำเร็จ": "green", 
            "ผิดพลาด": "red", 
            "รอดำเนินการ": "white", 
        }

        color = status_colors.get(status, "white")

        status_panel = Panel(
            Text.assemble(
                Text(f"📊 ขั้นตอน: {stage}", style = "bold"), 
                "\n", 
                Text(f"🔄 สถานะ: {status}", style = f"bold {color}"), 
                "\n" + details if details else "", 
            ), 
            title = "สถานะปัจจุบัน", 
            box = ROUNDED, 
            style = color, 
        )

        self.console.print(status_panel)

    def show_stage_summary(self, stage_results: Dict[str, Any]):
        """แสดงสรุปผลของแต่ละขั้นตอน"""

        summary_table = Table(title = "📈 สรุปผลการดำเนินงาน", show_header = True)
        summary_table.add_column("ขั้นตอน", style = "cyan")
        summary_table.add_column("สถานะ", style = "green")
        summary_table.add_column("เวลา (วินาที)", style = "yellow")
        summary_table.add_column("รายละเอียด", style = "white")

        for stage, result in stage_results.items():
            status = "✅ สำเร็จ" if result.get("success", True) else "❌ ผิดพลาด"
            duration = f"{result.get('duration', 0):.2f}"
            details = result.get("details", " - ")

            summary_table.add_row(stage, status, duration, details)

        panel = Panel(summary_table, box = ROUNDED, style = "blue")
        self.console.print(panel)

    def show_final_results(self, results: Dict[str, Any]):
        """แสดงผลลัพธ์สุดท้าย"""

        # สร้างตารางผลลัพธ์หลัก
        results_table = Table(title = "🎯 ผลลัพธ์สุดท้าย", show_header = True)
        results_table.add_column("รายการ", style = "cyan")
        results_table.add_column("ค่า", style = "green")
        results_table.add_column("หน่วย", style = "yellow")

        # เพิ่มข้อมูลผลลัพธ์
        if "model_performance" in results:
            perf = results["model_performance"]
            results_table.add_row("ความแม่นยำ", f"{perf.get('accuracy', 0):.3f}", "%")
            results_table.add_row("ค่า AUC", f"{perf.get('auc', 0):.3f}", "คะแนน")

        if "execution_time" in results:
            results_table.add_row(
                "เวลาทั้งหมด", f"{results['execution_time']:.2f}", "วินาที"
            )

        if "data_processed" in results:
            results_table.add_row(
                "ข้อมูลที่ประมวลผล", f"{results['data_processed']:, }", "แถว"
            )

        final_panel = Panel(
            Align.center(results_table), 
            title = "🏆 NICEGOLD Pipeline - เสร็จสมบูรณ์", 
            box = ROUNDED, 
            style = "gold1", 
            padding = (1, 2), 
        )

        self.console.print(final_panel)

        # แสดงข้อความสำเร็จ
        success_message = Text.assemble(
            Text("🎉 การประมวลผลเสร็จสมบูรณ์! ", style = "bold green"), 
            Text("ระบบพร้อมใช้งาน 🚀", style = "bold blue"), 
        )

        self.console.print(Align.center(success_message))

    def show_system_status(
        self, cpu_percent = 0, ram_percent = 0, gpu_available = False, gpu_usage = 0
    ):
        """แสดงสถานะระบบในปัจจุบัน"""

        # สร้างตารางสถานะระบบ
        status_table = Table(title = "🖥️ สถานะระบบ", show_header = True)
        status_table.add_column("ทรัพยากร", style = "cyan")
        status_table.add_column("การใช้งาน", style = "yellow")
        status_table.add_column("สถานะ", style = "green")

        # สถานะ CPU
        cpu_status = (
            "🟢 ปกติ" if cpu_percent < 70 else "🟡 ระวัง" if cpu_percent < 90 else "🔴 สูง"
        )
        status_table.add_row("CPU", f"{cpu_percent:.1f}%", cpu_status)

        # สถานะ RAM
        ram_status = (
            "🟢 ปกติ" if ram_percent < 70 else "🟡 ระวัง" if ram_percent < 90 else "🔴 สูง"
        )
        status_table.add_row("RAM", f"{ram_percent:.1f}%", ram_status)

        # สถานะ GPU
        if gpu_available:
            gpu_status = (
                "🟢 พร้อม"
                if gpu_usage < 70
                else "🟡 ใช้งาน" if gpu_usage < 90 else "🔴 เต็ม"
            )
            status_table.add_row("GPU", f"{gpu_usage:.1f}%", gpu_status)
        else:
            status_table.add_row("GPU", "ไม่มี", "➖ ไม่ใช้งาน")

        # แสดงในกรอบ
        system_panel = Panel(
            status_table, 
            title = "📊 การติดตามระบบ", 
            box = ROUNDED, 
            style = "blue", 
            padding = (1, 2), 
        )

        self.console.print(system_panel)


class EnhancedReportGenerator:
    """ระบบสร้างรายงานขั้นสูง"""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok = True)

    def generate_performance_report(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """สร้างรายงานประสิทธิภาพ"""
        report = {
            "timestamp": datetime.now().isoformat(), 
            "report_type": "performance_analysis", 
            "metrics": metrics, 
            "execution_summary": {
                "total_execution_time": metrics.get("total_time", 0), 
                "successful_stages": metrics.get("successful_stages", 0), 
                "total_stages": metrics.get("total_stages", 0), 
                "success_rate": (
                    metrics.get("successful_stages", 0)
                    / max(metrics.get("total_stages", 1), 1)
                )
                * 100, 
                "status": (
                    "สำเร็จ" if metrics.get("all_successful", True) else "มีข้อผิดพลาด"
                ), 
            }, 
            "stage_breakdown": metrics.get("stage_times", {}), 
            "resource_usage": metrics.get("resource_usage", {}), 
            "recommendations": self._generate_performance_recommendations(metrics), 
        }

        # บันทึกรายงาน
        report_path = os.path.join(self.output_dir, "performance_report.json")
        with open(report_path, "w", encoding = "utf - 8") as f:
            json.dump(report, f, ensure_ascii = False, indent = 2)

        return report

    def generate_data_quality_report(
        self, data_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """สร้างรายงานคุณภาพข้อมูล"""

        data_health_score = self._calculate_data_health_score(data_metrics)

        report = {
            "timestamp": datetime.now().isoformat(), 
            "report_type": "data_quality_analysis", 
            "data_health_score": data_health_score, 
            "metrics": data_metrics, 
            "quality_indicators": {
                "completeness": data_metrics.get("completeness_score", 100), 
                "consistency": data_metrics.get("consistency_score", 100), 
                "accuracy": data_metrics.get("accuracy_score", 100), 
                "freshness": data_metrics.get("freshness_score", 100), 
            }, 
            "data_issues": data_metrics.get("issues", []), 
            "recommendations": self._generate_data_recommendations(data_metrics), 
        }

        # บันทึกรายงาน
        report_path = os.path.join(self.output_dir, "data_quality_report.json")
        with open(report_path, "w", encoding = "utf - 8") as f:
            json.dump(report, f, ensure_ascii = False, indent = 2)

        return report

    def generate_html_dashboard(self, all_reports: Dict[str, Any]) -> str:
        """สร้าง Dashboard HTML ที่สวยงาม"""

        # เตรียมข้อมูลสำหรับ template
        performance_data = all_reports.get("performance", {})
        data_quality_data = all_reports.get("data_quality", {})
        execution_summary = performance_data.get("execution_summary", {})

        # Extract data for the HTML
        success_rate = execution_summary.get("success_rate", 100)
        execution_time = execution_summary.get("total_execution_time", 0)
        data_health_score = data_quality_data.get("data_health_score", 95)
        status = execution_summary.get("status", "สำเร็จ")
        successful_stages = execution_summary.get("successful_stages", "N/A")
        total_stages = execution_summary.get("total_stages", "N/A")

        html_content = f"""
<!DOCTYPE html>
<html lang = "th">
<head>
    <meta charset = "UTF - 8">
    <meta name = "viewport" content = "width = device - width, initial - scale = 1.0">
    <title>🏆 NICEGOLD Pipeline Dashboard</title>
    <style>
        body {{ font - family: Arial, sans - serif; margin: 20px; background: #f0f8ff; }}
        .container {{ max - width: 1000px; margin: 0 auto; background: white; padding: 30px; border - radius: 10px; }}
        .header {{ text - align: center; color: #2c3e50; margin - bottom: 30px; }}
        .stats {{ display: grid; grid - template - columns: repeat(auto - fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat - card {{ background: #f8f9fa; padding: 20px; border - radius: 8px; text - align: center; border - left: 4px solid #FFD700; }}
        .stat - value {{ font - size: 2em; font - weight: bold; color: #27ae60; }}
        .section {{ margin: 30px 0; padding: 20px; background: #f8f9fa; border - radius: 8px; }}
        .recommendations {{ background: #e8f5e8; padding: 15px; border - radius: 8px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class = "container">
        <div class = "header">
            <h1>🏆 NICEGOLD Pipeline Dashboard</h1>
            <p>รายงานผลการดำเนินงาน - {datetime.now().strftime('%Y - %m - %d %H:%M:%S')}</p>
        </div>

        <div class = "stats">
            <div class = "stat - card">
                <div class = "stat - value">{success_rate:.1f}%</div>
                <div>อัตราความสำเร็จ</div>
            </div>
            <div class = "stat - card">
                <div class = "stat - value">{execution_time:.1f}s</div>
                <div>เวลาการทำงาน</div>
            </div>
            <div class = "stat - card">
                <div class = "stat - value">{data_health_score:.1f}%</div>
                <div>คุณภาพข้อมูล</div>
            </div>
        </div>

        <div class = "section">
            <h2>📊 สรุปผลการดำเนินงาน</h2>
            <p><strong>สถานะ:</strong> {status}</p>
            <p><strong>ขั้นตอนที่สำเร็จ:</strong> {successful_stages}</p>
            <p><strong>ขั้นตอนทั้งหมด:</strong> {total_stages}</p>
        </div>

        <div class = "section">
            <h2>💡 คำแนะนำการปรับปรุง</h2>
            <div class = "recommendations">
                <p>• ระบบทำงานได้อย่างเหมาะสม ไม่มีข้อเสนอแนะเพิ่มเติม</p>
            </div>
        </div>

        <div class = "section">
            <h2>🎯 สรุป</h2>
            <p>ระบบ NICEGOLD Pipeline ทำงานเสร็จสมบูรณ์แล้ว พร้อมใช้งานสำหรับการเทรดทองคำอัจฉริยะ</p>
        </div>

        <div style = "text - align: center; margin - top: 30px; color: #7f8c8d;">
            <p>🚀 NICEGOLD ProjectP - Enhanced AI Trading System</p>
        </div>
    </div>
</body>
</html>
        """

        # บันทึกไฟล์ HTML
        html_path = os.path.join(self.output_dir, "pipeline_dashboard.html")
        with open(html_path, "w", encoding = "utf - 8") as f:
            f.write(html_content)

        return html_path

    def _generate_performance_recommendations(
        self, metrics: Dict[str, Any]
    ) -> List[str]:
        """สร้างคำแนะนำประสิทธิภาพ"""
        recommendations = []

        if metrics.get("total_time", 0) > 300:  # มากกว่า 5 นาที
            recommendations.append("ควรพิจารณาเพิ่มประสิทธิภาพการประมวลผล")

        if metrics.get("memory_usage", 0) > 80:  # ใช้ memory มากกว่า 80%
            recommendations.append("ควรเพิ่ม RAM หรือปรับปรุงการจัดการหน่วยความจำ")

        if not metrics.get("all_successful", True):
            recommendations.append("มีขั้นตอนที่ล้มเหลว ควรตรวจสอบ log เพื่อวิเคราะห์ปัญหา")

        if not recommendations:
            recommendations.append("ระบบทำงานได้อย่างเหมาะสม ไม่มีข้อเสนอแนะเพิ่มเติม")

        return recommendations

    def _generate_data_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """สร้างคำแนะนำคุณภาพข้อมูล"""
        recommendations = []

        completeness = metrics.get("completeness_score", 100)
        if completeness < 90:
            recommendations.append("ข้อมูลมีความครบถ้วนต่ำ ควรตรวจสอบการเก็บข้อมูล")

        missing_values = metrics.get("missing_values", 0)
        if missing_values > 0.1:  # มากกว่า 10%
            recommendations.append("พบข้อมูลสูญหายเกิน 10% ควรปรับปรุงการรวบรวมข้อมูล")

        if not recommendations:
            recommendations.append("คุณภาพข้อมูลอยู่ในเกณฑ์ที่ดี")

        return recommendations

    def _calculate_data_health_score(self, metrics: Dict[str, Any]) -> float:
        """คำนวณคะแนนสุขภาพข้อมูล"""

        # คะแนนพื้นฐาน
        base_score = 100.0

        # หักคะแนนจาก missing values
        missing_ratio = metrics.get("missing_values", 0)
        missing_penalty = missing_ratio * 50  # หัก 50 คะแนนต่อ 100% missing

        # หักคะแนนจาก outliers
        outlier_ratio = metrics.get("outliers", 0)
        outlier_penalty = outlier_ratio * 100 * 0.5

        score = base_score - missing_penalty - outlier_penalty

        return max(0, min(100, score))


# ฟังก์ชันสำหรับการใช้งาน
def create_enhanced_display() -> ThaiVisualDisplay:
    """สร้างระบบแสดงผลที่ปรับปรุงแล้ว"""
    return ThaiVisualDisplay()


def create_report_generator(
    output_dir: str = "output_default", 
) -> EnhancedReportGenerator:
    """สร้างระบบสร้างรายงาน"""
    return EnhancedReportGenerator(output_dir)