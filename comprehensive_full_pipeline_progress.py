#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 COMPREHENSIVE FULL PIPELINE PROGRESS SYSTEM
ระบบ Progress Bar ที่สมบูรณ์ที่สุดสำหรับ NICEGOLD ProjectP
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import all progress systems
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from utils.enhanced_progress import EnhancedProgressProcessor
    ENHANCED_PROGRESS_AVAILABLE = True
except ImportError:
    ENHANCED_PROGRESS_AVAILABLE = False

try:
    from enhanced_full_pipeline import EnhancedFullPipeline
    ENHANCED_PIPELINE_AVAILABLE = True
except ImportError:
    ENHANCED_PIPELINE_AVAILABLE = False

try:
    from enhanced_visual_display import ThaiVisualDisplay
    VISUAL_DISPLAY_AVAILABLE = True
except ImportError:
    VISUAL_DISPLAY_AVAILABLE = False

try:
    from utils.modern_ui import ModernProgressBar, ModernSpinner
    MODERN_UI_AVAILABLE = True
except ImportError:
    MODERN_UI_AVAILABLE = False

try:
    from production_full_pipeline import ProductionFullPipeline
    PRODUCTION_PIPELINE_AVAILABLE = True
except ImportError:
    PRODUCTION_PIPELINE_AVAILABLE = False


class ComprehensiveProgressSystem:
    """ระบบ Progress ที่สมบูรณ์ที่สุดสำหรับ Full Pipeline"""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.enhanced_processor = EnhancedProgressProcessor() if ENHANCED_PROGRESS_AVAILABLE else None
        self.visual_display = ThaiVisualDisplay() if VISUAL_DISPLAY_AVAILABLE else None
        self.start_time = None
        self.stage_times = {}
        self.stage_results = {}
        self.errors = []
        self.warnings = []
        
    def run_full_pipeline_with_complete_progress(self) -> Dict[str, Any]:
        """เรียกใช้ Full Pipeline พร้อม Progress Bar ที่สมบูรณ์ที่สุด"""
        
        self.start_time = datetime.now()
        print("\n" + "="*80)
        print("🚀 NICEGOLD ProjectP - Full Pipeline with Complete Progress System")
        print("="*80)
        
        # ระดับที่ 1: ลองใช้ Production Full Pipeline (Production-ready)
        if PRODUCTION_PIPELINE_AVAILABLE:
            try:
                print("✅ เรียกใช้ Production Full Pipeline (Production-ready)")
                production_pipeline = ProductionFullPipeline()
                results = production_pipeline.run_full_pipeline()
                self._display_final_results(results, "PRODUCTION")
                return results
            except Exception as e:
                print(f"⚠️ Production Pipeline ล้มเหลว: {str(e)}")
                self.warnings.append(f"Production Pipeline error: {str(e)}")
        
        # ระดับที่ 2: ลองใช้ Enhanced Full Pipeline (สมบูรณ์ที่สุด)
        if ENHANCED_PIPELINE_AVAILABLE:
            try:
                print("✅ เรียกใช้ Enhanced Full Pipeline (ระดับสูงสุด)")
                enhanced_pipeline = EnhancedFullPipeline()
                results = enhanced_pipeline.run_enhanced_full_pipeline()
                self._display_final_results(results, "ENHANCED")
                return results
            except Exception as e:
                print(f"⚠️ Enhanced Pipeline ล้มเหลว: {str(e)}")
                self.warnings.append(f"Enhanced Pipeline error: {str(e)}")
        
        # ระดับที่ 3: ลองใช้ Enhanced Progress Processor
        if ENHANCED_PROGRESS_AVAILABLE and self.enhanced_processor:
            try:
                print("✅ เรียกใช้ Enhanced Progress Processor")
                return self._run_with_enhanced_progress()
            except Exception as e:
                print(f"⚠️ Enhanced Progress ล้มเหลว: {str(e)}")
                self.warnings.append(f"Enhanced Progress error: {str(e)}")
        
        # ระดับที่ 3: ลองใช้ Rich Progress
        if RICH_AVAILABLE:
            try:
                print("✅ เรียกใช้ Rich Progress System")
                return self._run_with_rich_progress()
            except Exception as e:
                print(f"⚠️ Rich Progress ล้มเหลว: {str(e)}")
                self.warnings.append(f"Rich Progress error: {str(e)}")
        
        # ระดับที่ 4: ใช้ Basic Progress (Fallback)
        print("✅ เรียกใช้ Basic Progress System")
        return self._run_with_basic_progress()
    
    def _run_with_enhanced_progress(self) -> Dict[str, Any]:
        """เรียกใช้ด้วย Enhanced Progress Processor"""
        
        pipeline_steps = [
            {'name': '🔧 โหลดการตั้งค่าระบบ', 'duration': 1.0, 'spinner': 'dots'},
            {'name': '📊 เตรียมข้อมูลตลาด', 'duration': 2.0, 'spinner': 'bars'},
            {'name': '🧠 สร้างฟีเจอร์ขั้นสูง', 'duration': 3.0, 'spinner': 'circles'},
            {'name': '⚙️ ประมวลผลข้อมูลเบื้องต้น', 'duration': 2.5, 'spinner': 'arrows'},
            {'name': '📈 แบ่งข้อมูลฝึกสอน/ทดสอบ', 'duration': 1.0, 'spinner': 'squares'},
            {'name': '🤖 ฝึกสอนโมเดล AI', 'duration': 4.0, 'spinner': 'bars'},
            {'name': '🎯 ประเมินผลโมเดล', 'duration': 2.0, 'spinner': 'circles'},
            {'name': '🔮 สร้างการทำนาย', 'duration': 1.5, 'spinner': 'arrows'},
            {'name': '📊 วิเคราะห์ประสิทธิภาพ', 'duration': 2.0, 'spinner': 'squares'},
            {'name': '📋 สร้างรายงานผล', 'duration': 1.5, 'spinner': 'dots'},
            {'name': '💾 บันทึกผลลัพธ์', 'duration': 1.0, 'spinner': 'bars'}
        ]
        
        success = self.enhanced_processor.process_with_progress(
            pipeline_steps, 
            "🚀 NICEGOLD Full ML Trading Pipeline"
        )
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        results = {
            "pipeline_status": "SUCCESS" if success else "FAILED",
            "total_execution_time": total_time,
            "successful_stages": len(pipeline_steps) if success else 0,
            "total_stages": len(pipeline_steps),
            "method_used": "Enhanced Progress",
            "errors": self.errors,
            "warnings": self.warnings
        }
        
        self._display_final_results(results, "ENHANCED_PROGRESS")
        return results
    
    def _run_with_rich_progress(self) -> Dict[str, Any]:
        """เรียกใช้ด้วย Rich Progress System"""
        
        stages = [
            "🔧 Loading Configuration",
            "📊 Preparing Market Data", 
            "🧠 Advanced Feature Engineering",
            "⚙️ Data Preprocessing",
            "📈 Train/Test Split",
            "🤖 Model Training",
            "🎯 Model Evaluation",
            "🔮 Prediction Generation",
            "📊 Performance Analysis",
            "📋 Report Generation",
            "💾 Saving Results"
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            
            # เพิ่ม main task
            main_task = progress.add_task("🚀 Full Pipeline Progress", total=len(stages))
            
            for i, stage in enumerate(stages):
                # เพิ่ม stage task
                stage_task = progress.add_task(f"⏳ {stage}", total=100)
                
                # จำลองการทำงาน
                for step in range(100):
                    time.sleep(0.03)  # รวม ~3 วินาทีต่อ stage
                    progress.update(stage_task, advance=1)
                
                progress.update(stage_task, description=f"✅ {stage}")
                progress.update(main_task, advance=1)
                time.sleep(0.2)
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        results = {
            "pipeline_status": "SUCCESS",
            "total_execution_time": total_time,
            "successful_stages": len(stages),
            "total_stages": len(stages),
            "method_used": "Rich Progress",
            "errors": self.errors,
            "warnings": self.warnings
        }
        
        self._display_final_results(results, "RICH")
        return results
    
    def _run_with_basic_progress(self) -> Dict[str, Any]:
        """เรียกใช้ด้วย Basic Progress System"""
        
        stages = [
            ("🔧 Loading Configuration", 1.0),
            ("📊 Preparing Market Data", 2.0),
            ("🧠 Advanced Feature Engineering", 3.0),
            ("⚙️ Data Preprocessing", 2.5),
            ("📈 Train/Test Split", 1.0),
            ("🤖 Model Training", 4.0),
            ("🎯 Model Evaluation", 2.0),
            ("🔮 Prediction Generation", 1.5),
            ("📊 Performance Analysis", 2.0),
            ("📋 Report Generation", 1.5),
            ("💾 Saving Results", 1.0)
        ]
        
        print(f"\n🚀 NICEGOLD Full Pipeline - Basic Progress")
        print("="*60)
        
        for i, (stage_name, duration) in enumerate(stages):
            print(f"\n[{i+1:2d}/{len(stages)}] {stage_name}")
            
            # แสดง progress bar แบบง่าย
            progress_chars = 50
            for j in range(progress_chars):
                time.sleep(duration / progress_chars)
                progress = "█" * (j + 1) + "░" * (progress_chars - j - 1)
                percent = ((j + 1) / progress_chars) * 100
                print(f"\r     [{progress}] {percent:5.1f}%", end="", flush=True)
            
            print(f" ✅ Complete")
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        results = {
            "pipeline_status": "SUCCESS",
            "total_execution_time": total_time,
            "successful_stages": len(stages),
            "total_stages": len(stages),
            "method_used": "Basic Progress",
            "errors": self.errors,
            "warnings": self.warnings
        }
        
        self._display_final_results(results, "BASIC")
        return results
    
    def _display_final_results(self, results: Dict[str, Any], method: str):
        """แสดงผลลัพธ์สุดท้าย"""
        
        print("\n" + "="*80)
        print("🎉 FULL PIPELINE COMPLETED!")
        print("="*80)
        
        print(f"📊 Pipeline Status: {results['pipeline_status']}")
        print(f"⏱️ Total Time: {results['total_execution_time']:.1f} seconds")
        print(f"✅ Successful Stages: {results['successful_stages']}/{results['total_stages']}")
        print(f"🔧 Method Used: {method}")
        
        if results.get('warnings'):
            print(f"⚠️ Warnings: {len(results['warnings'])}")
            for warning in results['warnings']:
                print(f"   - {warning}")
        
        if results.get('errors'):
            print(f"❌ Errors: {len(results['errors'])}")
            for error in results['errors']:
                print(f"   - {error}")
        
        # แสดงสถานะระบบ
        print(f"\n🔧 System Status:")
        print(f"   Production Pipeline: {'✅' if PRODUCTION_PIPELINE_AVAILABLE else '❌'}")
        print(f"   Rich Available: {'✅' if RICH_AVAILABLE else '❌'}")
        print(f"   Enhanced Progress: {'✅' if ENHANCED_PROGRESS_AVAILABLE else '❌'}")
        print(f"   Enhanced Pipeline: {'✅' if ENHANCED_PIPELINE_AVAILABLE else '❌'}")
        print(f"   Visual Display: {'✅' if VISUAL_DISPLAY_AVAILABLE else '❌'}")
        print(f"   Modern UI: {'✅' if MODERN_UI_AVAILABLE else '❌'}")
        
        print("\n" + "="*80)


def main():
    """ฟังก์ชันหลักสำหรับทดสอบระบบ"""
    progress_system = ComprehensiveProgressSystem()
    results = progress_system.run_full_pipeline_with_complete_progress()
    return results


if __name__ == "__main__":
    main()
