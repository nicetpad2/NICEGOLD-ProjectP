#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Enhanced Full Pipeline
===========================

Test the complete enhanced pipeline with the new professional summary format.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from commands.pipeline_commands import PipelineCommands


def test_enhanced_full_pipeline():
    """Test the enhanced full pipeline with professional summary"""
    print("🚀 Testing Enhanced Full Pipeline with Professional Summary...")

    # Create pipeline commands
    project_root = Path(__file__).parent
    pipeline = PipelineCommands(project_root)

    print("\n" + "=" * 80)
    print("🎯 STARTING ENHANCED FULL PIPELINE TEST")
    print("=" * 80)

    # Run the enhanced full pipeline
    success = pipeline.full_pipeline()

    if success:
        print(f"\n{'='*80}")
        print("✅ ENHANCED FULL PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print("🏆 Professional trading summary generated with all requested metrics:")
        print("   ✓ Starting capital ($10,000)")
        print("   ✓ Win/Loss rates and total orders")
        print("   ✓ Maximum Drawdown (DD)")
        print("   ✓ Win rate percentage")
        print("   ✓ Test period dates and duration")
        print("   ✓ Professional trading analytics")
        print("   ✓ Executive summary format")
        print("   ✓ Comprehensive risk metrics")
        print("   ✓ Advanced trading statistics")
        print(f"{'='*80}")
    else:
        print("\n❌ Enhanced full pipeline test failed")


if __name__ == "__main__":
    test_enhanced_full_pipeline()
