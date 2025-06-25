#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix Double Braces in Pipeline Commands
"""

import re
from pathlib import Path


def fix_double_braces():
    """Fix double braces in pipeline_commands.py"""
    file_path = Path("src/commands/pipeline_commands.py")

    print("🔧 Fixing double braces in pipeline_commands.py...")

    try:
        # Read the file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Replace double braces with single braces
        # But be careful not to replace legitimate double braces in f-strings
        fixes = [
            ("stage_results = {{}}", "stage_results = {}"),
            (
                "stage_results['environment_setup'] = {{",
                "stage_results['environment_setup'] = {",
            ),
            ("preprocessing_metrics = {{", "preprocessing_metrics = {"),
            (
                "stage_results['preprocessing'] = {{",
                "stage_results['preprocessing'] = {",
            ),
            ("model_metrics = {{", "model_metrics = {"),
            ("model_results = {{", "model_results = {"),
            (
                "stage_results['model_training'] = {{",
                "stage_results['model_training'] = {",
            ),
            ("optimization_results = {{", "optimization_results = {"),
            ("stage_results['optimization'] = {{", "stage_results['optimization'] = {"),
            ("final_results = {{", "final_results = {"),
            ("        }}", "        }"),
            ("    }}", "    }"),
            ("}}", "}"),
        ]

        # Apply fixes
        for old, new in fixes:
            content = content.replace(old, new)

        # Write back
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print("✅ Double braces fixed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error fixing double braces: {e}")
        return False


if __name__ == "__main__":
    fix_double_braces()
