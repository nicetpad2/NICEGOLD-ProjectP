#!/usr/bin/env python3
"""
Fix f-string issues in pipeline_commands.py
This script fixes the f-string template issues where there are missing closing braces
"""

import re


def fix_fstring_issues():
    file_path = "src/commands/pipeline_commands.py"

    print("Reading pipeline_commands.py...")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    print("Original content length:", len(content))

    # Find the f-string template and fix the issues
    fixes = [
        # Fix missing closing braces in f-string expressions
        (
            r'print\(f\'❌ Missing some files: \{\{", ".join\(missing_files\)\}\}\'\)',
            r'print(f\'❌ Missing some files: {", ".join(missing_files)}\')',
        ),
        (
            r"\'outputs\': \{\{\'processed_data\': \'datacsv/processed_data\.csv\'\}",
            r"\'outputs\': {\'processed_data\': \'datacsv/processed_data.csv\'}",
        ),
        (
            r"print\(f\'📊 Loaded processed data: \{\{len\(df\)\} rows\'\)",
            r"print(f\'📊 Loaded processed data: {len(df)} rows\')",
        ),
        (
            r"print\(f\'📊 Loaded raw data: \{\{len\(df\)\} rows\'\)",
            r"print(f\'📊 Loaded raw data: {len(df)} rows\')",
        ),
        (
            r"print\(f\'📊 Loaded M15 data: \{\{len\(df\)\} rows\'\)",
            r"print(f\'📊 Loaded M15 data: {len(df)} rows\')",
        ),
        (r"df = pd\.DataFrame\(\{\{", r"df = pd.DataFrame({"),
        (r"\}\}\)", r"})"),
        (
            r"\'outputs\': \{\{\'feature_count\': len\(df\.columns\)\}",
            r"\'outputs\': {\'feature_count\': len(df.columns)}",
        ),
        (
            r"print\(f\'✅ Stage 2 completed: \{\{len\(df\)\} samples, \{\{len\(df\.columns\)\} features\'\)",
            r"print(f\'✅ Stage 2 completed: {len(df)} samples, {len(df.columns)} features\')",
        ),
        (
            r"print\(f\'⚠️ Preprocessing warning: \{\{e\}\'\)",
            r"print(f\'⚠️ Preprocessing warning: {e}\')",
        ),
        (
            r"print\(f\'⚠️ Model training warning: \{\{e\}\'\)",
            r"print(f\'⚠️ Model training warning: {e}\')",
        ),
        (
            r"\'outputs\': \{\{\'model_file\': \'results_model_object\.pkl\', \'results_file\': \'results_model_data\.pkl\'\}",
            r"\'outputs\': {\'model_file\': \'results_model_object.pkl\', \'results_file\': \'results_model_data.pkl\'}",
        ),
        (
            r"\'metrics\': \{\{\'accuracy\': 0\.75, \'f1_score\': 0\.70, \'train_samples\': 8000, \'test_samples\': 2000\}",
            r"\'metrics\': {\'accuracy\': 0.75, \'f1_score\': 0.70, \'train_samples\': 8000, \'test_samples\': 2000}",
        ),
        (
            r"\'metrics\': \{\{\'accuracy\': 0\.65, \'f1_score\': 0\.60, \'train_samples\': 5000, \'test_samples\': 1000\}",
            r"\'metrics\': {\'accuracy\': 0.65, \'f1_score\': 0.60, \'train_samples\': 5000, \'test_samples\': 1000}",
        ),
        (r"\'best_params\': \{\{", r"\'best_params\': {"),
        (
            r"\'outputs\': \{\{\'best_params\': optimization_results\[\'best_params\'\]\}",
            r"\'outputs\': {\'best_params\': optimization_results[\'best_params\']}",
        ),
        (
            r"print\(f\'⚠️ Optimization warning: \{\{e\}\'\)",
            r"print(f\'⚠️ Optimization warning: {e}\')",
        ),
        (r"\'pipeline_info\': \{\{", r"\'pipeline_info\': {"),
        (r"        \},", r"        },"),
        (
            r"print\(f\'❌ Pipeline failed with error: \{\{e\}\'\)",
            r"print(f\'❌ Pipeline failed with error: {e}\')",
        ),
    ]

    print(f"Applying {len(fixes)} fixes...")

    modified = False
    for i, (pattern, replacement) in enumerate(fixes):
        old_content = content
        content = re.sub(pattern, replacement, content)
        if content != old_content:
            print(f"✓ Applied fix {i+1}: {pattern[:50]}...")
            modified = True
        else:
            print(f"- Fix {i+1} not needed: {pattern[:50]}...")

    if modified:
        # Write the corrected content back
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Fixed f-string issues in {file_path}")
        print("New content length:", len(content))
    else:
        print("❌ No modifications made")

    # Test the syntax
    print("\n🔍 Testing syntax...")
    try:
        import ast

        ast.parse(content)
        print("✅ Syntax is now valid!")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error still exists: {e}")
        print(f"Error at line {e.lineno}: {e.text}")
        return False


if __name__ == "__main__":
    fix_fstring_issues()
