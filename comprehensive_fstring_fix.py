#!/usr/bin/env python3
"""
Comprehensive fix for f-string issues in pipeline_commands.py
This script identifies and fixes all the syntax issues in the f-string template
"""

import re


def fix_pipeline_fstring():
    file_path = "src/commands/pipeline_commands.py"

    print("Reading pipeline_commands.py...")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    print("Original content length:", len(content))

    # Find and replace the problematic f-string template
    # The issue is in the triple-quoted f-string that contains code for subprocess execution

    # The main issue is that there are incorrect brace escaping patterns
    # Let's fix the specific problematic areas step by step

    fixes_applied = 0

    # Fix 1: Missing closing brace in the missing files message
    old_pattern = r"print\(f'❌ Missing some files: \{\{.*?\}'\)"
    new_pattern = "print(f'❌ Missing some files: {\", \".join(missing_files)}')"
    if re.search(old_pattern, content):
        content = re.sub(old_pattern, new_pattern, content)
        fixes_applied += 1
        print("✓ Fixed missing files message")

    # Fix 2: Dictionary syntax issues in outputs
    old_pattern = r"'outputs': \{\{'processed_data': 'datacsv/processed_data\.csv'\}"
    new_pattern = "'outputs': {'processed_data': 'datacsv/processed_data.csv'}"
    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        fixes_applied += 1
        print("✓ Fixed outputs dictionary 1")

    # Fix 3: Fix data loading print statements
    print_fixes = [
        (
            r"print\(f'📊 Loaded processed data: \{\{len\(df\)\} rows'\)",
            "print(f'📊 Loaded processed data: {len(df)} rows')",
        ),
        (
            r"print\(f'📊 Loaded raw data: \{\{len\(df\)\} rows'\)",
            "print(f'📊 Loaded raw data: {len(df)} rows')",
        ),
        (
            r"print\(f'📊 Loaded M15 data: \{\{len\(df\)\} rows'\)",
            "print(f'📊 Loaded M15 data: {len(df)} rows')",
        ),
    ]

    for old, new in print_fixes:
        if re.search(old, content):
            content = re.sub(old, new, content)
            fixes_applied += 1
            print(f"✓ Fixed print statement: {old[:30]}...")

    # Fix 4: DataFrame creation syntax
    old_pattern = r"df = pd\.DataFrame\(\{\{"
    new_pattern = "df = pd.DataFrame({"
    if re.search(old_pattern, content):
        content = re.sub(old_pattern, new_pattern, content)
        fixes_applied += 1
        print("✓ Fixed DataFrame creation")

    # Fix 5: Closing braces for DataFrame
    old_pattern = r"\}\}\)"
    # Only replace the ones that are actually }}), not legitimate ones
    # We need to be more specific about this
    content = re.sub(
        r"'volume': np\.random\.randint\(100, 1000, 10000\)\s*\}\}\)",
        "'volume': np.random.randint(100, 1000, 10000)\n            })",
        content,
    )
    fixes_applied += 1
    print("✓ Fixed DataFrame closing braces")

    # Fix 6: More output dictionary fixes
    output_fixes = [
        (
            "'outputs': {{'feature_count': len(df.columns)}",
            "'outputs': {'feature_count': len(df.columns)}",
        ),
        (
            "'outputs': {{'model_file': 'results_model_object.pkl', 'results_file': 'results_model_data.pkl'}",
            "'outputs': {'model_file': 'results_model_object.pkl', 'results_file': 'results_model_data.pkl'}",
        ),
        (
            "'outputs': {{'best_params': optimization_results['best_params']}",
            "'outputs': {'best_params': optimization_results['best_params']}",
        ),
    ]

    for old, new in output_fixes:
        if old in content:
            content = content.replace(old, new)
            fixes_applied += 1
            print(f"✓ Fixed output dictionary: {old[:40]}...")

    # Fix 7: Stage 2 completion message
    old_pattern = r"print\(f'✅ Stage 2 completed: \{\{len\(df\)\} samples, \{\{len\(df\.columns\)\} features'\)"
    new_pattern = (
        "print(f'✅ Stage 2 completed: {len(df)} samples, {len(df.columns)} features')"
    )
    if re.search(old_pattern, content):
        content = re.sub(old_pattern, new_pattern, content)
        fixes_applied += 1
        print("✓ Fixed Stage 2 completion message")

    # Fix 8: Warning messages
    warning_fixes = [
        (
            r"print\(f'⚠️ Preprocessing warning: \{\{e\}\'\)",
            "print(f'⚠️ Preprocessing warning: {e}')",
        ),
        (
            r"print\(f'⚠️ Model training warning: \{\{e\}\'\)",
            "print(f'⚠️ Model training warning: {e}')",
        ),
        (
            r"print\(f'⚠️ Optimization warning: \{\{e\}\'\)",
            "print(f'⚠️ Optimization warning: {e}')",
        ),
        (
            r"print\(f'❌ Pipeline failed with error: \{\{e\}\'\)",
            "print(f'❌ Pipeline failed with error: {e}')",
        ),
    ]

    for old, new in warning_fixes:
        if re.search(old, content):
            content = re.sub(old, new, content)
            fixes_applied += 1
            print(f"✓ Fixed warning message: {old[:30]}...")

    # Fix 9: Metrics dictionary issues
    metrics_fixes = [
        (
            "'metrics': {{'accuracy': 0.75, 'f1_score': 0.70, 'train_samples': 8000, 'test_samples': 2000}",
            "'metrics': {'accuracy': 0.75, 'f1_score': 0.70, 'train_samples': 8000, 'test_samples': 2000}",
        ),
        (
            "'metrics': {{'accuracy': 0.65, 'f1_score': 0.60, 'train_samples': 5000, 'test_samples': 1000}",
            "'metrics': {'accuracy': 0.65, 'f1_score': 0.60, 'train_samples': 5000, 'test_samples': 1000}",
        ),
    ]

    for old, new in metrics_fixes:
        if old in content:
            content = content.replace(old, new)
            fixes_applied += 1
            print(f"✓ Fixed metrics dictionary: {old[:40]}...")

    # Fix 10: Best params dictionary
    old_pattern = r"'best_params': \{\{"
    new_pattern = "'best_params': {"
    if re.search(old_pattern, content):
        content = re.sub(old_pattern, new_pattern, content)
        fixes_applied += 1
        print("✓ Fixed best_params dictionary")

    # Fix 11: Pipeline info dictionary
    old_pattern = r"'pipeline_info': \{\{"
    new_pattern = "'pipeline_info': {"
    if re.search(old_pattern, content):
        content = re.sub(old_pattern, new_pattern, content)
        fixes_applied += 1
        print("✓ Fixed pipeline_info dictionary")

    # Fix 12: Dictionary closing issues - fix stray closing braces
    content = re.sub(
        r"        \},\s*'stage_results': stage_results",
        "        },\n        'stage_results': stage_results",
        content,
    )
    fixes_applied += 1
    print("✓ Fixed dictionary closing braces")

    print(f"\n📊 Applied {fixes_applied} fixes total")

    # Write the corrected content back
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Fixed f-string issues in {file_path}")
    print("New content length:", len(content))

    return content


def test_syntax(content):
    """Test if the fixed content has valid syntax"""
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
    fixed_content = fix_pipeline_fstring()
    test_syntax(fixed_content)
