#!/usr/bin/env python3
"""
Comprehensive fix for ProjectP.py indentation issues
"""

import ast
import re


def fix_indentation_errors(file_path):
    """Fix common indentation errors in Python file"""

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    fixed_lines = []
    in_except_block = False
    expect_indent = 0

    for i, line in enumerate(lines):
        original_line = line
        stripped = line.lstrip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith("#"):
            fixed_lines.append(line)
            continue

        # Get current indentation
        current_indent = len(line) - len(stripped)

        # Special handling for except blocks
        if stripped.startswith("except "):
            in_except_block = True
            # Find the matching try block indentation
            for j in range(i - 1, -1, -1):
                if lines[j].strip().startswith("try:"):
                    try_indent = len(lines[j]) - len(lines[j].lstrip())
                    expect_indent = try_indent
                    break
            fixed_lines.append(" " * expect_indent + stripped)
            continue

        elif in_except_block and stripped.startswith(
            ("if ", "else:", "print(", "logger.", "return")
        ):
            # Content inside except block should be indented relative to except
            content_indent = expect_indent + 4
            fixed_lines.append(" " * content_indent + stripped)
            continue

        elif stripped.startswith(
            (
                "def ",
                "class ",
                "try:",
                "if ",
                "elif ",
                "else:",
                "for ",
                "while ",
                "with ",
            )
        ):
            in_except_block = False

        # Regular line
        fixed_lines.append(line)

    return fixed_lines


def main():
    file_path = "/home/nicetpad2/nicegold_data/NICEGOLD-ProjectP/ProjectP.py"

    print("Analyzing ProjectP.py for indentation issues...")

    try:
        # First, try to parse the current file
        with open(file_path, "r") as f:
            content = f.read()

        ast.parse(content)
        print("✓ No syntax errors found!")
        return

    except SyntaxError as e:
        print(f"Found syntax error at line {e.lineno}: {e.msg}")

        # Read the file and identify problematic lines
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Look for common indentation patterns that need fixing
        fixes_needed = []

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Pattern 1: except block with wrong indentation
            if i < len(lines) - 1:
                current_line = line.rstrip()
                next_line = lines[i].rstrip() if i < len(lines) else ""

                # Check for except followed by incorrectly indented content
                if (
                    current_line.strip().startswith("except ")
                    and next_line
                    and not next_line.startswith(
                        " " * (len(current_line) - len(current_line.lstrip()) + 4)
                    )
                ):
                    fixes_needed.append((i, "Incorrect indentation after except"))

        print(f"Found {len(fixes_needed)} areas that need fixing:")
        for line_num, issue in fixes_needed:
            print(f"  Line {line_num}: {issue}")

        # Create a backup
        backup_path = file_path + ".backup"
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✓ Created backup: {backup_path}")

        # Apply automated fixes for common patterns
        fixed_content = content

        # Fix pattern: except block followed by incorrectly indented print
        pattern1 = re.compile(
            r'(\s+except [^:]+:\n)(\s+)print\(f"⚠️ Error: \{e\}"\)\n(\s+)if logger:',
            re.MULTILINE,
        )

        def fix_except_block(match):
            except_line = match.group(1)
            except_indent = len(except_line) - len(except_line.lstrip())
            content_indent = " " * (except_indent + 4)

            return (
                except_line
                + content_indent
                + 'print(f"⚠️ Error: {e}")\n'
                + content_indent
                + "if logger:"
            )

        fixed_content = pattern1.sub(fix_except_block, fixed_content)

        # Try to parse the fixed content
        try:
            ast.parse(fixed_content)
            print("✓ Automated fixes successful!")

            # Write the fixed content
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(fixed_content)
            print(f"✓ Applied fixes to {file_path}")

        except SyntaxError as e2:
            print(
                f"❌ Automated fixes didn't resolve all issues. Still have error at line {e2.lineno}: {e2.msg}"
            )
            print("Manual intervention may be required.")


if __name__ == "__main__":
    main()
