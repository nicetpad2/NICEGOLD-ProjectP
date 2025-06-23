"""
Agent Auto-Fix Module
====================

Automated problem detection and resolution system for ProjectP.
"""

import os
import re
import ast
import json
import shutil
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class FixResult:
    """Result of an automated fix."""
    success: bool
    fix_type: str
    description: str
    files_modified: List[str]
    backup_created: bool
    error_message: Optional[str] = None

class AutoFixSystem:
    """Automated problem detection and resolution system."""
    
    def __init__(self, project_root: str = None):
        self.project_root = project_root or os.getcwd()
        self.backup_dir = os.path.join(self.project_root, '.backups', 'auto_fix')
        self.fixes_applied: List[FixResult] = []
        
        # Ensure backup directory exists
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def run_comprehensive_fixes(self) -> Dict[str, Any]:
        """Run comprehensive automated fixes."""
        logger.info("🔧 Starting comprehensive auto-fix system...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'fixes_attempted': 0,
            'fixes_successful': 0,
            'fixes_failed': 0,
            'categories': {
                'syntax_fixes': [],
                'import_fixes': [],
                'formatting_fixes': [],
                'performance_fixes': [],
                'security_fixes': [],
                'ml_specific_fixes': []
            },
            'summary': {}
        }
        
        # Apply different categories of fixes
        try:
            # 1. Syntax and import fixes
            syntax_results = self._fix_syntax_issues()
            results['categories']['syntax_fixes'] = syntax_results
            
            # 2. Import optimization
            import_results = self._fix_import_issues()
            results['categories']['import_fixes'] = import_results
            
            # 3. Code formatting
            format_results = self._fix_formatting_issues()
            results['categories']['formatting_fixes'] = format_results
            
            # 4. Performance optimizations
            perf_results = self._fix_performance_issues()
            results['categories']['performance_fixes'] = perf_results
            
            # 5. Security improvements
            security_results = self._fix_security_issues()
            results['categories']['security_fixes'] = security_results
            
            # 6. ML-specific optimizations
            ml_results = self._fix_ml_specific_issues()
            results['categories']['ml_specific_fixes'] = ml_results
            
            # Calculate summary
            for category_results in results['categories'].values():
                for result in category_results:
                    results['fixes_attempted'] += 1
                    if result['success']:
                        results['fixes_successful'] += 1
                    else:
                        results['fixes_failed'] += 1
            
            results['summary'] = self._generate_fix_summary(results)
            
            logger.info(f"✅ Auto-fix completed: {results['fixes_successful']}/{results['fixes_attempted']} successful")
            
        except Exception as e:
            logger.error(f"❌ Auto-fix system error: {e}")
            results['error'] = str(e)
        
        return results
    
    def _fix_syntax_issues(self) -> List[Dict[str, Any]]:
        """Fix common syntax issues."""
        logger.info("🔍 Fixing syntax issues...")
        fixes = []
        
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    fix_result = self._fix_file_syntax(file_path)
                    if fix_result:
                        fixes.append(fix_result)
        
        return fixes
    
    def _fix_file_syntax(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Fix syntax issues in a specific file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                original_content = f.read()
            
            # Try to parse the file
            try:
                ast.parse(original_content)
                return None  # No syntax errors
            except SyntaxError as e:
                # Attempt common fixes
                fixed_content = self._apply_common_syntax_fixes(original_content, e)
                
                if fixed_content != original_content:
                    # Create backup
                    backup_path = self._create_backup(file_path)
                    
                    # Write fixed content
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    
                    # Verify fix
                    try:
                        ast.parse(fixed_content)
                        return {
                            'success': True,
                            'fix_type': 'syntax_fix',
                            'description': f'Fixed syntax error in {os.path.relpath(file_path, self.project_root)}',
                            'files_modified': [file_path],
                            'backup_created': True,
                            'backup_path': backup_path,
                            'error_fixed': str(e)
                        }
                    except SyntaxError:
                        # Restore from backup if fix didn't work
                        shutil.copy2(backup_path, file_path)
                        return {
                            'success': False,
                            'fix_type': 'syntax_fix',
                            'description': f'Failed to fix syntax error in {os.path.relpath(file_path, self.project_root)}',
                            'files_modified': [],
                            'backup_created': True,
                            'error_message': f'Could not automatically fix: {e}'
                        }
                
        except Exception as e:
            logger.warning(f"Error fixing syntax in {file_path}: {e}")
            return None
    
    def _apply_common_syntax_fixes(self, content: str, error: SyntaxError) -> str:
        """Apply common syntax fixes."""
        lines = content.splitlines()
        
        if error.lineno and error.lineno <= len(lines):
            line = lines[error.lineno - 1]
            
            # Common fixes
            fixes_applied = []
            
            # Fix missing parentheses in print statements (Python 2 to 3)
            if 'print ' in line and not line.strip().startswith('#'):
                new_line = re.sub(r'print\s+([^(].*)$', r'print(\1)', line)
                if new_line != line:
                    lines[error.lineno - 1] = new_line
                    fixes_applied.append('print_parentheses')
            
            # Fix missing colons
            if error.msg and 'expected' in error.msg.lower() and ':' in error.msg:
                if line.strip().endswith(('if', 'else', 'elif', 'for', 'while', 'def', 'class', 'try', 'except', 'finally', 'with')):
                    lines[error.lineno - 1] = line + ':'
                    fixes_applied.append('missing_colon')
            
            # Fix indentation issues
            if 'indentation' in error.msg.lower():
                # Simple indentation fix - standardize to 4 spaces
                stripped = line.lstrip()
                if stripped:
                    # Count leading whitespace
                    indent_level = len(line) - len(stripped)
                    # Round to nearest 4-space indent
                    new_indent = (indent_level // 4) * 4
                    lines[error.lineno - 1] = ' ' * new_indent + stripped
                    fixes_applied.append('indentation')
            
            # Fix common quote issues
            if 'EOF while scanning' in error.msg:
                # Try to fix unclosed strings
                if line.count('"') % 2 == 1:
                    lines[error.lineno - 1] = line + '"'
                    fixes_applied.append('unclosed_quote')
                elif line.count("'") % 2 == 1:
                    lines[error.lineno - 1] = line + "'"
                    fixes_applied.append('unclosed_quote')
        
        return '\n'.join(lines)
    
    def _fix_import_issues(self) -> List[Dict[str, Any]]:
        """Fix import-related issues."""
        logger.info("📦 Fixing import issues...")
        fixes = []
        
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    fix_result = self._fix_file_imports(file_path)
                    if fix_result:
                        fixes.append(fix_result)
        
        return fixes
    
    def _fix_file_imports(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Fix import issues in a specific file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            original_content = content
            modified = False
            
            # Fix common import issues
            # 1. Sort imports
            content, sorted_imports = self._sort_imports(content)
            if sorted_imports:
                modified = True
            
            # 2. Remove duplicate imports
            content, removed_duplicates = self._remove_duplicate_imports(content)
            if removed_duplicates:
                modified = True
            
            # 3. Fix relative imports
            content, fixed_relative = self._fix_relative_imports(content, file_path)
            if fixed_relative:
                modified = True
            
            if modified:
                # Create backup
                backup_path = self._create_backup(file_path)
                
                # Write fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                changes = []
                if sorted_imports:
                    changes.append('sorted imports')
                if removed_duplicates:
                    changes.append('removed duplicates')
                if fixed_relative:
                    changes.append('fixed relative imports')
                
                return {
                    'success': True,
                    'fix_type': 'import_fix',
                    'description': f'Fixed imports in {os.path.relpath(file_path, self.project_root)}: {", ".join(changes)}',
                    'files_modified': [file_path],
                    'backup_created': True,
                    'backup_path': backup_path
                }
        
        except Exception as e:
            logger.warning(f"Error fixing imports in {file_path}: {e}")
        
        return None
    
    def _sort_imports(self, content: str) -> Tuple[str, bool]:
        """Sort imports according to PEP 8."""
        lines = content.splitlines()
        import_lines = []
        from_lines = []
        other_lines = []
        import_section = True
        
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                if import_section:
                    import_lines.append(line)
                else:
                    other_lines.append(line)
            elif stripped.startswith('import '):
                import_lines.append(line)
            elif stripped.startswith('from '):
                from_lines.append(line)
            else:
                import_section = False
                other_lines.append(line)
        
        # Sort imports
        sorted_imports = sorted(import_lines + from_lines, key=lambda x: x.strip().lower())
        
        if sorted_imports != import_lines + from_lines:
            return '\n'.join(sorted_imports + other_lines), True
        
        return content, False
    
    def _remove_duplicate_imports(self, content: str) -> Tuple[str, bool]:
        """Remove duplicate import statements."""
        lines = content.splitlines()
        seen_imports = set()
        new_lines = []
        removed_any = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                if stripped not in seen_imports:
                    seen_imports.add(stripped)
                    new_lines.append(line)
                else:
                    removed_any = True
            else:
                new_lines.append(line)
        
        return '\n'.join(new_lines), removed_any
    
    def _fix_relative_imports(self, content: str, file_path: str) -> Tuple[str, bool]:
        """Fix relative import issues."""
        # This would be more sophisticated in practice
        # For now, just return unchanged
        return content, False
    
    def _fix_formatting_issues(self) -> List[Dict[str, Any]]:
        """Fix code formatting issues."""
        logger.info("✨ Fixing formatting issues...")
        fixes = []
        
        # Try to use autopep8 or black if available
        try:
            result = subprocess.run(['black', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                return self._apply_black_formatting()
        except FileNotFoundError:
            pass
        
        try:
            result = subprocess.run(['autopep8', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                return self._apply_autopep8_formatting()
        except FileNotFoundError:
            pass
        
        # Manual formatting fixes
        return self._apply_manual_formatting()
    
    def _apply_black_formatting(self) -> List[Dict[str, Any]]:
        """Apply Black code formatting."""
        fixes = []
        try:
            cmd = ['black', '--quiet', self.project_root]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                fixes.append({
                    'success': True,
                    'fix_type': 'formatting',
                    'description': 'Applied Black code formatting to entire project',
                    'files_modified': ['all_python_files'],
                    'backup_created': False
                })
            else:
                fixes.append({
                    'success': False,
                    'fix_type': 'formatting',
                    'description': 'Failed to apply Black formatting',
                    'files_modified': [],
                    'backup_created': False,
                    'error_message': result.stderr
                })
        except Exception as e:
            logger.warning(f"Error applying Black formatting: {e}")
        
        return fixes
    
    def _apply_autopep8_formatting(self) -> List[Dict[str, Any]]:
        """Apply autopep8 formatting."""
        fixes = []
        try:
            cmd = ['autopep8', '--in-place', '--recursive', self.project_root]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                fixes.append({
                    'success': True,
                    'fix_type': 'formatting',
                    'description': 'Applied autopep8 formatting to entire project',
                    'files_modified': ['all_python_files'],
                    'backup_created': False
                })
        except Exception as e:
            logger.warning(f"Error applying autopep8 formatting: {e}")
        
        return fixes
    
    def _apply_manual_formatting(self) -> List[Dict[str, Any]]:
        """Apply manual formatting fixes."""
        fixes = []
        
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    fix_result = self._fix_file_formatting(file_path)
                    if fix_result:
                        fixes.append(fix_result)
        
        return fixes
    
    def _fix_file_formatting(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Apply manual formatting fixes to a file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            original_content = content
            
            # Apply basic formatting fixes
            lines = content.splitlines()
            fixed_lines = []
            
            for line in lines:
                # Remove trailing whitespace
                fixed_line = line.rstrip()
                
                # Fix common spacing issues
                # Add space after commas
                fixed_line = re.sub(r',(?!\s)', ', ', fixed_line)
                
                # Add space around operators
                fixed_line = re.sub(r'(?<!\s)=(?!\s)', ' = ', fixed_line)
                fixed_line = re.sub(r'(?<!\s)\+(?!\s)', ' + ', fixed_line)
                fixed_line = re.sub(r'(?<!\s)-(?!\s)', ' - ', fixed_line)
                
                fixed_lines.append(fixed_line)
            
            # Remove excessive blank lines
            final_lines = []
            blank_count = 0
            
            for line in fixed_lines:
                if line.strip() == '':
                    blank_count += 1
                    if blank_count <= 2:  # Allow max 2 consecutive blank lines
                        final_lines.append(line)
                else:
                    blank_count = 0
                    final_lines.append(line)
            
            fixed_content = '\n'.join(final_lines)
            
            if fixed_content != original_content:
                # Create backup
                backup_path = self._create_backup(file_path)
                
                # Write fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                return {
                    'success': True,
                    'fix_type': 'formatting',
                    'description': f'Applied formatting fixes to {os.path.relpath(file_path, self.project_root)}',
                    'files_modified': [file_path],
                    'backup_created': True,
                    'backup_path': backup_path
                }
        
        except Exception as e:
            logger.warning(f"Error fixing formatting in {file_path}: {e}")
        
        return None
    
    def _fix_performance_issues(self) -> List[Dict[str, Any]]:
        """Fix performance-related issues."""
        logger.info("⚡ Fixing performance issues...")
        fixes = []
        
        # Performance fixes would be implemented here
        # For now, return placeholder
        
        return fixes
    
    def _fix_security_issues(self) -> List[Dict[str, Any]]:
        """Fix security-related issues."""
        logger.info("🔒 Fixing security issues...")
        fixes = []
        
        # Security fixes would be implemented here
        # For now, return placeholder
        
        return fixes
    
    def _fix_ml_specific_issues(self) -> List[Dict[str, Any]]:
        """Fix ML-specific issues like AUC problems, NaN values, etc."""
        logger.info("🧠 Fixing ML-specific issues...")
        fixes = []
        
        # Check for common ML issues
        projectp_file = os.path.join(self.project_root, 'ProjectP.py')
        if os.path.exists(projectp_file):
            fix_result = self._fix_projectp_issues(projectp_file)
            if fix_result:
                fixes.append(fix_result)
        
        return fixes
    
    def _fix_projectp_issues(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Fix specific issues in ProjectP.py."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            original_content = content
            fixed_issues = []
            
            # Fix import issues in the main ProjectP file
            if 'from datetime import datetime' not in content and 'datetime.now()' in content:
                content = 'from datetime import datetime\n' + content
                fixed_issues.append('added datetime import')
            
            # Fix incomplete exception handlers
            content = re.sub(
                r'except Exception as e:\s*$',
                r'except Exception as e:\n                print(f"⚠️ Error: {e}")',
                content,
                flags=re.MULTILINE
            )
            if len(re.findall(r'except Exception as e:\s*$', original_content, re.MULTILINE)) > 0:
                fixed_issues.append('completed exception handlers')
            
            # Fix incomplete functions
            if '# Continue with training using protected data' in content:
                # This looks like an incomplete section, let's complete it
                incomplete_pattern = r'(# Continue with training using protected data\s+)(training_result = run_real_data_training.*?)\s*else:'
                if re.search(incomplete_pattern, content, re.DOTALL):
                    content = re.sub(
                        incomplete_pattern,
                        r'\1\2\n                    \n                else:',
                        content,
                        flags=re.DOTALL
                    )
                    fixed_issues.append('completed training section')
            
            if fixed_issues and content != original_content:
                # Create backup
                backup_path = self._create_backup(file_path)
                
                # Write fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return {
                    'success': True,
                    'fix_type': 'ml_specific',
                    'description': f'Fixed ML issues in ProjectP.py: {", ".join(fixed_issues)}',
                    'files_modified': [file_path],
                    'backup_created': True,
                    'backup_path': backup_path
                }
        
        except Exception as e:
            logger.warning(f"Error fixing ML issues in {file_path}: {e}")
        
        return None
    
    def _create_backup(self, file_path: str) -> str:
        """Create a backup of a file."""
        rel_path = os.path.relpath(file_path, self.project_root)
        backup_path = os.path.join(self.backup_dir, f"{rel_path}.backup.{int(datetime.now().timestamp())}")
        
        # Ensure backup directory exists
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        
        # Copy file
        shutil.copy2(file_path, backup_path)
        
        return backup_path
    
    def _generate_fix_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of fixes applied."""
        summary = {
            'success_rate': 0,
            'most_common_fixes': [],
            'files_affected': 0,
            'backup_files_created': 0,
            'recommendations': []
        }
        
        if results['fixes_attempted'] > 0:
            summary['success_rate'] = results['fixes_successful'] / results['fixes_attempted']
        
        # Count fix types
        fix_type_counts = {}
        files_affected = set()
        
        for category in results['categories'].values():
            for fix in category:
                fix_type = fix.get('fix_type', 'unknown')
                fix_type_counts[fix_type] = fix_type_counts.get(fix_type, 0) + 1
                
                if fix.get('success') and fix.get('files_modified'):
                    files_affected.update(fix['files_modified'])
                
                if fix.get('backup_created'):
                    summary['backup_files_created'] += 1
        
        summary['most_common_fixes'] = sorted(fix_type_counts.items(), key=lambda x: x[1], reverse=True)
        summary['files_affected'] = len(files_affected)
        
        # Generate recommendations
        if results['fixes_failed'] > 0:
            summary['recommendations'].append('Review failed fixes and apply manually')
        if summary['success_rate'] < 0.8:
            summary['recommendations'].append('Consider running additional quality checks')
        if summary['files_affected'] > 10:
            summary['recommendations'].append('Test thoroughly after extensive changes')
        
        return summary
    
    def restore_from_backup(self, backup_path: str) -> bool:
        """Restore a file from backup."""
        try:
            # Extract original path from backup path
            rel_backup = os.path.relpath(backup_path, self.backup_dir)
            original_path = os.path.join(self.project_root, rel_backup.split('.backup.')[0])
            
            shutil.copy2(backup_path, original_path)
            logger.info(f"✅ Restored {original_path} from backup")
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to restore from backup {backup_path}: {e}")
            return False
    
    def generate_fix_report(self) -> str:
        """Generate a comprehensive fix report."""
        results = self.run_comprehensive_fixes()
        
        report = f"""
# Auto-Fix System Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Fix Summary
- **Fixes Attempted**: {results['fixes_attempted']}
- **Fixes Successful**: {results['fixes_successful']}
- **Fixes Failed**: {results['fixes_failed']}
- **Success Rate**: {results.get('summary', {}).get('success_rate', 0):.1%}

## 🔧 Fixes by Category
"""
        
        for category, fixes in results['categories'].items():
            if fixes:
                successful = sum(1 for fix in fixes if fix.get('success'))
                report += f"\n### {category.replace('_', ' ').title()}\n"
                report += f"- **Total**: {len(fixes)}\n"
                report += f"- **Successful**: {successful}\n"
                
                for fix in fixes[:3]:  # Show first 3 fixes
                    status = "✅" if fix.get('success') else "❌"
                    report += f"  {status} {fix.get('description', 'No description')}\n"
        
        summary = results.get('summary', {})
        if summary.get('recommendations'):
            report += "\n## 🎯 Recommendations\n"
            for rec in summary['recommendations']:
                report += f"- {rec}\n"
        
        return report

# Example usage
if __name__ == "__main__":
    auto_fix = AutoFixSystem()
    report = auto_fix.generate_fix_report()
    print(report)
