"""
Agent Code Analysis Module
==========================

Advanced code analysis capabilities for understanding patterns,
identifying issues, and suggesting improvements.
"""

import ast
import os
import re
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)

@dataclass
class CodeIssue:
    """Represents a code issue found during analysis."""
    file_path: str
    line_number: int
    issue_type: str
    severity: str
    description: str
    suggestion: str
    code_snippet: str

@dataclass
class CodeMetrics:
    """Code metrics for a file or project."""
    cyclomatic_complexity: int
    lines_of_code: int
    comment_ratio: float
    function_count: int
    class_count: int
    import_count: int
    duplicate_code_ratio: float
    maintainability_index: float

class CodeAnalyzer:
    """Advanced code analysis system."""
    
    def __init__(self, project_root: str = None):
        self.project_root = project_root or os.getcwd()
        self.issues: List[CodeIssue] = []
        self.metrics: Dict[str, CodeMetrics] = {}
        self.patterns: Dict[str, List[str]] = defaultdict(list)
        
    def analyze_code_quality(self) -> Dict[str, Any]:
        """Comprehensive code quality analysis."""
        logger.info("🔍 Starting code quality analysis...")
        
        results = {
            'overview': self._get_project_overview(),
            'issues': self._find_code_issues(),
            'metrics': self._calculate_metrics(),
            'patterns': self._identify_patterns(),
            'duplicates': self._find_duplicate_code(),
            'complexity': self._analyze_complexity(),
            'maintainability': self._assess_maintainability(),
            'recommendations': self._generate_recommendations()
        }
        
        logger.info("✅ Code quality analysis completed")
        return results
    
    def _get_project_overview(self) -> Dict[str, Any]:
        """Get high-level project overview."""
        overview = {
            'total_files': 0,
            'total_lines': 0,
            'total_functions': 0,
            'total_classes': 0,
            'language_distribution': Counter(),
            'file_size_distribution': {'small': 0, 'medium': 0, 'large': 0, 'huge': 0}
        }
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip hidden directories and cache
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            lines = len(content.splitlines())
                            
                        overview['total_files'] += 1
                        overview['total_lines'] += lines
                        overview['language_distribution']['Python'] += 1
                        
                        # File size categorization
                        if lines < 50:
                            overview['file_size_distribution']['small'] += 1
                        elif lines < 200:
                            overview['file_size_distribution']['medium'] += 1
                        elif lines < 500:
                            overview['file_size_distribution']['large'] += 1
                        else:
                            overview['file_size_distribution']['huge'] += 1
                            
                        # Count functions and classes
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                overview['total_functions'] += 1
                            elif isinstance(node, ast.ClassDef):
                                overview['total_classes'] += 1
                                
                    except Exception as e:
                        logger.warning(f"Error analyzing {file_path}: {e}")
        
        return overview
    
    def _find_code_issues(self) -> List[Dict[str, Any]]:
        """Find various code issues."""
        self.issues = []
        
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    self._analyze_file_issues(file_path)
        
        # Convert to serializable format
        return [
            {
                'file_path': issue.file_path,
                'line_number': issue.line_number,
                'issue_type': issue.issue_type,
                'severity': issue.severity,
                'description': issue.description,
                'suggestion': issue.suggestion,
                'code_snippet': issue.code_snippet
            }
            for issue in self.issues
        ]
    
    def _analyze_file_issues(self, file_path: str) -> None:
        """Analyze issues in a specific file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.splitlines()
            
            rel_path = os.path.relpath(file_path, self.project_root)
            
            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                self.issues.append(CodeIssue(
                    file_path=rel_path,
                    line_number=e.lineno or 0,
                    issue_type='syntax_error',
                    severity='critical',
                    description=f'Syntax error: {e.msg}',
                    suggestion='Fix syntax error',
                    code_snippet=lines[e.lineno-1] if e.lineno and e.lineno <= len(lines) else ''
                ))
                return
            
            # Check for various issues
            self._check_function_issues(tree, rel_path, lines)
            self._check_naming_conventions(tree, rel_path, lines)
            self._check_complexity_issues(tree, rel_path, lines)
            self._check_import_issues(tree, rel_path, lines)
            self._check_error_handling(tree, rel_path, lines)
            self._check_code_smells(content, rel_path, lines)
            
        except Exception as e:
            logger.warning(f"Error analyzing file issues in {file_path}: {e}")
    
    def _check_function_issues(self, tree: ast.AST, file_path: str, lines: List[str]) -> None:
        """Check function-related issues."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Long functions
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    func_lines = node.end_lineno - node.lineno
                    if func_lines > 50:
                        self.issues.append(CodeIssue(
                            file_path=file_path,
                            line_number=node.lineno,
                            issue_type='long_function',
                            severity='medium',
                            description=f'Function "{node.name}" is too long ({func_lines} lines)',
                            suggestion='Consider breaking down into smaller functions',
                            code_snippet=lines[node.lineno-1] if node.lineno <= len(lines) else ''
                        ))
                
                # Too many parameters
                if len(node.args.args) > 7:
                    self.issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        issue_type='too_many_parameters',
                        severity='medium',
                        description=f'Function "{node.name}" has too many parameters ({len(node.args.args)})',
                        suggestion='Consider using a configuration object or reducing parameters',
                        code_snippet=lines[node.lineno-1] if node.lineno <= len(lines) else ''
                    ))
                
                # Missing docstring
                if not ast.get_docstring(node):
                    self.issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        issue_type='missing_docstring',
                        severity='low',
                        description=f'Function "{node.name}" is missing a docstring',
                        suggestion='Add descriptive docstring',
                        code_snippet=lines[node.lineno-1] if node.lineno <= len(lines) else ''
                    ))
    
    def _check_naming_conventions(self, tree: ast.AST, file_path: str, lines: List[str]) -> None:
        """Check naming convention issues."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Function naming (should be snake_case)
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.name) and not node.name.startswith('_'):
                    self.issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        issue_type='naming_convention',
                        severity='low',
                        description=f'Function "{node.name}" doesn\'t follow snake_case convention',
                        suggestion='Use snake_case for function names',
                        code_snippet=lines[node.lineno-1] if node.lineno <= len(lines) else ''
                    ))
            
            elif isinstance(node, ast.ClassDef):
                # Class naming (should be PascalCase)
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    self.issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        issue_type='naming_convention',
                        severity='low',
                        description=f'Class "{node.name}" doesn\'t follow PascalCase convention',
                        suggestion='Use PascalCase for class names',
                        code_snippet=lines[node.lineno-1] if node.lineno <= len(lines) else ''
                    ))
    
    def _check_complexity_issues(self, tree: ast.AST, file_path: str, lines: List[str]) -> None:
        """Check complexity-related issues."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_cyclomatic_complexity(node)
                if complexity > 10:
                    self.issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        issue_type='high_complexity',
                        severity='high',
                        description=f'Function "{node.name}" has high cyclomatic complexity ({complexity})',
                        suggestion='Reduce complexity by breaking down logic or using early returns',
                        code_snippet=lines[node.lineno-1] if node.lineno <= len(lines) else ''
                    ))
    
    def _check_import_issues(self, tree: ast.AST, file_path: str, lines: List[str]) -> None:
        """Check import-related issues."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(node)
        
        # Too many imports
        if len(imports) > 20:
            self.issues.append(CodeIssue(
                file_path=file_path,
                line_number=1,
                issue_type='too_many_imports',
                severity='medium',
                description=f'File has too many imports ({len(imports)})',
                suggestion='Consider refactoring or organizing imports better',
                code_snippet=''
            ))
        
        # Check for unused imports (simplified check)
        # This would require more sophisticated analysis in practice
    
    def _check_error_handling(self, tree: ast.AST, file_path: str, lines: List[str]) -> None:
        """Check error handling patterns."""
        has_try_except = False
        bare_except_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                has_try_except = True
                for handler in node.handlers:
                    if handler.type is None:  # bare except
                        bare_except_count += 1
                        self.issues.append(CodeIssue(
                            file_path=file_path,
                            line_number=handler.lineno,
                            issue_type='bare_except',
                            severity='medium',
                            description='Bare except clause found',
                            suggestion='Specify exception type or use "except Exception"',
                            code_snippet=lines[handler.lineno-1] if handler.lineno <= len(lines) else ''
                        ))
        
        # Check if file has functions but no error handling
        has_functions = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
        if has_functions and not has_try_except and 'test' not in file_path.lower():
            self.issues.append(CodeIssue(
                file_path=file_path,
                line_number=1,
                issue_type='no_error_handling',
                severity='low',
                description='File contains functions but no error handling',
                suggestion='Consider adding try-except blocks for robustness',
                code_snippet=''
            ))
    
    def _check_code_smells(self, content: str, file_path: str, lines: List[str]) -> None:
        """Check for code smells."""
        # Long lines
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                self.issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=i,
                    issue_type='long_line',
                    severity='low',
                    description=f'Line too long ({len(line)} characters)',
                    suggestion='Break line or refactor for better readability',
                    code_snippet=line[:100] + '...' if len(line) > 100 else line
                ))
        
        # TODO comments
        for i, line in enumerate(lines, 1):
            if 'TODO' in line.upper() or 'FIXME' in line.upper():
                self.issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=i,
                    issue_type='todo_comment',
                    severity='low',
                    description='TODO/FIXME comment found',
                    suggestion='Complete the implementation or create a proper issue',
                    code_snippet=line.strip()
                ))
    
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.Comprehension):
                complexity += 1
        
        return complexity
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate various code metrics."""
        metrics = {
            'files': {},
            'overall': {
                'average_complexity': 0,
                'average_function_length': 0,
                'comment_ratio': 0,
                'maintainability_index': 0
            }
        }
        
        total_complexity = 0
        total_functions = 0
        total_lines = 0
        total_comment_lines = 0
        
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.project_root)
                    
                    file_metrics = self._calculate_file_metrics(file_path)
                    metrics['files'][rel_path] = file_metrics.__dict__
                    
                    total_complexity += file_metrics.cyclomatic_complexity
                    total_functions += file_metrics.function_count
                    total_lines += file_metrics.lines_of_code
                    total_comment_lines += file_metrics.lines_of_code * file_metrics.comment_ratio
        
        # Calculate overall metrics
        if total_functions > 0:
            metrics['overall']['average_complexity'] = total_complexity / total_functions
        if total_lines > 0:
            metrics['overall']['comment_ratio'] = total_comment_lines / total_lines
            
        return metrics
    
    def _calculate_file_metrics(self, file_path: str) -> CodeMetrics:
        """Calculate metrics for a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.splitlines()
            
            tree = ast.parse(content)
            
            # Count various elements
            function_count = 0
            class_count = 0
            import_count = 0
            total_complexity = 0
            comment_lines = 0
            
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('#') or '"""' in stripped or "'''" in stripped:
                    comment_lines += 1
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_count += 1
                    total_complexity += self._calculate_cyclomatic_complexity(node)
                elif isinstance(node, ast.ClassDef):
                    class_count += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_count += 1
            
            comment_ratio = comment_lines / len(lines) if lines else 0
            
            # Simple maintainability index calculation
            maintainability_index = max(0, 100 - total_complexity * 2 - (1 - comment_ratio) * 10)
            
            return CodeMetrics(
                cyclomatic_complexity=total_complexity,
                lines_of_code=len(lines),
                comment_ratio=comment_ratio,
                function_count=function_count,
                class_count=class_count,
                import_count=import_count,
                duplicate_code_ratio=0.0,  # Would need more sophisticated analysis
                maintainability_index=maintainability_index
            )
            
        except Exception as e:
            logger.warning(f"Error calculating metrics for {file_path}: {e}")
            return CodeMetrics(0, 0, 0, 0, 0, 0, 0, 0)
    
    def _identify_patterns(self) -> Dict[str, List[str]]:
        """Identify code patterns and anti-patterns."""
        patterns = {
            'design_patterns': [],
            'anti_patterns': [],
            'common_imports': [],
            'naming_patterns': []
        }
        
        # This would be expanded with actual pattern recognition
        return patterns
    
    def _find_duplicate_code(self) -> Dict[str, Any]:
        """Find duplicate code blocks."""
        duplicates = {
            'blocks': [],
            'similarity_threshold': 0.8,
            'total_duplicates': 0
        }
        
        # This would require more sophisticated duplicate detection
        return duplicates
    
    def _analyze_complexity(self) -> Dict[str, Any]:
        """Analyze code complexity at different levels."""
        complexity = {
            'file_complexity': {},
            'function_complexity': {},
            'class_complexity': {},
            'highest_complexity_files': [],
            'complexity_distribution': {'low': 0, 'medium': 0, 'high': 0, 'very_high': 0}
        }
        
        # This would be populated with actual complexity analysis
        return complexity
    
    def _assess_maintainability(self) -> Dict[str, Any]:
        """Assess code maintainability."""
        maintainability = {
            'overall_score': 0,
            'factors': {
                'readability': 0,
                'complexity': 0,
                'documentation': 0,
                'test_coverage': 0,
                'modularity': 0
            },
            'recommendations': []
        }
        
        # This would be expanded with actual maintainability assessment
        return maintainability
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate improvement recommendations based on analysis."""
        recommendations = []
        
        # Group issues by type and severity
        issue_counts = Counter(issue.issue_type for issue in self.issues)
        severity_counts = Counter(issue.severity for issue in self.issues)
        
        # Generate recommendations based on most common issues
        for issue_type, count in issue_counts.most_common(5):
            if count > 5:  # Threshold for recommendation
                recommendations.append({
                    'type': 'code_quality',
                    'priority': 'high' if count > 20 else 'medium',
                    'description': f'Address {count} instances of {issue_type.replace("_", " ")}',
                    'action': self._get_action_for_issue_type(issue_type),
                    'impact': 'improved code quality and maintainability'
                })
        
        return recommendations
    
    def _get_action_for_issue_type(self, issue_type: str) -> str:
        """Get recommended action for issue type."""
        actions = {
            'long_function': 'Break down long functions into smaller, focused functions',
            'too_many_parameters': 'Reduce parameter count or use configuration objects',
            'missing_docstring': 'Add comprehensive docstrings to functions and classes',
            'naming_convention': 'Follow Python naming conventions (PEP 8)',
            'high_complexity': 'Simplify complex functions using early returns and smaller functions',
            'bare_except': 'Specify exception types instead of using bare except clauses',
            'long_line': 'Break long lines for better readability',
            'todo_comment': 'Complete TODOs or convert to proper issues'
        }
        return actions.get(issue_type, 'Address this code quality issue')
    
    def generate_quality_report(self) -> str:
        """Generate a comprehensive code quality report."""
        analysis = self.analyze_code_quality()
        
        report = f"""
# Code Quality Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Project Overview
- **Total Files**: {analysis['overview']['total_files']}
- **Total Lines**: {analysis['overview']['total_lines']:,}
- **Total Functions**: {analysis['overview']['total_functions']}
- **Total Classes**: {analysis['overview']['total_classes']}

## 🚨 Issues Summary
- **Total Issues**: {len(analysis['issues'])}
- **Critical**: {sum(1 for issue in analysis['issues'] if issue['severity'] == 'critical')}
- **High**: {sum(1 for issue in analysis['issues'] if issue['severity'] == 'high')}
- **Medium**: {sum(1 for issue in analysis['issues'] if issue['severity'] == 'medium')}
- **Low**: {sum(1 for issue in analysis['issues'] if issue['severity'] == 'low')}

## 📈 Quality Metrics
- **Average Complexity**: {analysis['metrics']['overall']['average_complexity']:.2f}
- **Comment Ratio**: {analysis['metrics']['overall']['comment_ratio']:.2%}

## 🎯 Top Recommendations
"""
        
        for i, rec in enumerate(analysis['recommendations'][:5], 1):
            report += f"""
{i}. **{rec['type'].title()}** (Priority: {rec['priority'].upper()})
   - {rec['description']}
   - Action: {rec['action']}
   - Impact: {rec['impact']}
"""
        
        return report

# Example usage
if __name__ == "__main__":
    analyzer = CodeAnalyzer()
    report = analyzer.generate_quality_report()
    print(report)
