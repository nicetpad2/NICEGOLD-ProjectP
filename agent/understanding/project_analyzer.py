from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import ast
import json
import logging
import os
import time
"""
Agent Project Understanding System
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

This module provides comprehensive project understanding capabilities
for the AI agent to better comprehend and improve the ProjectP system.
"""


# Setup logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProjectComponent:
    """Represents a project component with metadata."""
    name: str
    path: str
    type: str  # 'module', 'class', 'function', 'config', 'data'
    dependencies: List[str]
    description: str
    complexity_score: float
    last_modified: datetime
    lines_of_code: int
    functions: List[str]
    classes: List[str]
    imports: List[str]

@dataclass
class ProjectInsight:
    """Project analysis insights."""
    component: str
    insight_type: str  # 'performance', 'bug', 'optimization', 'pattern'
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    suggested_action: str
    confidence: float

class ProjectUnderstanding:
    """Main project understanding system."""

    def __init__(self, project_root: str = None):
        self.project_root = project_root or os.getcwd()
        self.components: Dict[str, ProjectComponent] = {}
        self.insights: List[ProjectInsight] = []
        self.dependency_graph: Dict[str, List[str]] = {}
        self.performance_metrics: Dict[str, Any] = {}

    def analyze_project_structure(self) -> Dict[str, Any]:
        """Analyze the entire project structure."""
        logger.info("🔍 Starting comprehensive project analysis...")

        analysis_result = {
            'structure': self._analyze_directory_structure(), 
            'code_quality': self._analyze_code_quality(), 
            'dependencies': self._analyze_dependencies(), 
            'performance': self._analyze_performance_patterns(), 
            'ml_pipeline': self._analyze_ml_pipeline(), 
            'data_flow': self._analyze_data_flow(), 
            'configuration': self._analyze_configuration(), 
            'documentation': self._analyze_documentation()
        }

        # Generate insights
        self._generate_insights(analysis_result)

        logger.info("✅ Project analysis completed")
        return analysis_result

    def _analyze_directory_structure(self) -> Dict[str, Any]:
        """Analyze project directory structure."""
        structure = {}

        for root, dirs, files in os.walk(self.project_root):
            # Skip hidden directories and cache
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

            rel_path = os.path.relpath(root, self.project_root)
            if rel_path == '.':
                rel_path = 'root'

            structure[rel_path] = {
                'directories': dirs, 
                'files': files, 
                'python_files': [f for f in files if f.endswith('.py')], 
                'config_files': [f for f in files if f.endswith(('.yaml', '.yml', '.json', '.ini', '.toml'))], 
                'data_files': [f for f in files if f.endswith(('.csv', '.parquet', '.pkl', '.pickle'))], 
                'doc_files': [f for f in files if f.endswith(('.md', '.rst', '.txt'))], 
                'file_count': len(files)
            }

        return structure

    def _analyze_code_quality(self) -> Dict[str, Any]:
        """Analyze code quality across the project."""
        quality_metrics = {
            'total_files': 0, 
            'total_lines': 0, 
            'total_functions': 0, 
            'total_classes': 0, 
            'complexity_scores': [], 
            'file_analysis': {}
        }

        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    analysis = self._analyze_python_file(file_path)

                    quality_metrics['total_files'] += 1
                    quality_metrics['total_lines'] += analysis['lines_of_code']
                    quality_metrics['total_functions'] += len(analysis['functions'])
                    quality_metrics['total_classes'] += len(analysis['classes'])
                    quality_metrics['complexity_scores'].append(analysis['complexity_score'])

                    rel_path = os.path.relpath(file_path, self.project_root)
                    quality_metrics['file_analysis'][rel_path] = analysis

        return quality_metrics

    def _analyze_python_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding = 'utf - 8', errors = 'ignore') as f:
                content = f.read()

            # Parse AST
            tree = ast.parse(content)

            # Extract information
            analysis = {
                'lines_of_code': len(content.splitlines()), 
                'functions': [], 
                'classes': [], 
                'imports': [], 
                'complexity_score': 0, 
                'docstring_coverage': 0, 
                'has_main': False, 
                'error_handling': False
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis['functions'].append(node.name)
                    if node.name == 'main':
                        analysis['has_main'] = True
                elif isinstance(node, ast.ClassDef):
                    analysis['classes'].append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        analysis['imports'].append(node.module)
                elif isinstance(node, ast.Try):
                    analysis['error_handling'] = True

            # Calculate complexity (simplified)
            analysis['complexity_score'] = (
                len(analysis['functions']) * 2 +
                len(analysis['classes']) * 3 +
                analysis['lines_of_code'] * 0.01
            )

            return analysis

        except Exception as e:
            logger.warning(f"Error analyzing {file_path}: {e}")
            return {
                'lines_of_code': 0, 
                'functions': [], 
                'classes': [], 
                'imports': [], 
                'complexity_score': 0, 
                'docstring_coverage': 0, 
                'has_main': False, 
                'error_handling': False
            }

    def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze project dependencies."""
        dependencies = {
            'requirements_files': [], 
            'internal_dependencies': {}, 
            'external_dependencies': set(), 
            'dependency_tree': {}
        }

        # Find requirements files
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if 'requirements' in file.lower() or file in ['setup.py', 'pyproject.toml']:
                    dependencies['requirements_files'].append(os.path.join(root, file))

        # Analyze imports
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    file_deps = self._extract_file_dependencies(file_path)

                    rel_path = os.path.relpath(file_path, self.project_root)
                    dependencies['internal_dependencies'][rel_path] = file_deps['internal']
                    dependencies['external_dependencies'].update(file_deps['external'])

        dependencies['external_dependencies'] = list(dependencies['external_dependencies'])
        return dependencies

    def _extract_file_dependencies(self, file_path: str) -> Dict[str, List[str]]:
        """Extract dependencies from a Python file."""
        internal_deps = []
        external_deps = []

        try:
            with open(file_path, 'r', encoding = 'utf - 8', errors = 'ignore') as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if self._is_internal_module(alias.name):
                            internal_deps.append(alias.name)
                        else:
                            external_deps.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        if self._is_internal_module(node.module):
                            internal_deps.append(node.module)
                        else:
                            external_deps.append(node.module)

        except Exception as e:
            logger.warning(f"Error extracting dependencies from {file_path}: {e}")

        return {'internal': internal_deps, 'external': external_deps}

    def _is_internal_module(self, module_name: str) -> bool:
        """Check if a module is internal to the project."""
        # Check if module exists in project directories
        common_internal = ['src', 'projectp', 'agent', 'core', 'protection', 'analytics']
        return any(module_name.startswith(internal) for internal in common_internal)

    def _analyze_performance_patterns(self) -> Dict[str, Any]:
        """Analyze performance patterns in the code."""
        patterns = {
            'potential_bottlenecks': [], 
            'optimization_opportunities': [], 
            'resource_intensive_operations': [], 
            'async_patterns': [], 
            'caching_opportunities': []
        }

        # This would be expanded with actual pattern analysis
        # For now, return structure
        return patterns

    def _analyze_ml_pipeline(self) -> Dict[str, Any]:
        """Analyze ML pipeline components."""
        pipeline_analysis = {
            'pipeline_files': [], 
            'model_files': [], 
            'data_processing': [], 
            'training_scripts': [], 
            'evaluation_metrics': [], 
            'pipeline_stages': []
        }

        # Find ML - related files
        ml_keywords = ['pipeline', 'model', 'train', 'predict', 'preprocess', 'feature']

        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if file.endswith('.py'):
                    file_lower = file.lower()
                    if any(keyword in file_lower for keyword in ml_keywords):
                        file_path = os.path.relpath(os.path.join(root, file), self.project_root)

                        if 'pipeline' in file_lower:
                            pipeline_analysis['pipeline_files'].append(file_path)
                        elif 'model' in file_lower:
                            pipeline_analysis['model_files'].append(file_path)
                        elif 'train' in file_lower:
                            pipeline_analysis['training_scripts'].append(file_path)
                        elif any(kw in file_lower for kw in ['preprocess', 'feature']):
                            pipeline_analysis['data_processing'].append(file_path)

        return pipeline_analysis

    def _analyze_data_flow(self) -> Dict[str, Any]:
        """Analyze data flow patterns."""
        data_flow = {
            'data_sources': [], 
            'data_sinks': [], 
            'transformation_points': [], 
            'data_validation': [], 
            'data_formats': set()
        }

        # Find data files
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if file.endswith(('.csv', '.parquet', '.pkl', '.pickle', '.json')):
                    ext = os.path.splitext(file)[1]
                    data_flow['data_formats'].add(ext)
                    data_flow['data_sources'].append(os.path.relpath(os.path.join(root, file), self.project_root))

        data_flow['data_formats'] = list(data_flow['data_formats'])
        return data_flow

    def _analyze_configuration(self) -> Dict[str, Any]:
        """Analyze configuration files and patterns."""
        config_analysis = {
            'config_files': [], 
            'environment_files': [], 
            'settings_patterns': [], 
            'configuration_management': 'basic'
        }

        config_extensions = ['.yaml', '.yml', '.json', '.ini', '.toml', '.env']

        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if any(file.endswith(ext) for ext in config_extensions) or 'config' in file.lower():
                    file_path = os.path.relpath(os.path.join(root, file), self.project_root)

                    if file.startswith('.env') or 'env' in file.lower():
                        config_analysis['environment_files'].append(file_path)
                    else:
                        config_analysis['config_files'].append(file_path)

        return config_analysis

    def _analyze_documentation(self) -> Dict[str, Any]:
        """Analyze project documentation."""
        doc_analysis = {
            'readme_files': [], 
            'doc_files': [], 
            'code_comments_ratio': 0, 
            'docstring_coverage': 0, 
            'documentation_quality': 'needs_improvement'
        }

        # Find documentation files
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if file.endswith(('.md', '.rst', '.txt')):
                    file_path = os.path.relpath(os.path.join(root, file), self.project_root)

                    if 'readme' in file.lower():
                        doc_analysis['readme_files'].append(file_path)
                    else:
                        doc_analysis['doc_files'].append(file_path)

        return doc_analysis

    def _generate_insights(self, analysis_result: Dict[str, Any]) -> None:
        """Generate actionable insights from analysis."""
        self.insights = []

        # Code quality insights
        code_quality = analysis_result['code_quality']
        if code_quality['total_functions'] > 500:
            self.insights.append(ProjectInsight(
                component = 'codebase', 
                insight_type = 'optimization', 
                description = 'Large number of functions detected', 
                severity = 'medium', 
                suggested_action = 'Consider modularization and refactoring', 
                confidence = 0.8
            ))

        # Documentation insights
        doc_analysis = analysis_result['documentation']
        if len(doc_analysis['readme_files']) == 0:
            self.insights.append(ProjectInsight(
                component = 'documentation', 
                insight_type = 'bug', 
                description = 'No README file found', 
                severity = 'high', 
                suggested_action = 'Create comprehensive README.md', 
                confidence = 0.9
            ))

        # ML Pipeline insights
        ml_analysis = analysis_result['ml_pipeline']
        if len(ml_analysis['pipeline_files']) > 5:
            self.insights.append(ProjectInsight(
                component = 'ml_pipeline', 
                insight_type = 'optimization', 
                description = 'Multiple pipeline files detected', 
                severity = 'low', 
                suggested_action = 'Consider consolidating pipeline logic', 
                confidence = 0.7
            ))

    def get_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """Get specific improvement suggestions."""
        suggestions = []

        for insight in self.insights:
            suggestion = {
                'area': insight.component, 
                'priority': insight.severity, 
                'description': insight.description, 
                'action': insight.suggested_action, 
                'impact': 'high' if insight.confidence > 0.8 else 'medium', 
                'effort': 'low' if 'documentation' in insight.component else 'medium'
            }
            suggestions.append(suggestion)

        # Sort by priority and impact
        priority_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        suggestions.sort(key = lambda x: priority_order.get(x['priority'], 0), reverse = True)

        return suggestions

    def generate_understanding_report(self) -> str:
        """Generate a comprehensive understanding report."""
        analysis = self.analyze_project_structure()
        suggestions = self.get_improvement_suggestions()

        report = f"""
# ProjectP Understanding Report
Generated: {datetime.now().strftime('%Y - %m - %d %H:%M:%S')}

## 📊 Project Overview
- **Total Python Files**: {analysis['code_quality']['total_files']}
- **Total Lines of Code**: {analysis['code_quality']['total_lines']:, }
- **Total Functions**: {analysis['code_quality']['total_functions']}
- **Total Classes**: {analysis['code_quality']['total_classes']}

## 🏗️ Structure Analysis
- **ML Pipeline Files**: {len(analysis['ml_pipeline']['pipeline_files'])}
- **Model Files**: {len(analysis['ml_pipeline']['model_files'])}
- **Training Scripts**: {len(analysis['ml_pipeline']['training_scripts'])}
- **Data Processing**: {len(analysis['ml_pipeline']['data_processing'])}

## 📝 Documentation Status
- **README Files**: {len(analysis['documentation']['readme_files'])}
- **Documentation Files**: {len(analysis['documentation']['doc_files'])}

## 🔧 Configuration
- **Config Files**: {len(analysis['configuration']['config_files'])}
- **Environment Files**: {len(analysis['configuration']['environment_files'])}

## 🎯 Improvement Suggestions
"""

        for i, suggestion in enumerate(suggestions[:10], 1):
            report += f"""
{i}. **{suggestion['area'].title()}** (Priority: {suggestion['priority'].upper()})
   - Issue: {suggestion['description']}
   - Action: {suggestion['action']}
   - Impact: {suggestion['impact']} | Effort: {suggestion['effort']}
"""

        report += f"""
## 🔍 Key Insights
- Found {len(self.insights)} areas for improvement
- Primary focus should be on {suggestions[0]['area'] if suggestions else 'general optimization'}
- Code complexity is {'high' if analysis['code_quality']['total_lines'] > 10000 else 'manageable'}

## 📈 Next Steps
1. Address high - priority issues first
2. Implement suggested optimizations
3. Enhance documentation coverage
4. Set up automated quality checks
5. Monitor performance improvements
"""

        return report

    def save_analysis_results(self, output_path: str = None) -> str:
        """Save analysis results to file."""
        if output_path is None:
            output_path = os.path.join(self.project_root, 'reports', 'project_understanding.json')

        os.makedirs(os.path.dirname(output_path), exist_ok = True)

        analysis = self.analyze_project_structure()
        suggestions = self.get_improvement_suggestions()

        results = {
            'timestamp': datetime.now().isoformat(), 
            'analysis': analysis, 
            'insights': [
                {
                    'component': insight.component, 
                    'type': insight.insight_type, 
                    'description': insight.description, 
                    'severity': insight.severity, 
                    'action': insight.suggested_action, 
                    'confidence': insight.confidence
                }
                for insight in self.insights
            ], 
            'suggestions': suggestions
        }

        with open(output_path, 'w', encoding = 'utf - 8') as f:
            json.dump(results, f, indent = 2, ensure_ascii = False)

        return output_path

# Example usage
if __name__ == "__main__":
    # Initialize understanding system
    understanding = ProjectUnderstanding()

    # Generate comprehensive report
    report = understanding.generate_understanding_report()
    print(report)

    # Save analysis results
    results_path = understanding.save_analysis_results()
    print(f"\n📊 Analysis results saved to: {results_path}")