"""
Dependency Mapper
================

Maps project dependencies, imports, and component relationships for better understanding.
"""

import os
import ast
import json
from typing import Dict, List, Any, Set, Tuple
from pathlib import Path
import networkx as nx
from collections import defaultdict

class DependencyMapper:
    """
    Maps and analyzes project dependencies, imports, and component relationships.
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.dependency_graph = nx.DiGraph()
        self.import_map = defaultdict(list)
        self.circular_dependencies = []
        
    def analyze_dependencies(self) -> Dict[str, Any]:
        """
        Analyze project dependencies including:
        - Import relationships
        - Circular dependencies
        - Missing dependencies
        - Unused imports
        - External vs internal dependencies
        """
        return {
            'import_map': self._build_import_map(),
            'dependency_graph': self._build_dependency_graph(),
            'circular_dependencies': self._detect_circular_dependencies(),
            'missing_dependencies': self._detect_missing_dependencies(),
            'unused_imports': self._detect_unused_imports(),
            'external_dependencies': self._analyze_external_dependencies(),
            'dependency_health': self._assess_dependency_health()
        }
    
    def _build_import_map(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build a map of all import statements."""
        import_map = defaultdict(list)
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                file_imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            file_imports.append({
                                'type': 'import',
                                'module': alias.name,
                                'alias': alias.asname,
                                'line': node.lineno
                            })
                    
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            for alias in node.names:
                                file_imports.append({
                                    'type': 'from_import',
                                    'module': node.module,
                                    'name': alias.name,
                                    'alias': alias.asname,
                                    'line': node.lineno
                                })
                
                import_map[str(py_file.relative_to(self.project_root))] = file_imports
                
            except Exception as e:
                print(f"Error processing {py_file}: {e}")
                
        return dict(import_map)
    
    def _build_dependency_graph(self) -> Dict[str, Any]:
        """Build a dependency graph showing file relationships."""
        self.dependency_graph.clear()
        
        for py_file in self.project_root.rglob("*.py"):
            file_path = str(py_file.relative_to(self.project_root))
            self.dependency_graph.add_node(file_path)
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.module and not node.module.startswith('.'):
                            # Check if it's an internal import
                            module_path = node.module.replace('.', '/') + '.py'
                            if (self.project_root / module_path).exists():
                                self.dependency_graph.add_edge(file_path, module_path)
                                
            except Exception as e:
                continue
        
        return {
            'nodes': list(self.dependency_graph.nodes()),
            'edges': list(self.dependency_graph.edges()),
            'node_count': self.dependency_graph.number_of_nodes(),
            'edge_count': self.dependency_graph.number_of_edges()
        }
    
    def _detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies in the project."""
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            return cycles
        except:
            return []
    
    def _detect_missing_dependencies(self) -> List[Dict[str, Any]]:
        """Detect potentially missing dependencies."""
        missing = []
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            try:
                                __import__(alias.name)
                            except ImportError:
                                missing.append({
                                    'file': str(py_file.relative_to(self.project_root)),
                                    'module': alias.name,
                                    'line': node.lineno,
                                    'type': 'import'
                                })
                    
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            try:
                                __import__(node.module)
                            except ImportError:
                                missing.append({
                                    'file': str(py_file.relative_to(self.project_root)),
                                    'module': node.module,
                                    'line': node.lineno,
                                    'type': 'from_import'
                                })
                                
            except Exception as e:
                continue
                
        return missing
    
    def _detect_unused_imports(self) -> List[Dict[str, Any]]:
        """Detect potentially unused imports."""
        unused = []
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                # Simple heuristic: check if imported name appears in code
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            name = alias.asname if alias.asname else alias.name
                            imports.append({
                                'name': name,
                                'module': alias.name,
                                'line': node.lineno,
                                'type': 'import'
                            })
                    
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            for alias in node.names:
                                name = alias.asname if alias.asname else alias.name
                                imports.append({
                                    'name': name,
                                    'module': node.module,
                                    'line': node.lineno,
                                    'type': 'from_import'
                                })
                
                # Check usage
                for imp in imports:
                    if imp['name'] not in content.split('\n')[imp['line']:]:
                        # Simple check - not foolproof but gives indication
                        count = content.count(imp['name'])
                        if count <= 1:  # Only appears in import line
                            unused.append({
                                'file': str(py_file.relative_to(self.project_root)),
                                'import': imp['name'],
                                'module': imp['module'],
                                'line': imp['line']
                            })
                            
            except Exception as e:
                continue
                
        return unused
    
    def _analyze_external_dependencies(self) -> Dict[str, Any]:
        """Analyze external (third-party) dependencies."""
        external_deps = set()
        dep_usage = defaultdict(int)
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            root_module = alias.name.split('.')[0]
                            if not self._is_internal_module(root_module):
                                external_deps.add(root_module)
                                dep_usage[root_module] += 1
                    
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            root_module = node.module.split('.')[0]
                            if not self._is_internal_module(root_module):
                                external_deps.add(root_module)
                                dep_usage[root_module] += 1
                                
            except Exception as e:
                continue
        
        return {
            'external_dependencies': list(external_deps),
            'dependency_usage_count': dict(dep_usage),
            'most_used_dependencies': sorted(dep_usage.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def _is_internal_module(self, module_name: str) -> bool:
        """Check if a module is internal to the project."""
        # Check if module exists as a file or directory in project
        module_path = self.project_root / f"{module_name}.py"
        module_dir = self.project_root / module_name
        
        return module_path.exists() or module_dir.exists()
    
    def _assess_dependency_health(self) -> Dict[str, Any]:
        """Assess overall dependency health."""
        circular_count = len(self.circular_dependencies)
        missing_count = len(self._detect_missing_dependencies())
        unused_count = len(self._detect_unused_imports())
        
        # Simple health score
        total_issues = circular_count + missing_count + unused_count
        health_score = max(0.0, 1.0 - (total_issues / 100.0))  # Normalize
        
        return {
            'health_score': health_score,
            'circular_dependency_count': circular_count,
            'missing_dependency_count': missing_count,
            'unused_import_count': unused_count,
            'total_issues': total_issues,
            'recommendations': self._generate_dependency_recommendations()
        }
    
    def _generate_dependency_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations for dependency improvements."""
        recommendations = []
        
        if self.circular_dependencies:
            recommendations.append({
                'type': 'circular_dependencies',
                'severity': 'high',
                'message': f"Found {len(self.circular_dependencies)} circular dependencies",
                'action': 'Refactor code to break circular imports'
            })
        
        missing = self._detect_missing_dependencies()
        if missing:
            recommendations.append({
                'type': 'missing_dependencies',
                'severity': 'critical',
                'message': f"Found {len(missing)} missing dependencies",
                'action': 'Install missing packages or fix import statements'
            })
        
        unused = self._detect_unused_imports()
        if unused:
            recommendations.append({
                'type': 'unused_imports',
                'severity': 'low',
                'message': f"Found {len(unused)} potentially unused imports",
                'action': 'Remove unused imports to clean up code'
            })
        
        return recommendations
    
    def visualize_dependencies(self, output_path: str = None) -> str:
        """Generate a dependency visualization."""
        if not output_path:
            output_path = self.project_root / "dependency_graph.html"
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(self.dependency_graph)
            
            # Draw nodes
            nx.draw_networkx_nodes(self.dependency_graph, pos, 
                                 node_color='lightblue', 
                                 node_size=500)
            
            # Draw edges
            nx.draw_networkx_edges(self.dependency_graph, pos, 
                                 edge_color='gray', 
                                 arrows=True)
            
            # Draw labels
            nx.draw_networkx_labels(self.dependency_graph, pos, 
                                  font_size=8)
            
            plt.title("Project Dependency Graph")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except ImportError:
            # Fallback: Generate text-based representation
            with open(output_path, 'w') as f:
                f.write("# Project Dependency Graph\n\n")
                f.write("## Nodes (Files)\n")
                for node in self.dependency_graph.nodes():
                    f.write(f"- {node}\n")
                
                f.write("\n## Dependencies\n")
                for source, target in self.dependency_graph.edges():
                    f.write(f"- {source} → {target}\n")
            
            return str(output_path)
        
        except Exception as e:
            print(f"Error generating visualization: {e}")
            return ""
