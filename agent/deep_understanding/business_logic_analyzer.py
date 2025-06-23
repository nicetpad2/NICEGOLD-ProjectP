"""
Business Logic Analyzer
======================

Analyzes business logic, trading strategies, and domain-specific patterns in ProjectP.
"""

import os
import ast
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import json

class BusinessLogicAnalyzer:
    """
    Analyzes business logic, trading strategies, and domain-specific patterns.
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.trading_patterns = {}
        self.business_rules = []
        self.strategy_components = {}
        
    def analyze_business_logic(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of business logic including:
        - Trading strategies and patterns
        - Business rules and constraints
        - Decision logic flow
        - Risk management components
        - Performance metrics and KPIs
        """
        return {
            'trading_strategies': self._analyze_trading_strategies(),
            'business_rules': self._extract_business_rules(),
            'decision_logic': self._analyze_decision_logic(),
            'risk_management': self._analyze_risk_management(),
            'performance_metrics': self._analyze_performance_metrics(),
            'strategy_patterns': self._identify_strategy_patterns(),
            'business_logic_quality': self._assess_business_logic_quality()
        }
    
    def _analyze_trading_strategies(self) -> Dict[str, Any]:
        """Analyze trading strategy implementations."""
        strategies = {
            'strategy_files': [],
            'strategy_functions': [],
            'entry_signals': [],
            'exit_signals': [],
            'indicators_used': [],
            'timeframes': [],
            'strategy_complexity': 'unknown'
        }
        
        # Search for strategy-related files and functions
        strategy_keywords = ['strategy', 'signal', 'trade', 'entry', 'exit', 'indicator']
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if file contains strategy-related content
                if any(keyword in content.lower() for keyword in strategy_keywords):
                    strategies['strategy_files'].append(str(py_file.relative_to(self.project_root)))
                    
                    # Parse AST to find functions
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            func_name = node.name.lower()
                            
                            # Categorize functions
                            if any(keyword in func_name for keyword in ['entry', 'buy', 'long']):
                                strategies['entry_signals'].append({
                                    'function': node.name,
                                    'file': str(py_file.relative_to(self.project_root)),
                                    'line': node.lineno
                                })
                            
                            elif any(keyword in func_name for keyword in ['exit', 'sell', 'short', 'close']):
                                strategies['exit_signals'].append({
                                    'function': node.name,
                                    'file': str(py_file.relative_to(self.project_root)),
                                    'line': node.lineno
                                })
                            
                            elif any(keyword in func_name for keyword in strategy_keywords):
                                strategies['strategy_functions'].append({
                                    'function': node.name,
                                    'file': str(py_file.relative_to(self.project_root)),
                                    'line': node.lineno,
                                    'category': self._categorize_strategy_function(func_name)
                                })
                
                # Extract indicators and timeframes
                indicators = self._extract_indicators(content)
                strategies['indicators_used'].extend(indicators)
                
                timeframes = self._extract_timeframes(content)
                strategies['timeframes'].extend(timeframes)
                
            except Exception as e:
                continue
        
        # Remove duplicates and assess complexity
        strategies['indicators_used'] = list(set(strategies['indicators_used']))
        strategies['timeframes'] = list(set(strategies['timeframes']))
        strategies['strategy_complexity'] = self._assess_strategy_complexity(strategies)
        
        return strategies
    
    def _extract_business_rules(self) -> List[Dict[str, Any]]:
        """Extract business rules and constraints."""
        business_rules = []
        
        rule_patterns = [
            r'if\s+.*(price|auc|profit|loss|risk)',
            r'assert\s+.*(threshold|limit|minimum|maximum)',
            r'raise\s+.*if\s+.*(condition|rule|constraint)',
            r'(min|max)_?(price|auc|profit|loss|risk|threshold)',
            r'stop_?(loss|profit)',
            r'risk_?(management|limit|threshold)'
        ]
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    for pattern in rule_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            business_rules.append({
                                'rule_type': self._classify_business_rule(line),
                                'rule_text': line.strip(),
                                'file': str(py_file.relative_to(self.project_root)),
                                'line': line_num,
                                'pattern_matched': pattern
                            })
                
            except Exception as e:
                continue
        
        return business_rules
    
    def _analyze_decision_logic(self) -> Dict[str, Any]:
        """Analyze decision-making logic flow."""
        decision_points = []
        logic_complexity = {}
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                file_complexity = {
                    'if_statements': 0,
                    'nested_depth': 0,
                    'decision_functions': []
                }
                
                for node in ast.walk(tree):
                    # Count if statements
                    if isinstance(node, ast.If):
                        file_complexity['if_statements'] += 1
                        
                        # Analyze decision criteria
                        decision_info = self._analyze_if_statement(node, py_file)
                        if decision_info:
                            decision_points.append(decision_info)
                    
                    # Find decision-making functions
                    elif isinstance(node, ast.FunctionDef):
                        if any(keyword in node.name.lower() for keyword in 
                               ['decide', 'choose', 'select', 'determine', 'evaluate']):
                            file_complexity['decision_functions'].append({
                                'function': node.name,
                                'line': node.lineno
                            })
                
                # Calculate nesting depth
                file_complexity['nested_depth'] = self._calculate_nesting_depth(tree)
                logic_complexity[str(py_file.relative_to(self.project_root))] = file_complexity
                
            except Exception as e:
                continue
        
        return {
            'decision_points': decision_points,
            'logic_complexity': logic_complexity,
            'complexity_score': self._calculate_decision_complexity_score(logic_complexity),
            'optimization_suggestions': self._suggest_decision_optimizations(decision_points, logic_complexity)
        }
    
    def _analyze_risk_management(self) -> Dict[str, Any]:
        """Analyze risk management components."""
        risk_components = {
            'stop_loss_implementations': [],
            'risk_limits': [],
            'position_sizing': [],
            'risk_metrics': [],
            'validation_checks': []
        }
        
        risk_keywords = {
            'stop_loss': ['stop', 'loss', 'stoploss'],
            'risk_limit': ['risk', 'limit', 'threshold', 'maximum'],
            'position_size': ['position', 'size', 'sizing', 'allocation'],
            'validation': ['validate', 'check', 'verify', 'assert']
        }
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    line_lower = line.lower()
                    
                    for category, keywords in risk_keywords.items():
                        if any(keyword in line_lower for keyword in keywords):
                            component = {
                                'text': line.strip(),
                                'file': str(py_file.relative_to(self.project_root)),
                                'line': line_num,
                                'category': category
                            }
                            
                            if category == 'stop_loss':
                                risk_components['stop_loss_implementations'].append(component)
                            elif category == 'risk_limit':
                                risk_components['risk_limits'].append(component)
                            elif category == 'position_size':
                                risk_components['position_sizing'].append(component)
                            elif category == 'validation':
                                risk_components['validation_checks'].append(component)
                
            except Exception as e:
                continue
        
        # Assess risk management maturity
        risk_components['risk_management_score'] = self._calculate_risk_management_score(risk_components)
        risk_components['risk_recommendations'] = self._generate_risk_recommendations(risk_components)
        
        return risk_components
    
    def _analyze_performance_metrics(self) -> Dict[str, Any]:
        """Analyze performance metrics and KPIs."""
        metrics = {
            'auc_metrics': [],
            'accuracy_metrics': [],
            'trading_metrics': [],
            'custom_metrics': [],
            'metric_calculations': []
        }
        
        metric_patterns = {
            'auc': [r'auc', r'roc_auc', r'area.*under.*curve'],
            'accuracy': [r'accuracy', r'correct', r'precision', r'recall'],
            'trading': [r'profit', r'return', r'drawdown', r'sharpe', r'sortino'],
            'custom': [r'def.*metric', r'def.*score', r'def.*performance']
        }
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for category, patterns in metric_patterns.items():
                    for pattern in patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            # Find line number
                            line_num = content[:match.start()].count('\n') + 1
                            
                            metric_info = {
                                'metric_type': category,
                                'text': match.group(),
                                'file': str(py_file.relative_to(self.project_root)),
                                'line': line_num,
                                'pattern': pattern
                            }
                            
                            metrics[f'{category}_metrics'].append(metric_info)
                
            except Exception as e:
                continue
        
        metrics['metric_coverage_score'] = self._calculate_metric_coverage_score(metrics)
        metrics['metric_recommendations'] = self._generate_metric_recommendations(metrics)
        
        return metrics
    
    def _identify_strategy_patterns(self) -> Dict[str, Any]:
        """Identify common strategy patterns and anti-patterns."""
        patterns = {
            'good_patterns': [],
            'anti_patterns': [],
            'missing_patterns': [],
            'pattern_recommendations': []
        }
        
        # Good patterns to look for
        good_pattern_checks = [
            ('error_handling', r'try\s*:.*except', 'Proper error handling'),
            ('logging', r'log|print.*info|debug', 'Logging for debugging'),
            ('validation', r'assert|validate|check', 'Input validation'),
            ('modular_design', r'def\s+\w+.*:', 'Modular function design'),
            ('type_hints', r'def\s+\w+.*->.*:', 'Type hints usage')
        ]
        
        # Anti-patterns to avoid
        anti_pattern_checks = [
            ('hardcoded_values', r'\d+\.\d+(?!\s*[*/+-])', 'Hardcoded magic numbers'),
            ('global_variables', r'global\s+\w+', 'Global variable usage'),
            ('deep_nesting', r'(\s{8,}if|\s{12,}for)', 'Deep nesting levels'),
            ('long_functions', None, 'Functions with >50 lines'),  # Special check
            ('no_docstrings', None, 'Functions without docstrings')  # Special check
        ]
        
        total_files = 0
        for py_file in self.project_root.rglob("*.py"):
            total_files += 1
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                # Check good patterns
                for pattern_name, pattern_regex, description in good_pattern_checks:
                    if pattern_regex and re.search(pattern_regex, content, re.IGNORECASE):
                        patterns['good_patterns'].append({
                            'pattern': pattern_name,
                            'description': description,
                            'file': str(py_file.relative_to(self.project_root))
                        })
                
                # Check anti-patterns
                for pattern_name, pattern_regex, description in anti_pattern_checks:
                    if pattern_name == 'long_functions':
                        # Special check for long functions
                        long_funcs = self._find_long_functions(content)
                        for func in long_funcs:
                            patterns['anti_patterns'].append({
                                'pattern': pattern_name,
                                'description': f'Function {func["name"]} has {func["lines"]} lines',
                                'file': str(py_file.relative_to(self.project_root)),
                                'line': func['line']
                            })
                    
                    elif pattern_name == 'no_docstrings':
                        # Special check for missing docstrings
                        missing_docs = self._find_functions_without_docstrings(content)
                        for func in missing_docs:
                            patterns['anti_patterns'].append({
                                'pattern': pattern_name,
                                'description': f'Function {func["name"]} lacks docstring',
                                'file': str(py_file.relative_to(self.project_root)),
                                'line': func['line']
                            })
                    
                    elif pattern_regex and re.search(pattern_regex, content, re.IGNORECASE):
                        patterns['anti_patterns'].append({
                            'pattern': pattern_name,
                            'description': description,
                            'file': str(py_file.relative_to(self.project_root))
                        })
                
            except Exception as e:
                continue
        
        # Identify missing patterns
        patterns['missing_patterns'] = self._identify_missing_patterns(patterns['good_patterns'])
        patterns['pattern_recommendations'] = self._generate_pattern_recommendations(patterns)
        
        return patterns
    
    def _assess_business_logic_quality(self) -> Dict[str, Any]:
        """Assess overall business logic quality."""
        strategies = self._analyze_trading_strategies()
        rules = self._extract_business_rules()
        decision_logic = self._analyze_decision_logic()
        risk_mgmt = self._analyze_risk_management()
        metrics = self._analyze_performance_metrics()
        patterns = self._identify_strategy_patterns()
        
        # Calculate quality scores
        strategy_score = min(1.0, len(strategies['strategy_functions']) / 10.0)
        rules_score = min(1.0, len(rules) / 20.0)
        decision_score = 1.0 - min(1.0, decision_logic['complexity_score'] / 100.0)
        risk_score = risk_mgmt.get('risk_management_score', 0.5)
        metrics_score = metrics.get('metric_coverage_score', 0.5)
        
        good_patterns = len(patterns['good_patterns'])
        anti_patterns = len(patterns['anti_patterns'])
        pattern_score = good_patterns / max(1, good_patterns + anti_patterns)
        
        overall_score = (
            strategy_score * 0.2 +
            rules_score * 0.15 +
            decision_score * 0.2 +
            risk_score * 0.2 +
            metrics_score * 0.15 +
            pattern_score * 0.1
        )
        
        return {
            'overall_quality_score': overall_score,
            'component_scores': {
                'strategy_implementation': strategy_score,
                'business_rules': rules_score,
                'decision_logic': decision_score,
                'risk_management': risk_score,
                'metrics_coverage': metrics_score,
                'code_patterns': pattern_score
            },
            'quality_level': self._classify_quality_level(overall_score),
            'improvement_priorities': self._prioritize_improvements(overall_score, {
                'strategy': strategy_score,
                'rules': rules_score,
                'decision': decision_score,
                'risk': risk_score,
                'metrics': metrics_score,
                'patterns': pattern_score
            })
        }
    
    def _categorize_strategy_function(self, func_name: str) -> str:
        """Categorize strategy function by name."""
        if any(keyword in func_name for keyword in ['signal', 'indicator']):
            return 'signal_generation'
        elif any(keyword in func_name for keyword in ['entry', 'buy']):
            return 'entry_logic'
        elif any(keyword in func_name for keyword in ['exit', 'sell']):
            return 'exit_logic'
        elif any(keyword in func_name for keyword in ['strategy', 'trade']):
            return 'strategy_main'
        else:
            return 'other'
    
    def _extract_indicators(self, content: str) -> List[str]:
        """Extract technical indicators from content."""
        indicator_patterns = [
            r'sma', r'ema', r'rsi', r'macd', r'bollinger', r'stochastic',
            r'moving_average', r'relative_strength', r'momentum',
            r'atr', r'adx', r'cci', r'williams'
        ]
        
        indicators = []
        for pattern in indicator_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                indicators.append(pattern.upper())
        
        return indicators
    
    def _extract_timeframes(self, content: str) -> List[str]:
        """Extract timeframes from content."""
        timeframe_patterns = [
            r'M1', r'M5', r'M15', r'M30', r'H1', r'H4', r'D1', r'W1',
            r'1min', r'5min', r'15min', r'30min', r'1hour', r'4hour', r'daily'
        ]
        
        timeframes = []
        for pattern in timeframe_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                timeframes.append(pattern)
        
        return timeframes
    
    def _assess_strategy_complexity(self, strategies: Dict[str, Any]) -> str:
        """Assess strategy implementation complexity."""
        total_functions = len(strategies['strategy_functions'])
        total_indicators = len(strategies['indicators_used'])
        total_signals = len(strategies['entry_signals']) + len(strategies['exit_signals'])
        
        complexity_score = total_functions + total_indicators * 2 + total_signals
        
        if complexity_score < 5:
            return 'simple'
        elif complexity_score < 15:
            return 'moderate'
        elif complexity_score < 30:
            return 'complex'
        else:
            return 'very_complex'
    
    def _classify_business_rule(self, rule_text: str) -> str:
        """Classify business rule by type."""
        rule_lower = rule_text.lower()
        
        if any(keyword in rule_lower for keyword in ['risk', 'limit', 'maximum']):
            return 'risk_constraint'
        elif any(keyword in rule_lower for keyword in ['profit', 'target']):
            return 'profit_target'
        elif any(keyword in rule_lower for keyword in ['stop', 'loss']):
            return 'stop_loss'
        elif any(keyword in rule_lower for keyword in ['threshold', 'minimum']):
            return 'threshold_rule'
        elif any(keyword in rule_lower for keyword in ['auc', 'accuracy']):
            return 'performance_rule'
        else:
            return 'other'
    
    def _analyze_if_statement(self, node: ast.If, py_file: Path) -> Optional[Dict[str, Any]]:
        """Analyze an if statement for decision logic."""
        try:
            # Extract condition information
            condition_text = ast.unparse(node.test) if hasattr(ast, 'unparse') else 'complex_condition'
            
            return {
                'condition': condition_text,
                'file': str(py_file.relative_to(self.project_root)),
                'line': node.lineno,
                'has_else': node.orelse is not None,
                'nested_depth': self._get_node_depth(node)
            }
        except:
            return None
    
    def _calculate_nesting_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth in AST."""
        max_depth = 0
        
        def visit_node(node, depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    visit_node(child, depth + 1)
                else:
                    visit_node(child, depth)
        
        visit_node(tree)
        return max_depth
    
    def _get_node_depth(self, node: ast.AST) -> int:
        """Get the nesting depth of a specific node."""
        # Simplified depth calculation
        return 1  # Placeholder implementation
    
    def _calculate_decision_complexity_score(self, logic_complexity: Dict[str, Any]) -> float:
        """Calculate decision complexity score."""
        total_ifs = sum(file_data['if_statements'] for file_data in logic_complexity.values())
        total_depth = sum(file_data['nested_depth'] for file_data in logic_complexity.values())
        total_decisions = sum(len(file_data['decision_functions']) for file_data in logic_complexity.values())
        
        return total_ifs + total_depth * 2 + total_decisions
    
    def _suggest_decision_optimizations(self, decision_points: List[Dict[str, Any]], 
                                       logic_complexity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest optimizations for decision logic."""
        suggestions = []
        
        # Check for high complexity files
        for file_path, complexity in logic_complexity.items():
            if complexity['nested_depth'] > 5:
                suggestions.append({
                    'type': 'reduce_nesting',
                    'file': file_path,
                    'description': f"Reduce nesting depth (current: {complexity['nested_depth']})",
                    'priority': 'high'
                })
            
            if complexity['if_statements'] > 20:
                suggestions.append({
                    'type': 'simplify_conditions',
                    'file': file_path,
                    'description': f"Simplify complex conditional logic ({complexity['if_statements']} if statements)",
                    'priority': 'medium'
                })
        
        return suggestions
    
    def _calculate_risk_management_score(self, risk_components: Dict[str, Any]) -> float:
        """Calculate risk management maturity score."""
        score = 0.0
        
        # Points for having different risk components
        if risk_components['stop_loss_implementations']:
            score += 0.3
        if risk_components['risk_limits']:
            score += 0.2
        if risk_components['position_sizing']:
            score += 0.2
        if risk_components['validation_checks']:
            score += 0.3
        
        return min(1.0, score)
    
    def _generate_risk_recommendations(self, risk_components: Dict[str, Any]) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        if not risk_components['stop_loss_implementations']:
            recommendations.append("Implement stop-loss mechanisms to limit downside risk")
        
        if not risk_components['risk_limits']:
            recommendations.append("Add risk limits and thresholds for position sizing")
        
        if not risk_components['position_sizing']:
            recommendations.append("Implement position sizing algorithms")
        
        if not risk_components['validation_checks']:
            recommendations.append("Add input validation and data quality checks")
        
        if len(recommendations) == 0:
            recommendations.append("Risk management appears comprehensive")
        
        return recommendations
    
    def _calculate_metric_coverage_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate metrics coverage score."""
        total_metrics = sum(len(metrics[key]) for key in metrics if key.endswith('_metrics'))
        
        # Normalize score
        return min(1.0, total_metrics / 20.0)
    
    def _generate_metric_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate metrics recommendations."""
        recommendations = []
        
        if not metrics['auc_metrics']:
            recommendations.append("Add AUC/ROC metrics for model evaluation")
        
        if not metrics['accuracy_metrics']:
            recommendations.append("Implement accuracy and precision/recall metrics")
        
        if not metrics['trading_metrics']:
            recommendations.append("Add trading-specific metrics (Sharpe ratio, drawdown, etc.)")
        
        if not metrics['custom_metrics']:
            recommendations.append("Consider adding custom business-specific metrics")
        
        return recommendations or ["Metrics coverage appears adequate"]
    
    def _find_long_functions(self, content: str) -> List[Dict[str, Any]]:
        """Find functions longer than 50 lines."""
        long_functions = []
        
        try:
            tree = ast.parse(content)
            lines = content.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Calculate function length
                    if hasattr(node, 'end_lineno'):
                        func_length = node.end_lineno - node.lineno + 1
                    else:
                        # Fallback: count lines until next function or end
                        func_length = 50  # Default assumption
                    
                    if func_length > 50:
                        long_functions.append({
                            'name': node.name,
                            'line': node.lineno,
                            'lines': func_length
                        })
        
        except Exception:
            pass
        
        return long_functions
    
    def _find_functions_without_docstrings(self, content: str) -> List[Dict[str, Any]]:
        """Find functions without docstrings."""
        missing_docs = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if function has docstring
                    has_docstring = (
                        node.body and
                        isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, ast.Str)
                    )
                    
                    if not has_docstring:
                        missing_docs.append({
                            'name': node.name,
                            'line': node.lineno
                        })
        
        except Exception:
            pass
        
        return missing_docs
    
    def _identify_missing_patterns(self, good_patterns: List[Dict[str, Any]]) -> List[str]:
        """Identify missing good patterns."""
        pattern_types = set(pattern['pattern'] for pattern in good_patterns)
        expected_patterns = {'error_handling', 'logging', 'validation', 'modular_design', 'type_hints'}
        
        return list(expected_patterns - pattern_types)
    
    def _generate_pattern_recommendations(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate pattern improvement recommendations."""
        recommendations = []
        
        for missing in patterns['missing_patterns']:
            if missing == 'error_handling':
                recommendations.append("Add comprehensive error handling with try-except blocks")
            elif missing == 'logging':
                recommendations.append("Implement logging for better debugging and monitoring")
            elif missing == 'validation':
                recommendations.append("Add input validation and assertion checks")
            elif missing == 'type_hints':
                recommendations.append("Add type hints for better code documentation")
        
        if patterns['anti_patterns']:
            recommendations.append(f"Address {len(patterns['anti_patterns'])} code anti-patterns")
        
        return recommendations or ["Code patterns appear good"]
    
    def _classify_quality_level(self, score: float) -> str:
        """Classify quality level based on score."""
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.7:
            return 'good'
        elif score >= 0.5:
            return 'moderate'
        elif score >= 0.3:
            return 'poor'
        else:
            return 'very_poor'
    
    def _prioritize_improvements(self, overall_score: float, component_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Prioritize improvement areas."""
        improvements = []
        
        # Sort components by score (lowest first = highest priority)
        sorted_components = sorted(component_scores.items(), key=lambda x: x[1])
        
        for component, score in sorted_components[:3]:  # Top 3 priorities
            if score < 0.6:  # Only suggest if below threshold
                improvements.append({
                    'area': component,
                    'current_score': score,
                    'priority': 'high' if score < 0.3 else 'medium',
                    'improvement_potential': 0.8 - score  # Target score - current
                })
        
        return improvements
