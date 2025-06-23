"""
ProjectP AI Agent System
========================

A comprehensive AI agent system designed to understand, analyze, 
and improve the ProjectP machine learning trading system.

Modules:
- understanding: Project structure and component analysis
- analysis: Code quality and pattern analysis  
- auto_fix: Automated problem detection and resolution
- optimization: Performance and efficiency improvements
"""

from .agent_controller import AgentController
from .understanding.project_analyzer import ProjectUnderstanding
from .analysis.code_analyzer import CodeAnalyzer
from .auto_fix.auto_fixer import AutoFixSystem
from .optimization.project_optimizer import ProjectOptimizer

__version__ = "1.0.0"
__author__ = "ProjectP AI Agent System"

# Main interface
def run_comprehensive_analysis(project_root=None):
    """Run comprehensive project analysis using all agent systems."""
    agent = AgentController(project_root)
    return agent.run_comprehensive_analysis()

def generate_executive_summary(project_root=None):
    """Generate executive summary of project status."""
    agent = AgentController(project_root)
    return agent.generate_executive_summary()

def quick_health_check(project_root=None):
    """Quick project health assessment."""
    agent = AgentController(project_root)
    results = agent.run_comprehensive_analysis()
    health_score = results.get('summary', {}).get('project_health_score', 0)
    return {
        'health_score': health_score,
        'status': 'healthy' if health_score > 70 else 'needs_attention',
        'critical_issues': len(results.get('summary', {}).get('critical_issues', [])),
        'recommendations_count': len(results.get('recommendations', []))
    }

__all__ = [
    'AgentController',
    'ProjectUnderstanding', 
    'CodeAnalyzer',
    'AutoFixSystem',
    'ProjectOptimizer',
    'run_comprehensive_analysis',
    'generate_executive_summary',
    'quick_health_check'
]
