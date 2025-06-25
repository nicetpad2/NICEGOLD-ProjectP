from datetime import datetime
from typing import Dict, Any, List
"""
Improvement Planner for Recommendations
วางแผนการปรับปรุงตาม analysis results
"""


class ImprovementPlanner:
    """วางแผนการปรับปรุงโปรเจกต์"""

    def __init__(self):
        self.improvement_plans = []
        self.priority_matrix = {
            'critical': 1, 
            'high': 2, 
            'medium': 3, 
            'low': 4
        }

    def create_improvement_plan(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """สร้างแผนการปรับปรุง"""
        plan = {
            'plan_id': f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
            'created_at': datetime.now().isoformat(), 
            'analysis_summary': self._summarize_analysis(analysis_results), 
            'improvement_items': self._generate_improvement_items(analysis_results), 
            'timeline': self._estimate_timeline(analysis_results), 
            'resources_needed': self._estimate_resources(analysis_results)
        }

        self.improvement_plans.append(plan)
        return plan

    def _summarize_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """สรุปผลการวิเคราะห์"""
        return {
            'total_issues': len(results.get('issues', [])), 
            'performance_score': results.get('performance_score', 0), 
            'critical_areas': results.get('critical_areas', []), 
            'improvement_potential': results.get('improvement_potential', 'medium')
        }

    def _generate_improvement_items(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """สร้างรายการปรับปรุง"""
        items = []

        # Add performance improvements
        if results.get('performance_score', 100) < 70:
            items.append({
                'type': 'performance', 
                'priority': 'high', 
                'title': 'Improve Model Performance', 
                'description': 'Optimize model to achieve AUC ≥ 70%', 
                'estimated_effort': 'medium', 
                'expected_impact': 'high'
            })

        # Add code quality improvements
        issues = results.get('issues', [])
        if issues:
            items.append({
                'type': 'code_quality', 
                'priority': 'medium', 
                'title': 'Fix Code Quality Issues', 
                'description': f'Address {len(issues)} code quality issues', 
                'estimated_effort': 'low', 
                'expected_impact': 'medium'
            })

        # Add optimization improvements
        if 'optimization_opportunities' in results:
            items.append({
                'type': 'optimization', 
                'priority': 'medium', 
                'title': 'Performance Optimization', 
                'description': 'Implement performance optimizations', 
                'estimated_effort': 'medium', 
                'expected_impact': 'high'
            })

        # Sort by priority
        items.sort(key = lambda x: self.priority_matrix.get(x['priority'], 5))

        return items

    def _estimate_timeline(self, results: Dict[str, Any]) -> Dict[str, str]:
        """ประเมินระยะเวลา"""
        total_issues = len(results.get('issues', []))
        performance_score = results.get('performance_score', 100)

        if performance_score < 50:
            duration = '2 - 3 weeks'
        elif performance_score < 70:
            duration = '1 - 2 weeks'
        elif total_issues > 10:
            duration = '1 week'
        else:
            duration = '2 - 3 days'

        return {
            'estimated_duration': duration, 
            'start_date': 'immediately', 
            'phases': ['analysis', 'implementation', 'testing', 'validation']
        }

    def _estimate_resources(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """ประเมินทรัพยากรที่ต้องการ"""
        return {
            'human_resources': 'Data scientist + ML engineer', 
            'computational_resources': 'Standard development environment', 
            'tools_needed': ['Python', 'ML libraries', 'monitoring tools'], 
            'estimated_cost': 'Low to Medium'
        }

    def get_latest_plan(self) -> Dict[str, Any]:
        """ดึงแผนล่าสุด"""
        if not self.improvement_plans:
            return {}
        return self.improvement_plans[ - 1]

    def get_plan_summary(self) -> Dict[str, Any]:
        """สรุปแผนทั้งหมด"""
        if not self.improvement_plans:
            return {'total_plans': 0}

        latest = self.improvement_plans[ - 1]
        return {
            'total_plans': len(self.improvement_plans), 
            'latest_plan_id': latest['plan_id'], 
            'total_improvement_items': len(latest.get('improvement_items', [])), 
            'estimated_duration': latest.get('timeline', {}).get('estimated_duration', 'unknown')
        }