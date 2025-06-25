from datetime import datetime
from typing import Dict, Any, List, Tuple
"""
Action Prioritizer for Recommendations
จัดลำดับความสำคัญของ actions และ recommendations
"""


class ActionPrioritizer:
    """จัดลำดับความสำคัญของ actions"""

    def __init__(self):
        self.priority_weights = {
            'impact': 0.4,      # ผลกระทบ
            'urgency': 0.3,     # ความเร่งด่วน
            'effort': 0.2,      # ความพยายามที่ต้องใช้ (น้อยกว่า = ดีกว่า)
            'feasibility': 0.1  # ความเป็นไปได้
        }

        self.impact_scores = {
            'critical': 10, 
            'high': 8, 
            'medium': 5, 
            'low': 2, 
            'minimal': 1
        }

        self.urgency_scores = {
            'immediate': 10, 
            'urgent': 8, 
            'medium': 5, 
            'low': 3, 
            'later': 1
        }

        self.effort_scores = {
            'minimal': 10,  # น้อย = ดี
            'low': 8, 
            'medium': 5, 
            'high': 3, 
            'very_high': 1
        }

        self.feasibility_scores = {
            'very_easy': 10, 
            'easy': 8, 
            'medium': 5, 
            'difficult': 3, 
            'very_difficult': 1
        }

    def prioritize_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """จัดลำดับความสำคัญของ actions"""
        if not actions:
            return []

        # คำนวณ priority score สำหรับแต่ละ action
        scored_actions = []
        for action in actions:
            score = self._calculate_priority_score(action)
            action_with_score = action.copy()
            action_with_score['priority_score'] = score
            action_with_score['priority_rank'] = self._get_priority_rank(score)
            scored_actions.append(action_with_score)

        # จัดเรียงตาม priority score (สูงไปต่ำ)
        scored_actions.sort(key = lambda x: x.get('priority_score', 0), reverse = True)

        # เพิ่ม rank number
        for i, action in enumerate(scored_actions):
            action['rank'] = i + 1

        return scored_actions

    def _calculate_priority_score(self, action: Dict[str, Any]) -> float:
        """คำนวณ priority score"""
        impact = action.get('impact', 'medium')
        urgency = action.get('urgency', 'medium')
        effort = action.get('effort', 'medium')
        feasibility = action.get('feasibility', 'medium')

        impact_score = self.impact_scores.get(impact, 5)
        urgency_score = self.urgency_scores.get(urgency, 5)
        effort_score = self.effort_scores.get(effort, 5)
        feasibility_score = self.feasibility_scores.get(feasibility, 5)

        # คำนวณ weighted score
        total_score = (
            impact_score * self.priority_weights['impact'] +
            urgency_score * self.priority_weights['urgency'] +
            effort_score * self.priority_weights['effort'] +
            feasibility_score * self.priority_weights['feasibility']
        )

        return round(total_score, 2)

    def _get_priority_rank(self, score: float) -> str:
        """แปลง score เป็น priority rank"""
        if score >= 8.0:
            return 'critical'
        elif score >= 6.5:
            return 'high'
        elif score >= 4.5:
            return 'medium'
        elif score >= 2.5:
            return 'low'
        else:
            return 'minimal'

    def analyze_priorities(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """วิเคราะห์ priorities"""
        if not actions:
            return {}

        prioritized = self.prioritize_actions(actions)

        # นับจำนวนตาม priority rank
        rank_counts = {}
        for action in prioritized:
            rank = action.get('priority_rank', 'unknown')
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        # หา top priorities
        top_actions = prioritized[:5]  # Top 5

        # คำนวณ average score
        scores = [action.get('priority_score', 0) for action in prioritized]
        avg_score = sum(scores) / len(scores) if scores else 0

        return {
            'total_actions': len(prioritized), 
            'rank_distribution': rank_counts, 
            'average_priority_score': round(avg_score, 2), 
            'top_priorities': top_actions, 
            'critical_count': rank_counts.get('critical', 0), 
            'high_priority_count': rank_counts.get('high', 0)
        }

    def suggest_execution_order(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """แนะนำลำดับการดำเนินการ"""
        prioritized = self.prioritize_actions(actions)

        # จัดกลุ่มตาม priority rank
        critical = [a for a in prioritized if a.get('priority_rank') == 'critical']
        high = [a for a in prioritized if a.get('priority_rank') == 'high']
        medium = [a for a in prioritized if a.get('priority_rank') == 'medium']
        low = [a for a in prioritized if a.get('priority_rank') in ['low', 'minimal']]

        execution_plan = []

        # Phase 1: Critical (ทำทันที)
        if critical:
            execution_plan.append({
                'phase': 1, 
                'description': 'Critical Issues - Execute Immediately', 
                'actions': critical, 
                'estimated_time': self._estimate_phase_time(critical), 
                'priority': 'immediate'
            })

        # Phase 2: High Priority (ทำในสัปดาห์นี้)
        if high:
            execution_plan.append({
                'phase': 2, 
                'description': 'High Priority - Execute This Week', 
                'actions': high, 
                'estimated_time': self._estimate_phase_time(high), 
                'priority': 'this_week'
            })

        # Phase 3: Medium Priority (ทำในเดือนนี้)
        if medium:
            execution_plan.append({
                'phase': 3, 
                'description': 'Medium Priority - Execute This Month', 
                'actions': medium, 
                'estimated_time': self._estimate_phase_time(medium), 
                'priority': 'this_month'
            })

        # Phase 4: Low Priority (ทำเมื่อมีเวลา)
        if low:
            execution_plan.append({
                'phase': 4, 
                'description': 'Low Priority - Execute When Available', 
                'actions': low, 
                'estimated_time': self._estimate_phase_time(low), 
                'priority': 'when_available'
            })

        return execution_plan

    def _estimate_phase_time(self, actions: List[Dict[str, Any]]) -> str:
        """ประเมินเวลาที่ใช้สำหรับ phase"""
        if not actions:
            return '0 days'

        effort_mapping = {
            'minimal': 0.5, 
            'low': 1, 
            'medium': 3, 
            'high': 7, 
            'very_high': 14
        }

        total_days = 0
        for action in actions:
            effort = action.get('effort', 'medium')
            days = effort_mapping.get(effort, 3)
            total_days += days

        if total_days < 1:
            return 'Less than 1 day'
        elif total_days == 1:
            return '1 day'
        elif total_days <= 7:
            return f'{int(total_days)} days'
        elif total_days <= 30:
            return f'{int(total_days/7)} weeks'
        else:
            return f'{int(total_days/30)} months'

    def update_priority_weights(self, new_weights: Dict[str, float]):
        """อัพเดท priority weights"""
        # Validate weights sum to 1.0
        total = sum(new_weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

        self.priority_weights.update(new_weights)

    def get_priority_insights(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ดึง insights เกี่ยวกับ priorities"""
        analysis = self.analyze_priorities(actions)
        execution_plan = self.suggest_execution_order(actions)

        insights = []

        # Critical insights
        critical_count = analysis.get('critical_count', 0)
        if critical_count > 0:
            insights.append(f"🚨 {critical_count} critical issues require immediate attention")

        # High priority insights
        high_count = analysis.get('high_priority_count', 0)
        if high_count > 3:
            insights.append(f"⚡ {high_count} high - priority items - consider parallel execution")

        # Execution insights
        if len(execution_plan) > 2:
            insights.append("📅 Multi - phase execution recommended for optimal results")

        # Effort insights
        avg_score = analysis.get('average_priority_score', 0)
        if avg_score > 7:
            insights.append("🎯 Overall high priority level - focus on quick wins first")
        elif avg_score < 4:
            insights.append("💡 Lower priority level - good time for optimization and improvement")

        return {
            'insights': insights, 
            'recommended_focus': self._get_recommended_focus(analysis), 
            'execution_summary': f"{len(execution_plan)} phases recommended", 
            'timeline_estimate': self._get_total_timeline(execution_plan)
        }

    def _get_recommended_focus(self, analysis: Dict[str, Any]) -> str:
        """แนะนำจุดโฟกัส"""
        critical_count = analysis.get('critical_count', 0)
        high_count = analysis.get('high_priority_count', 0)

        if critical_count > 0:
            return 'immediate_critical_fixes'
        elif high_count > 2:
            return 'high_priority_batch'
        else:
            return 'steady_improvement'

    def _get_total_timeline(self, execution_plan: List[Dict[str, Any]]) -> str:
        """คำนวณ timeline รวม"""
        if not execution_plan:
            return 'No timeline required'

        # ใช้ phase ที่ยาวที่สุดเป็น baseline
        max_phase = len(execution_plan)

        if max_phase == 1:
            return 'Within 1 week'
        elif max_phase == 2:
            return '2 - 4 weeks'
        elif max_phase == 3:
            return '1 - 2 months'
        else:
            return '2 - 3 months'