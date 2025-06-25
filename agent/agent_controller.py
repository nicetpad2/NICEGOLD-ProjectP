import argparse
import glob
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .analysis.code_analyzer import CodeAnalyzer
from .auto_fix.auto_fixer import AutoFixSystem
from .deep_understanding.business_logic_analyzer import BusinessLogicAnalyzer
from .deep_understanding.dependency_mapper import DependencyMapper
from .deep_understanding.ml_pipeline_analyzer import MLPipelineAnalyzer
from .deep_understanding.performance_profiler import PerformanceProfiler
from .integration import AutoImprovement, PipelineMonitor, ProjectPIntegrator
from .optimization.project_optimizer import ProjectOptimizer
from .recommendations.recommendation_engine import RecommendationEngine
from .smart_monitoring.realtime_monitor import RealtimeMonitor
from .understanding.project_analyzer import ProjectUnderstanding

"""
Agent Main Controller
===========================

Central control system for the AI Agent that coordinates all sub-systems
to provide comprehensive project understanding and improvement capabilities.
"""


# Import agent modules
try:
    # Import modules with relative imports for package execution
    pass  # All imports are already done above
except ImportError:
    # Fallback imports for standalone execution
    sys.path.append(os.path.dirname(__file__))

    # Smart monitoring

    # Recommendations

    # Integration systems

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AgentController:
    """
    Main AI Agent Controller that coordinates all project improvement activities.
    """

    def __init__(self, project_root: str = None):
        self.project_root = project_root or os.getcwd()
        self.reports_dir = os.path.join(self.project_root, "agent_reports")
        self.session_id = f"session_{int(datetime.now().timestamp())}"

        # Initialize original sub - systems
        self.understanding_system = ProjectUnderstanding(self.project_root)
        self.code_analyzer = CodeAnalyzer(self.project_root)
        self.auto_fixer = AutoFixSystem(self.project_root)
        self.optimizer = ProjectOptimizer(self.project_root)

        # Initialize new deep understanding sub - systems
        self.ml_pipeline_analyzer = MLPipelineAnalyzer(self.project_root)
        self.dependency_mapper = DependencyMapper(self.project_root)
        self.performance_profiler = PerformanceProfiler(self.project_root)
        self.business_logic_analyzer = BusinessLogicAnalyzer(self.project_root)

        # Initialize smart monitoring
        self.realtime_monitor = RealtimeMonitor(self.project_root)

        # Initialize recommendation engine
        self.recommendation_engine = RecommendationEngine(self.project_root)

        # Initialize integration systems
        self.projectp_integrator = ProjectPIntegrator(self.project_root)
        self.pipeline_monitor = PipelineMonitor(self.project_root)
        self.auto_improvement = AutoImprovement(self.project_root)

        # Create reports directory
        os.makedirs(self.reports_dir, exist_ok=True)

        # Initialize integration
        self.projectp_integrator.initialize()

        logger.info(
            f"🤖 Enhanced AI Agent Controller initialized for: {self.project_root}"
        )

    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive project analysis using all agent systems.
        """
        logger.info("🚀 Starting comprehensive project analysis...")

        analysis_results = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "project_root": self.project_root,
            "phases": {
                "understanding": {},
                "code_analysis": {},
                "auto_fixes": {},
                "optimization": {},
            },
            "summary": {},
            "recommendations": [],
            "next_steps": [],
        }

        try:
            # Phase 1: Project Understanding
            logger.info("📋 Phase 1: Project Understanding Analysis")
            understanding_results = (
                self.understanding_system.analyze_project_structure()
            )
            analysis_results["phases"]["understanding"] = understanding_results

            # Phase 2: Code Quality Analysis
            logger.info("🔍 Phase 2: Code Quality Analysis")
            code_analysis_results = self.code_analyzer.analyze_code_quality()
            analysis_results["phases"]["code_analysis"] = code_analysis_results

            # Phase 3: Auto - Fix Application
            logger.info("🔧 Phase 3: Automated Fixes")
            auto_fix_results = self.auto_fixer.run_comprehensive_fixes()
            analysis_results["phases"]["auto_fixes"] = auto_fix_results

            # Phase 4: Performance Optimization
            logger.info("⚡ Phase 4: Performance Optimization")
            optimization_results = self.optimizer.run_comprehensive_optimization()
            analysis_results["phases"]["optimization"] = optimization_results

            # Generate comprehensive summary
            analysis_results["summary"] = self._generate_comprehensive_summary(
                analysis_results
            )
            analysis_results["recommendations"] = self._generate_recommendations(
                analysis_results
            )
            analysis_results["next_steps"] = self._generate_next_steps(analysis_results)

            # Save results
            results_path = self._save_analysis_results(analysis_results)
            analysis_results["results_saved_to"] = results_path

            logger.info("✅ Comprehensive analysis completed successfully")

        except Exception as e:
            logger.error(f"❌ Error during comprehensive analysis: {e}")
            analysis_results["error"] = str(e)

        return analysis_results

    def run_comprehensive_deep_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive deep analysis using all available sub - systems.
        """
        logger.info(
            "🔍 Starting comprehensive deep analysis with enhanced capabilities..."
        )

        analysis_results = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "project_root": self.project_root,
            # Original analysis
            "project_understanding": self.understanding_system.analyze_project_structure(),
            "code_analysis": self.code_analyzer.analyze_code_quality(),
            "auto_fix_results": self.auto_fixer.run_auto_fixes(),
            "optimization_analysis": self.optimizer.analyze_project(),
            # Deep understanding analysis
            "ml_pipeline_analysis": self.ml_pipeline_analyzer.analyze_pipeline_structure(),
            "ml_pipeline_insights": self.ml_pipeline_analyzer.generate_pipeline_insights(),
            "auc_performance_analysis": self.ml_pipeline_analyzer.analyze_auc_performance(),
            "data_quality_analysis": self.ml_pipeline_analyzer.analyze_data_quality(),
            "dependency_analysis": self.dependency_mapper.analyze_dependencies(),
            "performance_analysis": self.performance_profiler.profile_project_pipeline(),
            "business_logic_analysis": self.business_logic_analyzer.analyze_business_logic(),
            # Smart monitoring setup
            "monitoring_status": self._setup_smart_monitoring(),
            # Generate comprehensive recommendations
            "intelligent_recommendations": None,  # Will be populated below
        }

        # Generate intelligent recommendations based on all analysis
        logger.info("🧠 Generating intelligent recommendations...")
        comprehensive_recommendations = (
            self.recommendation_engine.generate_comprehensive_recommendations(
                analysis_results
            )
        )
        analysis_results["intelligent_recommendations"] = comprehensive_recommendations

        # Generate enhanced summary
        analysis_results["enhanced_summary"] = self._generate_enhanced_summary(
            analysis_results
        )
        analysis_results["priority_actions"] = self._extract_priority_actions(
            analysis_results
        )
        analysis_results["implementation_roadmap"] = comprehensive_recommendations.get(
            "implementation_roadmap", {}
        )

        # Save results
        results_path = self._save_enhanced_analysis_results(analysis_results)
        analysis_results["results_saved_to"] = results_path

        logger.info("✅ Comprehensive deep analysis completed!")
        return analysis_results

    def _setup_smart_monitoring(self) -> Dict[str, Any]:
        """Setup smart monitoring system."""
        try:
            logger.info("📊 Setting up smart monitoring...")

            # Add callback for monitoring alerts
            def handle_monitoring_alert(alert_data):
                logger.warning(f"⚠️ Monitoring Alert: {alert_data}")

            self.realtime_monitor.add_callback(
                "threshold_breach", handle_monitoring_alert
            )
            self.realtime_monitor.add_callback(
                "anomaly_detected", handle_monitoring_alert
            )

            # Start monitoring in background
            self.realtime_monitor.start_monitoring()

            return {
                "monitoring_enabled": True,
                "monitoring_interval": self.realtime_monitor.monitoring_interval,
                "status": "active",
            }
        except Exception as e:
            return {"monitoring_enabled": False, "error": str(e), "status": "failed"}

    def _generate_comprehensive_summary(
        self, results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive summary of all analysis phases."""
        summary = {
            "project_health_score": 0,
            "key_metrics": {},
            "critical_issues": [],
            "improvements_made": [],
            "performance_gains": {},
            "next_priorities": [],
        }

        try:
            # Extract key metrics from each phase
            understanding = results["phases"]["understanding"]
            code_analysis = results["phases"]["code_analysis"]
            auto_fixes = results["phases"]["auto_fixes"]
            optimization = results["phases"]["optimization"]

            # Calculate project health score (0 - 100)
            health_factors = []

            # Understanding factors
            if understanding:
                total_files = understanding.get("code_quality", {}).get(
                    "total_files", 0
                )
                if total_files > 0:
                    health_factors.append(
                        min(100, (total_files * 2))
                    )  # More files = better structure

            # Code quality factors
            if code_analysis and code_analysis.get("issues"):
                total_issues = len(code_analysis["issues"])
                critical_issues = sum(
                    1
                    for issue in code_analysis["issues"]
                    if issue.get("severity") == "critical"
                )

                if total_issues > 0:
                    quality_score = max(
                        0, 100 - (critical_issues * 20) - (total_issues * 2)
                    )
                    health_factors.append(quality_score)

                summary["critical_issues"] = [
                    issue
                    for issue in code_analysis["issues"]
                    if issue.get("severity") in ["critical", "high"]
                ][
                    :10
                ]  # Top 10 critical issues

            # Auto - fix factors
            if auto_fixes:
                fixes_successful = auto_fixes.get("fixes_successful", 0)
                fixes_attempted = auto_fixes.get("fixes_attempted", 1)
                fix_rate = (
                    (fixes_successful / fixes_attempted) * 100
                    if fixes_attempted > 0
                    else 0
                )
                health_factors.append(fix_rate)

                summary["improvements_made"] = [
                    f"{fixes_successful} successful automated fixes applied",
                    f"Fix success rate: {fix_rate:.1f}%",
                ]

            # Optimization factors
            if optimization:
                overall_improvement = optimization.get("overall_improvement", {})
                improvement_score = overall_improvement.get("overall_score", 0)
                health_factors.append(
                    max(0, 50 + improvement_score)
                )  # Base 50 + improvement

                summary["performance_gains"] = {
                    "memory_improvement": overall_improvement.get(
                        "memory_improvement", 0
                    ),
                    "performance_improvement": overall_improvement.get(
                        "performance_improvement", 0
                    ),
                    "overall_improvement": improvement_score,
                }

            # Calculate overall health score
            if health_factors:
                summary["project_health_score"] = sum(health_factors) / len(
                    health_factors
                )

            # Key metrics
            summary["key_metrics"] = {
                "total_files": understanding.get("code_quality", {}).get(
                    "total_files", 0
                ),
                "total_lines": understanding.get("code_quality", {}).get(
                    "total_lines", 0
                ),
                "total_issues_found": len(code_analysis.get("issues", [])),
                "critical_issues_count": len(summary["critical_issues"]),
                "fixes_applied": auto_fixes.get("fixes_successful", 0),
                "optimization_opportunities": len(
                    optimization.get("optimizations", {})
                ),
            }

            # Next priorities based on findings
            if summary["critical_issues"]:
                summary["next_priorities"].append(
                    "Address critical code quality issues"
                )

            if summary["project_health_score"] < 60:
                summary["next_priorities"].append("Improve overall project health")

            if auto_fixes.get("fixes_failed", 0) > 0:
                summary["next_priorities"].append(
                    "Review and manually fix failed auto - fixes"
                )

        except Exception as e:
            logger.warning(f"Error generating summary: {e}")
            summary["error"] = str(e)

        return summary

    def _generate_recommendations(
        self, results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        try:
            summary = results.get("summary", {})
            health_score = summary.get("project_health_score", 0)

            # Health - based recommendations
            if health_score < 30:
                recommendations.append(
                    {
                        "priority": "critical",
                        "category": "project_health",
                        "title": "Critical Project Health Issues",
                        "description": "Project health score is critically low",
                        "action": "Address all critical issues immediately and consider major refactoring",
                        "estimated_effort": "high",
                        "expected_impact": "high",
                    }
                )
            elif health_score < 60:
                recommendations.append(
                    {
                        "priority": "high",
                        "category": "project_health",
                        "title": "Improve Project Health",
                        "description": "Project health score needs improvement",
                        "action": "Focus on fixing critical issues and applying optimizations",
                        "estimated_effort": "medium",
                        "expected_impact": "high",
                    }
                )

            # Code quality recommendations
            code_analysis = results["phases"].get("code_analysis", {})
            if code_analysis.get("issues"):
                critical_count = len(
                    [
                        i
                        for i in code_analysis["issues"]
                        if i.get("severity") == "critical"
                    ]
                )
                if critical_count > 0:
                    recommendations.append(
                        {
                            "priority": "high",
                            "category": "code_quality",
                            "title": "Fix Critical Code Issues",
                            "description": f"Found {critical_count} critical code issues",
                            "action": "Review and fix all critical issues manually",
                            "estimated_effort": "medium",
                            "expected_impact": "high",
                        }
                    )

            # Auto - fix recommendations
            auto_fixes = results["phases"].get("auto_fixes", {})
            failed_fixes = auto_fixes.get("fixes_failed", 0)
            if failed_fixes > 0:
                recommendations.append(
                    {
                        "priority": "medium",
                        "category": "maintenance",
                        "title": "Review Failed Auto - Fixes",
                        "description": f"{failed_fixes} automated fixes failed",
                        "action": "Manually review and apply failed fixes",
                        "estimated_effort": "low",
                        "expected_impact": "medium",
                    }
                )

            # Optimization recommendations
            optimization = results["phases"].get("optimization", {})
            if optimization.get("optimizations"):
                recommendations.append(
                    {
                        "priority": "medium",
                        "category": "performance",
                        "title": "Apply Performance Optimizations",
                        "description": "Several optimization opportunities identified",
                        "action": "Implement suggested performance optimizations",
                        "estimated_effort": "medium",
                        "expected_impact": "medium",
                    }
                )

            # ML - specific recommendations
            key_metrics = summary.get("key_metrics", {})
            if key_metrics.get("total_files", 0) > 0:
                recommendations.append(
                    {
                        "priority": "medium",
                        "category": "ml_pipeline",
                        "title": "Enhance ML Pipeline",
                        "description": "Opportunities for ML pipeline improvements",
                        "action": "Review and enhance ML - specific components",
                        "estimated_effort": "medium",
                        "expected_impact": "high",
                    }
                )

            # Sort by priority
            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            recommendations.sort(key=lambda x: priority_order.get(x["priority"], 4))

        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")

        return recommendations

    def _generate_next_steps(self, results: Dict[str, Any]) -> List[str]:
        """Generate specific next steps for project improvement."""
        next_steps = []

        try:
            summary = results.get("summary", {})
            recommendations = results.get("recommendations", [])

            # High - priority next steps
            critical_recs = [
                r for r in recommendations if r.get("priority") == "critical"
            ]
            if critical_recs:
                next_steps.append(
                    "🚨 URGENT: Address critical project health issues immediately"
                )

            high_recs = [r for r in recommendations if r.get("priority") == "high"]
            if high_recs:
                next_steps.append("⚠️ HIGH: Fix critical code quality issues")

            # Specific action steps
            critical_issues = summary.get("critical_issues", [])
            if critical_issues:
                next_steps.append(
                    f"🔧 Fix {len(critical_issues)} critical code issues identified"
                )

            auto_fixes = results["phases"].get("auto_fixes", {})
            if auto_fixes.get("fixes_successful", 0) > 0:
                next_steps.append("✅ Review and validate applied automated fixes")

            # Performance steps
            health_score = summary.get("project_health_score", 0)
            if health_score < 80:
                next_steps.append(
                    "📈 Implement performance optimizations to improve health score"
                )

            # Long - term steps
            next_steps.extend(
                [
                    "🔄 Set up continuous monitoring for project health",
                    "📚 Enhance documentation based on analysis findings",
                    "🧪 Implement automated testing for stability",
                    "🚀 Consider CI/CD pipeline improvements",
                ]
            )

        except Exception as e:
            logger.warning(f"Error generating next steps: {e}")

        return next_steps[:10]  # Limit to top 10 steps

    def _save_analysis_results(self, results: Dict[str, Any]) -> str:
        """Save comprehensive analysis results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_analysis_{timestamp}.json"
        filepath = os.path.join(self.reports_dir, filename)

        try:
            with open(filepath, "w", encoding="utf - 8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"📊 Analysis results saved to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"❌ Failed to save analysis results: {e}")
            return ""

    def generate_executive_summary(self) -> str:
        """Generate executive summary report."""
        results = self.run_comprehensive_analysis()
        summary = results.get("summary", {})
        recommendations = results.get("recommendations", [])
        next_steps = results.get("next_steps", [])

        report = f"""
# 🤖 AI Agent Analysis Report
**Project**: {os.path.basename(self.project_root)}
**Generated**: {datetime.now().strftime('%Y - %m - %d %H:%M:%S')}
**Session**: {self.session_id}

## 📊 Executive Summary

### Project Health Score: {summary.get('project_health_score', 0):.1f}/100
{self._get_health_status_emoji(summary.get('project_health_score', 0))} **Status**: {self._get_health_status(summary.get('project_health_score', 0))}

### Key Metrics
- **Total Files Analyzed**: {summary.get('key_metrics', {}).get('total_files', 0):, }
- **Lines of Code**: {summary.get('key_metrics', {}).get('total_lines', 0):, }
- **Issues Found**: {summary.get('key_metrics', {}).get('total_issues_found', 0)}
- **Critical Issues**: {summary.get('key_metrics', {}).get('critical_issues_count', 0)}
- **Fixes Applied**: {summary.get('key_metrics', {}).get('fixes_applied', 0)}

### Performance Improvements
- **Memory Optimization**: {summary.get('performance_gains', {}).get('memory_improvement', 0):.1f}%
- **Performance Gain**: {summary.get('performance_gains', {}).get('performance_improvement', 0):.1f}%
- **Overall Improvement**: {summary.get('performance_gains', {}).get('overall_improvement', 0):.1f}%

## 🎯 Priority Recommendations
"""

        # Show top 5 recommendations
        for i, rec in enumerate(recommendations[:5], 1):
            priority_emoji = {
                "critical": "🚨",
                "high": "⚠️",
                "medium": "📋",
                "low": "💡",
            }.get(rec["priority"], "📋")
            report += f"""
{i}. {priority_emoji} **{rec['title']}** ({rec['priority'].upper()})
   - {rec['description']}
   - **Action**: {rec['action']}
   - **Effort**: {rec['estimated_effort']} | **Impact**: {rec['expected_impact']}
"""

        report += f"""
## 🚀 Immediate Next Steps
"""
        for i, step in enumerate(next_steps[:7], 1):
            report += f"{i}. {step}\n"

        report += f"""
## 📈 Analysis Phases Completed
✅ **Project Understanding**: Structure and component analysis
✅ **Code Quality Analysis**: Issue detection and quality metrics
✅ **Automated Fixes**: {results['phases'].get('auto_fixes', {}).get('fixes_successful', 0)} fixes applied
✅ **Performance Optimization**: Optimization opportunities identified

## 📁 Detailed Reports
- **Full Analysis Results**: `{results.get('results_saved_to', 'Not saved')}`
- **Session ID**: `{self.session_id}`

 -  -  - 
*Generated by ProjectP AI Agent System*
"""

        return report

    def _get_health_status(self, score: float) -> str:
        """Get health status description."""
        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Needs Improvement"
        elif score >= 20:
            return "Poor"
        else:
            return "Critical"

    def _get_health_status_emoji(self, score: float) -> str:
        """Get health status emoji."""
        if score >= 80:
            return "🟢"
        elif score >= 60:
            return "🟡"
        elif score >= 40:
            return "🟠"
        else:
            return "🔴"

    def save_executive_summary(self) -> str:
        """Save executive summary to file."""
        summary = self.generate_executive_summary()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"executive_summary_{timestamp}.md"
        filepath = os.path.join(self.reports_dir, filename)

        try:
            with open(filepath, "w", encoding="utf - 8") as f:
                f.write(summary)

            logger.info(f"📋 Executive summary saved to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"❌ Failed to save executive summary: {e}")
            return ""

    def run_with_monitoring(self, command: str, *args, **kwargs):
        """รัน ProjectP command พร้อม monitoring"""
        return self.projectp_integrator.run_with_monitoring(command, *args, **kwargs)

    def start_continuous_monitoring(self, interval: float = 10.0):
        """เริ่ม continuous monitoring"""
        self.pipeline_monitor.start_monitoring(interval)
        logger.info("Continuous monitoring started")

    def stop_continuous_monitoring(self):
        """หยุด continuous monitoring"""
        self.pipeline_monitor.stop_monitoring()
        logger.info("Continuous monitoring stopped")

    def get_monitoring_status(self) -> Dict[str, Any]:
        """ดึงสถานะ monitoring"""
        return {
            "pipeline_status": self.projectp_integrator.get_pipeline_status(),
            "current_metrics": self.pipeline_monitor.get_current_metrics(),
            "active_alerts": self.pipeline_monitor.get_active_alerts(),
            "improvement_opportunities": self.auto_improvement.identify_improvement_opportunities(
                self.auto_improvement.analyze_current_performance()
            ),
        }

    def auto_improve_project(self) -> Dict[str, Any]:
        """รันการปรับปรุงอัตโนมัติ"""
        # วิเคราะห์ประสิทธิภาพปัจจุบัน
        performance = self.auto_improvement.analyze_current_performance()

        # ระบุโอกาสในการปรับปรุง
        opportunities = self.auto_improvement.identify_improvement_opportunities(
            performance
        )

        # ดำเนินการปรับปรุง
        results = self.auto_improvement.implement_improvements(opportunities)

        # สร้างรายงาน
        report_path = self.auto_improvement.generate_improvement_report()

        return {
            "performance_before": results["performance_before"],
            "performance_after": results["performance_after"],
            "improvements_implemented": results["implemented"],
            "improvements_failed": results["failed"],
            "report_path": report_path,
            "timestamp": datetime.now().isoformat(),
        }

    def register_monitoring_hook(self, event: str, callback):
        """ลงทะเบียน monitoring hook"""
        self.projectp_integrator.register_hook(event, callback)

    def get_comprehensive_project_health(self) -> Dict[str, Any]:
        """ดึงสุขภาพโปรเจกต์ทั้งหมด"""
        return {
            "code_quality": self.code_analyzer.analyze_project(),
            "performance_metrics": self.auto_improvement.analyze_current_performance(),
            "monitoring_status": self.get_monitoring_status(),
            "integration_status": self.projectp_integrator.get_pipeline_status(),
            "last_analysis": datetime.now().isoformat(),
        }

    def _generate_enhanced_summary(self, results: Dict[str, Any]) -> str:
        """สร้าง enhanced summary"""
        summary_parts = []

        summary_parts.append("🔍 Enhanced Project Analysis Summary")
        summary_parts.append(" = " * 50)

        # Add performance metrics
        if "performance_metrics" in results:
            perf = results["performance_metrics"]
            auc_mean = perf.get("walkforward_auc_mean", 0)
            summary_parts.append(f"📊 Current AUC: {auc_mean:.3f}")

            if auc_mean >= 0.7:
                summary_parts.append("✅ AUC meets target (≥0.70)")
            else:
                summary_parts.append("⚠️ AUC below target (≥0.70)")

        # Add monitoring status
        if "monitoring_status" in results:
            monitoring = results["monitoring_status"]
            if monitoring.get("active_alerts"):
                summary_parts.append(
                    f"🚨 Active alerts: {len(monitoring['active_alerts'])}"
                )
            else:
                summary_parts.append("✅ No active monitoring alerts")

        return "\n".join(summary_parts)

    def _extract_priority_actions(self, results: Dict[str, Any]) -> List[str]:
        """สกัดความสำคัญของการกระทำ"""
        actions = []

        # Check performance
        if "performance_metrics" in results:
            perf = results["performance_metrics"]
            auc_mean = perf.get("walkforward_auc_mean", 0)

            if auc_mean < 0.7:
                actions.append(
                    "🎯 Priority: Improve AUC performance (current: {:.3f})".format(
                        auc_mean
                    )
                )

        # Check monitoring alerts
        if "monitoring_status" in results:
            monitoring = results["monitoring_status"]
            alerts = monitoring.get("active_alerts", [])

            critical_alerts = [a for a in alerts if a.get("severity") == "critical"]
            if critical_alerts:
                actions.append(
                    f"🚨 Urgent: Address {len(critical_alerts)} critical alerts"
                )

        # Add improvement opportunities
        if "improvement_opportunities" in results:
            opportunities = results.get("improvement_opportunities", [])
            high_priority = [o for o in opportunities if o.get("priority") == "high"]

            if high_priority:
                actions.append(
                    f"⚡ High Priority: {len(high_priority)} improvement opportunities"
                )

        return actions[:5]  # Top 5 priority actions

    def _save_enhanced_analysis_results(self, results: Dict[str, Any]) -> str:
        """บันทึกผลลัพธ์การวิเคราะห์ที่ขยาย"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_analysis_results_{timestamp}.json"
        filepath = os.path.join(self.project_root, "agent", "reports", filename)

        # Create reports directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Add metadata
        results["metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "agent_version": "2.0.0",
            "analysis_type": "enhanced_comprehensive",
        }

        with open(filepath, "w", encoding="utf - 8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Enhanced analysis results saved to {filepath}")
        return filepath

    def run_projectp_pipeline(
        self, mode: str = "full", wait_for_completion: bool = False
    ) -> Dict[str, Any]:
        """รัน ProjectP pipeline ในโหมดต่าง ๆ"""
        logger.info(f"🚀 Starting ProjectP pipeline in {mode} mode...")

        pipeline_results: Dict[str, Any] = {
            "mode": mode,
            "started_at": datetime.now().isoformat(),
            "status": "starting",
        }

        try:
            if mode == "full":
                command = "python ProjectP.py - - run_full_pipeline"
            elif mode == "debug":
                command = "python ProjectP.py - - run_debug_full_pipeline"
            elif mode == "ultimate":
                command = "python ProjectP.py - - run_ultimate_pipeline"
            else:
                command = "python ProjectP.py - - run_full_pipeline"

            if wait_for_completion:
                logger.info(f"Running command: {command}")
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                )
                pipeline_results["return_code"] = result.returncode
                pipeline_results["status"] = (
                    "completed" if result.returncode == 0 else "failed"
                )
            else:
                logger.info(f"Starting background process: {command}")
                process = subprocess.Popen(command, shell=True, cwd=self.project_root)
                pipeline_results["process_id"] = process.pid
                pipeline_results["status"] = "running_background"

            logger.info(f"✅ ProjectP {mode} pipeline started successfully")

        except Exception as e:
            logger.error(f"❌ Error starting ProjectP {mode} pipeline: {e}")
            pipeline_results["status"] = "failed"
            pipeline_results["error"] = str(e)

        return pipeline_results

    def monitor_projectp_execution(
        self, terminal_id: str = None, timeout: int = 3600
    ) -> Dict[str, Any]:
        """
        ตรวจสอบการดำเนินการของ ProjectP pipeline

        Args:
            terminal_id: ID ของ terminal ที่รัน (ถ้ามี)
            timeout: เวลาสูงสุดในการรอ (วินาที)

        Returns:
            สถานะและผลลัพธ์ปัจจุบัน
        """
        monitoring_results = {
            "monitoring_started": datetime.now().isoformat(),
            "terminal_id": terminal_id,
            "status": "monitoring",
            "output_snippets": [],
            "metrics_detected": {},
            "errors_detected": [],
            "warnings_detected": [],
            "auc_progress": [],
        }

        try:
            # ถ้ามี terminal_id ให้เช็ค output
            if terminal_id:
                try:
                    output = self.get_terminal_output(terminal_id)
                    monitoring_results["latest_output"] = output

                    # วิเคราะห์ output หา metrics
                    if output:
                        # หา AUC values
                        auc_matches = re.findall(
                            r"AUC[:\s]*([0 - 9] + \.?[0 - 9]*)", output
                        )
                        if auc_matches:
                            latest_auc = float(auc_matches[-1])
                            monitoring_results["metrics_detected"][
                                "latest_auc"
                            ] = latest_auc
                            monitoring_results["auc_progress"].append(
                                {
                                    "timestamp": datetime.now().isoformat(),
                                    "auc": latest_auc,
                                }
                            )

                        # หา error patterns
                        if "Error" in output or "Exception" in output:
                            error_lines = [
                                line
                                for line in output.split("\n")
                                if "Error" in line or "Exception" in line
                            ]
                            monitoring_results["errors_detected"].extend(
                                error_lines[-5:]
                            )  # เก็บ 5 ล่าสุด

                        # หา warning patterns
                        if "Warning" in output or "WARNING" in output:
                            warning_lines = [
                                line
                                for line in output.split("\n")
                                if "Warning" in line or "WARNING" in line
                            ]
                            monitoring_results["warnings_detected"].extend(
                                warning_lines[-5:]
                            )  # เก็บ 5 ล่าสุด

                except Exception as e:
                    monitoring_results["monitoring_errors"] = [
                        f"Failed to get terminal output: {e}"
                    ]

            # เช็ค file outputs ที่อาจมี
            results_patterns = [
                "classification_report.json",
                "features_main.json",
                "*.log",
                "backtest_*.csv",
            ]

            found_files = []
            for pattern in results_patterns:
                matches = glob.glob(os.path.join(self.project_root, pattern))
                found_files.extend(matches)

            monitoring_results["result_files_found"] = found_files

            # ถ้ามี classification_report.json ให้อ่าน
            classification_report_path = os.path.join(
                self.project_root, "classification_report.json"
            )
            if os.path.exists(classification_report_path):
                try:
                    with open(classification_report_path, "r") as f:
                        classification_data = json.load(f)
                        monitoring_results["latest_classification_report"] = (
                            classification_data
                        )
                except Exception as e:
                    monitoring_results["warnings_detected"].append(
                        f"Failed to read classification report: {e}"
                    )

            monitoring_results["monitoring_completed"] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"❌ Error during monitoring: {e}")
            monitoring_results["monitoring_error"] = str(e)

        return monitoring_results

    def get_projectp_status_summary(self) -> str:
        """
        สร้างสรุปสถานะ ProjectP แบบอ่านง่าย
        """
        try:
            # เช็คไฟล์ผลลัพธ์ล่าสุด
            classification_report_path = os.path.join(
                self.project_root, "classification_report.json"
            )
            features_main_path = os.path.join(self.project_root, "features_main.json")

            summary_parts = []
            summary_parts.append("🔍 ProjectP Pipeline Status Summary")
            summary_parts.append(" = " * 50)

            # เช็ค classification report
            if os.path.exists(classification_report_path):
                try:
                    with open(classification_report_path, "r") as f:
                        data = json.load(f)

                    # หา AUC จากข้อมูล
                    auc_value = None
                    if "walkforward_auc_mean" in data:
                        auc_value = data["walkforward_auc_mean"]
                    elif "auc" in data:
                        auc_value = data["auc"]

                    if auc_value:
                        summary_parts.append(f"📊 Latest AUC: {auc_value:.3f}")
                        if auc_value >= 0.7:
                            summary_parts.append("✅ AUC Target Achieved (≥0.70)")
                        else:
                            summary_parts.append("⚠️ AUC Below Target (≥0.70)")

                    # เพิ่มข้อมูลอื่น ๆ
                    if "total_trades" in data:
                        summary_parts.append(f"🔄 Total Trades: {data['total_trades']}")
                    if "win_rate" in data:
                        summary_parts.append(f"🎯 Win Rate: {data['win_rate']:.1%}")

                except Exception as e:
                    summary_parts.append(f"⚠️ Error reading classification report: {e}")
            else:
                summary_parts.append("📋 No classification report found yet")

            # เช็ค features
            if os.path.exists(features_main_path):
                summary_parts.append("✅ Feature engineering completed")
            else:
                summary_parts.append("⏳ Feature engineering in progress")

            # เช็ค log files
            log_files = []
            for file in os.listdir(self.project_root):
                if file.endswith(".log"):
                    log_files.append(file)

            if log_files:
                summary_parts.append(f"📝 Log files: {', '.join(log_files[:3])}")

            # เช็คเวลาล่าสุดของไฟล์
            if os.path.exists(classification_report_path):
                last_modified = os.path.getmtime(classification_report_path)
                last_modified_str = datetime.fromtimestamp(last_modified).strftime(
                    "%Y - %m - %d %H:%M:%S"
                )
                summary_parts.append(f"🕒 Last updated: {last_modified_str}")

            summary_parts.append("")
            summary_parts.append("💡 Quick Actions:")
            summary_parts.append(
                "- agent.run_projectp_pipeline('full') - Run full pipeline"
            )
            summary_parts.append(
                "- agent.run_projectp_pipeline('until_auc_70') - Run until AUC ≥ 70%"
            )
            summary_parts.append(
                "- agent.monitor_projectp_execution() - Check current status"
            )

            return "\n".join(summary_parts)

        except Exception as e:
            return f"❌ Error generating status summary: {e}"

    def wait_for_auc_target(
        self, target_auc: float = 0.7, max_attempts: int = 10, check_interval: int = 60
    ) -> Dict[str, Any]:
        """
        รอจนกว่า AUC จะถึงเป้าหมาย

        Args:
            target_auc: เป้าหมาย AUC
            max_attempts: จำนวนครั้งสูงสุดในการลอง
            check_interval: ช่วงเวลาในการเช็ค (วินาที)

        Returns:
            ผลลัพธ์การรอ
        """
        wait_results = {
            "target_auc": target_auc,
            "max_attempts": max_attempts,
            "started_at": datetime.now().isoformat(),
            "attempts": [],
            "final_status": "in_progress",
            "final_auc": None,
        }

        logger.info(f"🎯 Waiting for AUC target: {target_auc}")

        for attempt in range(max_attempts):
            try:
                # เช็ค AUC ปัจจุบัน
                classification_report_path = os.path.join(
                    self.project_root, "classification_report.json"
                )

                if os.path.exists(classification_report_path):
                    with open(classification_report_path, "r") as f:
                        data = json.load(f)

                    current_auc = data.get("walkforward_auc_mean", data.get("auc", 0))

                    attempt_data = {
                        "attempt": attempt + 1,
                        "timestamp": datetime.now().isoformat(),
                        "auc": current_auc,
                        "target_met": current_auc >= target_auc,
                    }

                    wait_results["attempts"].append(attempt_data)

                    logger.info(f"📊 Attempt {attempt + 1}: AUC = {current_auc:.3f}")

                    if current_auc >= target_auc:
                        wait_results["final_status"] = "success"
                        wait_results["final_auc"] = current_auc
                        logger.info(
                            f"🎉 Target achieved! AUC {current_auc:.3f} >= {target_auc}"
                        )
                        break
                else:
                    logger.info(f"⏳ Attempt {attempt + 1}: No results yet, waiting...")

                # รอก่อนลองใหม่
                if attempt < max_attempts - 1:
                    time.sleep(check_interval)

            except Exception as e:
                logger.error(f"❌ Error in attempt {attempt + 1}: {e}")
                wait_results["attempts"].append(
                    {
                        "attempt": attempt + 1,
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e),
                    }
                )

        if wait_results["final_status"] == "in_progress":
            wait_results["final_status"] = "timeout"
            logger.warning(
                f"⏰ Timeout: AUC target not reached after {max_attempts} attempts"
            )

        wait_results["completed_at"] = datetime.now().isoformat()
        return wait_results


# Example usage and CLI interface
def main():
    """Main CLI interface for the agent controller."""

    parser = argparse.ArgumentParser(description="ProjectP AI Agent Controller")
    parser.add_argument(
        " -  - project - root", default=None, help="Path to project root directory"
    )
    parser.add_argument(
        " -  - action",
        choices=["analyze", "summary", "fix", "optimize"],
        default="analyze",
        help="Action to perform",
    )
    parser.add_argument(
        " -  - verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize agent controller
    agent = AgentController(args.project_root)

    if args.action == "analyze":
        print("🤖 Running comprehensive analysis...")
        results = agent.run_comprehensive_analysis()
        print(
            f"✅ Analysis complete. Results saved to: {results.get('results_saved_to', 'Unknown')}"
        )

    elif args.action == "summary":
        print("📋 Generating executive summary...")
        summary_path = agent.save_executive_summary()
        print(f"✅ Executive summary saved to: {summary_path}")

    elif args.action == "fix":
        print("🔧 Running automated fixes...")
        fix_results = agent.auto_fixer.run_comprehensive_fixes()
        print(f"✅ Applied {fix_results.get('fixes_successful', 0)} fixes successfully")

    elif args.action == "optimize":
        print("⚡ Running optimizations...")
        opt_results = agent.optimizer.run_comprehensive_optimization()
        print(f"✅ Optimization complete. Check detailed results for improvements.")


if __name__ == "__main__":
    main()
