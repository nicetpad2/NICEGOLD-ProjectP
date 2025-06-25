from agent.agent_controller import AgentController
from datetime import datetime
import json
import logging
import os
import sys
import time
"""
🚀 Enhanced Agent System Demo
Demonstration of the advanced Agent capabilities for ProjectP

ตัวอย่างการใช้งาน Agent System ที่พัฒนาใหม่
รวมถึง deep understanding, monitoring, และ auto improvement
"""


# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


# Setup logging
logging.basicConfig(
    level = logging.INFO, 
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_basic_analysis():
    """Demo การวิเคราะห์พื้นฐาน"""
    print("🔍 DEMO: Basic Project Analysis")
    print(" = " * 60)

    agent = AgentController(project_root)

    # Run basic analysis
    results = agent.analyze_project()

    print(f"📊 Analysis completed!")
    print(f"🏗️ Project Structure: {len(results.get('project_structure', {}))} items analyzed")
    print(f"🔧 Code Quality Issues: {len(results.get('code_analysis', {}).get('issues', []))}")
    print(f"⚡ Optimization Suggestions: {len(results.get('optimization_suggestions', []))}")

    return results

def demo_deep_understanding():
    """Demo การวิเคราะห์เชิงลึก"""
    print("\n🧠 DEMO: Deep Understanding Analysis")
    print(" = " * 60)

    agent = AgentController(project_root)

    # Run deep analysis
    results = agent.run_comprehensive_deep_analysis()

    print(f"🔍 Deep Analysis completed!")

    # Show ML pipeline insights
    if 'ml_pipeline_analysis' in results:
        ml_analysis = results['ml_pipeline_analysis']
        print(f"🤖 ML Pipeline Components: {len(ml_analysis.get('components', []))}")
        print(f"📈 Performance Issues: {len(ml_analysis.get('performance_issues', []))}")

    # Show dependency insights
    if 'dependency_analysis' in results:
        dep_analysis = results['dependency_analysis']
        print(f"🔗 Dependencies: {len(dep_analysis.get('dependencies', []))}")
        print(f"⚠️ Dependency Issues: {len(dep_analysis.get('issues', []))}")

    return results

def demo_monitoring_system():
    """Demo ระบบ monitoring"""
    print("\n📊 DEMO: Real - time Monitoring System")
    print(" = " * 60)

    agent = AgentController(project_root)

    # Start monitoring
    print("🚀 Starting continuous monitoring...")
    agent.start_continuous_monitoring(interval = 2.0)

    # Wait a bit for metrics collection
    time.sleep(5)

    # Get monitoring status
    status = agent.get_monitoring_status()

    print(f"📈 Current Metrics:")
    current_metrics = status.get('current_metrics', {})
    if current_metrics:
        print(f"   💻 CPU Usage: {current_metrics.get('cpu_usage', 0):.1f}%")
        print(f"   🧠 Memory Usage: {current_metrics.get('memory_usage', 0):.1f}%")
        print(f"   💾 Disk Usage: {current_metrics.get('disk_usage', 0):.1f}%")

    # Check alerts
    alerts = status.get('active_alerts', [])
    print(f"🚨 Active Alerts: {len(alerts)}")

    # Stop monitoring
    agent.stop_continuous_monitoring()
    print("⏹️ Monitoring stopped")

    return status

def demo_auto_improvement():
    """Demo ระบบปรับปรุงอัตโนมัติ"""
    print("\n⚡ DEMO: Auto Improvement System")
    print(" = " * 60)

    agent = AgentController(project_root)

    # Analyze current performance
    print("📊 Analyzing current performance...")
    performance = agent.auto_improvement.analyze_current_performance()

    if performance:
        auc_mean = performance.get('walkforward_auc_mean', 0)
        f1_score = performance.get('threshold_best_f1', 0)
        overall_score = performance.get('overall_score', 0)

        print(f"🎯 Current Performance:")
        print(f"   AUC Score: {auc_mean:.3f}")
        print(f"   F1 Score: {f1_score:.3f}")
        print(f"   Overall Score: {overall_score:.3f}")

    # Identify improvement opportunities
    print("\n🔍 Identifying improvement opportunities...")
    opportunities = agent.auto_improvement.identify_improvement_opportunities(performance)

    print(f"💡 Found {len(opportunities)} improvement opportunities:")
    for i, opp in enumerate(opportunities[:3]):
        priority = opp.get('priority', 'unknown')
        desc = opp.get('description', 'No description')
        print(f"   {i + 1}. [{priority.upper()}] {desc}")

    # Run auto improvement (if opportunities exist)
    if opportunities:
        print("\n⚙️ Running auto improvement...")
        improvement_results = agent.auto_improve_project()

        implemented = len(improvement_results.get('improvements_implemented', []))
        failed = len(improvement_results.get('improvements_failed', []))

        print(f"✅ Improvements implemented: {implemented}")
        print(f"❌ Improvements failed: {failed}")

        return improvement_results
    else:
        print("✨ No improvements needed - project is already optimized!")
        return {}

def demo_comprehensive_health_check():
    """Demo การตรวจสุขภาพโปรเจกต์ทั้งหมด"""
    print("\n🏥 DEMO: Comprehensive Project Health Check")
    print(" = " * 60)

    agent = AgentController(project_root)

    # Get comprehensive health status
    health = agent.get_comprehensive_project_health()

    print("🏥 Project Health Status:")

    # Code quality
    code_quality = health.get('code_quality', {})
    issues = len(code_quality.get('issues', []))
    print(f"   📝 Code Quality: {issues} issues found")

    # Performance metrics
    perf_metrics = health.get('performance_metrics', {})
    auc_mean = perf_metrics.get('walkforward_auc_mean', 0)
    if auc_mean > 0:
        status = "✅ Good" if auc_mean >= 0.7 else "⚠️ Needs Improvement"
        print(f"   📈 Performance: {status} (AUC: {auc_mean:.3f})")
    else:
        print(f"   📈 Performance: 🔍 Not evaluated yet")

    # Integration status
    integration = health.get('integration_status', {})
    module_loaded = integration.get('module_loaded', False)
    status = "✅ Active" if module_loaded else "⚠️ Inactive"
    print(f"   🔗 Integration: {status}")

    return health

def save_demo_results(results: dict):
    """บันทึกผลลัพธ์ demo"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"agent_demo_results_{timestamp}.json"
    filepath = os.path.join(project_root, filename)

    with open(filepath, 'w', encoding = 'utf - 8') as f:
        json.dump(results, f, indent = 2, ensure_ascii = False, default = str)

    print(f"\n💾 Demo results saved to: {filename}")
    return filepath

def main():
    """Main demo function"""
    print("🚀 Enhanced Agent System Demo")
    print(" = " * 80)
    print("📝 Demonstrating advanced Agent capabilities for ProjectP")
    print("🎯 Features: Deep Analysis, Real - time Monitoring, Auto Improvement")
    print()

    demo_results = {}

    try:
        # Demo 1: Basic Analysis
        demo_results['basic_analysis'] = demo_basic_analysis()

        # Demo 2: Deep Understanding
        demo_results['deep_understanding'] = demo_deep_understanding()

        # Demo 3: Monitoring System
        demo_results['monitoring_system'] = demo_monitoring_system()

        # Demo 4: Auto Improvement
        demo_results['auto_improvement'] = demo_auto_improvement()

        # Demo 5: Health Check
        demo_results['health_check'] = demo_comprehensive_health_check()

        # Save results
        save_demo_results(demo_results)

        print("\n✅ All demos completed successfully!")
        print("🎉 Enhanced Agent System is ready for production use!")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.error(f"Demo error: {e}", exc_info = True)

if __name__ == "__main__":
    main()