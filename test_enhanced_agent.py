        from agent.agent_controller import AgentController
        from agent.deep_understanding import (
        from agent.integration import ProjectPIntegrator, PipelineMonitor, AutoImprovement
        from agent.integration.auto_improvement import AutoImprovement
        from agent.integration.pipeline_monitor import PipelineMonitor
        from agent.integration.projectp_integration import ProjectPIntegrator
        from agent.recommendations import RecommendationEngine
        from agent.smart_monitoring import RealtimeMonitor
from datetime import datetime
    import json
import os
import sys
import traceback
"""
🧪 Enhanced Agent System Integration Test
Quick test to verify all components work together
"""


# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """Test all imports work correctly"""
    print("🔍 Testing imports...")

    try:
        print("✅ AgentController import successful")

        print("✅ Integration modules import successful")

            MLPipelineAnalyzer, DependencyMapper, 
            PerformanceProfiler, BusinessLogicAnalyzer
        )
        print("✅ Deep understanding modules import successful")

        print("✅ Smart monitoring modules import successful")

        print("✅ Recommendations modules import successful")

        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_agent_initialization():
    """Test Agent initialization"""
    print("\n🚀 Testing Agent initialization...")

    try:

        agent = AgentController(project_root)
        print("✅ Agent initialized successfully")

        # Test basic attributes
        assert hasattr(agent, 'project_analyzer'), "Missing project_analyzer"
        assert hasattr(agent, 'code_analyzer'), "Missing code_analyzer"
        assert hasattr(agent, 'auto_fixer'), "Missing auto_fixer"
        assert hasattr(agent, 'optimizer'), "Missing optimizer"
        assert hasattr(agent, 'projectp_integrator'), "Missing projectp_integrator"
        assert hasattr(agent, 'pipeline_monitor'), "Missing pipeline_monitor"
        assert hasattr(agent, 'auto_improvement'), "Missing auto_improvement"

        print("✅ All required attributes present")
        return True

    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        traceback.print_exc()
        return False

def test_integration_components():
    """Test integration components"""
    print("\n🔗 Testing integration components...")

    try:

        # Test ProjectP Integrator
        integrator = ProjectPIntegrator(project_root)
        assert hasattr(integrator, 'initialize'), "Missing initialize method"
        assert hasattr(integrator, 'get_pipeline_status'), "Missing get_pipeline_status method"
        print("✅ ProjectPIntegrator works")

        # Test Pipeline Monitor
        monitor = PipelineMonitor(project_root)
        assert hasattr(monitor, 'start_monitoring'), "Missing start_monitoring method"
        assert hasattr(monitor, 'get_current_metrics'), "Missing get_current_metrics method"
        print("✅ PipelineMonitor works")

        # Test Auto Improvement
        improvement = AutoImprovement(project_root)
        assert hasattr(improvement, 'analyze_current_performance'), "Missing analyze_current_performance method"
        assert hasattr(improvement, 'identify_improvement_opportunities'), "Missing identify_improvement_opportunities method"
        print("✅ AutoImprovement works")

        return True

    except Exception as e:
        print(f"❌ Integration components test failed: {e}")
        traceback.print_exc()
        return False

def test_monitoring_system():
    """Test monitoring system functionality"""
    print("\n📊 Testing monitoring system...")

    try:

        monitor = PipelineMonitor(project_root)

        # Test metrics collection
        metrics = monitor._collect_system_metrics()
        assert 'cpu_usage' in metrics, "Missing cpu_usage metric"
        assert 'memory_usage' in metrics, "Missing memory_usage metric"
        assert 'timestamp' in metrics, "Missing timestamp"

        print(f"✅ Metrics collection works: CPU = {metrics['cpu_usage']:.1f}%, Memory = {metrics['memory_usage']:.1f}%")

        # Test thresholds
        monitor.update_thresholds({'cpu_usage': 95.0})
        assert monitor.thresholds['cpu_usage'] == 95.0, "Threshold update failed"
        print("✅ Threshold updates work")

        return True

    except Exception as e:
        print(f"❌ Monitoring system test failed: {e}")
        traceback.print_exc()
        return False

def test_improvement_analysis():
    """Test improvement analysis functionality"""
    print("\n⚡ Testing improvement analysis...")

    try:

        improvement = AutoImprovement(project_root)

        # Test performance analysis (might not have data yet)
        performance = improvement.analyze_current_performance()
        print(f"✅ Performance analysis completed: {len(performance)} metrics")

        # Test opportunity identification
        opportunities = improvement.identify_improvement_opportunities(performance)
        print(f"✅ Opportunity identification completed: {len(opportunities)} opportunities")

        # Test recommendation generation
        recommendations = improvement._generate_recommendations()
        assert isinstance(recommendations, list), "Recommendations should be a list"
        print(f"✅ Recommendations generated: {len(recommendations)} items")

        return True

    except Exception as e:
        print(f"❌ Improvement analysis test failed: {e}")
        traceback.print_exc()
        return False

def test_file_structure():
    """Test that all required files exist"""
    print("\n📁 Testing file structure...")

    required_files = [
        "agent/__init__.py", 
        "agent/agent_controller.py", 
        "agent/agent_config.yaml", 
        "agent/understanding/__init__.py", 
        "agent/analysis/__init__.py", 
        "agent/auto_fix/__init__.py", 
        "agent/optimization/__init__.py", 
        "agent/deep_understanding/__init__.py", 
        "agent/smart_monitoring/__init__.py", 
        "agent/recommendations/__init__.py", 
        "agent/integration/__init__.py", 
        "agent/integration/projectp_integration.py", 
        "agent/integration/pipeline_monitor.py", 
        "agent/integration/auto_improvement.py"
    ]

    missing_files = []
    for file_path in required_files:
        full_path = os.path.join(project_root, file_path)
        if not os.path.exists(full_path):
            missing_files.append(file_path)

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print(f"✅ All {len(required_files)} required files exist")
        return True

def generate_test_report(results):
    """Generate test report"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(project_root, f"agent_integration_test_report_{timestamp}.json")


    report = {
        'test_timestamp': datetime.now().isoformat(), 
        'project_root': project_root, 
        'test_results': results, 
        'overall_status': all(results.values()), 
        'passed_tests': sum(results.values()), 
        'total_tests': len(results)
    }

    with open(report_path, 'w', encoding = 'utf - 8') as f:
        json.dump(report, f, indent = 2, ensure_ascii = False)

    print(f"\n📄 Test report saved: {report_path}")
    return report_path

def main():
    """Run all integration tests"""
    print("🧪 Enhanced Agent System Integration Test")
    print(" = " * 60)
    print(f"📍 Project root: {project_root}")
    print(f"🕐 Test time: {datetime.now().isoformat()}")
    print()

    # Run all tests
    test_results = {}

    test_results['file_structure'] = test_file_structure()
    test_results['imports'] = test_imports()
    test_results['agent_initialization'] = test_agent_initialization()
    test_results['integration_components'] = test_integration_components()
    test_results['monitoring_system'] = test_monitoring_system()
    test_results['improvement_analysis'] = test_improvement_analysis()

    # Generate summary
    print("\n" + " = " * 60)
    print("📊 TEST SUMMARY")
    print(" = " * 60)

    passed = sum(test_results.values())
    total = len(test_results)

    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Enhanced Agent System is ready!")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")

    # Generate report
    generate_test_report(test_results)

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)