    from agent import (
        from agent.auto_fix import AutoFixSystem
        from agent.optimization import ProjectOptimizer
from datetime import datetime
import os
import sys
"""
Agent System Usage Examples
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

Examples of how to use the AI Agent System for ProjectP improvement.
"""


# Add agent to path
sys.path.append(os.path.dirname(__file__))

try:
        AgentController, 
        quick_health_check, 
        run_comprehensive_analysis, 
        generate_executive_summary
    )
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"❌ Agent system not available: {e}")
    AGENT_AVAILABLE = False

def example_quick_health_check():
    """Example: Quick project health assessment."""
    print("🔍 Example: Quick Health Check")
    print(" = " * 50)

    if not AGENT_AVAILABLE:
        print("❌ Agent system not available")
        return

    try:
        # Run quick health check
        health = quick_health_check()

        print(f"📊 Project Health Score: {health['health_score']:.1f}/100")
        print(f"📈 Status: {health['status']}")
        print(f"🚨 Critical Issues: {health['critical_issues']}")
        print(f"💡 Recommendations: {health['recommendations_count']}")

        # Health assessment
        if health['health_score'] >= 80:
            print("🟢 Excellent project health!")
        elif health['health_score'] >= 60:
            print("🟡 Good project health with room for improvement")
        elif health['health_score'] >= 40:
            print("🟠 Project needs attention")
        else:
            print("🔴 Critical issues require immediate attention")

    except Exception as e:
        print(f"❌ Error during health check: {e}")

def example_comprehensive_analysis():
    """Example: Full project analysis."""
    print("\n🔬 Example: Comprehensive Analysis")
    print(" = " * 50)

    if not AGENT_AVAILABLE:
        print("❌ Agent system not available")
        return

    try:
        # Initialize agent controller
        agent = AgentController()

        print("🚀 Starting comprehensive analysis...")

        # Run full analysis
        results = agent.run_comprehensive_analysis()

        # Display results
        summary = results.get('summary', {})

        print(f"\n📊 Analysis Results:")
        print(f"   Health Score: {summary.get('project_health_score', 0):.1f}/100")
        print(f"   Total Files: {summary.get('key_metrics', {}).get('total_files', 0):, }")
        print(f"   Lines of Code: {summary.get('key_metrics', {}).get('total_lines', 0):, }")
        print(f"   Issues Found: {summary.get('key_metrics', {}).get('total_issues_found', 0)}")
        print(f"   Critical Issues: {summary.get('key_metrics', {}).get('critical_issues_count', 0)}")
        print(f"   Fixes Applied: {summary.get('key_metrics', {}).get('fixes_applied', 0)}")

        # Show top recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"\n🎯 Top Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                priority_emoji = {'critical': '🚨', 'high': '⚠️', 'medium': '📋', 'low': '💡'}.get(rec['priority'], '📋')
                print(f"   {i}. {priority_emoji} {rec['title']} ({rec['priority'].upper()})")
                print(f"      Action: {rec['action']}")

        # Show next steps
        next_steps = results.get('next_steps', [])
        if next_steps:
            print(f"\n🚀 Next Steps:")
            for i, step in enumerate(next_steps[:5], 1):
                print(f"   {i}. {step}")

        return results

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None

def example_auto_fixes():
    """Example: Automated fixes."""
    print("\n🔧 Example: Automated Fixes")
    print(" = " * 50)

    if not AGENT_AVAILABLE:
        print("❌ Agent system not available")
        return

    try:

        # Initialize auto - fix system
        auto_fixer = AutoFixSystem()

        print("🔧 Running automated fixes...")

        # Run comprehensive fixes
        results = auto_fixer.run_comprehensive_fixes()

        # Display results
        print(f"\n📊 Fix Results:")
        print(f"   Fixes Attempted: {results.get('fixes_attempted', 0)}")
        print(f"   Fixes Successful: {results.get('fixes_successful', 0)}")
        print(f"   Fixes Failed: {results.get('fixes_failed', 0)}")

        success_rate = 0
        if results.get('fixes_attempted', 0) > 0:
            success_rate = (results.get('fixes_successful', 0) / results.get('fixes_attempted', 0)) * 100
        print(f"   Success Rate: {success_rate:.1f}%")

        # Show fix categories
        categories = results.get('categories', {})
        for category, fixes in categories.items():
            if fixes:
                successful = sum(1 for fix in fixes if fix.get('success'))
                print(f"\n🔧 {category.replace('_', ' ').title()}:")
                print(f"   Applied: {successful}/{len(fixes)} fixes")

                # Show first few fixes
                for fix in fixes[:3]:
                    status = "✅" if fix.get('success') else "❌"
                    print(f"   {status} {fix.get('description', 'No description')}")

        return results

    except Exception as e:
        print(f"❌ Error during auto - fixes: {e}")
        return None

def example_optimization():
    """Example: Performance optimization."""
    print("\n⚡ Example: Performance Optimization")
    print(" = " * 50)

    if not AGENT_AVAILABLE:
        print("❌ Agent system not available")
        return

    try:

        # Initialize optimizer
        optimizer = ProjectOptimizer()

        print("⚡ Running performance optimization...")

        # Run optimization
        results = optimizer.run_comprehensive_optimization()

        # Display results
        baseline = results.get('baseline_metrics', {})
        final = results.get('final_metrics', {})
        improvements = results.get('overall_improvement', {})

        print(f"\n📊 Optimization Results:")
        print(f"   Memory Improvement: {improvements.get('memory_improvement', 0):.1f}%")
        print(f"   Performance Improvement: {improvements.get('performance_improvement', 0):.1f}%")
        print(f"   Overall Score: {improvements.get('overall_score', 0):.1f}%")

        # Show optimization categories
        optimizations = results.get('optimizations', {})
        for category, opts in optimizations.items():
            if opts:
                print(f"\n⚡ {category.replace('_', ' ').title()}:")
                print(f"   Optimizations: {len(opts)}")

                # Show first few optimizations
                for opt in opts[:3]:
                    status = "✅" if opt.get('success') else "❌"
                    print(f"   {status} {opt.get('description', 'No description')}")

        return results

    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        return None

def example_executive_summary():
    """Example: Generate executive summary."""
    print("\n📋 Example: Executive Summary")
    print(" = " * 50)

    if not AGENT_AVAILABLE:
        print("❌ Agent system not available")
        return

    try:
        # Generate executive summary
        summary = generate_executive_summary()

        print("📋 Executive Summary Generated:")
        print(" - " * 30)
        print(summary)

        return summary

    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        return None

def example_monitoring_workflow():
    """Example: Complete monitoring workflow."""
    print("\n🔄 Example: Complete Monitoring Workflow")
    print(" = " * 50)

    if not AGENT_AVAILABLE:
        print("❌ Agent system not available")
        return

    print("🔄 Running complete project monitoring workflow...")

    # Step 1: Health check
    print("\n📊 Step 1: Initial Health Assessment")
    health = quick_health_check()
    initial_score = health.get('health_score', 0)
    print(f"Initial Health Score: {initial_score:.1f}/100")

    # Step 2: Auto - fixes if needed
    if initial_score < 70:
        print("\n🔧 Step 2: Applying Automated Fixes")
        fix_results = example_auto_fixes()
        if fix_results and fix_results.get('fixes_successful', 0) > 0:
            print(f"✅ Applied {fix_results.get('fixes_successful', 0)} fixes")
        else:
            print("⚠️ No fixes applied or available")

    # Step 3: Optimization
    print("\n⚡ Step 3: Performance Optimization")
    opt_results = example_optimization()
    if opt_results:
        improvement = opt_results.get('overall_improvement', {}).get('overall_score', 0)
        print(f"📈 Overall improvement: {improvement:.1f}%")

    # Step 4: Final health check
    print("\n📊 Step 4: Final Health Assessment")
    final_health = quick_health_check()
    final_score = final_health.get('health_score', 0)
    print(f"Final Health Score: {final_score:.1f}/100")

    # Calculate improvement
    improvement = final_score - initial_score
    if improvement > 0:
        print(f"🎉 Health improvement: +{improvement:.1f} points")
    elif improvement < 0:
        print(f"⚠️ Health decrease: {improvement:.1f} points")
    else:
        print("➡️ No change in health score")

    # Step 5: Generate summary
    print("\n📋 Step 5: Generate Executive Summary")
    summary = example_executive_summary()

    return {
        'initial_score': initial_score, 
        'final_score': final_score, 
        'improvement': improvement, 
        'summary_generated': summary is not None
    }

def main():
    """Run all examples."""
    print(f"🤖 ProjectP AI Agent System Examples")
    print(f"Generated: {datetime.now().strftime('%Y - %m - %d %H:%M:%S')}")
    print(" = " * 70)

    if not AGENT_AVAILABLE:
        print("❌ AI Agent System is not available. Please check installation.")
        return

    try:
        # Run examples
        example_quick_health_check()
        example_comprehensive_analysis()
        example_auto_fixes()
        example_optimization()
        example_executive_summary()

        # Complete workflow
        print("\n" + " = " * 70)
        workflow_results = example_monitoring_workflow()

        print(f"\n🎯 Workflow Complete!")
        if workflow_results:
            print(f"   Initial Score: {workflow_results['initial_score']:.1f}/100")
            print(f"   Final Score: {workflow_results['final_score']:.1f}/100")
            print(f"   Improvement: {workflow_results['improvement']: + .1f} points")
            print(f"   Summary Generated: {'✅' if workflow_results['summary_generated'] else '❌'}")

    except Exception as e:
        print(f"❌ Error in examples: {e}")

if __name__ == "__main__":
    main()