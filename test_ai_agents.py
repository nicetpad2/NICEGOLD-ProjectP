#!/usr/bin/env python3
from pathlib import Path
import json
import os
import subprocess
import sys
        import yaml
"""
AI Agents System Test
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

Test script to verify AI Agents system integration and functionality.
"""


def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing imports...")

    tests = [
        ("ai_agents_menu", "AI Agents menu integration"), 
        ("streamlit", "Streamlit web framework"), 
        ("plotly", "Plotly visualization"), 
        ("pandas", "Pandas data processing"), 
        ("psutil", "System monitoring"), 
        ("yaml", "YAML configuration"), 
    ]

    results = {}

    for module, description in tests:
        try:
            __import__(module)
            print(f"  ✅ {description}: OK")
            results[module] = True
        except ImportError as e:
            print(f"  ❌ {description}: FAILED - {e}")
            results[module] = False

    return results


def test_files_exist():
    """Test if all required files exist."""
    print("\n📁 Testing file existence...")

    required_files = [
        "ai_agents_menu.py", 
        "ai_agents_web.py", 
        "ai_agents_web_enhanced.py", 
        "run_ai_agents.py", 
        "ai_agents_config.yaml", 
        "AI_AGENTS_DOCUMENTATION.md", 
        "start_ai_agents.sh", 
        "ProjectP.py", 
    ]

    results = {}

    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}: EXISTS")
            results[file_path] = True
        else:
            print(f"  ❌ {file_path}: MISSING")
            results[file_path] = False

    return results


def test_menu_integration():
    """Test if AI Agents are integrated into ProjectP menu."""
    print("\n🎯 Testing menu integration...")

    try:
        # Check if ProjectP.py contains AI Agents menu options
        with open("ProjectP.py", "r", encoding = "utf - 8") as f:
            content = f.read()

        checks = [
            ("🤖 AI Agents", "AI Agents section in menu"), 
            ("handle_project_analysis", "Project analysis function"), 
            ("handle_auto_fix", "Auto - fix function"), 
            ("handle_optimization", "Optimization function"), 
            ("handle_executive_summary", "Executive summary function"), 
            ("handle_web_dashboard", "Web dashboard function"), 
            ("AI_AGENTS_AVAILABLE", "AI Agents availability flag"), 
        ]

        results = {}

        for check_text, description in checks:
            if check_text in content:
                print(f"  ✅ {description}: FOUND")
                results[check_text] = True
            else:
                print(f"  ❌ {description}: NOT FOUND")
                results[check_text] = False

        return results

    except Exception as e:
        print(f"  ❌ Error testing menu integration: {e}")
        return {}


def test_configuration():
    """Test configuration file."""
    print("\n⚙️ Testing configuration...")

    try:

        with open("ai_agents_config.yaml", "r", encoding = "utf - 8") as f:
            config = yaml.safe_load(f)

        required_sections = [
            "agent_controller", 
            "analysis", 
            "auto_fix", 
            "optimization", 
            "monitoring", 
            "web_interface", 
        ]

        results = {}

        for section in required_sections:
            if section in config:
                print(f"  ✅ {section}: CONFIGURED")
                results[section] = True
            else:
                print(f"  ❌ {section}: MISSING")
                results[section] = False

        return results

    except Exception as e:
        print(f"  ❌ Error testing configuration: {e}")
        return {}


def test_web_interface_syntax():
    """Test web interface files for syntax errors."""
    print("\n🌐 Testing web interface syntax...")

    web_files = ["ai_agents_web.py", "ai_agents_web_enhanced.py"]

    results = {}

    for file_path in web_files:
        try:
            # Try to compile the file
            with open(file_path, "r", encoding = "utf - 8") as f:
                content = f.read()

            compile(content, file_path, "exec")
            print(f"  ✅ {file_path}: SYNTAX OK")
            results[file_path] = True

        except SyntaxError as e:
            print(f"  ❌ {file_path}: SYNTAX ERROR - {e}")
            results[file_path] = False
        except Exception as e:
            print(f"  ⚠️ {file_path}: WARNING - {e}")
            results[file_path] = False

    return results


def test_cli_runner():
    """Test CLI runner functionality."""
    print("\n💻 Testing CLI runner...")

    try:
        # Test help command
        result = subprocess.run(
            [sys.executable, "run_ai_agents.py", " -  - help"], 
            capture_output = True, 
            text = True, 
            timeout = 10, 
        )

        if result.returncode == 0:
            print("  ✅ CLI runner help: OK")
            return True
        else:
            print(f"  ❌ CLI runner help: FAILED - {result.stderr}")
            return False

    except Exception as e:
        print(f"  ❌ CLI runner test failed: {e}")
        return False


def generate_test_report(results):
    """Generate comprehensive test report."""
    print("\n📊 GENERATING TEST REPORT...")
    print(" = " * 60)

    total_tests = 0
    passed_tests = 0

    for category, test_results in results.items():
        print(f"\n{category.upper()}:")

        if isinstance(test_results, dict):
            for test_name, result in test_results.items():
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"  {test_name}: {status}")
                total_tests += 1
                if result:
                    passed_tests += 1
        elif isinstance(test_results, bool):
            status = "✅ PASS" if test_results else "❌ FAIL"
            print(f"  {category}: {status}")
            total_tests += 1
            if test_results:
                passed_tests += 1

    # Calculate success rate
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

    print("\n" + " = " * 60)
    print("📈 TEST SUMMARY")
    print(" = " * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")

    if success_rate >= 80:
        print("\n🎉 AI AGENTS SYSTEM STATUS: READY FOR USE!")
    elif success_rate >= 60:
        print("\n⚠️ AI AGENTS SYSTEM STATUS: PARTIALLY READY (some issues detected)")
    else:
        print("\n❌ AI AGENTS SYSTEM STATUS: NEEDS ATTENTION (multiple issues)")

    # Save report
    report_data = {
        "timestamp": str(os.path.getmtime(__file__)), 
        "total_tests": total_tests, 
        "passed_tests": passed_tests, 
        "success_rate": success_rate, 
        "detailed_results": results, 
    }

    try:
        os.makedirs("agent_reports", exist_ok = True)
        with open(
            "agent_reports/ai_agents_test_report.json", "w", encoding = "utf - 8"
        ) as f:
            json.dump(report_data, f, indent = 2, ensure_ascii = False)
        print(f"\n💾 Test report saved to: agent_reports/ai_agents_test_report.json")
    except Exception as e:
        print(f"\n⚠️ Could not save test report: {e}")

    return success_rate


def main():
    """Main test function."""
    print("🤖 NICEGOLD ProjectP - AI Agents System Test")
    print(" = " * 60)

    # Run all tests
    test_results = {
        "imports": test_imports(), 
        "files": test_files_exist(), 
        "menu_integration": test_menu_integration(), 
        "configuration": test_configuration(), 
        "web_interface": test_web_interface_syntax(), 
        "cli_runner": test_cli_runner(), 
    }

    # Generate report
    success_rate = generate_test_report(test_results)

    # Exit with appropriate code
    if success_rate >= 80:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    main()