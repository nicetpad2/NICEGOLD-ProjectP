#!/usr/bin/env python3
    from agent.agent_controller import AgentController
    import traceback
"""
Test script สำหรับทดสอบ Agent Controller
"""

try:
    print("✅ Agent Controller imported successfully")

    # Initialize agent
    agent = AgentController()
    print("✅ Agent Controller initialized successfully")

    # Test run ProjectP pipeline
    print("\n🚀 Testing ProjectP pipeline execution...")
    result = agent.run_projectp_pipeline(mode = "full", wait_for_completion = False)
    print(f"Pipeline result: {result}")

    # Test status summary
    print("\n📊 Testing status summary...")
    status = agent.get_projectp_status_summary()
    print(status)

except Exception as e:
    print(f"❌ Error: {e}")
    traceback.print_exc()