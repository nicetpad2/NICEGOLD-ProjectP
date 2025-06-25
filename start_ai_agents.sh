#!/bin/bash
# AI Agents Quick Start Script
# ============================

set -e

echo "🤖 NICEGOLD ProjectP - AI Agents Quick Start"
echo "============================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    echo -e "${1}${2}${NC}"
}

# Check Python version
check_python() {
    print_color $BLUE "🔍 Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_color $RED "❌ Python not found. Please install Python 3.8+"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    print_color $GREEN "✅ Python $PYTHON_VERSION found"
}

# Install required packages
install_dependencies() {
    print_color $BLUE "📦 Installing AI Agents dependencies..."
    
    # Core dependencies
    $PYTHON_CMD -m pip install --upgrade pip
    
    # Essential packages for AI Agents
    PACKAGES=(
        "streamlit>=1.20.0"
        "plotly>=5.0.0"
        "pandas>=1.3.0"
        "psutil>=5.8.0"
        "pyyaml>=6.0"
        "requests>=2.25.0"
    )
    
    for package in "${PACKAGES[@]}"; do
        print_color $YELLOW "  Installing $package..."
        $PYTHON_CMD -m pip install "$package" --quiet
    done
    
    print_color $GREEN "✅ Dependencies installed successfully"
}

# Setup directories
setup_directories() {
    print_color $BLUE "📁 Setting up directories..."
    
    mkdir -p agent_reports
    mkdir -p logs
    mkdir -p output_default
    
    print_color $GREEN "✅ Directories created"
}

# Check agent modules
check_agent_modules() {
    print_color $BLUE "🔍 Checking AI Agent modules..."
    
    if [ ! -d "agent" ]; then
        print_color $RED "❌ Agent directory not found"
        print_color $YELLOW "💡 Creating basic agent structure..."
        mkdir -p agent/{understanding,analysis,auto_fix,optimization}
        
        # Create placeholder files if they don't exist
        if [ ! -f "agent/agent_controller.py" ]; then
            print_color $YELLOW "⚠️  agent_controller.py not found - some features may not work"
        fi
    else
        print_color $GREEN "✅ Agent modules found"
    fi
}

# Test basic functionality
test_basic_functionality() {
    print_color $BLUE "🧪 Testing basic functionality..."
    
    # Test Python imports
    $PYTHON_CMD -c "
import sys
import os
try:
    import streamlit
    import plotly
    import pandas
    import psutil
    print('✅ All required packages imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_color $GREEN "✅ Basic functionality test passed"
    else
        print_color $RED "❌ Basic functionality test failed"
        exit 1
    fi
}

# Show available commands
show_commands() {
    print_color $MAGENTA "🎯 Available AI Agents Commands:"
    echo ""
    
    print_color $CYAN "📋 Command Line Interface:"
    echo "  python run_ai_agents.py --action analyze     # Run comprehensive analysis"
    echo "  python run_ai_agents.py --action fix         # Run auto-fix"
    echo "  python run_ai_agents.py --action optimize    # Run optimization"
    echo "  python run_ai_agents.py --action summary     # Generate executive summary"
    echo "  python run_ai_agents.py --action web         # Launch web interface"
    echo ""
    
    print_color $CYAN "🌐 Web Interface:"
    echo "  streamlit run ai_agents_web.py               # Basic web interface"
    echo "  streamlit run ai_agents_web_enhanced.py      # Enhanced web interface"
    echo ""
    
    print_color $CYAN "🎛️ Main Menu Integration:"
    echo "  python ProjectP.py                           # Use options 16-20 for AI Agents"
    echo ""
}

# Launch options
launch_options() {
    print_color $MAGENTA "🚀 Quick Launch Options:"
    echo ""
    echo "1) Launch Web Dashboard (Enhanced)"
    echo "2) Launch Web Dashboard (Basic)"
    echo "3) Run Quick Analysis"
    echo "4) Show Main Menu"
    echo "5) Exit"
    echo ""
    
    read -p "Choose an option (1-5): " choice
    
    case $choice in
        1)
            print_color $GREEN "🌐 Launching Enhanced Web Dashboard..."
            $PYTHON_CMD -m streamlit run ai_agents_web_enhanced.py --server.port 8501 &
            sleep 3
            print_color $CYAN "🔗 Dashboard available at: http://localhost:8501"
            ;;
        2)
            print_color $GREEN "🌐 Launching Basic Web Dashboard..."
            $PYTHON_CMD -m streamlit run ai_agents_web.py --server.port 8502 &
            sleep 3
            print_color $CYAN "🔗 Dashboard available at: http://localhost:8502"
            ;;
        3)
            print_color $GREEN "🔍 Running Quick Analysis..."
            $PYTHON_CMD run_ai_agents.py --action analyze --verbose
            ;;
        4)
            print_color $GREEN "📋 Starting Main Menu..."
            $PYTHON_CMD ProjectP.py
            ;;
        5)
            print_color $YELLOW "👋 Goodbye!"
            exit 0
            ;;
        *)
            print_color $RED "❌ Invalid option"
            ;;
    esac
}

# Main setup function
main() {
    echo ""
    print_color $BLUE "Starting AI Agents setup..."
    echo ""
    
    # Run setup steps
    check_python
    install_dependencies
    setup_directories
    check_agent_modules
    test_basic_functionality
    
    echo ""
    print_color $GREEN "🎉 AI Agents setup completed successfully!"
    echo ""
    
    # Show available commands
    show_commands
    
    # Launch options
    launch_options
}

# Handle script arguments
case "${1:-}" in
    --install-only)
        check_python
        install_dependencies
        setup_directories
        print_color $GREEN "✅ Installation completed"
        ;;
    --test-only)
        check_python
        test_basic_functionality
        print_color $GREEN "✅ Tests completed"
        ;;
    --web)
        print_color $GREEN "🌐 Launching web interface..."
        $PYTHON_CMD -m streamlit run ai_agents_web_enhanced.py --server.port "${2:-8501}"
        ;;
    --analyze)
        print_color $GREEN "🔍 Running analysis..."
        $PYTHON_CMD run_ai_agents.py --action analyze --verbose
        ;;
    --help|-h)
        echo "AI Agents Quick Start Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --install-only    Only install dependencies"
        echo "  --test-only       Only run tests"
        echo "  --web [port]      Launch web interface (default port: 8501)"
        echo "  --analyze         Run quick analysis"
        echo "  --help, -h        Show this help message"
        echo ""
        echo "No options: Run full interactive setup"
        ;;
    *)
        main
        ;;
esac
