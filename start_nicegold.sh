#!/bin/bash
# NICEGOLD ProjectP - Easy Startup Script
# This script provides an easy way to start the NICEGOLD ProjectP system

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                                ║${NC}"
echo -e "${CYAN}║           🚀 NICEGOLD ProjectP Startup Script 🚀              ║${NC}"
echo -e "${CYAN}║                   Production Ready Version                    ║${NC}"
echo -e "${CYAN}║                                                                ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"

echo ""
echo -e "${BLUE}📂 Project Directory: ${NC}${PROJECT_DIR}"
echo -e "${BLUE}🐍 Python Version: ${NC}$(python3 --version 2>/dev/null || python --version)"
echo ""

# Function to check if Python is available
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}❌ Python not found! Please install Python 3.8 or higher.${NC}"
        exit 1
    fi
}

# Function to check if ProjectP_refactored.py exists
check_main_file() {
    if [ ! -f "$PROJECT_DIR/ProjectP_refactored.py" ]; then
        echo -e "${RED}❌ ProjectP_refactored.py not found in $PROJECT_DIR${NC}"
        echo -e "${YELLOW}💡 Please ensure you're running this script from the correct directory.${NC}"
        exit 1
    fi
}

# Function to install dependencies
install_dependencies() {
    echo -e "${YELLOW}📦 Checking dependencies...${NC}"
    
    if [ -f "$PROJECT_DIR/requirements.txt" ]; then
        echo -e "${BLUE}🔍 Found requirements.txt, checking if installation is needed...${NC}"
        
        # Check if pip is available
        if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
            echo -e "${YELLOW}⚠️ pip not found. Please install pip to manage dependencies.${NC}"
            return 1
        fi
        
        # Try to install requirements
        PIP_CMD="pip3"
        if ! command -v pip3 &> /dev/null; then
            PIP_CMD="pip"
        fi
        
        echo -e "${BLUE}🔧 Installing/updating dependencies...${NC}"
        $PIP_CMD install -r "$PROJECT_DIR/requirements.txt" --quiet --upgrade
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Dependencies installed successfully!${NC}"
        else
            echo -e "${YELLOW}⚠️ Some dependencies might not have installed correctly.${NC}"
            echo -e "${YELLOW}💡 You may need to install them manually.${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️ requirements.txt not found. Proceeding without dependency check.${NC}"
    fi
}

# Function to show menu
show_menu() {
    echo ""
    echo -e "${PURPLE}┌─────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${PURPLE}│                      🎛️ Options Menu                        │${NC}"
    echo -e "${PURPLE}├─────────────────────────────────────────────────────────────┤${NC}"
    echo -e "${PURPLE}│  1. 🚀 Start NICEGOLD ProjectP (Interactive)               │${NC}"
    echo -e "${PURPLE}│  2. 📖 Show Help                                           │${NC}"
    echo -e "${PURPLE}│  3. 📋 Show Version                                        │${NC}"
    echo -e "${PURPLE}│  4. 🔧 Install/Update Dependencies                         │${NC}"
    echo -e "${PURPLE}│  5. 🏥 System Health Check Only                           │${NC}"
    echo -e "${PURPLE}│  0. 🚪 Exit                                                │${NC}"
    echo -e "${PURPLE}└─────────────────────────────────────────────────────────────┘${NC}"
    echo ""
}

# Function to run the main system
run_main_system() {
    echo -e "${GREEN}🚀 Starting NICEGOLD ProjectP...${NC}"
    echo ""
    cd "$PROJECT_DIR"
    $PYTHON_CMD ProjectP_refactored.py
}

# Function to show help
show_help() {
    echo -e "${CYAN}📖 Showing help information...${NC}"
    echo ""
    cd "$PROJECT_DIR"
    $PYTHON_CMD ProjectP_refactored.py --help
}

# Function to show version
show_version() {
    echo -e "${CYAN}📋 Showing version information...${NC}"
    echo ""
    cd "$PROJECT_DIR"
    $PYTHON_CMD ProjectP_refactored.py --version
}

# Function to run system health check
run_health_check() {
    echo -e "${BLUE}🏥 Running system health check...${NC}"
    echo ""
    cd "$PROJECT_DIR"
    
    # Try to import and run a basic health check
    $PYTHON_CMD -c "
import sys
sys.path.insert(0, 'src')
try:
    from system.health_monitor import SystemHealthMonitor
    monitor = SystemHealthMonitor('.')
    health = monitor.check_system_health()
    if health:
        print('✅ Basic health check passed!')
    else:
        print('⚠️ Health check completed with warnings.')
except Exception as e:
    print(f'❌ Health check failed: {e}')
"
}

# Main execution
main() {
    # Initial checks
    check_python
    check_main_file
    
    # If arguments provided, handle them directly
    if [ $# -gt 0 ]; then
        case $1 in
            "start"|"run"|"1")
                run_main_system
                ;;
            "help"|"--help"|"-h"|"2")
                show_help
                ;;
            "version"|"--version"|"-v"|"3")
                show_version
                ;;
            "install"|"deps"|"4")
                install_dependencies
                ;;
            "health"|"check"|"5")
                run_health_check
                ;;
            *)
                echo -e "${YELLOW}❓ Unknown option: $1${NC}"
                show_menu
                ;;
        esac
        exit 0
    fi
    
    # Interactive mode
    while true; do
        show_menu
        echo -e -n "${CYAN}👉 Please select an option (0-5): ${NC}"
        read -r choice
        
        case $choice in
            1)
                run_main_system
                break
                ;;
            2)
                show_help
                echo ""
                echo -e "${GREEN}Press Enter to continue...${NC}"
                read -r
                ;;
            3)
                show_version
                echo ""
                echo -e "${GREEN}Press Enter to continue...${NC}"
                read -r
                ;;
            4)
                install_dependencies
                echo ""
                echo -e "${GREEN}Press Enter to continue...${NC}"
                read -r
                ;;
            5)
                run_health_check
                echo ""
                echo -e "${GREEN}Press Enter to continue...${NC}"
                read -r
                ;;
            0)
                echo -e "${GREEN}👋 Goodbye! Thanks for using NICEGOLD ProjectP!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}❌ Invalid option. Please choose 0-5.${NC}"
                echo ""
                echo -e "${GREEN}Press Enter to continue...${NC}"
                read -r
                ;;
        esac
    done
}

# Run main function
main "$@"
