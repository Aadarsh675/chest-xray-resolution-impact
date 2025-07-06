#!/bin/bash
#
# DETECTION SERVICE VENV SETUP
#
# This script creates a virtual environment for the detection service.
#
# Usage: bash scripts/create_detection_venv.sh
#

# ANSI color codes for output formatting
COLOR_RED='\033[0;31m'     # Error messages
COLOR_GREEN='\033[0;32m'   # Success messages
COLOR_YELLOW='\033[1;33m'  # Warning/note messages
COLOR_BLUE='\033[0;34m'    # Info messages
COLOR_CYAN='\033[0;36m'    # Action messages
COLOR_PURPLE='\033[0;35m'  # Section headers
COLOR_RESET='\033[0m'      # Reset color

# ----- UTILITY FUNCTIONS -----

print_color() {
    local color="$1"
    local message="$2"
    
    case $color in
        "red")    color_code=$COLOR_RED ;;
        "green")  color_code=$COLOR_GREEN ;;
        "yellow") color_code=$COLOR_YELLOW ;;
        "blue")   color_code=$COLOR_BLUE ;;
        "cyan")   color_code=$COLOR_CYAN ;;
        "purple") color_code=$COLOR_PURPLE ;;
        *)        color_code=$COLOR_RESET ;;
    esac
    
    echo -e "${color_code}${message}${COLOR_RESET}"
}

print_header() {
    local title="$1"
    echo
    print_color "purple" "=== $title ==="
    echo
}

# ----- PROJECT INITIALIZATION -----

init_project() {
    # Get the absolute path to the project root directory
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    
    print_header "NEXTERA ROBOTICS DETECTION SERVICE SETUP"
    print_color "blue" "Project root: $PROJECT_ROOT"
    
    # Export for use in subshells
    export PROJECT_ROOT
}

# ----- SERVICE ENVIRONMENT SETUP -----

create_detection_venv() {
    print_header "CREATING DETECTION SERVICE VIRTUAL ENVIRONMENT"
    
    local service="detection"
    local create_venv_script="$PROJECT_ROOT/services/$service/create_venv.sh"
    
    if [ -f "$create_venv_script" ]; then
        print_color "cyan" "[$service] Starting setup..."
        
        if bash "$create_venv_script"; then
            print_color "green" "[$service] ✓ Setup completed successfully"
        else
            print_color "red" "[$service] ✗ Setup failed"
            exit 1
        fi
    else
        print_color "red" "[$service] ✗ create_venv.sh script not found at: $create_venv_script"
        exit 1
    fi
}

# ----- MAIN SCRIPT EXECUTION -----

main() {
    init_project
    create_detection_venv
    
    print_header "SETUP COMPLETE"
    print_color "green" "✓ Detection service virtual environment has been created!"
    print_color "blue" "Run 'source venv/detection/bin/activate' to use the detection environment."
    echo
}

# Execute main function
main