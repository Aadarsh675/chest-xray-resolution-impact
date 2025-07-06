#!/bin/bash

# ==============================================================
# Script to set up the 'detection' environment.
#
# This script will:
#   - Create a virtual environment in venv/detection.
#   - Install required Python packages, including PyTorch and MMCV.
#   - Download pretrained weights from S3 based on model and dataset.
#   - Set up AWS CLI and other dependencies using environment variables.
#
# Example Usage:
#   bash services/detection/create_venv.sh --model=co_dino --dataset=ppe
#
# Author: Andrew Kent
# Modified: 2025-04-18
# ==============================================================

# Function to print colorful messages
print_color() {
    local COLOR="$1"
    local MESSAGE="$2"
    local RESET='\033[0m'
    case $COLOR in
        "red") COLOR_CODE='\033[0;31m' ;;
        "green") COLOR_CODE='\033[0;32m' ;;
        "yellow") COLOR_CODE='\033[1;33m' ;;
        "blue") COLOR_CODE='\033[0;34m' ;;
        "cyan") COLOR_CODE='\033[0;36m' ;;
        *) COLOR_CODE='\033[0m' ;;
    esac
    echo -e "${COLOR_CODE}${MESSAGE}${RESET}"
}

# Function to load environment variables
load_environment_variables() {
    local ENV_FILE=".env"
    if [ -f "$ENV_FILE" ]; then
        export $(grep -v '^#' "$ENV_FILE" | xargs)
        print_color "green" "Loaded environment variables from $ENV_FILE"

        # Print the loaded environment variables
        print_color "blue" "The following variables were loaded from $ENV_FILE:"
        grep -v '^#' "$ENV_FILE" | awk -F= '{print $1}' | while read -r var; do
            print_color "cyan" "$var=${!var}"
        done
        return 0
    else
        print_color "red" "Environment file not found: $ENV_FILE"
        return 1
    fi
}

# Function to validate AWS credentials
validate_aws_credentials() {
    if [ -z "${AWS_ACCESS_KEY_ID}" ] || [ -z "${AWS_SECRET_ACCESS_KEY}" ]; then
        print_color "red" "AWS credentials are not set. Please check your .env file."
        return 1
    fi

    print_color "blue" "Using AWS configuration from .env file"
    return 0
}

# Function to set up project directories
setup_project_directories() {
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
    ENV_NAME="detection"
    ENV_PATH="$PROJECT_ROOT/venv/$ENV_NAME"
    REQUIREMENTS_FILE="$PROJECT_ROOT/services/$ENV_NAME/requirements.txt"

    print_color "blue" "Project root: $PROJECT_ROOT"
    print_color "blue" "Environment path: $ENV_PATH"
    print_color "blue" "Requirements file: $REQUIREMENTS_FILE"

    # Export variables for other functions to use
    export PROJECT_ROOT ENV_NAME ENV_PATH REQUIREMENTS_FILE
}

# Function to create and check virtual environment
setup_virtual_environment() {
    local python_cmd="python3.9"

    # Check Python version
    local python_version=$($python_cmd --version 2>&1 | cut -d' ' -f2)
    print_color "blue" "Using Python version: $python_version"

    # Check if venv module is available
    if ! $python_cmd -c "import venv" 2>/dev/null; then
        print_color "red" "Python venv module not available."
        print_color "yellow" "Trying to install python3-venv package..."
        sudo apt-get update && sudo apt-get install -y python3-venv || {
            print_color "red" "Failed to install python3-venv. Trying virtualenv instead..."
            pip install virtualenv
            local use_virtualenv=true
        }
    fi

    # Check if the virtual environment exists; if not, create it
    if [ ! -d "$ENV_PATH" ]; then
        print_color "blue" "Creating new virtual environment at: $ENV_PATH"

        if [ "$use_virtualenv" = true ]; then
            virtualenv "$ENV_PATH"
        else
            $python_cmd -m venv "$ENV_PATH"
        fi

        if [ $? -eq 0 ]; then
            print_color "green" "Created virtual environment: $ENV_PATH"
        else
            print_color "red" "Failed to create virtual environment"
            return 1
        fi
    else
        print_color "yellow" "Virtual environment already exists: $ENV_PATH"

        # Check if activate script exists
        if [ ! -f "$ENV_PATH/bin/activate" ]; then
            print_color "red" "Virtual environment is incomplete (missing activate script)"
            print_color "yellow" "Recreating virtual environment..."
            rm -rf "$ENV_PATH"

            if [ "$use_virtualenv" = true ]; then
                virtualenv "$ENV_PATH"
            else
                $python_cmd -m venv "$ENV_PATH"
            fi

            if [ ! -f "$ENV_PATH/bin/activate" ]; then
                print_color "red" "Failed to create proper virtual environment. Exiting."
                return 1
            fi
        fi
    fi

    return 0
}

# Function to install Python packages
install_packages() {
    # Activate the virtual environment
    print_color "blue" "Activating virtual environment..."
    source "$ENV_PATH/bin/activate"

    if [ $? -ne 0 ]; then
        print_color "red" "Failed to activate virtual environment."
        return 1
    fi

    # Verify activation
    if [ -z "$VIRTUAL_ENV" ]; then
        print_color "red" "Virtual environment not properly activated."
        return 1
    fi

    print_color "green" "Virtual environment activated: $VIRTUAL_ENV"

    # Upgrade pip
    print_color "blue" "Upgrading pip, setuptools, and wheel..."
    pip install --upgrade pip setuptools wheel

    # Install requirements
    if [ -f "$REQUIREMENTS_FILE" ]; then
        print_color "blue" "Installing requirements from $REQUIREMENTS_FILE..."
        pip install -r "$REQUIREMENTS_FILE"
    else
        print_color "red" "Requirements file not found: $REQUIREMENTS_FILE"
        return 1
    fi

    # Install PyTorch and MMCV for CUDA 11.3
    print_color "blue" "Installing PyTorch and MMCV..."
    # Install PyTorch 1.12.0 with CUDA 11.6
    pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html || error_exit "Failed to install PyTorch."

    # Install MMCV >=2.0.0rc4 compatible with PyTorch 1.12.0 and CUDA 11.6
    pip install "mmcv>=2.0.0rc4,<2.2.0" -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html || error_exit "Failed to install MMCV."
    return 0
}

# Function to set up AWS CLI
setup_aws_cli() {
    print_color "blue" "Setting up AWS CLI..."
    pip install --upgrade awscli

    aws configure set aws_access_key_id "${AWS_ACCESS_KEY_ID}"
    aws configure set aws_secret_access_key "${AWS_SECRET_ACCESS_KEY}"
    aws configure set default.region "${AWS_REGION}"
    aws configure set default.output json

    print_color "green" "AWS CLI configured successfully"
    return 0
}

# Function to download model weights
download_model_weights() {
    local MODEL="${1:-co_dino}"  # Default to co_dino if not specified
    local DATASET="${2:-ppe}"    # Default to ppe if not specified

    print_color "blue" "Downloading pretrained weights for model='$MODEL' and dataset='$DATASET'..."

    # Define mapping from model/dataset to S3 paths
    local S3_PATH=""
    local WEIGHTS_FILE=""

    case "$MODEL" in
        "co_dino")
            case "$DATASET" in
                "ppe")
                    S3_PATH="s3://didge-cv-models/Co-DETR/ppe_1.1"
                    WEIGHTS_FILE="best_bbox_mAP.pth"
                    ;;
                "gh")
                    S3_PATH="s3://didge-cv-models/Co-DETR/gh_1.2"
                    WEIGHTS_FILE="best_bbox_mAP.pth"
                    ;;
                "sse")
                    S3_PATH="s3://didge-cv-models/Co-DETR/sse_1.0"
                    WEIGHTS_FILE="best_bbox_mAP.pth"
                    ;;
                "hlr")
                    S3_PATH="s3://didge-cv-models/Co-DETR/hlr_1.1"
                    WEIGHTS_FILE="best_bbox_mAP.pth"
                    ;;
                *)
                    print_color "red" "Unknown dataset '$DATASET' for model 'co_dino'"
                    return 1
                    ;;
            esac
            ;;
        "yolox")
            if [ "$DATASET" = "heavy-equipment" ]; then
                S3_PATH="s3://didge-cv-models/YOLOX/heavy-equipment/yolox_tiny_8x8_300e_coco"
                WEIGHTS_FILE="best_bbox_mAP.pth"
            else
                print_color "red" "Unknown dataset '$DATASET' for model 'yolox'"
                return 1
            fi
            ;;
        "solov2")
            if [ "$DATASET" = "heavy-equipment" ]; then
                S3_PATH="s3://didge-cv-models/SOLOv2/heavy-equipment/solov2_x101_dcn_fpn_3x_coco"
                WEIGHTS_FILE="best_mIoU.pth"
            else
                print_color "red" "Unknown dataset '$DATASET' for model 'solov2'"
                return 1
            fi
            ;;
        *)
            print_color "red" "Unknown model: $MODEL"
            return 1
            ;;
    esac

    # Create appropriate weights directory
    local WEIGHTS_DIR="$PROJECT_ROOT/weights/$MODEL/$DATASET"
    mkdir -p "$WEIGHTS_DIR"

    # Download weights from S3
    local S3_WEIGHTS_PATH="$S3_PATH/$WEIGHTS_FILE"

    print_color "blue" "Downloading from: $S3_WEIGHTS_PATH"
    print_color "blue" "Downloading to: $WEIGHTS_DIR/$WEIGHTS_FILE"

    aws s3 cp "$S3_WEIGHTS_PATH" "$WEIGHTS_DIR/$WEIGHTS_FILE"
    if [ $? -ne 0 ]; then
        print_color "red" "Failed to download weights from S3. Ensure the file exists and bucket is accessible."
        return 1
    fi

    print_color "green" "Downloaded weights to $WEIGHTS_DIR/$WEIGHTS_FILE"
    return 0
}

# Main function to run the script
main() {
    clear

    # Parse command line arguments
    local MODEL="co_dino"  # Default model
    local DATASET="ppe"    # Default dataset

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model=*)
                MODEL="${1#*=}"
                shift
                ;;
            --dataset=*)
                DATASET="${1#*=}"
                shift
                ;;
            *)
                print_color "yellow" "Unknown parameter: $1"
                shift
                ;;
        esac
    done

    print_color "blue" "Starting setup for detection environment (model=$MODEL, dataset=$DATASET)..."

    # Load environment variables
    load_environment_variables || return 1

    # Validate AWS credentials
    validate_aws_credentials || return 1

    # Set up project directories
    setup_project_directories

    # Create and check virtual environment
    setup_virtual_environment || return 1

    # Install Python packages
    install_packages || return 1

    # Set up AWS CLI
    setup_aws_cli || return 1

    # Download model weights
    download_model_weights "$MODEL" "$DATASET" || return 1

    # Deactivate the virtual environment
    deactivate
    print_color "green" "Setup completed successfully!"

    return 0
}

# Execute main function with all command line arguments
main "$@"
exit $?
