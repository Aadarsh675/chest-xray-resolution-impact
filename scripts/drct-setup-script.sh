#!/bin/bash

# DRCT Setup Script for GCP Instance
# This script automates the entire setup process

set -e  # Exit on error

echo "======================================"
echo "DRCT Super-Resolution Setup Script"
echo "======================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running with sudo
if [ "$EUID" -eq 0 ]; then 
   print_error "Please don't run this script with sudo"
   exit 1
fi

# Check CUDA availability
print_status "Checking CUDA installation..."
if ! command -v nvidia-smi &> /dev/null; then
    print_error "CUDA not found. Please ensure you're on a GPU instance with CUDA installed."
    exit 1
else
    nvidia-smi
    print_status "CUDA is available"
fi

# Update system packages
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y git wget curl python3-pip python3-venv build-essential

# Create project directory
PROJECT_DIR="$HOME/drct-project"
print_status "Creating project directory at $PROJECT_DIR..."
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Clone DRCT repository
print_status "Cloning DRCT repository..."
if [ -d "DRCT" ]; then
    print_warning "DRCT directory already exists. Pulling latest changes..."
    cd DRCT
    git pull
else
    git clone https://github.com/ming053l/DRCT.git
    cd DRCT
fi

# Install PyTorch with CUDA 11.8
print_status "Installing PyTorch with CUDA 11.8 support..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install requirements
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Fix numpy compatibility
print_status "Fixing numpy compatibility..."
pip uninstall numpy -y
pip install numpy==1.23.0

# Install DRCT in development mode
print_status "Installing DRCT in development mode..."
python setup.py develop

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p weights input output

# Download model weights (placeholder - update with actual links)
print_status "Downloading model weights..."
print_warning "Please manually download model weights and place them in the 'weights' directory"
echo "You can download weights from:"
echo "  - DRCT-L_X2.pth for 2x upscaling"
echo "  - DRCT-L_X3.pth for 3x upscaling"
echo "  - DRCT-L_X4.pth for 4x upscaling"

# Create a test script
print_status "Creating test script..."
cat > test_drct.py << 'EOF'
#!/usr/bin/env python
import os
import torch
import sys

def check_setup():
    print("Checking DRCT setup...")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA is available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("✗ CUDA is not available")
        return False
    
    # Check if weights exist
    weights_dir = "weights"
    if os.path.exists(weights_dir) and os.listdir(weights_dir):
        print(f"✓ Weights directory exists with files: {os.listdir(weights_dir)}")
    else:
        print("✗ No weights found in weights directory")
        print("  Please download model weights")
        return False
    
    # Check if DRCT can be imported
    try:
        import drct
        print("✓ DRCT module can be imported")
    except ImportError:
        print("✗ Cannot import DRCT module")
        return False
    
    print("\nSetup complete! You can now run:")
    print("  python inference.py --input input --output output --model_path weights/DRCT-L_X4.pth --scale 4")
    
    return True

if __name__ == "__main__":
    success = check_setup()
    sys.exit(0 if success else 1)
EOF

chmod +x test_drct.py

# Create convenience run script
print_status "Creating convenience run script..."
cat > run_inference.sh << 'EOF'
#!/bin/bash
# Convenience script to run DRCT inference

# Activate virtual environment
source ~/drct-project/venv/bin/activate
cd ~/drct-project/DRCT

# Default values
INPUT_DIR="input"
OUTPUT_DIR="output"
MODEL="weights/DRCT-L_X4.pth"
SCALE=4

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -s|--scale)
            SCALE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-i input_dir] [-o output_dir] [-m model_path] [-s scale]"
            exit 1
            ;;
    esac
done

# Check if model exists
if [ ! -f "$MODEL" ]; then
    echo "Error: Model file not found: $MODEL"
    exit 1
fi

# Check if input directory has images
if [ ! -d "$INPUT_DIR" ] || [ -z "$(ls -A $INPUT_DIR 2>/dev/null)" ]; then
    echo "Error: No images found in input directory: $INPUT_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Running DRCT inference..."
echo "  Input: $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Model: $MODEL"
echo "  Scale: ${SCALE}x"

python inference.py --input "$INPUT_DIR" --output "$OUTPUT_DIR" --model_path "$MODEL" --scale "$SCALE"

echo "Inference complete! Check $OUTPUT_DIR for results."
EOF

chmod +x run_inference.sh

# Final setup check
print_status "Running setup verification..."
source venv/bin/activate
python test_drct.py

# Print summary
echo ""
echo "======================================"
echo "Setup Summary"
echo "======================================"
echo "Project directory: $PROJECT_DIR"
echo "Virtual environment: $PROJECT_DIR/venv"
echo "DRCT directory: $PROJECT_DIR/DRCT"
echo ""
echo "To activate the environment:"
echo "  source $PROJECT_DIR/venv/bin/activate"
echo ""
echo "To run inference:"
echo "  cd $PROJECT_DIR/DRCT"
echo "  ./run_inference.sh"
echo ""
print_warning "Remember to download model weights and place them in: $PROJECT_DIR/DRCT/weights/"
echo "======================================"
