# DRCT Super-Resolution on GCP Instance

This guide provides step-by-step instructions for setting up and running DRCT (Dense Residual Connected Transformer) for image super-resolution on a Google Cloud Platform (GCP) instance.

## Prerequisites

- GCP instance with GPU (recommended: NVIDIA T4, V100, or A100)
- Ubuntu 20.04 or 22.04
- CUDA 11.8 compatible GPU drivers installed
- SSH access to your GCP instance
- Python 3.8+ installed

## Quick Start

```bash
# One-liner to get started (run after SSH'ing into your instance)
curl -sSL https://raw.githubusercontent.com/your-repo/drct-setup/main/setup.sh | bash
```

## Detailed Setup Instructions

### 1. System Dependencies

First, ensure your system is up to date and has the necessary dependencies:

```bash
# Update package list
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y git wget curl python3-pip python3-venv build-essential

# Verify CUDA installation
nvidia-smi
```

### 2. Create Project Directory

```bash
# Create a working directory
mkdir -p ~/drct-project
cd ~/drct-project

# Create a Python virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
```

### 3. Clone DRCT Repository

```bash
# Clone the DRCT repository
git clone https://github.com/ming053l/DRCT.git
cd DRCT
```

### 4. Install PyTorch with CUDA Support

```bash
# Install PyTorch 2.0.1 with CUDA 11.8 support
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

### 5. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Fix numpy version compatibility issue
pip uninstall numpy -y
pip install numpy==1.23.0

# Install DRCT in development mode
python setup.py develop
```

### 6. Download Model Weights

Create a weights directory and download the pre-trained model:

```bash
# Create weights directory
mkdir -p weights

# Option 1: Download from Google Drive (if you have the link)
# Replace YOUR_FILE_ID with actual Google Drive file ID
pip install gdown
gdown --id YOUR_FILE_ID -O weights/DRCT-L_X4.pth

# Option 2: Download from URL (if available)
# wget https://example.com/path/to/DRCT-L_X4.pth -O weights/DRCT-L_X4.pth

# Option 3: Copy from GCS bucket (if you've uploaded there)
# gsutil cp gs://your-bucket/DRCT-L_X4.pth weights/
```

### 7. Prepare Input Images

```bash
# Create input directory
mkdir -p input

# Copy your images to the input directory
# Example: Copy from local machine to GCP instance
# scp local_image.jpg username@instance-ip:~/drct-project/DRCT/input/

# Or download sample images
wget https://example.com/sample_image.jpg -O input/sample.jpg
```

### 8. Run Inference

```bash
# Run super-resolution with 4x upscaling
python inference.py --input input --output output --model_path weights/DRCT-L_X4.pth --scale 4

# The upscaled images will be saved in the 'output' directory
```

## Advanced Usage

### Different Scale Factors

```bash
# For 2x upscaling (requires corresponding model weights)
python inference.py --input input --output output --model_path weights/DRCT-L_X2.pth --scale 2

# For 3x upscaling
python inference.py --input input --output output --model_path weights/DRCT-L_X3.pth --scale 3
```

### Batch Processing

```bash
# Process all images in a directory
for img in input/*; do
    echo "Processing $img..."
    python inference.py --input "$img" --output output --model_path weights/DRCT-L_X4.pth --scale 4
done
```

### GPU Memory Management

If you encounter GPU memory issues:

```bash
# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

# Or modify the inference command with reduced batch size
python inference.py --input input --output output --model_path weights/DRCT-L_X4.pth --scale 4 --tile 256
```

## Monitoring and Optimization

### Monitor GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or use nvtop for better visualization
sudo apt install nvtop
nvtop
```

### Performance Tips

1. **Use SSD Storage**: Store input/output on local SSD for faster I/O
2. **Batch Processing**: Process multiple images together if memory allows
3. **Tile Processing**: For very large images, use tile-based processing

## Troubleshooting

### Common Issues and Solutions

1. **CUDA Out of Memory**
   ```bash
   # Reduce tile size or process smaller images
   python inference.py --input input --output output --model_path weights/DRCT-L_X4.pth --scale 4 --tile 128
   ```

2. **Module Import Errors**
   ```bash
   # Ensure you're in the virtual environment
   source ~/drct-project/venv/bin/activate
   
   # Reinstall in development mode
   cd ~/drct-project/DRCT
   python setup.py develop
   ```

3. **NumPy Version Conflicts**
   ```bash
   # Force reinstall numpy 1.23.0
   pip install --force-reinstall numpy==1.23.0
   ```

4. **Missing CUDA Libraries**
   ```bash
   # Add CUDA to PATH
   export PATH=/usr/local/cuda-11.8/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
   ```

## Automation Script

Save this as `run_drct.sh` for easy execution:

```bash
#!/bin/bash

# Activate virtual environment
source ~/drct-project/venv/bin/activate

# Change to DRCT directory
cd ~/drct-project/DRCT

# Check if weights exist
if [ ! -f "weights/DRCT-L_X4.pth" ]; then
    echo "Error: Model weights not found. Please download them first."
    exit 1
fi

# Check if input directory has images
if [ -z "$(ls -A input 2>/dev/null)" ]; then
    echo "Error: No images found in input directory."
    exit 1
fi

# Run inference
echo "Starting DRCT super-resolution..."
python inference.py --input input --output output --model_path weights/DRCT-L_X4.pth --scale 4

echo "Processing complete! Check the output directory for results."
```

Make it executable:
```bash
chmod +x run_drct.sh
./run_drct.sh
```

## Clean Up

When you're done:

```bash
# Deactivate virtual environment
deactivate

# Optional: Remove the project (be careful!)
# rm -rf ~/drct-project
```

## Additional Resources

- [DRCT Paper](https://arxiv.org/abs/xxxxx)
- [Original GitHub Repository](https://github.com/ming053l/DRCT)
- [GCP GPU Documentation](https://cloud.google.com/compute/docs/gpus)

## Support

If you encounter issues not covered in this guide:
1. Check the original DRCT repository issues
2. Verify your CUDA installation with `nvidia-smi`
3. Ensure all dependencies are correctly installed
4. Check GCP instance logs for system-level errors
