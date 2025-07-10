#!/bin/bash

# NIH Chest X-ray Pipeline Runner
# This script sets up and runs the complete pipeline

echo "======================================"
echo "NIH Chest X-ray Pipeline Runner"
echo "======================================"

# Check if running in Google Colab
if [ -d "/content" ]; then
    echo "Detected Google Colab environment"
    GDRIVE_DIR="/content/drive/MyDrive/nih-chest-xrays"
else
    echo "Running in local environment"
    GDRIVE_DIR=""
fi

# Default parameters
DATA_DIR="./data"
EPOCHS=10
BATCH_SIZE=32
MODEL="yolov8n"
TRAIN_SPLIT=0.8
VAL_SPLIT=0.1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --quick)
            # Quick test mode
            EPOCHS=2
            echo "Running in quick test mode (2 epochs)"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories
echo "Creating directories..."
mkdir -p "$DATA_DIR"
mkdir -p "$DATA_DIR/logs"

# Check Python version
echo "Checking Python version..."
python3 --version

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Set up Kaggle credentials if they exist
if [ -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "Found Kaggle credentials"
    export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
else
    echo "No Kaggle credentials found in $HOME/.kaggle/"
    echo "Please set up Kaggle API credentials if needed"
fi

# Run the pipeline
echo ""
echo "Starting pipeline with parameters:"
echo "  Data directory: $DATA_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Model: $MODEL"
echo "  Train split: $TRAIN_SPLIT"
echo "  Val split: $VAL_SPLIT"
echo ""

# Run with timestamp logging
LOG_FILE="$DATA_DIR/logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"

python3 main.py \
    --data-dir "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --model "$MODEL" \
    --train-split "$TRAIN_SPLIT" \
    --val-split "$VAL_SPLIT" \
    2>&1 | tee "$LOG_FILE"

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Pipeline completed successfully!"
    echo "======================================"
    echo "Results saved in: $DATA_DIR/results/"
    echo "Model weights in: $DATA_DIR/runs/"
    echo "Log file: $LOG_FILE"
else
    echo ""
    echo "======================================"
    echo "Pipeline failed! Check the log file:"
    echo "$LOG_FILE"
    echo "======================================"
    exit 1
fi