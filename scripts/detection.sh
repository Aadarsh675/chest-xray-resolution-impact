#!/bin/bash
# -*- coding: utf-8 -*-
#
# ==============================================================================
# Detection Inference Script
#
# Description:
# This script runs the detection inference pipeline using CodeTR model
# on the NIH Chest X-ray dataset
#
# Usage:
#   bash scripts/detection.sh
#
# ==============================================================================

# Clear the terminal screen for a clean output view
clear

# Pull the latest code from the Git repository silently
git pull > /dev/null 2>&1

# Get the absolute path to the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# =======================================================
# Configuration
# =======================================================

MODEL="codetr"
DATASET="nih-chest-xray"

# =======================================================
# Execution
# =======================================================

# Activate the 'detection' virtual environment
source "$PROJECT_ROOT/venv/detection/bin/activate"

# Run the inference script with the specified model and dataset
echo "Running detection inference with model: $MODEL and dataset: $DATASET"
python "$PROJECT_ROOT/services/detection/detection_inference.py" \
  --model "$MODEL" \
  --dataset "$DATASET"

# Check if the inference script executed successfully
if [ $? -ne 0 ]; then
  echo "Error: Inference script failed."
  exit 1
fi

echo "Inference completed successfully!"