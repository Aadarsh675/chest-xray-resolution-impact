#!/usr/bin/env python
#-*- coding: utf-8 -*-
#
# ==============================================================================
# Main entry point for the inference script
#
# Created on: Thu Mar 07 2024 5:47:50 PM
# ==============================================================================

# Add project root to the Python path
import sys
import signal
import os
import torch
from multiprocessing import Process, set_start_method
from argparse import ArgumentParser
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

# Import from common modules
from common.logger import get_logger

# Import custom modules
from configs.detection_config import setup_paths, get_gpu_config
from services.detection.src.worker import worker_process

# Initialize the logger
logger = get_logger(__name__)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='Dataset to use for inference',
                        choices=['nih-chest-xray'],
                        default='nih-chest-xray', required=False)
    parser.add_argument('--model', help='Model to be used for inference',
                        choices=['codetr'],
                        required=True)
    parser.add_argument('--data-path', help='Path to dataset images',
                        default='data/datasets/nih-chest-xray/images', required=False)
    parser.add_argument('--output-path', help='Path to save detection results',
                        default='data/detection/results', required=False)
    args = parser.parse_args()
    return args

def get_image_files(data_path):
    """Get all image files from the dataset directory"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    path = Path(data_path)
    if not path.exists():
        raise ValueError(f"Data path does not exist: {data_path}")
    
    for ext in image_extensions:
        image_files.extend(path.glob(f'*{ext}'))
        image_files.extend(path.glob(f'*{ext.upper()}'))
    
    return sorted(image_files)

def main():
    # Parse arguments
    args = parse_args()

    # Set the start method for multiprocessing
    set_start_method('spawn')

    # Setup paths and configurations
    setup_paths(args)

    # Get GPU configuration based on model and dataset
    gpu_config = get_gpu_config(args.model, args.dataset)

    logger.info(f"Using checkpoint: {args.checkpoint}")
    logger.info(f"Processing images from: {args.data_path}")

    # Create necessary directories
    os.makedirs(args.output_path, exist_ok=True)

    # Get list of images to process
    image_files = get_image_files(args.data_path)
    logger.info(f"Found {len(image_files)} images to process")

    if not image_files:
        logger.error("No images found in the specified directory")
        return

    # Configure worker processes
    gpu_multiple = gpu_config['gpu_multiple']
    num_gpus = gpu_multiple * torch.cuda.device_count()
    logger.info(f"Using {num_gpus} worker processes across {torch.cuda.device_count()} GPUs (multiple: {gpu_multiple})")
    
    # Split images among workers
    images_per_worker = len(image_files) // num_gpus
    remainder = len(image_files) % num_gpus
    
    # graceful shut down
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    
    # Start worker processes
    processes = []
    try:
        start_idx = 0
        for worker_id in range(num_gpus):
            # Calculate image range for this worker
            end_idx = start_idx + images_per_worker
            if worker_id < remainder:
                end_idx += 1
            
            worker_images = image_files[start_idx:end_idx]
            
            p = Process(target=worker_process, 
                       args=(args, worker_id, worker_images, args.output_path))
            p.start()
            processes.append(p)
            
            start_idx = end_idx

        # Wait for all processes to complete
        for p in processes:
            p.join()
    finally:
        # Ensure a clean shutdown on Ctrl-C
        for p in processes:
            if p.is_alive():
                p.terminate()
        logger.info("All worker processes have completed.")

if __name__ == "__main__":
    main()