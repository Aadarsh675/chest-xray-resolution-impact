#!/usr/bin/env python3
"""
NIH Chest X-ray Dataset Pipeline
Handles downloading, organizing, training, and testing with COCO format
"""

import os
import json
import shutil
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse

# Import helper modules
from src.data_formatter import create_coco_dataset, split_dataset
from src.gdrive_handler import GDriveHandler
from src.model_trainer import ModelTrainer
from src.dataset_downloader import download_nih_dataset


def main():
    parser = argparse.ArgumentParser(description='NIH Chest X-ray Pipeline')
    parser.add_argument('--data-dir', default='./data', help='Local data directory')
    parser.add_argument('--gdrive-dir', default='/content/drive/MyDrive/nih-chest-xrays', 
                        help='Google Drive directory')
    parser.add_argument('--train-split', type=float, default=0.8, help='Training split ratio')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--model', default='yolov8n', help='Model to use (yolov8n, yolov8s, etc.)')
    
    args = parser.parse_args()
    
    # Initialize Google Drive handler
    gdrive = GDriveHandler(args.gdrive_dir)
    
    print("=" * 80)
    print("NIH Chest X-ray Dataset Pipeline")
    print("=" * 80)
    
    # Step 1: Check if dataset exists in Google Drive
    dataset_path = None
    if gdrive.is_mounted():
        print("Checking Google Drive for existing dataset...")
        if gdrive.dataset_exists():
            print("✓ Dataset found in Google Drive!")
            dataset_path = gdrive.get_dataset_path()
        else:
            print("✗ Dataset not found in Google Drive")
    
    # Step 2: Download dataset if needed
    if dataset_path is None:
        print("\nDownloading NIH Chest X-ray dataset...")
        dataset_path = download_nih_dataset(args.data_dir)
        
        # Upload to Google Drive if mounted
        if gdrive.is_mounted():
            print("\nUploading dataset to Google Drive...")
            gdrive.upload_dataset(dataset_path)
            dataset_path = gdrive.get_dataset_path()
    
    # Step 3: Load metadata and prepare COCO format
    print("\nPreparing COCO format dataset...")
    
    # Load the CSV file with bounding box information
    csv_path = os.path.join(dataset_path, 'Data_Entry_2017_v2020.csv')
    bbox_csv_path = os.path.join(dataset_path, 'BBox_List_2017.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found at {csv_path}")
    
    # Load metadata
    metadata_df = pd.read_csv(csv_path)
    
    # Load bounding boxes if available
    bbox_df = None
    if os.path.exists(bbox_csv_path):
        bbox_df = pd.read_csv(bbox_csv_path)
        print(f"✓ Loaded {len(bbox_df)} bounding box annotations")
    else:
        print("! No bounding box annotations found, will use full images")
    
    # Create COCO format dataset
    coco_dir = os.path.join(args.data_dir, 'coco_format')
    os.makedirs(coco_dir, exist_ok=True)
    
    # Get image paths
    images_dir = os.path.join(dataset_path, 'images')
    if not os.path.exists(images_dir):
        # Try to find images in subdirectories
        for subdir in os.listdir(dataset_path):
            subpath = os.path.join(dataset_path, subdir)
            if os.path.isdir(subpath) and 'images' in subdir.lower():
                images_dir = subpath
                break
    
    print(f"Images directory: {images_dir}")
    
    # Create COCO dataset
    create_coco_dataset(
        metadata_df=metadata_df,
        bbox_df=bbox_df,
        images_dir=images_dir,
        output_dir=coco_dir
    )
    
    # Step 4: Split dataset
    print("\nSplitting dataset...")
    split_dataset(
        coco_dir=coco_dir,
        train_ratio=args.train_split,
        val_ratio=args.val_split
    )
    
    # Step 5: Train model
    print("\nTraining model...")
    trainer = ModelTrainer(
        data_dir=coco_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Train the model
    results = trainer.train()
    
    # Step 6: Test model
    print("\nTesting model...")
    test_results = trainer.test()
    
    # Step 7: Save results
    results_dir = os.path.join(args.data_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save training results
    with open(os.path.join(results_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save test results
    with open(os.path.join(results_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Pipeline completed successfully!")
    print(f"Results saved to: {results_dir}")
    print("=" * 80)
    
    # Upload results to Google Drive if mounted
    if gdrive.is_mounted():
        print("\nUploading results to Google Drive...")
        gdrive.upload_results(results_dir)


if __name__ == "__main__":
    main()