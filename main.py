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
from PIL import Image

# Import helper modules
from src.data_formatter import create_coco_dataset, split_dataset
from src.gdrive_handler import GDriveHandler
from src.model_trainer import ModelTrainer
from src.dataset_downloader import download_nih_dataset

def downsample_images(images_dir, scale=0.5):
    print("\nDownsampling images...")
    for img_file in tqdm(os.listdir(images_dir)):
        img_path = os.path.join(images_dir, img_file)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = Image.open(img_path)
                new_size = (int(img.width * scale), int(img.height * scale))
                img = img.resize(new_size, Image.LANCZOS)
                img.save(img_path)
            except Exception as e:
                print(f"Failed to process {img_file}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='nih_data', help='Path to NIH data directory')
    parser.add_argument('--coco_dir', type=str, default='coco_format', help='Path to COCO output directory')
    parser.add_argument('--model', type=str, default='fasterrcnn_resnet50_fpn', help='Model name')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--train_split', type=float, default=0.7, help='Training split')
    parser.add_argument('--val_split', type=float, default=0.15, help='Validation split')
    args = parser.parse_args()

    # Step 1: Download the dataset
    dataset_path = os.path.abspath(args.data_dir)
    download_nih_dataset(dataset_path)

    # Step 2: Load metadata
    csv_path = os.path.join(dataset_path, 'Data_Entry_2017_v2020.csv')
    bbox_csv_path = os.path.join(dataset_path, 'BBox_List_2017.csv')
    metadata_df = pd.read_csv(csv_path)
    bbox_df = pd.read_csv(bbox_csv_path) if os.path.exists(bbox_csv_path) else None

    # Step 3: Locate images
    images_dir = os.path.join(dataset_path, 'images')

    # Step 4: Convert to COCO format
    print("\nCreating COCO format dataset...")
    create_coco_dataset(
        metadata_df=metadata_df,
        bbox_df=bbox_df,
        images_dir=images_dir,
        output_dir=args.coco_dir
    )

    split_dataset(
        coco_dir=args.coco_dir,
        train_ratio=args.train_split,
        val_ratio=args.val_split
    )

    # Step 5: Train model on original images
    print("\nTraining model on original images...")
    trainer = ModelTrainer(
        data_dir=args.coco_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    results_original = trainer.train()
    test_original = trainer.test()

    results_dir = os.path.join(args.data_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'original_training_results.json'), 'w') as f:
        json.dump(results_original, f, indent=2)
    with open(os.path.join(results_dir, 'original_test_results.json'), 'w') as f:
        json.dump(test_original, f, indent=2)

    # Step 6: Downsample all images
    downsample_images(images_dir=images_dir, scale=0.5)

    # Re-create COCO dataset with new image sizes
    print("\nRecreating COCO format dataset for downsampled images...")
    create_coco_dataset(
        metadata_df=metadata_df,
        bbox_df=bbox_df,
        images_dir=images_dir,
        output_dir=args.coco_dir
    )

    split_dataset(
        coco_dir=args.coco_dir,
        train_ratio=args.train_split,
        val_ratio=args.val_split
    )

    # Step 7: Retrain model on downsampled images
    print("\nTraining model on downsampled images...")
    trainer_downsampled = ModelTrainer(
        data_dir=args.coco_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    results_down = trainer_downsampled.train()
    test_down = trainer_downsampled.test()

    with open(os.path.join(results_dir, 'downsampled_training_results.json'), 'w') as f:
        json.dump(results_down, f, indent=2)
    with open(os.path.join(results_dir, 'downsampled_test_results.json'), 'w') as f:
        json.dump(test_down, f, indent=2)

if __name__ == '__main__':
    main()
