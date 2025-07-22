#!/usr/bin/env python3
"""
NIH Chest X-ray Dataset Pipeline with Multi-Scale Training
"""

import os
import json
import pandas as pd
import argparse
from PIL import Image
from tqdm import tqdm

from src.data_formatter import create_coco_dataset, split_dataset
from src.dataset_downloader import download_nih_dataset
from src.model_trainer import ModelTrainer

def downsample_images(images_dir, output_dir, scale=1.0):
    os.makedirs(output_dir, exist_ok=True)
    for img_file in tqdm(os.listdir(images_dir), desc=f"Downsampling x{scale}"):
        src_path = os.path.join(images_dir, img_file)
        dst_path = os.path.join(output_dir, img_file)
        if src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = Image.open(src_path)
                new_size = (int(img.width * scale), int(img.height * scale))
                img = img.resize(new_size, Image.LANCZOS)
                img.save(dst_path)
            except Exception as e:
                print(f"Failed on {img_file}: {e}")

def scale_bounding_boxes(bbox_df, scale):
    bbox_df = bbox_df.copy()
    bbox_df['Bbox [x'] = bbox_df['Bbox [x'] * scale
    bbox_df['y'] = bbox_df['y'] * scale
    bbox_df['width'] = bbox_df['width'] * scale
    bbox_df['height]'] = bbox_df['height]'] * scale
    return bbox_df

def run_iteration(scale, args, metadata_df, bbox_df, original_img_dir, results_dict):
    # Define dirs
    scale_tag = f"{int(scale * 100)}pct"
    img_out_dir = os.path.join(args.data_dir, f'images_{scale_tag}')
    coco_out_dir = os.path.join(args.coco_dir, f'coco_{scale_tag}')

    # Step 1: Downsample images
    downsample_images(original_img_dir, img_out_dir, scale)

    # Step 2: Scale bboxes
    bbox_scaled = scale_bounding_boxes(bbox_df, scale) if bbox_df is not None else None

    # Step 3: Create COCO dataset
    create_coco_dataset(
        metadata_df=metadata_df,
        bbox_df=bbox_scaled,
        images_dir=img_out_dir,
        output_dir=coco_out_dir
    )
    split_dataset(
        coco_dir=coco_out_dir,
        train_ratio=args.train_split,
        val_ratio=args.val_split
    )

    # Step 4: Train & Test
    print(f"\n--- Training at {scale_tag} ---")
    trainer = ModelTrainer(
        data_dir=coco_out_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    train_metrics = trainer.train()
    test_metrics = trainer.test()

    # Step 5: Save result
    results_dict[scale_tag] = {
        'scale': scale,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='nih_data')
    parser.add_argument('--coco_dir', type=str, default='coco_multi')
    parser.add_argument('--model', type=str, default='fasterrcnn_resnet50_fpn')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--train_split', type=float, default=0.7)
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--scales', nargs='+', type=float, default=[1.0, 0.5, 0.25])
    args = parser.parse_args()

    # Download dataset if needed
    dataset_path = os.path.abspath(args.data_dir)
    download_nih_dataset(dataset_path)

    # Load metadata
    metadata_df = pd.read_csv(os.path.join(dataset_path, 'Data_Entry_2017_v2020.csv'))
    bbox_path = os.path.join(dataset_path, 'BBox_List_2017.csv')
    bbox_df = pd.read_csv(bbox_path) if os.path.exists(bbox_path) else None
    original_img_dir = os.path.join(dataset_path, 'images')

    # Track results
    results_dict = {}

    for scale in args.scales:
        run_iteration(scale, args, metadata_df, bbox_df, original_img_dir, results_dict)

    # Save all results
    result_dir = os.path.join(args.data_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, 'resolution_vs_accuracy.json')
    with open(result_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n✅ Saved all results to {result_path}")

if __name__ == '__main__':
    main()
