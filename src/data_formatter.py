"""
Data formatter for converting NIH Chest X-rays to COCO format
"""

import os
import json
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm


# NIH Chest X-ray disease classes
DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia', 'No Finding'
]


def create_coco_dataset(metadata_df, bbox_df, images_dir, output_dir):
    """
    Create COCO format dataset from NIH Chest X-rays
    
    Args:
        metadata_df: DataFrame with image metadata
        bbox_df: DataFrame with bounding box annotations (optional)
        images_dir: Directory containing images
        output_dir: Output directory for COCO format dataset
    """
    print("Creating COCO format dataset...")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
    
    # Initialize COCO format
    coco_format = {
        "info": {
            "description": "NIH Chest X-rays Dataset in COCO Format",
            "version": "1.0",
            "year": 2024,
            "contributor": "NIH Clinical Center",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [{
            "id": 1,
            "name": "NIH License",
            "url": "https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community"
        }],
        "categories": [],
        "images": [],
        "annotations": []
    }
    
    # Create categories
    for idx, disease in enumerate(DISEASE_CLASSES):
        coco_format["categories"].append({
            "id": idx + 1,
            "name": disease,
            "supercategory": "disease"
        })
    
    # Create disease to ID mapping
    disease_to_id = {disease: idx + 1 for idx, disease in enumerate(DISEASE_CLASSES)}
    
    # Process images
    print(f"Processing {len(metadata_df)} images...")
    
    annotation_id = 1
    image_id = 1
    
    # If we have bounding boxes, create a lookup dictionary
    bbox_lookup = {}
    if bbox_df is not None:
        for _, row in bbox_df.iterrows():
            if row['Image Index'] not in bbox_lookup:
                bbox_lookup[row['Image Index']] = []
            bbox_lookup[row['Image Index']].append(row)
    
    # Process each image
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
        image_name = row['Image Index']
        
        # Find image file
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = os.path.join(images_dir, image_name)
            if not image_name.endswith(ext):
                potential_path += ext
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path is None:
            # Try subdirectories
            for root, dirs, files in os.walk(images_dir):
                if image_name in files or f"{image_name}.png" in files:
                    for ext in ['', '.png', '.jpg', '.jpeg']:
                        potential_path = os.path.join(root, f"{image_name}{ext}")
                        if os.path.exists(potential_path):
                            image_path = potential_path
                            break
                    if image_path:
                        break
        
        if image_path is None:
            continue
        
        # Get image dimensions
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error reading image {image_name}: {e}")
            continue
        
        # Copy image to output directory
        output_image_path = os.path.join(output_dir, 'images', os.path.basename(image_path))
        if not os.path.exists(output_image_path):
            shutil.copy2(image_path, output_image_path)
        
        # Add image to COCO format
        coco_format["images"].append({
            "id": image_id,
            "file_name": os.path.basename(image_path),
            "width": width,
            "height": height,
            "date_captured": "",
            "license": 1
        })
        
        # Parse findings
        findings = row['Finding Labels'].split('|')
        
        # Add annotations
        if image_name in bbox_lookup:
            # Use bounding box annotations
            for bbox_row in bbox_lookup[image_name]:
                if bbox_row['Finding Label'] in disease_to_id:
                    # Convert bbox format (x, y, width, height)
                    x = bbox_row['Bbox [x']
                    y = bbox_row['y']
                    w = bbox_row['w']
                    h = bbox_row['h]']
                    
                    coco_format["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": disease_to_id[bbox_row['Finding Label']],
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "segmentation": [],
                        "iscrowd": 0
                    })
                    annotation_id += 1
        else:
            # No bounding boxes - use full image as bbox for each finding
            for finding in findings:
                finding = finding.strip()
                if finding in disease_to_id:
                    # Use full image as bounding box
                    coco_format["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": disease_to_id[finding],
                        "bbox": [0, 0, width, height],
                        "area": width * height,
                        "segmentation": [],
                        "iscrowd": 0,
                        "full_image": True  # Custom field to indicate full image annotation
                    })
                    annotation_id += 1
        
        image_id += 1
    
    # Save annotations
    annotations_path = os.path.join(output_dir, 'annotations', 'instances_all.json')
    with open(annotations_path, 'w') as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"✓ Created COCO dataset with {len(coco_format['images'])} images and {len(coco_format['annotations'])} annotations")
    
    # Create dataset statistics
    create_dataset_stats(coco_format, output_dir)
    
    return coco_format


def split_dataset(coco_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Split COCO dataset into train, validation, and test sets
    
    Args:
        coco_dir: Directory containing COCO format dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
    """
    print("Splitting dataset into train/val/test...")
    
    # Load annotations
    annotations_path = os.path.join(coco_dir, 'annotations', 'instances_all.json')
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    # Get all image IDs
    image_ids = [img['id'] for img in coco_data['images']]
    
    # Split image IDs
    train_ids, test_ids = train_test_split(
        image_ids, 
        train_size=train_ratio + val_ratio, 
        random_state=42
    )
    
    train_ids, val_ids = train_test_split(
        train_ids,
        train_size=train_ratio / (train_ratio + val_ratio),
        random_state=42
    )
    
    # Create splits
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    # Create annotation files for each split
    for split_name, split_ids in splits.items():
        split_data = {
            "info": coco_data["info"],
            "licenses": coco_data["licenses"],
            "categories": coco_data["categories"],
            "images": [],
            "annotations": []
        }
        
        # Filter images
        split_data["images"] = [
            img for img in coco_data["images"] 
            if img["id"] in split_ids
        ]
        
        # Filter annotations
        split_data["annotations"] = [
            ann for ann in coco_data["annotations"]
            if ann["image_id"] in split_ids
        ]
        
        # Save split annotations
        split_path = os.path.join(coco_dir, 'annotations', f'instances_{split_name}.json')
        with open(split_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        print(f"✓ {split_name}: {len(split_data['images'])} images, {len(split_data['annotations'])} annotations")
    
    # Create data.yaml for YOLO training
    create_yolo_yaml(coco_dir, coco_data["categories"])


def create_yolo_yaml(coco_dir, categories):
    """
    Create data.yaml file for YOLO training
    
    Args:
        coco_dir: Directory containing COCO dataset
        categories: List of categories
    """
    yaml_content = f"""# NIH Chest X-rays Dataset
path: {os.path.abspath(coco_dir)}
train: annotations/instances_train.json
val: annotations/instances_val.json
test: annotations/instances_test.json

# Classes
names:
"""
    
    # Add class names
    for cat in categories:
        yaml_content += f"  {cat['id'] - 1}: {cat['name']}\n"
    
    yaml_content += f"\n# Number of classes\nnc: {len(categories)}\n"
    
    # Save yaml file
    yaml_path = os.path.join(coco_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✓ Created data.yaml for YOLO training")


def create_dataset_stats(coco_data, output_dir):
    """
    Create dataset statistics
    
    Args:
        coco_data: COCO format data
        output_dir: Output directory
    """
    stats = {
        "total_images": len(coco_data["images"]),
        "total_annotations": len(coco_data["annotations"]),
        "categories": {}
    }
    
    # Count annotations per category
    for cat in coco_data["categories"]:
        cat_id = cat["id"]
        cat_name = cat["name"]
        count = sum(1 for ann in coco_data["annotations"] if ann["category_id"] == cat_id)
        stats["categories"][cat_name] = count
    
    # Count images with bounding boxes vs full image annotations
    bbox_count = sum(1 for ann in coco_data["annotations"] if not ann.get("full_image", False))
    full_image_count = sum(1 for ann in coco_data["annotations"] if ann.get("full_image", False))
    
    stats["bbox_annotations"] = bbox_count
    stats["full_image_annotations"] = full_image_count
    
    # Save statistics
    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\nDataset Statistics:")
    print(f"Total images: {stats['total_images']}")
    print(f"Total annotations: {stats['total_annotations']}")
    print(f"Bounding box annotations: {bbox_count}")
    print(f"Full image annotations: {full_image_count}")
    print("\nAnnotations per category:")
    for cat, count in sorted(stats["categories"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")