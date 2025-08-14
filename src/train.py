import os
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import argparse
import sys
from pathlib import Path

# Add Co-DETR repository to sys.path
sys.path.append('/content/chest-xray-resolution-impact/Co-DETR')  # Updated path
from models.detr import build_model  # Import from models/detr.py
from datasets.coco import build as build_dataset
from engine import train_one_epoch, evaluate
from util.misc import nested_tensor_from_tensor_list

# ==== CONFIGURATION ====
CSV_PATH = '/content/drive/My Drive/nih_chest_xray_dataset/BBox_List_2017.csv'
IMAGE_DIR = '/content/drive/My Drive/nih_chest_xray_dataset/images'
TRAIN_IMAGE_DIR = 'data/images/train'
TEST_IMAGE_DIR = 'data/images/test'
ANNOTATION_DIR = 'data/annotations'
TRAIN_ANNOTATION_FILE = os.path.join(ANNOTATION_DIR, 'train_annotations_coco.json')
TEST_ANNOTATION_FILE = os.path.join(ANNOTATION_DIR, 'test_annotations_coco.json')
OUTPUT_DIR = 'outputs'
MODEL_NAME = 'codetr'  # Co-DETR model variant, adjust based on Co-DETR configs
BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_QUERIES = 100  # Number of object queries for Co-DETR

def load_csv(csv_path):
    """Load CSV and return DataFrame."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df

def split_dataset(df, train_ratio=0.8, random_state=42):
    """Split DataFrame into train and test sets with stratification by disease label."""
    unique_images_df = df.groupby('Image Index')['Finding Label'].first().reset_index()
    print(f"Found {len(unique_images_df)} unique images")
    
    # Print original dataset label distribution
    original_counts = unique_images_df['Finding Label'].value_counts()
    total_images = len(unique_images_df)
    print("\nOriginal dataset label distribution:")
    for label, count in original_counts.items():
        percentage = (count / total_images) * 100
        print(f"{label}: {count} images ({percentage:.2f}%)")
    
    # Split images with stratification
    train_images, test_images = train_test_split(
        unique_images_df['Image Index'],
        train_size=train_ratio,
        random_state=random_state,
        stratify=unique_images_df['Finding Label']
    )
    
    train_df = df[df['Image Index'].isin(train_images)]
    test_df = df[df['Image Index'].isin(test_images)]
    
    print(f"\nTrain set: {len(train_df)} annotations for {len(train_images)} images")
    print(f"Test set: {len(test_df)} annotations for {len(test_images)} images")
    
    # Print train and test label distribution
    train_counts = train_df.groupby('Image Index')['Finding Label'].first().value_counts()
    test_counts = test_df.groupby('Image Index')['Finding Label'].first().value_counts()
    total_train_images = len(train_images)
    total_test_images = len(test_images)
    
    print("\nTrain set label distribution:")
    for label, count in train_counts.items():
        percentage = (count / total_train_images) * 100
        print(f"{label}: {count} images ({percentage:.2f}%)")
    
    print("\nTest set label distribution:")
    for label, count in test_counts.items():
        percentage = (count / total_test_images) * 100
        print(f"{label}: {count} images ({percentage:.2f}%)")
    
    return train_df, test_df

class ChestXrayDataset(torch.utils.data.Dataset):
    """Custom Dataset for NIH Chest X-ray in COCO format."""
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        
        # Get bounding boxes and labels
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])  # Convert to [x_min, y_min, x_max, y_max]
            labels.append(ann['category_id'])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': torch.tensor([ann['area'] for ann in anns], dtype=torch.float32),
            'iscrowd': torch.tensor([ann['iscrowd'] for ann in anns], dtype=torch.int64)
        }
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.ids)

def get_transforms():
    """Define transforms for training and evaluation."""
    def transform(img, target):
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # Convert to tensor and normalize
        return img, target
    return transform

def collate_fn(batch):
    """Collate function for DataLoader to handle images and targets."""
    images, targets = zip(*batch)
    images = torch.stack(images)
    images = nested_tensor_from_tensor_list(images)
    return images, targets

def main(args):
    # Load and split dataset
    df = load_csv(args.csv_path)
    train_df, test_df = split_dataset(df, train_ratio=args.train_ratio, random_state=args.random_state)
    
    # Verify annotation files exist
    if not os.path.exists(args.train_annotation_file):
        raise FileNotFoundError(f"Train annotation file {args.train_annotation_file} not found")
    if not os.path.exists(args.test_annotation_file):
        raise FileNotFoundError(f"Test annotation file {args.test_annotation_file} not found")
    
    # Initialize datasets
    train_dataset = ChestXrayDataset(args.train_image_dir, args.train_annotation_file, transforms=get_transforms())
    test_dataset = ChestXrayDataset(args.test_image_dir, args.test_annotation_file, transforms=get_transforms())
    
    # Initialize data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    # Initialize Co-DETR model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(train_dataset.coco.cats) + 1  # +1 for background
    model_args = {
        'backbone': 'resnet50',
        'position_embedding': 'sine',
        'num_classes': num_classes,
        'num_queries': args.num_queries,
        'aux_loss': True,
        'with_box_refine': True,
        'two_stage': False
    }
    model = build_model(model_args)
    model.to(device)
    
    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        
        # Evaluate on test set
        coco_evaluator = evaluate(model, test_loader, device=device)
        
        # Save checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1}.pth'))
        print(f"Saved checkpoint to {os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1}.pth')}")
    
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Co-DETR on NIH Chest X-ray dataset")
    parser.add_argument('--csv_path', default=CSV_PATH, help='Path to the CSV file')
    parser.add_argument('--train_image_dir', default=TRAIN_IMAGE_DIR, help='Path to train images')
    parser.add_argument('--test_image_dir', default=TEST_IMAGE_DIR, help='Path to test images')
    parser.add_argument('--train_annotation_file', default=TRAIN_ANNOTATION_FILE, help='Path to train COCO annotation file')
    parser.add_argument('--test_annotation_file', default=TEST_ANNOTATION_FILE, help='Path to test COCO annotation file')
    parser.add_argument('--output_dir', default=OUTPUT_DIR, help='Directory to save model checkpoints')
    parser.add_argument('--model_name', default=MODEL_NAME, help='Co-DETR model variant (e.g., codetr)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--num_queries', type=int, default=NUM_QUERIES, help='Number of object queries')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train split ratio')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for splitting')
    
    args = parser.parse_args()
    main(args)
