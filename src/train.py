import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from transformers import DetrImageProcessor, DetrForObjectDetection
import wandb
import numpy as np

# Specify data paths
DATA_DIR = "data"
ANNO_DIR = os.path.join(DATA_DIR, "annotations")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
TRAIN_ANNO_FILE = os.path.join(ANNO_DIR, "train_annotations_coco.json")
TRAIN_IMG_DIR = os.path.join(IMAGE_DIR, "train")

class SimpleCocoDataset(Dataset):
    def __init__(self, img_folder, anno_file, processor):
        self.coco = COCO(anno_file)
        self.img_folder = img_folder
        self.processor = processor
        # Only use images that have annotations
        self.ids = [img_id for img_id in self.coco.imgs.keys()
                   if len(self.coco.getAnnIds(imgIds=img_id)) > 0]
        print(f"Found {len(self.ids)} images with annotations")
  
    def __len__(self):
        return len(self.ids)
  
    def __getitem__(self, idx):
        # Get image
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_folder, img_info['file_name'])
      
        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise
        width, height = image.size
      
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
      
        # Convert annotations to DETR format
        boxes = []
        labels = []
      
        for ann in anns:
            # Get bbox and normalize
            x, y, w, h = ann['bbox']
            # Convert to center format and normalize
            x_center = (x + w/2) / width
            y_center = (y + h/2) / height
            w_norm = w / width
            h_norm = h / height
          
            # Clamp values to [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))
          
            boxes.append([x_center, y_center, w_norm, h_norm])
            # Assuming category_ids start from 1, subtract 1 for 0-indexing
            labels.append(ann['category_id'] - 1)
      
        # Process image
        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)
      
        # Create target
        target = {
            "class_labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "image_id": torch.tensor([img_id], dtype=torch.int64)
        }
      
        return pixel_values, target

def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return pixel_values, targets

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
  
    for batch_idx, (pixel_values, targets) in enumerate(dataloader):
        pixel_values = pixel_values.to(device)
      
        # Move targets to device
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
      
        # Forward pass
        outputs = model(pixel_values=pixel_values, labels=targets)
        loss = outputs.loss
      
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      
        total_loss += loss.item()
      
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
            wandb.log({"batch": batch_idx + epoch * num_batches, "train_loss": loss.item()})
  
    avg_loss = total_loss / num_batches
    return avg_loss

def main():
    # Initialize W&B
    wandb.init(project="detr-nih-chest-xray", name="detr_training_run", config={
        "num_epochs": 10,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "model": "facebook/detr-resnet-50"
    })
   
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
  
    # Load annotations to get number of classes
    with open(TRAIN_ANNO_FILE, 'r') as f:
        coco_data = json.load(f)
  
    # Get number of classes
    num_classes = len(coco_data['categories'])
    print(f"Number of classes: {num_classes}")
    print("Categories: ", coco_data['categories'])
  
    # Initialize processor and model
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", size=800)
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    ).to(device)
  
    # Create datasets
    print("\nLoading training dataset...")
    train_dataset = SimpleCocoDataset(TRAIN_IMG_DIR, TRAIN_ANNO_FILE, processor)
  
    # Test one sample
    print("\nTesting one sample...")
    sample_img, sample_target = train_dataset[0]
    print(f"Image shape: {sample_img.shape}")
    print(f"Number of objects: {len(sample_target['class_labels'])}")
    print(f"Boxes shape: {sample_target['boxes'].shape}")
  
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
  
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
  
    # Create weights folder
    os.makedirs("weights", exist_ok=True)
  
    # Training loop
    num_epochs = 10
    print(f"\nStarting training for {num_epochs} epochs...")
  
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Average loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "avg_train_loss": avg_loss})
      
        # Save weights
        weights_path = f"weights/epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), weights_path)
        print(f"Saved weights to {weights_path}")
  
    print("Training complete!")
   
    # Finish W&B run
    wandb.finish()

if __name__ == "__main__":
    main()
