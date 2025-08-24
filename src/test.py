import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import DetrImageProcessor, DetrForObjectDetection
import numpy as np

# Specify data paths
DATA_DIR = "data"
ANNO_DIR = os.path.join(DATA_DIR, "annotations")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
VAL_ANNO_FILE = os.path.join(ANNO_DIR, "test_annotations_coco.json")
VAL_IMG_DIR = os.path.join(IMAGE_DIR, "test")

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

def evaluate_model(model, dataloader, coco_val, device):
    model.eval()
    results = []
    with torch.no_grad():
        for pixel_values, targets in dataloader:
            pixel_values = pixel_values.to(device)
            outputs = model(pixel_values=pixel_values)
           
            # Process predictions
            orig_target_sizes = torch.tensor([t["image_id"].shape[0] for t in targets], device=device)
            pred_boxes = outputs.logits.softmax(-1)[..., :-1].max(-1).indices # Remove "no object" class
            pred_scores = outputs.logits.softmax(-1)[..., :-1].max(-1).values
           
            for i, (boxes, scores, target) in enumerate(zip(outputs.pred_boxes, pred_scores, targets)):
                img_id = target["image_id"].item()
                img_info = coco_val.loadImgs(img_id)[0]
                width, height = img_info["width"], img_info["height"]
               
                # Convert normalized boxes to COCO format [x, y, w, h]
                boxes = boxes.cpu().numpy()
                boxes = boxes * np.array([width, height, width, height])
                boxes[:, [0, 2]] = boxes[:, [0, 2]] - boxes[:, [2, 2]] / 2 # Convert center to top-left
                boxes[:, [1, 3]] = boxes[:, [1, 3]] - boxes[:, [3, 3]] / 2
               
                for box, score, label in zip(boxes, scores.cpu().numpy(), pred_boxes[i].cpu().numpy()):
                    if score > 0.5: # Threshold for predictions
                        results.append({
                            "image_id": img_id,
                            "category_id": label + 1, # Add 1 to match COCO 1-indexing
                            "bbox": box.tolist(),
                            "score": float(score)
                        })
   
    # Save predictions to a temporary file
    results_file = "eval_predictions.json"
    with open(results_file, "w") as f:
        json.dump(results, f)
   
    # Load predictions into COCO format
    coco_dt = coco_val.loadRes(results_file)
   
    # Run COCO evaluation
    coco_eval = COCOeval(coco_val, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
   
    # Extract key metrics
    metrics = {
        "mAP": coco_eval.stats[0], # AP @ IoU=0.5:0.95
        "AP@50": coco_eval.stats[1], # AP @ IoU=0.5
        "AP@75": coco_eval.stats[2], # AP @ IoU=0.75
    }
   
    return metrics

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
  
    # Load annotations to get number of classes
    with open(VAL_ANNO_FILE, 'r') as f:
        coco_data = json.load(f)
  
    # Get number of classes
    num_classes = len(coco_data['categories'])
    print(f"Number of classes: {num_classes}")
  
    # Load processor and model from saved directory
    model_dir = "./detr_final"
    processor = DetrImageProcessor.from_pretrained(model_dir)
    model = DetrForObjectDetection.from_pretrained(model_dir).to(device)
  
    # Create validation dataset and loader
    print("\nLoading test dataset...")
    val_dataset = SimpleCocoDataset(VAL_IMG_DIR, VAL_ANNO_FILE, processor)
  
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
  
    # Evaluate model
    print("\nEvaluating model on test set...")
    coco_val = COCO(VAL_ANNO_FILE)
    eval_metrics = evaluate_model(model, val_loader, coco_val, device)
    print(f"Evaluation metrics: {eval_metrics}")
  
    print("Evaluation complete!")

if __name__ == "__main__":
    main()
