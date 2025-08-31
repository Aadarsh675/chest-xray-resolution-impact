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
        self.ids = [img_id for img_id in self.coco.imgs.keys()
                    if len(self.coco.getAnnIds(imgIds=img_id)) > 0]
        print(f"Found {len(self.ids)} images with annotations")
   
    def __len__(self):
        return len(self.ids)
   
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_folder, img_info['file_name'])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise
        width, height = image.size
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            x_center = (x + w/2) / width
            y_center = (y + h/2) / height
            w_norm = w / width
            h_norm = h / height
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))
            boxes.append([x_center, y_center, w_norm, h_norm])
            labels.append(ann['category_id'] - 1)
        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)
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

def evaluate_model(model, dataloader, coco_val, device, num_classes):
    model.eval()
    results = []
    with torch.no_grad():
        for pixel_values, targets in dataloader:
            pixel_values = pixel_values.to(device)
            outputs = model(pixel_values=pixel_values)
            for i, target in enumerate(targets):
                img_id = target["image_id"].item()
                img_info = coco_val.loadImgs(img_id)[0]
                img_path = os.path.join(VAL_IMG_DIR, img_info['file_name'])
                width = img_info.get("width")
                height = img_info.get("height")
                # Load image dimensions if missing in annotations
                if width is None or height is None:
                    try:
                        image = Image.open(img_path).convert("RGB")
                        width, height = image.size
                    except Exception as e:
                        print(f"Error loading image {img_path} for dimensions: {e}")
                        continue
                # Proper post-processing: get scores and labels including no-obj
                logits = outputs.logits[i]  # (queries, num_classes + 1)
                pred_boxes = outputs.pred_boxes[i]  # (queries, 4)
                probs = logits.softmax(-1)
                scores, labels = probs.max(-1)
                keep = labels != num_classes  # no-obj class
                pred_scores = scores[keep]
                pred_labels = labels[keep]
                pred_boxes = pred_boxes[keep]
                # Filter by score threshold
                score_keep = pred_scores > 0.5
                pred_scores = pred_scores[score_keep]
                pred_labels = pred_labels[score_keep]
                pred_boxes = pred_boxes[score_keep]
                # Scale and convert boxes
                if len(pred_boxes) > 0:
                    boxes = pred_boxes.cpu().numpy()
                    boxes = boxes * np.array([width, height, width, height])
                    # Convert from [cx, cy, w, h] to [x, y, w, h]
                    boxes[:, 0] -= boxes[:, 2] / 2
                    boxes[:, 1] -= boxes[:, 3] / 2
                    # Now boxes is [xmin, ymin, w, h]
                    for box, score, label in zip(boxes, pred_scores.cpu().numpy(), pred_labels.cpu().numpy()):
                        results.append({
                            "image_id": img_id,
                            "category_id": label + 1,
                            "bbox": box.tolist(),
                            "score": float(score)
                        })
    results_file = "eval_predictions.json"
    with open(results_file, "w") as f:
        json.dump(results, f)
    coco_dt = coco_val.loadRes(results_file)
    coco_eval = COCOeval(coco_val, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    metrics = {
        "mAP": coco_eval.stats[0],
        "AP@50": coco_eval.stats[1],
        "AP@75": coco_eval.stats[2],
    }
    return metrics

def main():
    print(f"Current working directory: {os.getcwd()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    weights_dir = "weights"
    if not os.path.exists(weights_dir):
        raise FileNotFoundError(f"Weights directory {weights_dir} not found.")
    weight_files = [f for f in os.listdir(weights_dir) if f.startswith("epoch_") and f.endswith(".pth")]
    if not weight_files:
        raise FileNotFoundError(f"No weight files found in {weights_dir}")
    latest_weight = max(weight_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    weights_path = os.path.join(weights_dir, latest_weight)
    print(f"Using latest weights: {weights_path}")
    with open(VAL_ANNO_FILE, 'r') as f:
        coco_data = json.load(f)
    num_classes = len(coco_data['categories'])
    print(f"Number of classes: {num_classes}")
    print("Initializing processor and model...")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", size=800)
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    ).to(device)
    model.load_state_dict(torch.load(weights_path))
    print("\nLoading test dataset...")
    val_dataset = SimpleCocoDataset(VAL_IMG_DIR, VAL_ANNO_FILE, processor)
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    print("\nEvaluating model on test set...")
    coco_val = COCO(VAL_ANNO_FILE)
    eval_metrics = evaluate_model(model, val_loader, coco_val, device, num_classes)
    print(f"Evaluation metrics: {eval_metrics}")
    print("Evaluation complete!")

if __name__ == "__main__":
    main()
