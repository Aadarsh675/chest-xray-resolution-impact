# main.py
import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from transformers import DetrImageProcessor, DetrForObjectDetection

# -----------------------------
# Optional: Weights & Biases
# -----------------------------
USE_WANDB = False
if USE_WANDB:
    import wandb

# -----------------------------
# Paths & constants
# -----------------------------
DATA_DIR = "data"
ANNO_DIR = os.path.join(DATA_DIR, "annotations")
IMAGE_DIR = os.path.join(DATA_DIR, "images")

TRAIN_ANNO_FILE = os.path.join(ANNO_DIR, "train_annotations_coco.json")
VAL_ANNO_FILE   = os.path.join(ANNO_DIR, "test_annotations_coco.json")  # using test as validation
TRAIN_IMG_DIR   = os.path.join(IMAGE_DIR, "train")
VAL_IMG_DIR     = os.path.join(IMAGE_DIR, "test")

WEIGHTS_DIR = "weights"
MODEL_NAME  = "facebook/detr-resnet-50"

BATCH_SIZE  = 2
NUM_WORKERS = 0
NUM_EPOCHS  = 10
LEARNING_RATE = 1e-4

# For evaluation
TOPK = 100          # cap predictions per image
SCORE_THRESH = 0.0  # let COCOeval handle thresholds


# -----------------------------
# Dataset
# -----------------------------
class SimpleCocoDataset(Dataset):
    def __init__(self, img_folder, anno_file, processor):
        self.coco = COCO(anno_file)
        self.img_folder = img_folder
        self.processor = processor
        # Only use images that have annotations
        self.ids = [img_id for img_id in self.coco.imgs.keys()
                    if len(self.coco.getAnnIds(imgIds=img_id)) > 0]
        print(f"[Dataset] {os.path.basename(img_folder)}: {len(self.ids)} images with annotations")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_folder, img_info['file_name'])

        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Convert COCO bbox [x,y,w,h] -> DETR normalized [cx,cy,w,h]
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            cx = (x + w / 2.0) / width
            cy = (y + h / 2.0) / height
            w_n = w / width
            h_n = h / height
            # clamp
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            w_n = max(0, min(1, w_n))
            h_n = max(0, min(1, h_n))
            boxes.append([cx, cy, w_n, h_n])
            # COCO ids are 1-based; DETR expects 0-based
            labels.append(ann['category_id'] - 1)

        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)

        target = {
            "class_labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "image_id": torch.tensor([img_id], dtype=torch.int64),
        }
        return pixel_values, target


def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return pixel_values, targets


# -----------------------------
# Training/Eval helpers
# -----------------------------
def train_one_epoch(model, dataloader, optimizer, device, epoch, use_wandb=False):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, (pixel_values, targets) in enumerate(dataloader):
        pixel_values = pixel_values.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(pixel_values=pixel_values, labels=targets)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"[Train] Epoch {epoch+1} | Batch {batch_idx}/{num_batches} | Loss: {loss.item():.4f}")
            if use_wandb:
                wandb.log({"train/loss": loss.item(),
                           "train/step": batch_idx + epoch * num_batches})

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss


@torch.no_grad()
def evaluate_model(model, dataloader, coco_gt, device, num_classes,
                   img_dir, score_thresh=SCORE_THRESH, topk=TOPK):
    """
    Produces COCO-style results and returns mAP, AP@50, AP@75.
    """
    model.eval()
    results = []

    for pixel_values, targets in dataloader:
        pixel_values = pixel_values.to(device)
        outputs = model(pixel_values=pixel_values)

        B = pixel_values.shape[0]
        for i in range(B):
            target = targets[i]
            img_id = target["image_id"].item()
            img_info = coco_gt.loadImgs(img_id)[0]
            img_path = os.path.join(img_dir, img_info['file_name'])

            width = img_info.get("width")
            height = img_info.get("height")
            if width is None or height is None:
                try:
                    image = Image.open(img_path).convert("RGB")
                    width, height = image.size
                except Exception as e:
                    print(f"[Eval] Failed to read size for {img_path}: {e}")
                    continue

            logits = outputs.logits[i]       # (queries, num_classes+1)
            pred_boxes = outputs.pred_boxes[i]  # (queries, 4) normalized cx,cy,w,h

            probs = logits.softmax(-1)
            scores, labels = probs.max(-1)
            keep = labels != num_classes      # drop no-object
            scores = scores[keep]
            labels = labels[keep]
            boxes  = pred_boxes[keep]

            # Optional threshold (keep zero, let COCOeval handle)
            if score_thresh > 0.0:
                m = scores > score_thresh
                scores = scores[m]
                labels = labels[m]
                boxes  = boxes[m]

            if scores.numel() > 0 and scores.numel() > topk:
                topk_idx = torch.topk(scores, k=topk).indices
                scores = scores[topk_idx]
                labels = labels[topk_idx]
                boxes  = boxes[topk_idx]

            if scores.numel() == 0:
                continue

            # scale to pixels & convert to [x,y,w,h]
            boxes = boxes.cpu().numpy()
            boxes = boxes * np.array([width, height, width, height], dtype=np.float32)
            boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
            boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0

            # clip to image
            boxes[:, 0] = np.clip(boxes[:, 0], 0, max(0, width - 1))
            boxes[:, 1] = np.clip(boxes[:, 1], 0, max(0, height - 1))
            boxes[:, 2] = np.clip(boxes[:, 2], 1e-6, width)
            boxes[:, 3] = np.clip(boxes[:, 3], 1e-6, height)

            for box, score, label in zip(boxes, scores.cpu().numpy(), labels.cpu().numpy()):
                results.append({
                    "image_id": img_id,
                    "category_id": int(label) + 1,  # back to COCO 1-based ids
                    "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "score": float(score),
                })

    # If nothing predicted, avoid crash
    if len(results) == 0:
        print("[Eval] No predictions produced; returning zeros.")
        return {"mAP": 0.0, "AP@50": 0.0, "AP@75": 0.0}

    results_file = "val_predictions.json"
    with open(results_file, "w") as f:
        json.dump(results, f)

    coco_dt = coco_gt.loadRes(results_file)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        "mAP":  float(coco_eval.stats[0]),
        "AP@50": float(coco_eval.stats[1]),
        "AP@75": float(coco_eval.stats[2]),
    }
    return metrics


# -----------------------------
# Main training loop
# -----------------------------
def main():
    print(f"cwd: {os.getcwd()}")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Load annotations (to get number of classes)
    with open(TRAIN_ANNO_FILE, "r") as f:
        train_coco_data = json.load(f)
    num_classes = len(train_coco_data["categories"])
    print(f"num_classes: {num_classes}")

    # Init processor & model
    processor = DetrImageProcessor.from_pretrained(MODEL_NAME, size=800)
    model = DetrForObjectDetection.from_pretrained(
        MODEL_NAME, num_labels=num_classes, ignore_mismatched_sizes=True
    ).to(device)

    # Data
    train_dataset = SimpleCocoDataset(TRAIN_IMG_DIR, TRAIN_ANNO_FILE, processor)
    val_dataset   = SimpleCocoDataset(VAL_IMG_DIR,   VAL_ANNO_FILE,   processor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # COCO GT for validation
    coco_val = COCO(VAL_ANNO_FILE)

    # W&B
    if USE_WANDB:
        wandb.init(project="detr-nih-chest-xray", name="main-train-eval",
                   config={"epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE,
                           "lr": LEARNING_RATE, "model": MODEL_NAME})

    best_map = -1.0
    best_epoch = -1
    best_metrics = None

    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        # ---- Train ----
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, use_wandb=USE_WANDB)
        print(f"[Train] Epoch {epoch+1}/{NUM_EPOCHS} | avg loss: {avg_loss:.4f}")
        if USE_WANDB:
            wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch + 1})

        # Save epoch checkpoint
        epoch_path = os.path.join(WEIGHTS_DIR, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_path)
        print(f"[Checkpoint] Saved: {epoch_path}")

        # ---- Evaluate ----
        print("[Eval] Running COCOeval on validation split...")
        eval_metrics = evaluate_model(
            model, val_loader, coco_val, device, num_classes,
            img_dir=VAL_IMG_DIR, score_thresh=SCORE_THRESH, topk=TOPK
        )
        print(f"[Eval] Epoch {epoch+1} metrics: {eval_metrics}")
        if USE_WANDB:
            wandb.log({"val/mAP": eval_metrics["mAP"],
                       "val/AP@50": eval_metrics["AP@50"],
                       "val/AP@75": eval_metrics["AP@75"],
                       "epoch": epoch + 1})

        # Track best by mAP
        if eval_metrics["mAP"] > best_map:
            best_map = eval_metrics["mAP"]
            best_epoch = epoch + 1
            best_metrics = eval_metrics
            # Save best weights
            best_path = os.path.join(WEIGHTS_DIR, "best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"[Best] New best mAP={best_map:.4f} at epoch {best_epoch}. Saved: {best_path}")
            # Write metrics to a small json
            with open(os.path.join(WEIGHTS_DIR, "best_metrics.json"), "w") as f:
                json.dump({"epoch": best_epoch, "metrics": best_metrics}, f, indent=2)

    print("Training complete!")
    if best_epoch > 0:
        print(f"Best epoch: {best_epoch} | Best metrics: {best_metrics}")
    else:
        print("No best epoch recorded (check evaluation results).")

    if USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
