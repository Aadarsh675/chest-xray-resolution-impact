import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import DetrImageProcessor, DetrForObjectDetection
import wandb
import numpy as np

# -----------------------------
# Paths & constants
# -----------------------------
DATA_DIR = "data"
ANNO_DIR = os.path.join(DATA_DIR, "annotations")
IMAGE_DIR = os.path.join(DATA_DIR, "images")

TRAIN_ANNO_FILE = os.path.join(ANNO_DIR, "train_annotations_coco.json")
VAL_ANNO_FILE   = os.path.join(ANNO_DIR, "test_annotations_coco.json")   # using test as validation
TRAIN_IMG_DIR   = os.path.join(IMAGE_DIR, "train")
VAL_IMG_DIR     = os.path.join(IMAGE_DIR, "test")

MODEL_NAME   = "facebook/detr-resnet-50"
BATCH_SIZE   = 2
NUM_WORKERS  = 0
NUM_EPOCHS   = 10
LEARNING_RATE = 1e-4

# COCO eval settings
TOPK = 100
SCORE_THRESH = 0.0  # keep at 0; COCOeval handles thresholds


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
        print(f"Found {len(self.ids)} images with annotations in {os.path.basename(img_folder)}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Get image
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_folder, img_info['file_name'])

        # Load image
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Convert annotations to DETR format
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            # convert to center-format normalized
            x_center = (x + w/2) / width
            y_center = (y + h/2) / height
            w_norm = w / width
            h_norm = h / height
            # clamp
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))
            boxes.append([x_center, y_center, w_norm, h_norm])
            labels.append(ann['category_id'] - 1)  # 0-indexed

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


# -----------------------------
# Evaluation (COCO mAP)
# -----------------------------
@torch.no_grad()
def evaluate_model(model, dataloader, coco_gt, device, num_classes, img_dir,
                   score_thresh=SCORE_THRESH, topk=TOPK):
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
                    print(f"[Eval] Unable to read size for {img_path}: {e}")
                    continue

            logits = outputs.logits[i]          # (queries, num_classes+1)
            pred_boxes = outputs.pred_boxes[i]  # (queries, 4) normalized cx,cy,w,h

            probs = logits.softmax(-1)
            scores, labels = probs.max(-1)
            keep = labels != num_classes        # drop no-object
            scores = scores[keep]
            labels = labels[keep]
            boxes  = pred_boxes[keep]

            # optional threshold (keep at 0.0)
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

            # clip
            boxes[:, 0] = np.clip(boxes[:, 0], 0, max(0, width - 1))
            boxes[:, 1] = np.clip(boxes[:, 1], 0, max(0, height - 1))
            boxes[:, 2] = np.clip(boxes[:, 2], 1e-6, width)
            boxes[:, 3] = np.clip(boxes[:, 3], 1e-6, height)

            for box, score, label in zip(boxes, scores.cpu().numpy(), labels.cpu().numpy()):
                results.append({
                    "image_id": img_id,
                    "category_id": int(label) + 1,  # back to COCO 1-indexed
                    "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "score": float(score),
                })

    if len(results) == 0:
        return {"mAP": 0.0, "AP@50": 0.0, "AP@75": 0.0}

    results_file = "val_predictions.json"
    with open(results_file, "w") as f:
        json.dump(results, f)

    coco_dt = coco_gt.loadRes(results_file)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "mAP":  float(coco_eval.stats[0]),
        "AP@50": float(coco_eval.stats[1]),
        "AP@75": float(coco_eval.stats[2]),
    }


# -----------------------------
# Training loop (eval after EVERY batch)
# -----------------------------
def main():
    # Initialize W&B
    wandb.init(project="detr-nih-chest-xray", name="detr_train_eval_per_batch", config={
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "model": MODEL_NAME
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load annotations to get number of classes
    with open(TRAIN_ANNO_FILE, 'r') as f:
        coco_train_data = json.load(f)
    num_classes = len(coco_train_data['categories'])
    print(f"Number of classes: {num_classes}")

    # Processor & model
    processor = DetrImageProcessor.from_pretrained(MODEL_NAME, size=800)
    model = DetrForObjectDetection.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    ).to(device)

    # Datasets / loaders
    print("\nLoading datasets...")
    train_dataset = SimpleCocoDataset(TRAIN_IMG_DIR, TRAIN_ANNO_FILE, processor)
    val_dataset   = SimpleCocoDataset(VAL_IMG_DIR,   VAL_ANNO_FILE,   processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # COCO GT for validation
    coco_val = COCO(VAL_ANNO_FILE)

    # Weights dir
    os.makedirs("weights", exist_ok=True)

    best_map = -1.0
    best_info = {"epoch": None, "batch_global": None, "metrics": None}

    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (pixel_values, targets) in enumerate(train_loader):
            global_step += 1

            pixel_values = pixel_values.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(pixel_values=pixel_values, labels=targets)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Log batch loss
            if batch_idx % 1 == 0:
                print(f"[Train] Epoch {epoch+1}/{NUM_EPOCHS} | Batch {batch_idx+1}/{num_batches} | Loss: {loss.item():.4f}")
            wandb.log({"train/loss": loss.item(), "step": global_step, "epoch": epoch + 1})

            # ---- Evaluate AFTER EVERY BATCH ----
            with torch.no_grad():
                val_metrics = evaluate_model(
                    model, val_loader, coco_val, device, num_classes,
                    img_dir=VAL_IMG_DIR, score_thresh=SCORE_THRESH, topk=TOPK
                )
            print(f"[Eval@Batch] step={global_step} | mAP={val_metrics['mAP']:.4f} | AP50={val_metrics['AP@50']:.4f} | AP75={val_metrics['AP@75']:.4f}")
            wandb.log({
                "val/mAP": val_metrics["mAP"],
                "val/AP@50": val_metrics["AP@50"],
                "val/AP@75": val_metrics["AP@75"],
                "step": global_step,
                "epoch": epoch + 1
            })

            # Keep best weights
            if val_metrics["mAP"] > best_map:
                best_map = val_metrics["mAP"]
                best_info = {"epoch": epoch + 1, "batch_global": global_step, "metrics": val_metrics}
                torch.save(model.state_dict(), "weights/best.pth")
                with open("weights/best_metrics.json", "w") as f:
                    json.dump(best_info, f, indent=2)
                print(f"[Best] New best mAP={best_map:.4f} at step {global_step}. Saved weights/best.pth")

        avg_loss = total_loss / max(1, num_batches)
        print(f"[Train] Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        wandb.log({"train/epoch_avg_loss": avg_loss, "epoch": epoch + 1})

        # (Optional) save per-epoch snapshot
        epoch_path = f"weights/epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), epoch_path)
        print(f"[Checkpoint] Saved {epoch_path}")

    print("\nTraining complete!")
    if best_info["epoch"] is not None:
        print(f"Best checkpoint at epoch {best_info['epoch']}, step {best_info['batch_global']}, metrics: {best_info['metrics']}")
    else:
        print("No best checkpoint recorded (mAP never improved).")

    wandb.finish()


if __name__ == "__main__":
    main()
