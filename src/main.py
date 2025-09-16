import os
import json
import uuid
from datetime import datetime
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import DetrImageProcessor, DetrForObjectDetection
import wandb

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

MODEL_NAME    = "facebook/detr-resnet-50"
BATCH_SIZE    = 2
NUM_WORKERS   = 0
NUM_EPOCHS    = 10
LEARNING_RATE = 1e-4

# COCO eval settings
TOPK = 100
SCORE_THRESH = 0.0

# -----------------------------
# Dataset
# -----------------------------
class SimpleCocoDataset(Dataset):
    def __init__(self, img_folder, anno_file, processor):
        self.coco = COCO(anno_file)
        self.img_folder = img_folder
        self.processor = processor
        self.ids = [img_id for img_id in self.coco.imgs.keys()
                    if len(self.coco.getAnnIds(imgIds=img_id)) > 0]
        print(f"Found {len(self.ids)} images with annotations in {os.path.basename(img_folder)}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_folder, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            cx = (x + w/2) / width
            cy = (y + h/2) / height
            w_n = w / width
            h_n = h / height
            boxes.append([cx, cy, w_n, h_n])
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

# -----------------------------
# (keep your existing evaluate_model here — IoU, MAPE, mAP)
# -----------------------------
# ... paste the evaluate_model(...) function we wrote earlier ...

# -----------------------------
# Training loop — evaluate per EPOCH
# -----------------------------
def main():
    run_name = f"detr_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    wandb.init(project="detr-nih-chest-xray", name=run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(TRAIN_ANNO_FILE, 'r') as f:
        coco_train_data = json.load(f)
    num_classes = len(coco_train_data['categories'])

    processor = DetrImageProcessor.from_pretrained(MODEL_NAME, size=800)
    model = DetrForObjectDetection.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    ).to(device)

    train_dataset = SimpleCocoDataset(TRAIN_IMG_DIR, TRAIN_ANNO_FILE, processor)
    val_dataset   = SimpleCocoDataset(VAL_IMG_DIR, VAL_ANNO_FILE, processor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    coco_val = COCO(VAL_ANNO_FILE)
    os.makedirs("weights", exist_ok=True)

    best_map = -1.0
    for epoch in range(NUM_EPOCHS):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        for batch_idx, (pixel_values, targets) in enumerate(train_loader):
            pixel_values = pixel_values.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(pixel_values=pixel_values, labels=targets)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            wandb.log({"train/loss": loss.item(), "epoch": epoch+1})

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"[Train] Epoch {epoch+1}/{NUM_EPOCHS} | avg loss: {avg_loss:.4f}")
        wandb.log({"train/epoch_avg_loss": avg_loss, "epoch": epoch+1})

        # ---- Evaluate after epoch ----
        val_metrics = evaluate_model(
            model, val_loader, coco_val, device, num_classes,
            img_dir=VAL_IMG_DIR, score_thresh=SCORE_THRESH, topk=TOPK
        )
        print(f"[Eval@Epoch {epoch+1}] {val_metrics}")
        wandb.log(val_metrics)

        if val_metrics["mAP"] > best_map:
            best_map = val_metrics["mAP"]
            torch.save(model.state_dict(), "weights/best.pth")
            with open("weights/best_metrics.json", "w") as f:
                json.dump({"epoch": epoch+1, "metrics": val_metrics}, f, indent=2)
            print(f"[Best] New best mAP={best_map:.4f} at epoch {epoch+1}")

        # optional: save epoch snapshot
        torch.save(model.state_dict(), f"weights/epoch_{epoch+1}.pth")

    print("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    main()
