# train.py
import os
import json
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DetrImageProcessor
from pycocotools.coco import COCO
import wandb

# Import only evaluation (avoid circular training import)
from test import evaluate_model

# -----------------------------
# Dataset used for TRAIN/VAL
# -----------------------------
class SimpleCocoDataset(Dataset):
    def __init__(self, img_folder, anno_file, processor: DetrImageProcessor):
        self.coco = COCO(anno_file)
        self.img_folder = img_folder
        self.processor = processor
        # KEEP ALL IMAGES (even with zero anns → valid negatives)
        self.ids = list(self.coco.imgs.keys())
        print(f"[Dataset] {os.path.basename(img_folder)}: {len(self.ids)} images (incl. empty)")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_folder, img_info['file_name'])

        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            cx = (x + w / 2.0) / W
            cy = (y + h / 2.0) / H
            w_n = w / W
            h_n = h / H
            boxes.append([
                max(0.0, min(1.0, cx)),
                max(0.0, min(1.0, cy)),
                max(0.0, min(1.0, w_n)),
                max(0.0, min(1.0, h_n)),
            ])
            labels.append(ann['category_id'] - 1)  # COCO 1..K → 0..K-1

        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)
        target = {
            "class_labels": torch.tensor(labels, dtype=torch.long),       # can be empty
            "boxes": torch.tensor(boxes, dtype=torch.float32),            # can be empty
            "image_id": torch.tensor([img_id], dtype=torch.int64),
        }
        return pixel_values, target

def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return pixel_values, targets

# -----------------------------
# Training helpers
# -----------------------------
def train_one_epoch(model, dataloader, optimizer, device, epoch_idx):
    model.train()
    total_loss = 0.0
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
            print(f"[Train] Epoch {epoch_idx+1} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")
        if wandb.run is not None:
            wandb.log({"train/loss": loss.item(), "epoch": epoch_idx + 1})

    return total_loss / max(1, len(dataloader))

# -----------------------------
# Train + validate each epoch
# -----------------------------
def train_and_validate(
    model,
    processor,
    device,
    num_classes,
    train_img_dir,
    train_anno_file,
    val_img_dir,
    val_anno_file,
    weights_dir,
    num_epochs=10,
    batch_size=2,
    num_workers=2,
    learning_rate=1e-4,
    score_thresh=0.05,
    topk=100,
):
    # Data
    train_dataset = SimpleCocoDataset(train_img_dir, train_anno_file, processor)
    val_dataset   = SimpleCocoDataset(val_img_dir,   val_anno_file,   processor)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    val_loader   = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # COCO GT for validation
    coco_val = COCO(val_anno_file)

    best_map = -1.0
    best_epoch = -1
    best_metrics = None

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        print(f"[Train] Epoch {epoch+1}/{num_epochs} | avg loss: {avg_loss:.4f}")
        if wandb.run is not None:
            wandb.log({"train/epoch_avg_loss": avg_loss, "epoch": epoch + 1})

        # Validate
        val_metrics = evaluate_model(
            model, val_loader, coco_val, device, num_classes,
            img_dir=val_img_dir, score_thresh=score_thresh, topk=topk
        )

        print(
            f"[Val @ epoch {epoch+1}] "
            f"mAP={val_metrics['mAP']:.4f} | AP50={val_metrics['AP@50']:.4f} | AP75={val_metrics['AP@75']:.4f} | "
            f"Disease Acc={val_metrics['disease_acc_pct']:.2f}% | "
            f"BBox IoU Mean={val_metrics['bbox_iou_mean_pct']:.2f}% | "
            f"BBox Area MAPE={val_metrics['bbox_area_mape_pct']:.2f}% "
            f"(matched_pairs={val_metrics['matched_pairs']})"
        )
        if val_metrics.get("per_class"):
            print("Per-class AP (IoU=.50:.95) and AP50:")
            for cls_name, cls_metrics in val_metrics["per_class"].items():
                ap = cls_metrics["AP"]
                ap50 = cls_metrics["AP50"]
                print(f"  - {cls_name:20s} AP={ap:.4f} | AP50={ap50:.4f}")

        # Log to W&B
        if wandb.run is not None:
            wandb.log({**{f"val/{k}": v for k, v in val_metrics.items() if k != "per_class"}, "epoch": epoch + 1})
            if val_metrics.get("per_class"):
                log_dict = {}
                for cls_name, cls_metrics in val_metrics["per_class"].items():
                    log_dict[f"val/AP/{cls_name}"] = cls_metrics["AP"]
                    log_dict[f"val/AP50/{cls_name}"] = cls_metrics["AP50"]
                log_dict["epoch"] = epoch + 1
                wandb.log(log_dict)

        # Save snapshot
        os.makedirs(weights_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(weights_dir, f"epoch_{epoch+1}.pth"))

        # Track best by mAP
        if val_metrics["mAP"] > best_map:
            best_map = val_metrics["mAP"]
            best_epoch = epoch + 1
            best_metrics = val_metrics
            torch.save(model.state_dict(), os.path.join(weights_dir, "best.pth"))
            with open(os.path.join(weights_dir, "best_metrics.json"), "w") as f:
                json.dump({"epoch": best_epoch, "metrics": best_metrics}, f, indent=2)
            print(f"[Best] New best mAP={best_map:.4f} at epoch {best_epoch}. Saved best.pth")

    return best_epoch, best_metrics
