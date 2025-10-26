# train.py
import os
import sys
import json
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DetrImageProcessor
from pycocotools.coco import COCO
import wandb

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import only evaluation (avoid circular import)
from src.test import evaluate_model


# -----------------------------
# Dataset used for TRAIN/VAL
# -----------------------------
class SimpleCocoDataset(Dataset):
    def __init__(self, img_folder, anno_file, processor: DetrImageProcessor):
        self.coco = COCO(anno_file)
        self.img_folder = img_folder
        self.processor = processor
        # KEEP ALL IMAGES (even those with zero annotations)
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

        # If COCO width/height missing, update our local view (doesn't write file)
        if not img_info.get("width") or not img_info.get("height"):
            img_info["width"], img_info["height"] = W, H

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
            # COCO category ids are 1..K; DETR expects 0..K-1
            labels.append(ann['category_id'] - 1)

        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)
        
        # Ensure boxes and labels are 2D tensors even when empty
        # This is required for DETR to work properly
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        
        target = {
            "class_labels": labels,
            "boxes": boxes,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
        }
        return pixel_values, target


def collate_fn(batch):
    """
    Collate function that handles variable-sized images.
    DETR's processor already normalizes images, so we just need to handle batching.
    We'll pad to the maximum size in the batch for efficiency.
    """
    # Get max dimensions in batch
    max_height = max([item[0].shape[1] for item in batch])
    max_width = max([item[0].shape[2] for item in batch])
    
    # Ensure dimensions are multiples of 32 for better GPU efficiency
    max_height = ((max_height + 31) // 32) * 32
    max_width = ((max_width + 31) // 32) * 32
    
    # Pad all images to max dimensions (use mean pixel value instead of zeros)
    batch_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)  # ImageNet mean
    padded_images = []
    
    for item in batch:
        img = item[0]  # Shape: [3, H, W]
        h, w = img.shape[1], img.shape[2]
        
        # Create padded tensor with mean values (better than zeros)
        padded = batch_mean.repeat(1, max_height, max_width)
        padded[:, :h, :w] = img
        padded_images.append(padded)
    
    pixel_values = torch.stack(padded_images)
    targets = [item[1] for item in batch]
    return pixel_values, targets


# -----------------------------
# Training helpers
# -----------------------------
def train_one_epoch(model, dataloader, optimizer, device, epoch_idx):
    model.train()
    total_loss = 0.0
    global_step = epoch_idx * len(dataloader)  # Track global step for proper logging
    
    for batch_idx, (pixel_values, targets) in enumerate(dataloader):
        pixel_values = pixel_values.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(pixel_values=pixel_values, labels=targets)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients (common in transformer models)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()
        
        # Calculate global step for consistent logging
        current_step = global_step + batch_idx
        
        if batch_idx % 10 == 0:
            print(f"[Train] Epoch {epoch_idx+1} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")
        
        # Log to wandb every batch with proper step tracking
        if wandb.run is not None:
            wandb.log({
                "train/batch_loss": loss.item(),
                "train/epoch": epoch_idx + 1,
                "train/learning_rate": optimizer.param_groups[0]['lr']
            }, step=current_step)

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
    test_img_dir,
    test_anno_file,
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
    test_dataset  = SimpleCocoDataset(test_img_dir,  test_anno_file,  processor)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    val_loader   = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    test_loader  = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )

    # Optimizer with weight decay for better regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler (cosine annealing for better convergence)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # COCO GT for validation and test
    coco_val = COCO(val_anno_file)
    coco_test = COCO(test_anno_file)

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

        # Log to W&B with comprehensive metrics
        if wandb.run is not None:
            # Main validation metrics
            val_log = {
                "val/mAP": val_metrics.get("mAP", 0.0),
                "val/AP@50": val_metrics.get("AP@50", 0.0),
                "val/AP@75": val_metrics.get("AP@75", 0.0),
                "val/Disease_Accuracy": val_metrics.get("disease_acc_pct", 0.0),
                "val/BBox_IoU_Mean": val_metrics.get("bbox_iou_mean_pct", 0.0),
                "val/BBox_Area_MAPE": val_metrics.get("bbox_area_mape_pct", 0.0),
                "val/Matched_Pairs": val_metrics.get("matched_pairs", 0),
                "epoch": epoch + 1
            }
            wandb.log(val_log)
            
            # Per-class metrics for each disease
            if val_metrics.get("per_class"):
                per_class_log = {}
                for cls_name, cls_metrics in val_metrics["per_class"].items():
                    per_class_log[f"val/AP/{cls_name}"] = cls_metrics.get("AP", 0.0)
                    per_class_log[f"val/AP50/{cls_name}"] = cls_metrics.get("AP50", 0.0)
                    per_class_log[f"val/Precision/{cls_name}"] = cls_metrics.get("precision", 0.0)
                    per_class_log[f"val/Recall/{cls_name}"] = cls_metrics.get("recall", 0.0)
                    per_class_log[f"val/F1/{cls_name}"] = cls_metrics.get("f1", 0.0)
                    per_class_log[f"val/mIoU/{cls_name}"] = cls_metrics.get("miou", 0.0)
                per_class_log["epoch"] = epoch + 1
                wandb.log(per_class_log)

        # Test set evaluation after each epoch
        test_metrics = evaluate_model(
            model, test_loader, coco_test, device, num_classes,
            img_dir=test_img_dir, score_thresh=score_thresh, topk=topk
        )

        print(
            f"[Test @ epoch {epoch+1}] "
            f"mAP={test_metrics['mAP']:.4f} | AP50={test_metrics['AP@50']:.4f} | AP75={test_metrics['AP@75']:.4f} | "
            f"Disease Acc={test_metrics['disease_acc_pct']:.2f}% | "
            f"BBox IoU Mean={test_metrics['bbox_iou_mean_pct']:.2f}% | "
            f"BBox Area MAPE={test_metrics['bbox_area_mape_pct']:.2f}% "
            f"(matched_pairs={test_metrics['matched_pairs']})"
        )

        # Log test metrics to W&B
        if wandb.run is not None:
            # Main test epoch metrics
            test_epoch_log = {
                "test_epoch/mAP": test_metrics.get("mAP", 0.0),
                "test_epoch/AP@50": test_metrics.get("AP@50", 0.0),
                "test_epoch/AP@75": test_metrics.get("AP@75", 0.0),
                "test_epoch/Disease_Accuracy": test_metrics.get("disease_acc_pct", 0.0),
                "test_epoch/BBox_IoU_Mean": test_metrics.get("bbox_iou_mean_pct", 0.0),
                "test_epoch/BBox_Area_MAPE": test_metrics.get("bbox_area_mape_pct", 0.0),
                "test_epoch/Matched_Pairs": test_metrics.get("matched_pairs", 0),
                "epoch": epoch + 1
            }
            wandb.log(test_epoch_log)
            
            # Per-class test epoch metrics for each disease
            if test_metrics.get("per_class"):
                test_per_class_log = {}
                for cls_name, cls_metrics in test_metrics["per_class"].items():
                    test_per_class_log[f"test_epoch/AP/{cls_name}"] = cls_metrics.get("AP", 0.0)
                    test_per_class_log[f"test_epoch/AP50/{cls_name}"] = cls_metrics.get("AP50", 0.0)
                    test_per_class_log[f"test_epoch/Precision/{cls_name}"] = cls_metrics.get("precision", 0.0)
                    test_per_class_log[f"test_epoch/Recall/{cls_name}"] = cls_metrics.get("recall", 0.0)
                    test_per_class_log[f"test_epoch/F1/{cls_name}"] = cls_metrics.get("f1", 0.0)
                    test_per_class_log[f"test_epoch/mIoU/{cls_name}"] = cls_metrics.get("miou", 0.0)
                test_per_class_log["epoch"] = epoch + 1
                wandb.log(test_per_class_log)

        # Save snapshot
        os.makedirs(weights_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(weights_dir, f"epoch_{epoch+1}.pth"))

        # Track best by test mAP (not validation mAP)
        if test_metrics["mAP"] > best_map:
            best_map = test_metrics["mAP"]
            best_epoch = epoch + 1
            best_metrics = test_metrics
            torch.save(model.state_dict(), os.path.join(weights_dir, "best.pth"))
            with open(os.path.join(weights_dir, "best_metrics.json"), "w") as f:
                json.dump({"epoch": best_epoch, "metrics": best_metrics}, f, indent=2)
            print(f"[Best] New best test mAP={best_map:.4f} at epoch {best_epoch}. Saved best.pth")
        
        # Step scheduler at the end of each epoch
        scheduler.step()
        
        # Log learning rate after scheduler step
        if wandb.run is not None:
            wandb.log({"learning_rate": scheduler.get_last_lr()[0], "epoch": epoch + 1})

    return best_epoch, best_metrics


def main(args, wandb_run=None):
    """Main entry point for training from command line arguments."""
    import os
    import wandb
    from transformers import DetrForObjectDetection, DetrImageProcessor
    import torch
    from pycocotools.coco import COCO
    import uuid
    from datetime import datetime
    from zoneinfo import ZoneInfo
    
    # Load configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set up paths
    train_anno_file = os.path.join(args.anno_dir, "train_annotations_coco.json")
    val_anno_file = os.path.join(args.anno_dir, "val_annotations_coco.json")
    test_anno_file = os.path.join(args.anno_dir, "test_annotations_coco.json")
    
    # Check if val file exists, if not create split
    if not os.path.exists(val_anno_file):
        print("[INFO] Validation split not found, creating from train...")
        import json
        import random
        
        # Load and split the training data
        with open(train_anno_file, 'r') as f:
            data = json.load(f)
        
        images = data.get("images", [])
        anns = data.get("annotations", [])
        cats = data.get("categories", [])
        
        img_ids = [im["id"] for im in images]
        random.Random(42).shuffle(img_ids)
        n_val = max(1, int(round(len(img_ids) * 0.2)))
        val_ids = set(img_ids[:n_val])
        train_ids = set(img_ids[n_val:])
        
        # Create splits
        train_imgs = [im for im in images if im["id"] in train_ids]
        train_anns = [a for a in anns if a["image_id"] in train_ids]
        val_imgs = [im for im in images if im["id"] in val_ids]
        val_anns = [a for a in anns if a["image_id"] in val_ids]
        
        # Reassign annotation ids
        for i, a in enumerate(train_anns, 1): a["id"] = i
        for i, a in enumerate(val_anns, 1): a["id"] = i
        
        train_split = {"info": {}, "licenses": [], "images": train_imgs, "annotations": train_anns, "categories": cats}
        val_split = {"info": {}, "licenses": [], "images": val_imgs, "annotations": val_anns, "categories": cats}
        
        train_split_json = os.path.join(args.anno_dir, "train_annotations_coco_split.json")
        with open(train_split_json, "w") as f:
            json.dump(train_split, f, indent=2)
        with open(val_anno_file, "w") as f:
            json.dump(val_split, f, indent=2)
        
        print(f"[Split] Wrote train split -> {train_split_json} ({len(train_imgs)} imgs)")
        print(f"[Split] Wrote val split -> {val_anno_file} ({len(val_imgs)} imgs)")
        
        train_anno_file = train_split_json
    
    # Load class names
    coco = COCO(train_anno_file)
    cats = sorted(coco.loadCats(coco.getCatIds()), key=lambda c: c['id'])
    class_names = [c['name'] for c in cats]
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    
    # Initialize model
    MODEL_NAME = "facebook/detr-resnet-50"
    processor = DetrImageProcessor.from_pretrained(MODEL_NAME)
    model = DetrForObjectDetection.from_pretrained(
        MODEL_NAME, num_labels=num_classes, ignore_mismatched_sizes=True
    ).to(device)
    
    # Initialize wandb only if not provided externally
    if wandb_run is None:
        run_name = f"detr_vindr_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        wandb.init(
            project="chest-xray-resolution-impact",
            name=run_name,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "model": MODEL_NAME,
                "classes": class_names,
                "num_classes": num_classes,
            }
        )
        should_finish_wandb = True
    else:
        should_finish_wandb = False
    
    # Create weights directory
    if wandb_run is None:
        weights_dir = os.path.join(args.weights_dir, run_name)
    else:
        weights_dir = os.path.join(args.weights_dir, wandb_run.name)
    os.makedirs(weights_dir, exist_ok=True)
    
    # Train
    best_epoch, best_metrics = train_and_validate(
        model=model,
        processor=processor,
        device=device,
        num_classes=num_classes,
        train_img_dir=args.train_img_dir,
        train_anno_file=train_anno_file,
        val_img_dir=args.val_img_dir,
        val_anno_file=val_anno_file,
        test_img_dir=args.test_img_dir,
        test_anno_file=test_anno_file,
        weights_dir=weights_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=2,
        learning_rate=5e-5,  # Reduced from 1e-4 for more stable training
        score_thresh=0.05,
        topk=100,
    )
    
    print(f"\nTraining complete! Best epoch: {best_epoch}")
    print(f"Weights saved to: {weights_dir}")
    
    # Only finish wandb if we created the session
    if should_finish_wandb:
        wandb.finish()
    
    return best_epoch, best_metrics
