# main.py
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
VAL_ANNO_FILE   = os.path.join(ANNO_DIR, "val_annotations_coco.json")     # preferred
TEST_ANNO_FILE  = os.path.join(ANNO_DIR, "test_annotations_coco.json")

TRAIN_IMG_DIR   = os.path.join(IMAGE_DIR, "train")
VAL_IMG_DIR     = os.path.join(IMAGE_DIR, "val")
TEST_IMG_DIR    = os.path.join(IMAGE_DIR, "test")

MODEL_NAME    = "facebook/detr-resnet-50"
BATCH_SIZE    = 2
NUM_WORKERS   = 0
NUM_EPOCHS    = 10
LEARNING_RATE = 1e-4

# COCO eval settings
TOPK = 100
SCORE_THRESH = 0.0
IOU_MATCH_THRESH = 0.1  # used for IoU/MAPE pair matching (not COCOeval)


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
            cx = (x + w/2) / width
            cy = (y + h/2) / height
            w_n = w / width
            h_n = h / height
            boxes.append([max(0,min(1,cx)), max(0,min(1,cy)), max(0,min(1,w_n)), max(0,min(1,h_n))])
            labels.append(ann['category_id'] - 1)  # 0-indexed

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
# Utilities for extra metrics
# -----------------------------
def iou_xyxy(a, b):
    """IoU for boxes in [x1,y1,x2,y2]. a: (4,), b: (N,4) -> (N,)"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def xywh_to_xyxy(box_xywh):
    x, y, w, h = box_xywh
    return np.array([x, y, x + w, y + h], dtype=np.float32)


# -----------------------------
# Evaluation (COCO mAP + extra % metrics)
# -----------------------------
@torch.no_grad()
def evaluate_model(model, dataloader, coco_gt, device, num_classes, img_dir,
                   score_thresh=SCORE_THRESH, topk=TOPK, iou_match_thresh=IOU_MATCH_THRESH):
    """
    Returns:
      - mAP, AP@50, AP@75 (COCO)
      - disease_acc_pct: top-1 disease label accuracy (image-level; GT=first ann)
      - bbox_iou_mean_pct: mean IoU (%) over matched GT–Pred pairs
      - bbox_area_mape_pct: mean absolute percentage error (%) of bbox area
      - matched_pairs: count of matched GT–Pred pairs
    """
    model.eval()
    results = []

    y_true_dz, y_pred_dz = [], []
    iou_list, area_mape_list = [], []

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
            keep = labels != num_classes
            scores = scores[keep]
            labels = labels[keep]
            boxes  = pred_boxes[keep]

            if scores.numel() > 0 and scores.numel() > topk:
                topk_idx = torch.topk(scores, k=topk).indices
                scores = scores[topk_idx]
                labels = labels[topk_idx]
                boxes  = boxes[topk_idx]

            # to pixel [x,y,w,h]
            if scores.numel() > 0:
                boxes = boxes.cpu().numpy()
                boxes = boxes * np.array([width, height, width, height], dtype=np.float32)
                boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
                boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
                boxes[:, 0] = np.clip(boxes[:, 0], 0, max(0, width - 1))
                boxes[:, 1] = np.clip(boxes[:, 1], 0, max(0, height - 1))
                boxes[:, 2] = np.clip(boxes[:, 2], 1e-6, width)
                boxes[:, 3] = np.clip(boxes[:, 3], 1e-6, height)

                for box, score, label in zip(boxes, scores.cpu().numpy(), labels.cpu().numpy()):
                    results.append({
                        "image_id": img_id,
                        "category_id": int(label) + 1,
                        "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        "score": float(score),
                    })

            # disease accuracy (image-level)
            ann_ids = coco_gt.getAnnIds(imgIds=img_id)
            anns = coco_gt.loadAnns(ann_ids)
            if len(anns) > 0:
                gt_cat = anns[0]["category_id"] - 1  # 0-index
                if scores.numel() > 0:
                    j = int(torch.argmax(scores).item())
                    pred_cat = int(labels[j].item())
                    y_true_dz.append(gt_cat)
                    y_pred_dz.append(pred_cat)

            # bbox pair metrics (IoU / MAPE)
            if scores.numel() > 0 and len(anns) > 0:
                pred_xyxy = np.stack([boxes[:,0], boxes[:,1], boxes[:,0]+boxes[:,2], boxes[:,1]+boxes[:,3]], axis=1)
                used_pred = set()
                for ann in anns:
                    gx, gy, gw, gh = ann["bbox"]
                    gt_xyxy = xywh_to_xyxy([gx, gy, gw, gh])
                    ious = iou_xyxy(gt_xyxy, pred_xyxy)
                    k = int(np.argmax(ious))
                    if ious[k] < iou_match_thresh or k in used_pred:
                        continue
                    used_pred.add(k)

                    # IoU (%)
                    iou_list.append(float(ious[k] * 100.0))

                    # Area MAPE (%)
                    gt_area = max(1e-6, gw * gh)
                    pred_area = max(1e-6, (pred_xyxy[k,2]-pred_xyxy[k,0]) * (pred_xyxy[k,3]-pred_xyxy[k,1]))
                    area_mape_list.append(float(abs(pred_area - gt_area) / gt_area * 100.0))

    # COCOeval
    if len(results) == 0:
        coco_metrics = {"mAP": 0.0, "AP@50": 0.0, "AP@75": 0.0}
    else:
        results_file = "val_predictions.json"
        with open(results_file, "w") as f:
            json.dump(results, f)
        coco_dt = coco_gt.loadRes(results_file)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_metrics = {
            "mAP":  float(coco_eval.stats[0]),
            "AP@50": float(coco_eval.stats[1]),
            "AP@75": float(coco_eval.stats[2]),
        }

    disease_acc = float(np.mean(np.array(y_true_dz) == np.array(y_pred_dz)) * 100.0) if len(y_true_dz) else 0.0
    bbox_iou_mean_pct = float(np.mean(iou_list)) if len(iou_list) else 0.0
    bbox_area_mape_pct = float(np.mean(area_mape_list)) if len(area_mape_list) else 0.0

    coco_metrics.update({
        "disease_acc_pct": disease_acc,
        "bbox_iou_mean_pct": bbox_iou_mean_pct,
        "bbox_area_mape_pct": bbox_area_mape_pct,
        "matched_pairs": int(len(iou_list)),
    })
    return coco_metrics


# -----------------------------
# Main training + validation each epoch + final testing
# -----------------------------
def main():
    # ----- W&B run name -----
    run_name = f"detr_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    wandb.init(
        project="detr-nih-chest-xray",
        name=run_name,
        config={
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "model": MODEL_NAME
        }
    )
    print(f"W&B run name: {run_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Resolve val/test fallbacks if needed ----
    val_anno, val_img = VAL_ANNO_FILE, VAL_IMG_DIR
    test_anno, test_img = TEST_ANNO_FILE, TEST_IMG_DIR

    if not os.path.exists(val_anno) or not os.path.exists(val_img):
        print("[Warning] No validation split found. Using TEST split as validation.")
        val_anno, val_img = TEST_ANNO_FILE, TEST_IMG_DIR

    if not os.path.exists(test_anno) or not os.path.exists(test_img):
        print("[Warning] No test split found. Using VALIDATION split as test.")
        test_anno, test_img = val_anno, val_img

    # ---- Classes ----
    with open(TRAIN_ANNO_FILE, "r") as f:
        coco_train_data = json.load(f)
    num_classes = len(coco_train_data["categories"])
    print(f"num_classes: {num_classes}")

    # ---- Processor & model ----
    processor = DetrImageProcessor.from_pretrained(MODEL_NAME, size=800)
    model = DetrForObjectDetection.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    ).to(device)

    # ---- Data ----
    train_dataset = SimpleCocoDataset(TRAIN_IMG_DIR, TRAIN_ANNO_FILE, processor)
    val_dataset   = SimpleCocoDataset(val_img,    val_anno,          processor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS)

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # ---- COCO GT ----
    coco_val = COCO(val_anno)

    # ---- Weights dir ----
    os.makedirs("weights", exist_ok=True)

    best_map = -1.0
    best_epoch = -1
    best_metrics = None

    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        # ---------- Train ----------
        model.train()
        total_loss = 0.0
        for pixel_values, targets in train_loader:
            pixel_values = pixel_values.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(pixel_values=pixel_values, labels=targets)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            wandb.log({"train/loss": loss.item(), "epoch": epoch + 1})

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"[Train] Epoch {epoch+1}/{NUM_EPOCHS} | avg loss: {avg_loss:.4f}")
        wandb.log({"train/epoch_avg_loss": avg_loss, "epoch": epoch + 1})

        # ---------- Validate ----------
        val_metrics = evaluate_model(
            model, val_loader, coco_val, device, num_classes,
            img_dir=val_img, score_thresh=SCORE_THRESH, topk=TOPK
        )
        print(f"[Val @ epoch {epoch+1}] {val_metrics}")
        wandb.log({f"val/{k}": v for k, v in val_metrics.items()} | {"epoch": epoch + 1})

        # Save epoch snapshot
        epoch_path = f"weights/epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), epoch_path)

        # Track best by mAP
        if val_metrics["mAP"] > best_map:
            best_map = val_metrics["mAP"]
            best_epoch = epoch + 1
            best_metrics = val_metrics
            torch.save(model.state_dict(), "weights/best.pth")
            with open("weights/best_metrics.json", "w") as f:
                json.dump({"epoch": best_epoch, "metrics": best_metrics, "run_name": run_name}, f, indent=2)
            print(f"[Best] New best mAP={best_map:.4f} at epoch {best_epoch}. Saved weights/best.pth")

    print("\nTraining complete!")
    if best_epoch > 0:
        print(f"Best epoch: {best_epoch} | Best val metrics: {best_metrics}")
    else:
        print("No best epoch recorded (mAP never improved).")

    # ---------- FINAL TEST (with best weights) ----------
    print("\n=== Final testing with best checkpoint ===")
    # Load best weights
    best_path = "weights/best.pth"
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    else:
        print("[Warning] best.pth not found, using last epoch weights.")

    # Build test loader (may be same as val if no dedicated test split)
    test_dataset = SimpleCocoDataset(test_img, test_anno, processor)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS)
    coco_test = COCO(test_anno)

    test_metrics = evaluate_model(
        model, test_loader, coco_test, device, num_classes,
        img_dir=test_img, score_thresh=SCORE_THRESH, topk=TOPK
    )
    print(f"[TEST] {test_metrics}")
    with open("weights/test_metrics.json", "w") as f:
        json.dump({"metrics": test_metrics, "run_name": run_name}, f, indent=2)
    wandb.log({f"test/{k}": v for k, v in test_metrics.items()})

    wandb.finish()


if __name__ == "__main__":
    main()
