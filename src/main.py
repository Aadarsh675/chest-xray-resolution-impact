import os
import json
from datetime import datetime
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import DetrImageProcessor, DetrForObjectDetection

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =====================
# Speed / behavior knobs
# =====================
USE_AMP = True                      # Mixed precision (fast on GPUs)
USE_TORCH_COMPILE = False           # PyTorch 2+; can help, but adds a small compile warmup
EVAL_EVERY_N_EPOCHS = 1             # Run full validation every N epochs
EVAL_MAX_IMAGES = None              # e.g. 300 to sample a subset; None = all
TOPK = 50                           # Fewer predictions per image speeds COCOeval
SCORE_THRESH = 0.0                  # Keep 0, let COCOeval handle thresholds
NUM_EPOCHS = 10
BATCH_SIZE = 2
NUM_WORKERS = max(2, min(8, (os.cpu_count() or 4) // 2))  # tune for your VM
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

# =====================
# Paths
# =====================
DATA_DIR = "data"
ANNO_DIR = os.path.join(DATA_DIR, "annotations")
IMAGE_DIR = os.path.join(DATA_DIR, "images")

TRAIN_ANNO_FILE = os.path.join(ANNO_DIR, "train_annotations_coco.json")
TRAIN_IMG_DIR   = os.path.join(IMAGE_DIR, "train")

# NOTE: for proper splits, point VAL_* to a true val set.
VAL_ANNO_FILE   = os.path.join(ANNO_DIR, "test_annotations_coco.json")
VAL_IMG_DIR     = os.path.join(IMAGE_DIR, "test")

TEST_ANNO_FILE  = os.path.join(ANNO_DIR, "test_annotations_coco.json")
TEST_IMG_DIR    = os.path.join(IMAGE_DIR, "test")

WEIGHTS_DIR = "weights"
MODEL_NAME  = "facebook/detr-resnet-50"

# Confusion-matrix classes
DISEASE_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Pneumonia",
    "Infiltrate", "Pneumothorax", "Nodule", "Mass",
]

# =====================
# Dataset
# =====================
class SimpleCocoDataset(Dataset):
    def __init__(self, img_folder, anno_file, processor):
        self.coco = COCO(anno_file)
        self.img_folder = img_folder
        self.processor = processor
        self.ids = [img_id for img_id in self.coco.imgs.keys()
                    if len(self.coco.getAnnIds(imgIds=img_id)) > 0]
        # cache sizes to avoid re-opening images
        self.sizes = {img_id: (self.coco.imgs[img_id].get("width"),
                               self.coco.imgs[img_id].get("height"))
                      for img_id in self.ids}
        print(f"[Dataset] {os.path.basename(img_folder)}: {len(self.ids)} images with annotations")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_folder, img_info['file_name'])

        # Load once per batch item (only here, not during eval loops)
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        self.sizes[img_id] = (width, height)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            cx = (x + w/2) / width
            cy = (y + h/2) / height
            w_n = w / width
            h_n = h / height
            boxes.append([
                max(0, min(1, cx)),
                max(0, min(1, cy)),
                max(0, min(1, w_n)),
                max(0, min(1, h_n))
            ])
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

# =====================
# Utils
# =====================
def iou_xyxy(a, b):
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

# =====================
# Eval (COCO + extras)
# =====================
@torch.no_grad()
def evaluate_model(model, dataloader, coco_gt, device, num_classes, img_dir,
                   score_thresh=SCORE_THRESH, topk=TOPK, iou_match_thresh=0.1,
                   max_images=EVAL_MAX_IMAGES, use_amp=USE_AMP):
    model.eval()
    scaler_ctx = torch.cuda.amp.autocast if (use_amp and device.type == "cuda") else torch.cpu.amp.autocast
    results = []

    # extras
    y_true_dz, y_pred_dz = [], []
    iou_list, area_mape_list = []

    num_seen = 0
    for pixel_values, targets in dataloader:
        if max_images is not None and num_seen >= max_images:
            break

        pixel_values = pixel_values.to(device, memory_format=torch.channels_last if device.type=="cuda" else torch.contiguous_format)
        with scaler_ctx():
            outputs = model(pixel_values=pixel_values)

        B = pixel_values.shape[0]
        for i in range(B):
            if max_images is not None and num_seen >= max_images:
                break
            num_seen += 1

            target = targets[i]
            img_id = target["image_id"].item()
            img_info = coco_gt.loadImgs(img_id)[0]
            width = img_info.get("width")
            height = img_info.get("height")
            if width is None or height is None:
                # as a last resort (should not happen if your JSON includes sizes)
                img_path = os.path.join(img_dir, img_info['file_name'])
                wimg = Image.open(img_path).convert("RGB")
                width, height = wimg.size

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

            # disease % (image-level top-1)
            ann_ids = coco_gt.getAnnIds(imgIds=img_id)
            anns = coco_gt.loadAnns(ann_ids)
            if len(anns) > 0 and scores.numel() > 0:
                gt_cat = anns[0]["category_id"] - 1
                j = int(torch.argmax(scores).item())
                pred_cat = int(labels[j].item())
                y_true_dz.append(gt_cat)
                y_pred_dz.append(pred_cat)

            # bbox % diffs on matched pairs
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
                    iou_list.append(float(ious[k] * 100.0))
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
        "num_eval_images": int(num_seen)
    })
    return coco_metrics

# =====================
# Train / Val / Test
# =====================
def train_one_epoch(model, dataloader, optimizer, device, epoch_idx, use_amp=USE_AMP):
    model.train()
    total_loss = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type=="cuda")

    for batch_idx, (pixel_values, targets) in enumerate(dataloader):
        pixel_values = pixel_values.to(device, memory_format=torch.channels_last if device.type=="cuda" else torch.contiguous_format)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=use_amp and device.type=="cuda"):
            outputs = model(pixel_values=pixel_values, labels=targets)
            loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"[Train] Epoch {epoch_idx+1} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")

    return total_loss / max(1, len(dataloader))

def visualize_predictions(model, processor, coco_val, img_dir,
                          num_images=3, score_thresh=0.0, topk=8,
                          random_sample=True, use_grayscale=True):
    model.eval()
    device = next(model.parameters()).device
    cat_id_to_name = {c['id']: c['name'] for c in coco_val.dataset['categories']}

    img_ids = coco_val.getImgIds()
    if random_sample:
        import random
        random.shuffle(img_ids)
    img_ids = img_ids[:num_images]

    for img_id in img_ids:
        img_info = coco_val.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        enc = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**enc)

        target_sizes = torch.tensor([[H, W]], device=device)
        processed = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=score_thresh)[0]
        boxes_xyxy = processed["boxes"].cpu().numpy()
        scores = processed["scores"].cpu().numpy()
        labels = processed["labels"].cpu().numpy()

        if len(scores) > 0 and len(scores) > topk:
            idx = np.argsort(-scores)[:topk]
            boxes_xyxy, scores, labels = boxes_xyxy[idx], scores[idx], labels[idx]

        if len(boxes_xyxy) > 0:
            px = boxes_xyxy[:, 0]
            py = boxes_xyxy[:, 1]
            pw = np.maximum(1e-6, boxes_xyxy[:, 2] - boxes_xyxy[:, 0])
            ph = np.maximum(1e-6, boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
            boxes_xywh = np.stack([px, py, pw, ph], axis=1)
        else:
            boxes_xywh = boxes_xyxy

        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(image, cmap="gray" if use_grayscale else None)

        ann_ids = coco_val.getAnnIds(imgIds=img_id)
        anns = coco_val.loadAnns(ann_ids)
        for ann in anns:
            gx, gy, gw, gh = ann["bbox"]
            ax.add_patch(patches.Rectangle((gx, gy), gw, gh, linewidth=2,
                                           edgecolor="yellow", facecolor="none", zorder=3))
            gt_label = cat_id_to_name.get(ann["category_id"], str(ann["category_id"]))
            ax.text(gx, max(0, gy - 6), f"GT: {gt_label}",
                    color="yellow", fontsize=10, fontweight="bold",
                    backgroundcolor="black", zorder=4)

        for (bx, by, bw, bh), sc, lb in zip(boxes_xywh, scores, labels):
            ax.add_patch(patches.Rectangle((bx, by), bw, bh, linewidth=2,
                                           edgecolor="red", facecolor="none", zorder=3))
            pred_name = cat_id_to_name.get(int(lb) + 1, str(int(lb) + 1))
            ax.text(bx, min(H - 1, by + bh + 12), f"Pred: {pred_name} ({sc:.2f})",
                    color="red", fontsize=10, fontweight="bold",
                    backgroundcolor="black", zorder=4)

        ax.set_title(f"{img_info['file_name']}")
        ax.axis("off")
        plt.show()

def main():
    torch.backends.cudnn.benchmark = True
    print(f"cwd: {os.getcwd()}")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    with open(TRAIN_ANNO_FILE, "r") as f:
        train_coco_data = json.load(f)
    num_classes = len(train_coco_data["categories"])
    print(f"num_classes: {num_classes}")

    processor = DetrImageProcessor.from_pretrained(MODEL_NAME, size=800)
    model = DetrForObjectDetection.from_pretrained(
        MODEL_NAME, num_labels=num_classes, ignore_mismatched_sizes=True
    ).to(device)

    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if USE_TORCH_COMPILE and hasattr(torch, "compile") and device.type == "cuda":
        model = torch.compile(model)

    # Data
    train_dataset = SimpleCocoDataset(TRAIN_IMG_DIR, TRAIN_ANNO_FILE, processor)
    val_dataset   = SimpleCocoDataset(VAL_IMG_DIR,   VAL_ANNO_FILE,   processor)
    test_dataset  = SimpleCocoDataset(TEST_IMG_DIR,  TEST_ANNO_FILE,  processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=max(1, NUM_WORKERS//2),
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=max(1, NUM_WORKERS//2),
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    coco_val  = COCO(VAL_ANNO_FILE)
    coco_test = COCO(TEST_ANNO_FILE)

    best_map = -1.0
    best_epoch = -1
    best_metrics = None

    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, use_amp=USE_AMP)
        print(f"[Train] Avg loss: {avg_loss:.4f}")

        # Validate every N epochs (can also limit to subset for speed)
        if (epoch + 1) % EVAL_EVERY_N_EPOCHS == 0:
            val_metrics = evaluate_model(
                model, val_loader, coco_val, device, num_classes,
                img_dir=VAL_IMG_DIR, score_thresh=SCORE_THRESH, topk=TOPK,
                max_images=EVAL_MAX_IMAGES, use_amp=USE_AMP
            )
            print(
                f"[Val @ epoch {epoch+1}] "
                f"mAP={val_metrics['mAP']:.4f} | AP50={val_metrics['AP@50']:.4f} | AP75={val_metrics['AP@75']:.4f} | "
                f"DzAcc={val_metrics['disease_acc_pct']:.2f}% | IoUmean={val_metrics['bbox_iou_mean_pct']:.2f}% | "
                f"AreaMAPE={val_metrics['bbox_area_mape_pct']:.2f}% | "
                f"eval_imgs={val_metrics['num_eval_images']}"
            )

            # Save snapshot
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, f"epoch_{epoch+1}.pth"))

            # Track best by mAP
            if val_metrics["mAP"] > best_map:
                best_map = val_metrics["mAP"]
                best_epoch = epoch + 1
                best_metrics = val_metrics
                torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "best.pth"))
                with open(os.path.join(WEIGHTS_DIR, "best_metrics.json"), "w") as f:
                    json.dump({"epoch": best_epoch, "metrics": best_metrics}, f, indent=2)
                print(f"[Best] New best mAP={best_map:.4f} at epoch {best_epoch} — saved weights/best.pth")

    print("\nTraining complete!")
    if best_epoch <= 0:
        print("Warning: no best epoch recorded — check your validation.")
    else:
        print(f"Best epoch: {best_epoch} | Best val metrics: {best_metrics}")

    # ===== TEST with best weights =====
    print("\n=== TESTING with best weights ===")
    best_path = os.path.join(WEIGHTS_DIR, "best.pth")
    if not os.path.exists(best_path):
        # fallback: latest epoch
        epoch_ckpts = [f for f in os.listdir(WEIGHTS_DIR) if f.startswith("epoch_") and f.endswith(".pth")]
        if not epoch_ckpts:
            raise FileNotFoundError("No weights found for testing.")
        latest = max(epoch_ckpts, key=lambda x: int(x.split("_")[1].split(".")[0]))
        best_path = os.path.join(WEIGHTS_DIR, latest)
        print(f"[Test] best.pth not found; using latest epoch checkpoint: {best_path}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    test_metrics = evaluate_model(
        model, test_loader, coco_test, device, num_classes,
        img_dir=TEST_IMG_DIR, score_thresh=SCORE_THRESH, topk=TOPK,
        max_images=None, use_amp=USE_AMP
    )
    print(f"[TEST] {test_metrics}")

    # Optional: visual sanity checks on a few test images
    visualize_predictions(model, processor, coco_test, TEST_IMG_DIR,
                          num_images=3, score_thresh=0.0, topk=10,
                          random_sample=True, use_grayscale=True)

if __name__ == "__main__":
    main()
