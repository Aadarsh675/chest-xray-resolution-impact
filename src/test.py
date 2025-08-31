import os
import json
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from transformers import DetrImageProcessor, DetrForObjectDetection

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# -----------------------------
# Paths & constants
# -----------------------------
DATA_DIR = "data"
ANNO_DIR = os.path.join(DATA_DIR, "annotations")
IMAGE_DIR = os.path.join(DATA_DIR, "images")

VAL_ANNO_FILE = os.path.join(ANNO_DIR, "test_annotations_coco.json")
VAL_IMG_DIR = os.path.join(IMAGE_DIR, "test")

WEIGHTS_DIR = "weights"
MODEL_NAME = "facebook/detr-resnet-50"
BATCH_SIZE = 2
NUM_WORKERS = 0
TOPK = 100         # max predictions per image to keep for evaluation
SCORE_THRESH = 0.0 # keep 0.0 so COCOeval handles thresholds

# The 8 diseases requested for the disease confusion matrix
DISEASE_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Pneumonia",
    "Infiltrate",
    "Pneumothorax",
    "Nodule",
    "Mass",
]

# Location/size bins for bbox confusion matrices
LOC_BINS = ["TL","TC","TR","ML","MC","MR","BL","BC","BR"]  # 3x3 grid by center
SIZE_BINS = ["Small","Medium","Large"]  # by area fraction


# -----------------------------
# Dataset
# -----------------------------
class SimpleCocoDataset(Dataset):
    def __init__(self, img_folder, anno_file, processor):
        self.coco = COCO(anno_file)
        self.img_folder = img_folder
        self.processor = processor
        # Use only images that have at least one annotation
        self.ids = [img_id for img_id in self.coco.imgs.keys()
                    if len(self.coco.getAnnIds(imgIds=img_id)) > 0]
        print(f"Found {len(self.ids)} images with annotations")

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
            labels.append(ann['category_id'] - 1)  # make 0-indexed

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
# Utility: IoU, binning
# -----------------------------
def box_xywh_to_xyxy(box_xywh):
    x, y, w, h = box_xywh
    return np.array([x, y, x + w, y + h], dtype=np.float32)

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

def loc_bin_from_center(cx, cy, W, H):
    """Return one of the 3x3 bins based on center thirds."""
    x_bin = "L" if cx < W/3 else ("C" if cx < 2*W/3 else "R")
    y_bin = "T" if cy < H/3 else ("M" if cy < 2*H/3 else "B")
    return y_bin + x_bin  # e.g., "TL","MC",...

def size_bin_from_area(w, h, W, H):
    """Area fraction bins: S < 5%, M < 20%, else L."""
    frac = (w * h) / (W * H + 1e-9)
    if frac < 0.05:
        return "Small"
    elif frac < 0.20:
        return "Medium"
    else:
        return "Large"


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_model(model, dataloader, coco_val, device, num_classes, score_thresh=SCORE_THRESH, topk=TOPK):
    model.eval()
    results = []

    with torch.no_grad():
        for pixel_values, targets in dataloader:
            pixel_values = pixel_values.to(device)
            outputs = model(pixel_values=pixel_values)

            B = pixel_values.shape[0]
            for i in range(B):
                target = targets[i]
                img_id = target["image_id"].item()
                img_info = coco_val.loadImgs(img_id)[0]
                img_path = os.path.join(VAL_IMG_DIR, img_info['file_name'])

                width = img_info.get("width")
                height = img_info.get("height")
                if width is None or height is None:
                    try:
                        image = Image.open(img_path).convert("RGB")
                        width, height = image.size
                    except Exception as e:
                        print(f"Error loading image {img_path} for dimensions: {e}")
                        continue

                # outputs: logits (queries, num_classes+1) and pred_boxes (queries, 4) in cx,cy,w,h normalized
                logits = outputs.logits[i]
                pred_boxes = outputs.pred_boxes[i]

                probs = logits.softmax(-1)
                scores, labels = probs.max(-1)
                keep = labels != num_classes  # drop "no object" class
                scores = scores[keep]
                labels = labels[keep]
                boxes = pred_boxes[keep]

                # optional threshold (keep 0.0 for COCOeval)
                if score_thresh > 0.0:
                    m = scores > score_thresh
                    scores = scores[m]
                    labels = labels[m]
                    boxes = boxes[m]

                # top-k keep
                if scores.numel() > 0 and scores.numel() > topk:
                    topk_idx = torch.topk(scores, k=topk).indices
                    scores = scores[topk_idx]
                    labels = labels[topk_idx]
                    boxes = boxes[topk_idx]

                if scores.numel() == 0:
                    continue

                # scale to pixels & convert to [x,y,w,h]
                boxes = boxes.cpu().numpy()
                boxes = boxes * np.array([width, height, width, height], dtype=np.float32)
                boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0  # x
                boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0  # y

                # clip bounds & positive sizes
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

    results_file = "eval_predictions.json"
    with open(results_file, "w") as f:
        json.dump(results, f)

    if len(results) == 0:
        print("Warning: no predictions produced. "
              "Lower threshold or verify checkpoint/preprocessing.")
        return {"mAP": 0.0, "AP@50": 0.0, "AP@75": 0.0}

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


# -----------------------------
# Visualization
# -----------------------------
def visualize_predictions(
    model,
    processor,
    coco_val,
    img_dir,
    num_images=3,
    score_thresh=0.0,   # keep 0.0; we cap by topk
    topk=8,             # draw up to K red boxes per image
    random_sample=True,
    use_grayscale=True
):
    """
    GT boxes: yellow w/ 'GT: <label>' at TOP-LEFT.
    Pred boxes: red w/ 'Pred: <label> (score)' at BOTTOM-LEFT.
    Uses post-process first; if empty, falls back to top-K queries so red boxes show.
    """
    model.eval()
    device = next(model.parameters()).device
    cat_id_to_name = {c['id']: c['name'] for c in coco_val.dataset['categories']}

    img_ids = coco_val.getImgIds()
    if random_sample:
        random.shuffle(img_ids)
    img_ids = img_ids[:num_images]

    for img_id in img_ids:
        img_info = coco_val.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        # --- GT
        ann_ids = coco_val.getAnnIds(imgIds=img_id)
        anns = coco_val.loadAnns(ann_ids)

        # --- Forward
        enc = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**enc)

        # Path A: post-process (pixel-space)
        target_sizes = torch.tensor([[H, W]], device=device)
        processed = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=score_thresh
        )[0]
        boxes_xyxy = processed["boxes"].cpu().numpy()            # [x1,y1,x2,y2]
        scores = processed["scores"].cpu().numpy()
        labels = processed["labels"].cpu().numpy()               # 0-based

        if len(scores) > 0 and len(scores) > topk:
            idx = np.argsort(-scores)[:topk]
            boxes_xyxy, scores, labels = boxes_xyxy[idx], scores[idx], labels[idx]

        # Path B: fallback to top-K queries if nothing kept
        if len(scores) == 0:
            logits = outputs.logits[0].softmax(-1)               # (Q, C+1)
            pred_boxes = outputs.pred_boxes[0].cpu().numpy()     # (Q, 4) cx,cy,w,h (norm)
            cls_probs = logits[:, :-1].cpu().numpy()             # drop no-object
            best_scores = cls_probs.max(axis=1)
            best_labels = cls_probs.argmax(axis=1)
            order = np.argsort(-best_scores)[:topk]
            scores = best_scores[order]
            labels = best_labels[order]
            # to pixel xyxy
            boxes_cxcywh = pred_boxes[order] * np.array([W, H, W, H], dtype=np.float32)
            x = boxes_cxcywh[:,0] - boxes_cxcywh[:,2]/2.0
            y = boxes_cxcywh[:,1] - boxes_cxcywh[:,3]/2.0
            w = np.clip(boxes_cxcywh[:,2], 1e-6, W)
            h = np.clip(boxes_cxcywh[:,3], 1e-6, H)
            x2 = np.clip(x + w, 0, W - 1)
            y2 = np.clip(y + h, 0, H - 1)
            x1 = np.clip(x, 0, W - 1)
            y1 = np.clip(y, 0, H - 1)
            boxes_xyxy = np.stack([x1,y1,x2,y2], axis=1)

        print(f"[viz] img_id={img_id} preds={len(scores)}")

        # convert to xywh for drawing
        if len(boxes_xyxy) > 0:
            px = boxes_xyxy[:, 0]
            py = boxes_xyxy[:, 1]
            pw = np.maximum(1e-6, boxes_xyxy[:, 2] - boxes_xyxy[:, 0])
            ph = np.maximum(1e-6, boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
            boxes_xywh = np.stack([px, py, pw, ph], axis=1)
        else:
            boxes_xywh = boxes_xyxy

        # --- Plot
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(image, cmap="gray" if use_grayscale else None)

        # GT (yellow, label at TOP-LEFT)
        for ann in anns:
            gx, gy, gw, gh = ann["bbox"]
            ax.add_patch(patches.Rectangle((gx, gy), gw, gh, linewidth=2,
                                           edgecolor="yellow", facecolor="none", zorder=3))
            gt_label = cat_id_to_name.get(ann["category_id"], str(ann["category_id"]))
            ax.text(gx, max(0, gy - 6), f"GT: {gt_label}",
                    color="yellow", fontsize=10, fontweight="bold",
                    backgroundcolor="black", zorder=4)

        # Pred (red, label at BOTTOM-LEFT)
        for (bx, by, bw, bh), sc, lb in zip(boxes_xywh, scores, labels):
            ax.add_patch(patches.Rectangle((bx, by), bw, bh, linewidth=2,
                                           edgecolor="red", facecolor="none", zorder=3))
            pred_name = cat_id_to_name.get(int(lb) + 1, str(int(lb) + 1))  # map 0->category id 1
            ax.text(bx, min(H - 1, by + bh + 12), f"Pred: {pred_name} ({sc:.2f})",
                    color="red", fontsize=10, fontweight="bold",
                    backgroundcolor="black", zorder=4)

        ax.set_title(f"Image ID: {img_id} - {img_info['file_name']}")
        ax.axis("off")
        ax.plot([], [], color="yellow", linewidth=2, label="GT box")
        ax.plot([], [], color="red", linewidth=2, label="Pred box")
        ax.legend(loc="upper right")
        plt.show()


# -----------------------------
# Confusion matrices (disease, bbox location, bbox size)
# -----------------------------
def build_confusion_matrices(model, processor, coco_val, img_dir, max_images=None, iou_thresh=0.1):
    """
    Creates:
      1) Disease confusion matrix (8 classes). One GT label per image (first ann),
         vs top-score predicted label per image. Skips if either side not in the 8 classes.
      2) BBox Location confusion (9 bins). Matches each GT bbox to best-IoU pred (>= iou_thresh),
         compares location bins.
      3) BBox Size confusion (3 bins). Same matched pairs, compares size bins.
    """
    device = next(model.parameters()).device
    cat_id_to_name = {c['id']: c['name'] for c in coco_val.dataset['categories']}
    name_to_cat_id = {v: k for k, v in cat_id_to_name.items()}

    # disease indices (exactly these 8)
    disease_to_idx = {name: i for i, name in enumerate(DISEASE_CLASSES)}

    # accumulators
    y_true_dz = []
    y_pred_dz = []

    y_true_loc = []
    y_pred_loc = []

    y_true_sz = []
    y_pred_sz = []

    img_ids = coco_val.getImgIds()
    if max_images is not None:
        img_ids = img_ids[:max_images]

    for img_id in img_ids:
        img_info = coco_val.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        ann_ids = coco_val.getAnnIds(imgIds=img_id)
        anns = coco_val.loadAnns(ann_ids)
        if len(anns) == 0:
            continue

        # ---- forward
        enc = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**enc)

        # post-process preds at 0.0 so we can keep many, then cap later
        target_sizes = torch.tensor([[H, W]], device=device)
        processed = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.0)[0]
        pred_boxes_xyxy = processed["boxes"].cpu().numpy()   # [x1,y1,x2,y2]
        pred_scores = processed["scores"].cpu().numpy()
        pred_labels = processed["labels"].cpu().numpy()      # 0-based class idx

        # ===== 1) Disease confusion (image-level)
        # GT label: take the first annotation's category name
        gt_cat_id = anns[0]["category_id"]
        gt_name = cat_id_to_name.get(gt_cat_id)
        # Pred label: take the single best-score predicted name
        if len(pred_scores) > 0:
            j = int(np.argmax(pred_scores))
            pred_name = cat_id_to_name.get(int(pred_labels[j]) + 1)  # map 0->1-based
        else:
            pred_name = None

        # Only include if both are in the requested 8 classes
        if gt_name in disease_to_idx and pred_name in disease_to_idx:
            y_true_dz.append(disease_to_idx[gt_name])
            y_pred_dz.append(disease_to_idx[pred_name])

        # ===== 2&3) BBox confusion (matched pairs by IoU)
        if len(pred_boxes_xyxy) == 0:
            continue

        # compute pred xywh for binning later
        pw = np.maximum(1e-6, pred_boxes_xyxy[:,2] - pred_boxes_xyxy[:,0])
        ph = np.maximum(1e-6, pred_boxes_xyxy[:,3] - pred_boxes_xyxy[:,1])
        px = pred_boxes_xyxy[:,0]
        py = pred_boxes_xyxy[:,1]
        pred_xywh = np.stack([px, py, pw, ph], axis=1)

        used_pred = set()
        for ann in anns:
            gx, gy, gw, gh = ann["bbox"]
            gt_xyxy = box_xywh_to_xyxy([gx, gy, gw, gh])
            ious = iou_xyxy(gt_xyxy, pred_boxes_xyxy)
            j = int(np.argmax(ious))
            if ious[j] < iou_thresh or j in used_pred:
                continue  # no match for this GT
            used_pred.add(j)

            # bins
            g_cx, g_cy = gx + gw/2.0, gy + gh/2.0
            p_cx, p_cy = pred_xywh[j,0] + pred_xywh[j,2]/2.0, pred_xywh[j,1] + pred_xywh[j,3]/2.0
            gt_loc = loc_bin_from_center(g_cx, g_cy, W, H)
            pr_loc = loc_bin_from_center(p_cx, p_cy, W, H)
            gt_sz = size_bin_from_area(gw, gh, W, H)
            pr_sz = size_bin_from_area(pred_xywh[j,2], pred_xywh[j,3], W, H)

            if gt_loc in LOC_BINS and pr_loc in LOC_BINS:
                y_true_loc.append(LOC_BINS.index(gt_loc))
                y_pred_loc.append(LOC_BINS.index(pr_loc))
            if gt_sz in SIZE_BINS and pr_sz in SIZE_BINS:
                y_true_sz.append(SIZE_BINS.index(gt_sz))
                y_pred_sz.append(SIZE_BINS.index(pr_sz))

    # ----- Plot confusion matrices
    # Disease (8x8)
    if len(y_true_dz) > 0:
        cm_dz = confusion_matrix(y_true_dz, y_pred_dz, labels=list(range(len(DISEASE_CLASSES))))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_dz, display_labels=DISEASE_CLASSES)
        fig, ax = plt.subplots(figsize=(8, 7))
        disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
        ax.set_title("Confusion Matrix — Disease (Image-level)")
        plt.tight_layout()
        plt.show()
    else:
        print("Disease confusion matrix: not enough matched samples within the specified 8 classes.")

    # Location (9x9)
    if len(y_true_loc) > 0:
        cm_loc = confusion_matrix(y_true_loc, y_pred_loc, labels=list(range(len(LOC_BINS))))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_loc, display_labels=LOC_BINS)
        fig, ax = plt.subplots(figsize=(7, 7))
        disp.plot(ax=ax, xticks_rotation=45, cmap="Oranges", colorbar=False)
        ax.set_title("Confusion Matrix — BBox Location (3×3 grid)")
        plt.tight_layout()
        plt.show()
    else:
        print("BBox location confusion matrix: no matched GT–prediction pairs (IoU threshold too high?).")

    # Size (3x3)
    if len(y_true_sz) > 0:
        cm_sz = confusion_matrix(y_true_sz, y_pred_sz, labels=list(range(len(SIZE_BINS))))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_sz, display_labels=SIZE_BINS)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, xticks_rotation=0, cmap="Greens", colorbar=False)
        ax.set_title("Confusion Matrix — BBox Size (area fraction)")
        plt.tight_layout()
        plt.show()
    else:
        print("BBox size confusion matrix: no matched GT–prediction pairs.")


# -----------------------------
# Main
# -----------------------------
def main():
    print(f"Current working directory: {os.getcwd()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # pick latest weights
    if not os.path.exists(WEIGHTS_DIR):
        raise FileNotFoundError(f"Weights directory {WEIGHTS_DIR} not found.")
    weight_files = [f for f in os.listdir(WEIGHTS_DIR) if f.startswith("epoch_") and f.endswith(".pth")]
    if not weight_files:
        raise FileNotFoundError(f"No weight files found in {WEIGHTS_DIR}")
    latest_weight = max(weight_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    weights_path = os.path.join(WEIGHTS_DIR, latest_weight)
    print(f"Using latest weights: {weights_path}")

    # load anno to get num classes
    with open(VAL_ANNO_FILE, 'r') as f:
        coco_data = json.load(f)
    num_classes = len(coco_data['categories'])
    print(f"Number of classes: {num_classes}")

    # init processor & model
    print("Initializing processor and model...")
    processor = DetrImageProcessor.from_pretrained(MODEL_NAME, size=800)
    model = DetrForObjectDetection.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    ).to(device)
    # load finetuned weights
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # dataset & loader
    print("\nLoading test dataset...")
    val_dataset = SimpleCocoDataset(VAL_IMG_DIR, VAL_ANNO_FILE, processor)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS
    )

    # evaluate
    print("\nEvaluating model on test set...")
    coco_val = COCO(VAL_ANNO_FILE)
    eval_metrics = evaluate_model(model, val_loader, coco_val, device, num_classes,
                                  score_thresh=SCORE_THRESH, topk=TOPK)
    print(f"Evaluation metrics: {eval_metrics}")
    print("Evaluation complete!")

    # -------- Confusion matrices (BEFORE visual examples) --------
    print("\nBuilding confusion matrices...")
    build_confusion_matrices(model, processor, coco_val, VAL_IMG_DIR, max_images=None, iou_thresh=0.1)

    # -------- Visualize a few predictions vs GT --------
    print("\nVisualizing predictions vs ground truth...")
    visualize_predictions(
        model, processor, coco_val, VAL_IMG_DIR,
        num_images=3, score_thresh=0.0, topk=10, random_sample=True, use_grayscale=True
    )


if __name__ == "__main__":
    main()
