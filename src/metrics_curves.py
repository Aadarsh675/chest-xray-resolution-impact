# metrics_curves.py
import os, json, csv
from typing import List, Dict
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import wandb

from pycocotools.coco import COCO

# ---- Minimal dataset/collate (same structure as in train/test) ----
class SimpleCocoDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, anno_file, processor):
        self.coco = COCO(anno_file)
        self.img_folder = img_folder
        self.processor = processor
        self.ids = [i for i in self.coco.imgs.keys()
                    if len(self.coco.getAnnIds(imgIds=i)) > 0]

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_folder, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        enc = self.processor(images=image, return_tensors="pt")
        return enc["pixel_values"].squeeze(0), img_id, image.size  # (W,H)

def collate_fn(batch):
    pix = torch.stack([b[0] for b in batch])
    img_ids = [b[1] for b in batch]
    sizes = [b[2] for b in batch]
    return pix, img_ids, sizes

# ---- Utilities ----
def _xywh_to_xyxy(box_xywh):
    x, y, w, h = box_xywh
    return np.array([x, y, x + w, y + h], dtype=np.float32)

def _iou_xyxy(a, b):  # a: (4,), b: (N,4)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]
    inter_x1 = np.maximum(ax1, bx1); inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2); inter_y2 = np.minimum(ay2, by2)
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    area_b = np.maximum(0.0, bx2-bx1) * np.maximum(0.0, by2-by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def _match_per_image(pred, gts, iou_thr=0.5):
    """
    Greedy match predictions to GT by IoU. 
    pred: dict with 'boxes' (N,4 xyxy), 'labels'(N,), 'scores'(N,)
    gts: list of {'bbox':[x,y,w,h], 'category_id':int}
    Returns list of (pred_idx, gt_idx, iou) for matched pairs and arrays of unmatched idxs.
    """
    if len(pred["boxes"]) == 0 or len(gts) == 0:
        return [], np.arange(len(pred["boxes"])), np.arange(len(gts))
    P = pred["boxes"].shape[0]
    G = len(gts)
    gt_xyxy = np.stack([_xywh_to_xyxy(g["bbox"]) for g in gts], axis=0)
    used_pred, used_gt = set(), set()
    matches = []

    # Build IoU matrix
    ious = np.zeros((P, G), dtype=np.float32)
    for p in range(P):
        ious[p] = _iou_xyxy(pred["boxes"][p], gt_xyxy)

    # Greedy by best IoU globally
    flat = np.dstack(np.unravel_index(np.argsort(-ious, axis=None), (P, G)))[0]
    for p, g in flat:
        if p in used_pred or g in used_gt: 
            continue
        if ious[p, g] >= iou_thr:
            used_pred.add(p); used_gt.add(g)
            matches.append((p, g, ious[p, g]))
    unmatched_p = np.array([p for p in range(P) if p not in used_pred])
    unmatched_g = np.array([g for g in range(G) if g not in used_gt])
    return matches, unmatched_p, unmatched_g

# ---- Core: export curves vs confidence threshold ----
@torch.no_grad()
def export_threshold_curves(
    model,
    processor,
    device,
    test_img_dir: str,
    test_anno_file: str,
    out_dir: str,
    thresholds: List[float] = None,
    iou_match_thresh: float = 0.5,
    batch_size: int = 2,
    num_workers: int = 0,
) -> Dict:
    """
    For a grid of score thresholds, compute per-class Precision, Recall, and mIoU.
    Saves JSON and CSV (wide) in out_dir. Returns the metrics dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    if thresholds is None:
        thresholds = list(np.round(np.linspace(0.0, 0.9, 10), 3)) + [0.95, 0.99]

    coco = COCO(test_anno_file)
    cat_id_to_name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    cat_ids_sorted = sorted(cat_id_to_name.keys())
    cat_names_sorted = [cat_id_to_name[cid] for cid in cat_ids_sorted]

    ds = SimpleCocoDataset(test_img_dir, test_anno_file, processor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    # Per-threshold accumulators (per class): TP, FP, FN and IoUs (for matched pairs)
    per_thr = {
        t: {
            cname: {"TP": 0, "FP": 0, "FN": 0, "ious": []}
            for cname in cat_names_sorted
        } for t in thresholds
    }

    model.eval()
    for pix, img_ids, sizes in dl:
        pix = pix.to(device)
        outputs = model(pixel_values=pix)
        B = pix.shape[0]

        for i in range(B):
            W, H = sizes[i]
            target_sizes = torch.tensor([[H, W]], device=device)
            # Pull raw logits/boxes to build thresholded predictions on the fly
            logits = outputs.logits[i].softmax(-1)[:, :-1].cpu().numpy()   # (Q, K)
            boxes_cxcywh = outputs.pred_boxes[i].cpu().numpy() * np.array([W, H, W, H], dtype=np.float32)

            # Compose candidate predictions (all queries)
            scores_all = logits.max(axis=1)
            labels_all = logits.argmax(axis=1)

            # Convert to xyxy
            x = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2.0
            y = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2.0
            w = np.clip(boxes_cxcywh[:, 2], 1e-6, W)
            h = np.clip(boxes_cxcywh[:, 3], 1e-6, H)
            x1 = np.clip(x, 0, W-1); y1 = np.clip(y, 0, H-1)
            x2 = np.clip(x + w, 0, W-1); y2 = np.clip(y + h, 0, H-1)
            boxes_xyxy_all = np.stack([x1, y1, x2, y2], axis=1)

            # Ground truth for this image
            ann_ids = coco.getAnnIds(imgIds=img_ids[i])
            anns = coco.loadAnns(ann_ids)

            # Organize GT by class
            gt_by_cls = {}
            for a in anns:
                cname = cat_id_to_name[a["category_id"]]
                gt_by_cls.setdefault(cname, []).append(a)

            for t in thresholds:
                # Threshold predictions
                keep = scores_all >= t
                if keep.sum() == 0:
                    # All GT become FN
                    for cname, gtlist in gt_by_cls.items():
                        per_thr[t][cname]["FN"] += len(gtlist)
                    continue

                pred_boxes = boxes_xyxy_all[keep]
                pred_labels = labels_all[keep]
                pred_scores = scores_all[keep]

                # Group predictions per class
                pred_by_cls = {}
                for pb, pl, ps in zip(pred_boxes, pred_labels, pred_scores):
                    cname = cat_names_sorted[pl] if pl < len(cat_names_sorted) else str(pl)
                    pred_by_cls.setdefault(cname, {"boxes": [], "scores": []})
                    pred_by_cls[cname]["boxes"].append(pb)
                    pred_by_cls[cname]["scores"].append(ps)
                for cname in pred_by_cls.keys():
                    pred_by_cls[cname]["boxes"] = np.array(pred_by_cls[cname]["boxes"], dtype=np.float32)
                    pred_by_cls[cname]["scores"] = np.array(pred_by_cls[cname]["scores"], dtype=np.float32)

                # For each class, greedy match and accumulate
                for cname in cat_names_sorted:
                    preds = pred_by_cls.get(cname, {"boxes": np.zeros((0,4), np.float32), "scores": np.zeros((0,), np.float32)})
                    gts   = gt_by_cls.get(cname, [])
                    matches, unmatched_p, unmatched_g = _match_per_image(
                        {"boxes": preds["boxes"], "labels": None, "scores": preds["scores"]},
                        gts, iou_thr=iou_match_thresh
                    )
                    per_thr[t][cname]["TP"] += len(matches)
                    per_thr[t][cname]["FP"] += len(unmatched_p)
                    per_thr[t][cname]["FN"] += len(unmatched_g)
                    # mIoU: average IoU of matched pairs
                    for _, _, iou in matches:
                        per_thr[t][cname]["ious"].append(float(iou))

    # Reduce to precision/recall/miou
    result = {
        "thresholds": thresholds,
        "per_class": {
            cname: {
                "precision": [],
                "recall": [],
                "miou": [],
            } for cname in cat_names_sorted
        }
    }
    for t in thresholds:
        for cname in cat_names_sorted:
            TP = per_thr[t][cname]["TP"]
            FP = per_thr[t][cname]["FP"]
            FN = per_thr[t][cname]["FN"]
            ious = per_thr[t][cname]["ious"]
            prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            miou = float(np.mean(ious)) if len(ious) else 0.0
            result["per_class"][cname]["precision"].append(prec)
            result["per_class"][cname]["recall"].append(rec)
            result["per_class"][cname]["miou"].append(miou)

    # Save JSON
    with open(os.path.join(out_dir, "curves.json"), "w") as f:
        json.dump(result, f, indent=2)

    # Save CSV (wide format): threshold, class, precision, recall, miou
    csv_path = os.path.join(out_dir, "curves.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "class", "precision", "recall", "miou"])
        for t_idx, t in enumerate(thresholds):
            for cname in cat_names_sorted:
                writer.writerow([
                    t, cname,
                    result["per_class"][cname]["precision"][t_idx],
                    result["per_class"][cname]["recall"][t_idx],
                    result["per_class"][cname]["miou"][t_idx],
                ])

    # Create and save visualization plots
    # 1. Precision vs Threshold for all classes
    fig, ax = plt.subplots(figsize=(10, 6))
    for cname in cat_names_sorted:
        ax.plot(thresholds, result["per_class"][cname]["precision"], label=cname, marker='o')
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Confidence Threshold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    precision_path = os.path.join(out_dir, "precision_vs_threshold.png")
    plt.savefig(precision_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {precision_path}")
    if wandb.run is not None:
        wandb.log({"threshold_curves/precision": wandb.Image(precision_path)})
    plt.close()

    # 2. Recall vs Threshold for all classes
    fig, ax = plt.subplots(figsize=(10, 6))
    for cname in cat_names_sorted:
        ax.plot(thresholds, result["per_class"][cname]["recall"], label=cname, marker='o')
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Recall')
    ax.set_title('Recall vs Confidence Threshold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    recall_path = os.path.join(out_dir, "recall_vs_threshold.png")
    plt.savefig(recall_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {recall_path}")
    if wandb.run is not None:
        wandb.log({"threshold_curves/recall": wandb.Image(recall_path)})
    plt.close()

    # 3. mIoU vs Threshold for all classes
    fig, ax = plt.subplots(figsize=(10, 6))
    for cname in cat_names_sorted:
        ax.plot(thresholds, result["per_class"][cname]["miou"], label=cname, marker='o')
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Mean IoU')
    ax.set_title('Mean IoU vs Confidence Threshold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    miou_path = os.path.join(out_dir, "miou_vs_threshold.png")
    plt.savefig(miou_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {miou_path}")
    if wandb.run is not None:
        wandb.log({"threshold_curves/miou": wandb.Image(miou_path)})
    plt.close()

    return result
