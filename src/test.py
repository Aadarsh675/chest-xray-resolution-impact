# test.py
import os
import json
from PIL import Image
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import DetrImageProcessor, DetrForObjectDetection

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import wandb


# -----------------------------
# Utilities
# -----------------------------
def iou_xyxy(a, b):
    """IoU for boxes in [x1,y1,x2,y2]. a: (4,), b: (N,4) -> (N,)"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
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
# Dataset (keeps empty images)
# -----------------------------
class SimpleCocoDataset(Dataset):
    def __init__(self, img_folder, anno_file, processor):
        self.coco = COCO(anno_file)
        self.img_folder = img_folder
        self.processor = processor
        self.ids = list(self.coco.imgs.keys())  # keep all images
        print(f"[Dataset] {os.path.basename(img_folder)}: {len(self.ids)} images (incl. empty)")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_folder, img_info['file_name'])

        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        # Fill missing dims just in case
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
            boxes.append([max(0, min(1, cx)),
                          max(0, min(1, cy)),
                          max(0, min(1, w_n)),
                          max(0, min(1, h_n))])
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
    # Images may have different sizes, so we need to pad them
    # Get max dimensions in batch
    max_height = max([item[0].shape[1] for item in batch])
    max_width = max([item[0].shape[2] for item in batch])
    
    # Pad all images to max dimensions
    padded_images = []
    for item in batch:
        img = item[0]  # Shape: [3, H, W]
        h, w = img.shape[1], img.shape[2]
        
        # Create padded tensor
        padded = torch.zeros(3, max_height, max_width, dtype=img.dtype)
        padded[:, :h, :w] = img
        padded_images.append(padded)
    
    pixel_values = torch.stack(padded_images)
    targets = [item[1] for item in batch]
    return pixel_values, targets


# -----------------------------
# Evaluation (COCO + extras + per-class AP)
# -----------------------------
@torch.no_grad()
def evaluate_model(model, dataloader, coco_gt, device, num_classes, img_dir,
                   score_thresh=0.05, topk=100, iou_match_thresh=0.1):
    model.eval()
    results = []

    y_true_dz, y_pred_dz = [], []
    iou_list = []
    area_mape_list = []

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

            logits = outputs.logits[i]
            pred_boxes = outputs.pred_boxes[i]

            probs = logits.softmax(-1)[..., :-1]  # exclude no-object
            scores, labels = probs.max(-1)
            boxes = pred_boxes

            # threshold then top-k
            if scores.numel() > 0:
                keep = scores > score_thresh
                if keep.any():
                    scores = scores[keep]
                    labels = labels[keep]
                    boxes  = boxes[keep]
                else:
                    scores = scores[:0]; labels = labels[:0]; boxes = boxes[:0]

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
                        "category_id": int(label) + 1,  # back to COCO id space
                        "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        "score": float(score),
                    })

            # image-level proxy (skip empty-annotation images)
            ann_ids = coco_gt.getAnnIds(imgIds=img_id)
            anns = coco_gt.loadAnns(ann_ids)
            if len(anns) > 0 and scores.numel() > 0:
                gt_cat = anns[0]["category_id"] - 1
                j = int(torch.argmax(scores).item())
                pred_cat = int(labels[j].item())
                y_true_dz.append(gt_cat)
                y_pred_dz.append(pred_cat)

            # localization diffs on matched pairs
            if scores.numel() > 0 and len(anns) > 0:
                pred_xyxy = np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]], axis=1)
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
                    pred_area = max(1e-6, (pred_xyxy[k, 2] - pred_xyxy[k, 0]) * (pred_xyxy[k, 3] - pred_xyxy[k, 1]))
                    area_mape_list.append(float(abs(pred_area - gt_area) / gt_area * 100.0))

    per_class = {}
    if len(results) == 0:
        coco_metrics = {"mAP": 0.0, "AP@50": 0.0, "AP@75": 0.0}
    else:
        results_file = "predictions_test.json"
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

        precisions = coco_eval.eval['precision']  # [T, R, K, A, M]
        recalls = coco_eval.eval['recall']  # [T, K, A, M]
        iou_thrs = coco_eval.params.iouThrs
        cats = coco_gt.loadCats(coco_gt.getCatIds())
        
        for k, cat in enumerate(cats):
            # AP metrics
            p = precisions[:, :, k, 0, -1]
            p = p[p > -1]
            ap = float(np.mean(p)) if p.size else float('nan')

            t_50 = np.where(np.isclose(iou_thrs, 0.5))[0]
            if t_50.size:
                p50 = precisions[t_50[0], :, k, 0, -1]
                p50 = p50[p50 > -1]
                ap50 = float(np.mean(p50)) if p50.size else float('nan')
            else:
                ap50 = float('nan')
            
            # Precision and Recall at IoU=0.5
            t_50_idx = np.where(np.isclose(iou_thrs, 0.5))[0]
            if t_50_idx.size:
                precision_50 = precisions[t_50_idx[0], :, k, 0, -1]
                precision_50 = precision_50[precision_50 > -1]
                precision_50 = float(np.mean(precision_50)) if precision_50.size else 0.0
                
                recall_50 = recalls[t_50_idx[0], k, 0, -1]
                recall_50 = recall_50[recall_50 > -1]
                recall_50 = float(np.mean(recall_50)) if recall_50.size else 0.0
            else:
                precision_50 = 0.0
                recall_50 = 0.0
            
            # F1 score
            f1_50 = 2 * (precision_50 * recall_50) / (precision_50 + recall_50 + 1e-9)
            
            # mIoU calculation (average IoU across all IoU thresholds)
            ious = []
            for t_idx in range(len(iou_thrs)):
                p_t = precisions[t_idx, :, k, 0, -1]
                p_t = p_t[p_t > -1]
                if p_t.size > 0:
                    ious.append(float(np.mean(p_t)))
            miou = float(np.mean(ious)) if ious else 0.0

            per_class[cat['name']] = {
                "AP": ap, 
                "AP50": ap50,
                "precision": precision_50,
                "recall": recall_50,
                "f1": f1_50,
                "miou": miou
            }

    disease_acc = float(np.mean(np.array(y_true_dz) == np.array(y_pred_dz)) * 100.0) if len(y_true_dz) else 0.0
    bbox_iou_mean_pct = float(np.mean(iou_list)) if len(iou_list) else 0.0
    bbox_area_mape_pct = float(np.mean(area_mape_list)) if len(area_mape_list) else 0.0

    coco_metrics.update({
        "disease_acc_pct": disease_acc,
        "bbox_iou_mean_pct": bbox_iou_mean_pct,
        "bbox_area_mape_pct": bbox_area_mape_pct,
        "matched_pairs": int(len(iou_list)),
        "per_class": per_class,
    })
    return coco_metrics


# -----------------------------
# Confusion matrix + bbox diffs
# -----------------------------
def disease_confusion_and_bbox_diffs(model, processor, coco_val, img_dir, iou_thresh=0.1, max_images=None, class_names=None):
    device = next(model.parameters()).device
    cat_id_to_name = {c['id']: c['name'] for c in coco_val.dataset['categories']}

    if class_names is None:
        class_names = [cat_id_to_name[cid] for cid in sorted(cat_id_to_name.keys())]

    disease_to_idx = {name: i for i, name in enumerate(class_names)}
    y_true_dz, y_pred_dz = [], []

    loc_dx_abs_pix, loc_dy_abs_pix = [], []
    loc_dx_abs_norm, loc_dy_abs_norm = [], []
    size_dw_abs_pix, size_dh_abs_pix = [], []
    size_dw_abs_norm, size_dh_abs_norm = [], []

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
            continue  # skip images with no GT for this visualization

        enc = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**enc)

        target_sizes = torch.tensor([[H, W]], device=device)
        processed = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.0)[0]
        pred_boxes_xyxy = processed["boxes"].cpu().numpy()
        pred_scores = processed["scores"].cpu().numpy()
        pred_labels = processed["labels"].cpu().numpy()

        gt_name = cat_id_to_name.get(anns[0]["category_id"])
        if len(pred_scores) > 0:
            j = int(np.argmax(pred_scores))
            pred_name = cat_id_to_name.get(int(pred_labels[j]) + 1)
        else:
            pred_name = None

        if gt_name in disease_to_idx and pred_name in disease_to_idx:
            y_true_dz.append(disease_to_idx[gt_name])
            y_pred_dz.append(disease_to_idx[pred_name])

        if len(pred_boxes_xyxy) == 0:
            continue

        px1, py1, px2, py2 = pred_boxes_xyxy[:, 0], pred_boxes_xyxy[:, 1], pred_boxes_xyxy[:, 2], pred_boxes_xyxy[:, 3]
        pw = np.maximum(1e-6, px2 - px1)
        ph = np.maximum(1e-6, py2 - py1)
        pred_xywh = np.stack([px1, py1, pw, ph], axis=1)

        used_pred = set()
        for ann in anns:
            gx, gy, gw, gh = ann["bbox"]
            gt_xyxy = xywh_to_xyxy([gx, gy, gw, gh])
            ious = iou_xyxy(gt_xyxy, pred_boxes_xyxy)
            j = int(np.argmax(ious))
            if ious[j] < iou_thresh or j in used_pred:
                continue
            used_pred.add(j)

            g_cx, g_cy = gx + gw / 2.0, gy + gh / 2.0
            p_cx, p_cy = pred_xywh[j, 0] + pred_xywh[j, 2] / 2.0, pred_xywh[j, 1] + pred_xywh[j, 3] / 2.0

            dx_pix = abs(g_cx - p_cx); dy_pix = abs(g_cy - p_cy)
            dx_norm = dx_pix / (W + 1e-9); dy_norm = dy_pix / (H + 1e-9)

            loc_dx_abs_pix.append(dx_pix); loc_dy_abs_pix.append(dy_pix)
            loc_dx_abs_norm.append(dx_norm); loc_dy_abs_norm.append(dy_norm)

            dw_pix = abs(gw - pred_xywh[j, 2]); dh_pix = abs(gh - pred_xywh[j, 3])
            dw_norm = dw_pix / (W + 1e-9);     dh_norm = dh_pix / (H + 1e-9)

            size_dw_abs_pix.append(dw_pix); size_dh_abs_pix.append(dh_pix)
            size_dw_abs_norm.append(dw_norm); size_dh_abs_norm.append(dh_norm)

    if len(y_true_dz) > 0:
        cm_dz = confusion_matrix(y_true_dz, y_pred_dz, labels=list(range(len(class_names))))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_dz, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(class_names)), 7))
        disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
        ax.set_title("Confusion Matrix — Image-level (GT first ann vs top pred)")
        plt.tight_layout()
        
        # Save plot instead of showing
        plots_dir = os.path.join("data", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_filename = "confusion_matrix.png"
        plot_path = os.path.join(plots_dir, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {plot_path}")
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({"confusion_matrix": wandb.Image(plot_path)})
        
        plt.close()  # Close the figure to free memory
    else:
        print("Disease confusion matrix: not enough samples in the current class set.")

    def safe_mean(arr): return float(np.mean(arr)) if len(arr) > 0 else float("nan")
    print("\nAverage bounding-box differences over matched GT–prediction pairs:")
    print(f"  Location | mean |dx| (pixels): {safe_mean(loc_dx_abs_pix):.2f}   |dy| (pixels): {safe_mean(loc_dy_abs_pix):.2f}")
    print(f"           | mean |dx| (norm W): {safe_mean(loc_dx_abs_norm)::.4f}  |dy| (norm H): {safe_mean(loc_dy_abs_norm)::.4f}")
    print(f"  Size     | mean |dw| (pixels): {safe_mean(size_dw_abs_pix):.2f}  |dh| (pixels): {safe_mean(size_dh_abs_pix):.2f}")
    print(f"           | mean |dw| (norm W): {safe_mean(size_dw_abs_norm)::.4f} |dh| (norm H): {safe_mean(size_dh_abs_norm)::.4f}")


# -----------------------------
# Visualization
# -----------------------------
def visualize_predictions(model, processor, coco_val, img_dir,
                          num_images=3, score_thresh=0.05, topk=8,
                          random_sample=True, use_grayscale=True):
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

        ann_ids = coco_val.getAnnIds(imgIds=img_id)
        anns = coco_val.loadAnns(ann_ids)

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

        # GT (yellow)
        for ann in anns:
            gx, gy, gw, gh = ann["bbox"]
            ax.add_patch(patches.Rectangle((gx, gy), gw, gh, linewidth=2,
                                           edgecolor="yellow", facecolor="none", zorder=3))
            gt_label = cat_id_to_name.get(ann["category_id"], str(ann["category_id"]))
            ax.text(gx, max(0, gy - 6), f"GT: {gt_label}",
                    color="yellow", fontsize=10, fontweight="bold",
                    backgroundcolor="black", zorder=4)

        # Pred (red)
        for (bx, by, bw, bh), sc, lb in zip(boxes_xywh, scores, labels):
            ax.add_patch(patches.Rectangle((bx, by), bw, bh, linewidth=2,
                                           edgecolor="red", facecolor="none", zorder=3))
            pred_name = cat_id_to_name.get(int(lb) + 1, str(int(lb) + 1))
            ax.text(bx, min(H - 1, by + bh + 12), f"Pred: {pred_name} ({sc:.2f})",
                    color="red", fontsize=10, fontweight="bold",
                    backgroundcolor="black", zorder=4)

        ax.set_title(f"Image ID: {img_id} - {img_info['file_name']}")
        ax.axis("off")
        ax.plot([], [], color="yellow", linewidth=2, label="GT box")
        ax.plot([], [], color="red", linewidth=2, label="Pred box")
        ax.legend(loc="upper right")
        
        # Save plot instead of showing
        plots_dir = os.path.join("data", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_filename = f"prediction_visualization_{img_id}.png"
        plot_path = os.path.join(plots_dir, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {plot_path}")
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({f"prediction_visualization/img_{img_id}": wandb.Image(plot_path)})
        
        plt.close()  # Close the figure to free memory


# -----------------------------
# Convenience: full test run
# -----------------------------
def run_test(model, processor, device, num_classes,
             test_img_dir, test_anno_file, weights_dir,
             score_thresh=0.05, topk=100, do_confusion=True, do_viz=True,
             class_names=None):

    best_path = os.path.join(weights_dir, "best.pth")
    if not os.path.exists(best_path):
        ckpts = [f for f in os.listdir(weights_dir) if f.startswith("epoch_") and f.endswith(".pth")]
        if not ckpts:
            raise FileNotFoundError("No weights found for testing.")
        latest = max(ckpts, key=lambda x: int(x.split("_")[1].split(".")[0]))
        best_path = os.path.join(weights_dir, latest)
        print(f"[Test] best.pth not found; using latest epoch checkpoint: {best_path}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    test_dataset = SimpleCocoDataset(test_img_dir, test_anno_file, processor)
    test_loader  = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    coco_test = COCO(test_anno_file)

    test_metrics = evaluate_model(
        model, test_loader, coco_test, device, num_classes,
        img_dir=test_img_dir, score_thresh=score_thresh, topk=topk
    )
    print(f"[TEST] metrics: {test_metrics}")

    if wandb.run is not None:
        # Log main test metrics
        main_metrics = {
            "test/mAP": test_metrics.get("mAP", 0.0),
            "test/AP@50": test_metrics.get("AP@50", 0.0), 
            "test/AP@75": test_metrics.get("AP@75", 0.0),
            "test/Disease_Accuracy": test_metrics.get("disease_acc_pct", 0.0),
            "test/BBox_IoU_Mean": test_metrics.get("bbox_iou_mean_pct", 0.0),
            "test/BBox_Area_MAPE": test_metrics.get("bbox_area_mape_pct", 0.0),
            "test/Matched_Pairs": test_metrics.get("matched_pairs", 0),
        }
        wandb.log(main_metrics)
        
        # Log per-class metrics with comprehensive breakdown
        if test_metrics.get("per_class"):
            per_cls_log = {}
            for cls_name, cls_metrics in test_metrics["per_class"].items():
                # AP metrics
                per_cls_log[f"test/AP/{cls_name}"] = cls_metrics["AP"]
                per_cls_log[f"test/AP50/{cls_name}"] = cls_metrics["AP50"]
                
                # Additional metrics for paper analysis
                per_cls_log[f"test/Precision/{cls_name}"] = cls_metrics.get("precision", 0.0)
                per_cls_log[f"test/Recall/{cls_name}"] = cls_metrics.get("recall", 0.0)
                per_cls_log[f"test/F1/{cls_name}"] = cls_metrics.get("f1", 0.0)
                per_cls_log[f"test/mIoU/{cls_name}"] = cls_metrics.get("miou", 0.0)
                
            wandb.log(per_cls_log)
            
        # Log comprehensive summary metrics for paper
        summary_metrics = {
            "test/mAP": test_metrics.get("mAP", 0.0),
            "test/AP50": test_metrics.get("AP@50", 0.0), 
            "test/AP75": test_metrics.get("AP@75", 0.0),
            "test/Disease_Accuracy": test_metrics.get("disease_acc_pct", 0.0),
            "test/BBox_IoU_Mean": test_metrics.get("bbox_iou_mean_pct", 0.0),
            "test/BBox_Area_MAPE": test_metrics.get("bbox_area_mape_pct", 0.0),
            "test/Matched_Pairs": test_metrics.get("matched_pairs", 0),
        }
        wandb.log(summary_metrics)

    if do_confusion:
        disease_confusion_and_bbox_diffs(
            model, processor, coco_test, test_img_dir,
            iou_thresh=0.1, max_images=None, class_names=class_names
        )
    if do_viz:
        visualize_predictions(model, processor, coco_test, test_img_dir,
                              num_images=3, score_thresh=0.0, topk=10,
                              random_sample=True, use_grayscale=True)

    return test_metrics


def main(args, wandb_run=None):
    """Main entry point for testing from command line arguments."""
    import os
    import torch
    from transformers import DetrForObjectDetection, DetrImageProcessor
    from pycocotools.coco import COCO
    
    # Load configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set up paths
    test_anno_file = os.path.join(args.anno_dir, "test_annotations_coco.json")
    
    # Load class names
    coco = COCO(test_anno_file)
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
    
    # Find most recent weights directory
    weights_dirs = [d for d in os.listdir(args.weights_dir) if os.path.isdir(os.path.join(args.weights_dir, d))]
    if not weights_dirs:
        raise FileNotFoundError(f"No weights found in {args.weights_dir}. Please train a model first.")
    
    # Use most recent directory
    weights_dir = os.path.join(args.weights_dir, sorted(weights_dirs)[-1])
    print(f"Using weights from: {weights_dir}")
    
    # Run test
    test_metrics = run_test(
        model=model,
        processor=processor,
        device=device,
        num_classes=num_classes,
        test_img_dir=args.test_img_dir,
        test_anno_file=test_anno_file,
        weights_dir=weights_dir,
        score_thresh=0.05,
        topk=100,
        do_confusion=True,
        do_viz=True,
        class_names=class_names,
    )
    
    print(f"\nTesting complete!")
    print(f"Test metrics: {test_metrics}")
    
    return test_metrics


if __name__ == "__main__":
    pass
