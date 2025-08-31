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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image
import os, random

def visualize_predictions(
    model,
    processor,
    coco_val,
    img_dir,
    num_images=3,
    score_thresh=0.0,   # super low so you always see something
    topk=10,            # cap how many red boxes you draw
    random_sample=True,
    use_grayscale=True
):
    """
    GT boxes (yellow) with 'GT: <label>' at TOP-LEFT.
    Pred boxes (red) with 'Pred: <label> (score)' at BOTTOM-LEFT.
    Uses processor.post_process_object_detection for reliable boxes.
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

        # load image + size
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        # ground truth
        ann_ids = coco_val.getAnnIds(imgIds=img_id)
        anns = coco_val.loadAnns(ann_ids)

        # forward pass
        encoding = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encoding)

        # use built-in post-processing to get pixel-space boxes/scores/labels
        target_sizes = torch.tensor([[height, width]], device=device)  # (h, w)
        processed = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=score_thresh
        )[0]

        boxes = processed["boxes"].cpu().numpy()   # [x1, y1, x2, y2] in pixels
        scores = processed["scores"].cpu().numpy()
        labels = processed["labels"].cpu().numpy() # 0-based class indices

        # keep top-K
        if len(scores) > topk:
            idx = np.argsort(-scores)[:topk]
            boxes, scores, labels = boxes[idx], scores[idx], labels[idx]

        print(f"[viz] img_id={img_id} preds_kept={len(scores)} (thr={score_thresh}, topk={topk})")

        # convert to [x, y, w, h]
        if len(boxes) > 0:
            boxes_xywh = boxes.copy()
            boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # w
            boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # h
            boxes_xywh[:, 0] = boxes[:, 0]                # x
            boxes_xywh[:, 1] = boxes[:, 1]                # y
            # clip
            boxes_xywh[:, 0] = np.clip(boxes_xywh[:, 0], 0, width - 1)
            boxes_xywh[:, 1] = np.clip(boxes_xywh[:, 1], 0, height - 1)
            boxes_xywh[:, 2] = np.clip(boxes_xywh[:, 2], 1e-6, width)
            boxes_xywh[:, 3] = np.clip(boxes_xywh[:, 3], 1e-6, height)
        else:
            boxes_xywh = boxes

        # draw
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(image, cmap="gray" if use_grayscale else None)

        # GT (yellow) — label at TOP-LEFT
        for ann in anns:
            gx, gy, gw, gh = ann["bbox"]
            ax.add_patch(patches.Rectangle((gx, gy), gw, gh, linewidth=2,
                                           edgecolor="yellow", facecolor="none", zorder=3))
            gt_label = cat_id_to_name.get(ann["category_id"], str(ann["category_id"]))
            tx, ty = gx, max(0, gy - 6)
            ax.text(tx, ty, f"GT: {gt_label}",
                    color="yellow", fontsize=10, fontweight="bold",
                    backgroundcolor="black", zorder=4)

        # Pred (red) — label at BOTTOM-LEFT
        for (px, py, pw, ph), sc, lb in zip(boxes_xywh, scores, labels):
            ax.add_patch(patches.Rectangle((px, py), pw, ph, linewidth=2,
                                           edgecolor="red", facecolor="none", zorder=3))
            # labels from post_process are 0-based class indices
            pred_name = cat_id_to_name.get(int(lb) + 1, str(int(lb) + 1))
            tx, ty = px, min(height - 1, py + ph + 12)
            ax.text(tx, ty, f"Pred: {pred_name} ({sc:.2f})",
                    color="red", fontsize=10, fontweight="bold",
                    backgroundcolor="black", zorder=4)

        ax.set_title(f"Image ID: {img_id} - {img_info['file_name']}")
        ax.axis("off")
        # simple legend
        ax.plot([], [], color="yellow", linewidth=2, label="GT box")
        ax.plot([], [], color="red", linewidth=2, label="Pred box")
        ax.legend(loc="upper right")
        plt.show()

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

    # visualize a few predictions vs GT
    print("\nVisualizing predictions vs ground truth...")
    visualize_predictions(model, processor, coco_val, VAL_IMG_DIR, num_images=3, score_thresh=0.3, random_sample=True)


if __name__ == "__main__":
    main()
