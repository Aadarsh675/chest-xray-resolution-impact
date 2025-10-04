# main.py
import os, json, uuid, random
from zoneinfo import ZoneInfo
from datetime import datetime
from typing import Tuple, Dict, Any, List

import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
import wandb

from train import train_and_validate
import test as test_mod
from metrics_curves import export_threshold_curves

# -----------------------------
# Base paths / constants
# -----------------------------
DATA_DIR     = "data"
ANNO_DIR     = os.path.join(DATA_DIR, "annotations")
IMAGE_DIR    = os.path.join(DATA_DIR, "images")     # expects PNGs in images/{train,test}
WEIGHTS_DIR  = "weights"
MODEL_NAME   = "facebook/detr-resnet-50"

BATCH_SIZE    = 2
NUM_WORKERS   = 2
NUM_EPOCHS    = 10
LEARNING_RATE = 1e-4

TOPK         = 100
SCORE_THRESH = 0.05  # small threshold to reduce noise in eval

# -----------------------------
# Experiment controls
# -----------------------------
ENABLE_SCALES     = False          # keep off for ViNDr; add later if needed
REPEATS_PER_SCALE = 1
CURVE_THRESHOLDS  = None           # None -> default grid inside metrics_curves

# Files expected to exist prior to training (from your converter):
TRAIN_JSON_BASE = os.path.join(ANNO_DIR, "train_annotations_coco.json")
TEST_JSON       = os.path.join(ANNO_DIR, "test_annotations_coco.json")

# Files we will create (split train into train_split/val)
TRAIN_SPLIT_JSON = os.path.join(ANNO_DIR, "train_annotations_coco_split.json")
VAL_JSON         = os.path.join(ANNO_DIR, "val_annotations_coco.json")


def _load_json(p: str) -> Dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)


def _save_json(d: Dict[str, Any], p: str) -> None:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(d, f, indent=2)


def _load_category_names(coco_json_path: str) -> List[str]:
    data = _load_json(coco_json_path)
    cats = sorted(data.get("categories", []), key=lambda c: c["id"])
    return [c["name"] for c in cats]


def _ensure_train_val_split(
    train_json_in: str,
    out_train_split_json: str,
    out_val_json: str,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[str, str]:
    """
    Create a val split from train if not present.
    Splits at image level. Keeps all categories as-is.
    """
    if os.path.exists(out_train_split_json) and os.path.exists(out_val_json):
        return out_train_split_json, out_val_json

    data = _load_json(train_json_in)
    images = data.get("images", [])
    anns   = data.get("annotations", [])
    cats   = data.get("categories", [])

    img_ids = [im["id"] for im in images]
    random.Random(seed).shuffle(img_ids)
    n_val = max(1, int(round(len(img_ids) * val_ratio)))
    val_ids = set(img_ids[:n_val])
    train_ids = set(img_ids[n_val:])

    def _subset(images_src, anns_src, keep_ids):
        imgs = [im for im in images_src if im["id"] in keep_ids]
        anns_out = [a for a in anns_src if a["image_id"] in keep_ids]
        return imgs, anns_out

    train_imgs, train_anns = _subset(images, anns, train_ids)
    val_imgs,   val_anns   = _subset(images, anns, val_ids)

    # Reassign annotation ids (clean)
    for i, a in enumerate(train_anns, 1): a["id"] = i
    for i, a in enumerate(val_anns,   1): a["id"] = i

    train_split = {"info": {}, "licenses": [], "images": train_imgs, "annotations": train_anns, "categories": cats}
    val_split   = {"info": {}, "licenses": [], "images": val_imgs,   "annotations": val_anns,   "categories": cats}

    _save_json(train_split, out_train_split_json)
    _save_json(val_split, out_val_json)
    print(f"[Split] Wrote train split → {out_train_split_json}  ({len(train_imgs)} imgs)")
    print(f"[Split] Wrote val   split → {out_val_json}        ({len(val_imgs)} imgs)")
    return out_train_split_json, out_val_json


def main():
    print(f"cwd: {os.getcwd()}")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # Make/ensure a small validation split from train
    train_split_json, val_json = _ensure_train_val_split(
        TRAIN_JSON_BASE, TRAIN_SPLIT_JSON, VAL_JSON, val_ratio=0.1, seed=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Paths (no scaling for now)
    TRAIN_IMG_DIR = os.path.join(IMAGE_DIR, "train")
    VAL_IMG_DIR   = os.path.join(IMAGE_DIR, "train")  # same folder; val JSON picks a subset of IDs
    TEST_IMG_DIR  = os.path.join(IMAGE_DIR, "test")

    # Load class names from the (split) train JSON
    class_names = _load_category_names(train_split_json)
    num_classes = len(class_names)
    print(f"classes ({num_classes}): {class_names}")

    for rep in range(1, REPEATS_PER_SCALE + 1):
        # Per-run W&B and weights directory
        run_name = f"detr_vindr_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        wandb.init(
            project="detr-vindr-pcxr",
            name=run_name,
            config={
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LEARNING_RATE,
                "model": MODEL_NAME,
                "classes": class_names,
            }
        )
        run_dir = os.path.join(WEIGHTS_DIR, run_name)
        os.makedirs(run_dir, exist_ok=True)
        print(f"W&B run: {run_name} | weights dir: {run_dir}")

        # Model & processor
        processor = DetrImageProcessor.from_pretrained(MODEL_NAME)
        model = DetrForObjectDetection.from_pretrained(
            MODEL_NAME, num_labels=num_classes, ignore_mismatched_sizes=True
        ).to(device)

        # Train + Validate
        best_epoch, best_metrics = train_and_validate(
            model=model,
            processor=processor,
            device=device,
            num_classes=num_classes,
            train_img_dir=TRAIN_IMG_DIR,
            train_anno_file=train_split_json,
            val_img_dir=VAL_IMG_DIR,
            val_anno_file=val_json,
            weights_dir=run_dir,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            learning_rate=LEARNING_RATE,
            score_thresh=SCORE_THRESH,
            topk=TOPK,
        )

        print("\nTraining complete!")
        if best_epoch <= 0:
            print("Warning: no best epoch recorded — check your validation.")
        else:
            print(f"Best epoch: {best_epoch} | Best val metrics: {best_metrics}")

        # TEST with best.pth
        test_metrics = test_mod.run_test(
            model=model,
            processor=processor,
            device=device,
            num_classes=num_classes,
            test_img_dir=TEST_IMG_DIR,
            test_anno_file=TEST_JSON,
            weights_dir=run_dir,
            score_thresh=SCORE_THRESH,
            topk=TOPK,
            do_confusion=True,
            do_viz=True,
            class_names=class_names,
        )

        # Export threshold curves from TEST
        out_dir = os.path.join("analysis", "scale_100", f"rep_{rep}")
        curves = export_threshold_curves(
            model=model,
            processor=processor,
            device=device,
            test_img_dir=TEST_IMG_DIR,
            test_anno_file=TEST_JSON,
            out_dir=out_dir,
            thresholds=CURVE_THRESHOLDS,
            iou_match_thresh=0.5,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )
        if curves and wandb.run is not None:
            t_last = curves["thresholds"][-1]
            log_row = {f"curves/last_threshold": t_last}
            for cls, d in curves["per_class"].items():
                log_row[f"curve_precision/{cls}"] = d["precision"][-1]
                log_row[f"curve_recall/{cls}"] = d["recall"][-1]
                log_row[f"curve_mIoU/{cls}"] = d["miou"][-1]
            wandb.log(log_row)

        wandb.finish()

    print("\nAll done. Curves under ./analysis/scale_100/rep_<n>/")
    print("Weights saved under ./weights/<run_name>/best.pth")

if __name__ == "__main__":
    main()
