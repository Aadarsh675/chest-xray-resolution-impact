# main.py
import os, json, uuid
from zoneinfo import ZoneInfo
from datetime import datetime

import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from pycocotools.coco import COCO
import wandb

from train import train_and_validate
import test as test_mod
from scale_dataset import prepare_all_scales
from metrics_curves import export_threshold_curves

# -----------------------------
# Base paths / constants
# -----------------------------
DATA_DIR   = "data"
ANNO_DIR   = os.path.join(DATA_DIR, "annotations")
IMAGE_DIR  = os.path.join(DATA_DIR, "images")
WEIGHTS_DIR = "weights"
MODEL_NAME  = "facebook/detr-resnet-50"

BATCH_SIZE    = 2
NUM_WORKERS   = 0
NUM_EPOCHS    = 10
LEARNING_RATE = 1e-4

TOPK = 100
SCORE_THRESH = 0.0

# -----------------------------
# Experiment controls
# -----------------------------
SCALES_TO_BUILD   = [1.0, 0.5, 0.25]   # resolutions to prepare
ENABLE_SCALES     = True               # set False if pre-built
REPEATS_PER_SCALE = 1                  # how many training runs per scale
CURVE_THRESHOLDS  = None               # None -> default grid inside metrics_curves

def resolve_paths_for_scale(scale: float):
    """Return (train_img_dir, val_img_dir, test_img_dir, train_anno, val_anno, test_anno) for a given scale."""
    if abs(scale - 1.0) < 1e-6:
        train_img_dir = os.path.join(IMAGE_DIR, "train")
        val_img_dir   = os.path.join(IMAGE_DIR, "test")
        test_img_dir  = os.path.join(IMAGE_DIR, "test")
        train_anno = os.path.join(ANNO_DIR, "train_annotations_coco.json")
        val_anno   = os.path.join(ANNO_DIR, "test_annotations_coco.json")
        test_anno  = os.path.join(ANNO_DIR, "test_annotations_coco.json")
    else:
        pct = int(round(scale * 100))
        img_root = f"data/images_s{pct}"
        train_img_dir = os.path.join(img_root, "train")
        val_img_dir   = os.path.join(img_root, "test")
        test_img_dir  = os.path.join(img_root, "test")
        train_anno = os.path.join(ANNO_DIR, f"train_annotations_coco_s{pct}.json")
        val_anno   = os.path.join(ANNO_DIR, f"test_annotations_coco_s{pct}.json")
        test_anno  = os.path.join(ANNO_DIR, f"test_annotations_coco_s{pct}.json")
    return train_img_dir, val_img_dir, test_img_dir, train_anno, val_anno, test_anno

def main():
    print(f"cwd: {os.getcwd()}")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # Prepare scaled datasets
    if ENABLE_SCALES:
        prepare_all_scales(
            scales=SCALES_TO_BUILD,
            base_image_root=IMAGE_DIR,
            base_ann_dir=ANNO_DIR,
            splits=("train", "test"),
            overwrite=False
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    for scale in SCALES_TO_BUILD:
        pct = int(round(scale * 100))
        print(f"\n========== SCALE {pct}% ({scale}) ==========")
        TRAIN_IMG_DIR, VAL_IMG_DIR, TEST_IMG_DIR, TRAIN_ANNO_FILE, VAL_ANNO_FILE, TEST_ANNO_FILE = resolve_paths_for_scale(scale)

        # Load classes
        with open(TRAIN_ANNO_FILE, "r") as f:
            train_coco_data = json.load(f)
        num_classes = len(train_coco_data["categories"])
        print(f"num_classes: {num_classes}")

        for rep in range(1, REPEATS_PER_SCALE + 1):
            # W&B per-run init (Pacific time)
            run_name = f"detr_run_{pct}pct_rep{rep}_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            wandb.init(
                project="detr-nih-chest-xray",
                name=run_name,
                config={
                    "epochs": NUM_EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "lr": LEARNING_RATE,
                    "model": MODEL_NAME,
                    "scale": scale,
                    "repeat": rep
                }
            )
            print(f"W&B run name: {run_name}")

            # Model & processor
            processor = DetrImageProcessor.from_pretrained(MODEL_NAME, size=800)
            model = DetrForObjectDetection.from_pretrained(
                MODEL_NAME, num_labels=num_classes, ignore_mismatched_sizes=True
            ).to(device)

            # Train + validate each epoch
            best_epoch, best_metrics = train_and_validate(
                model=model,
                processor=processor,
                device=device,
                num_classes=num_classes,
                train_img_dir=TRAIN_IMG_DIR,
                train_anno_file=TRAIN_ANNO_FILE,
                val_img_dir=VAL_IMG_DIR,
                val_anno_file=VAL_ANNO_FILE,
                weights_dir=WEIGHTS_DIR,
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
                test_anno_file=TEST_ANNO_FILE,
                weights_dir=WEIGHTS_DIR,
                score_thresh=SCORE_THRESH,
                topk=TOPK,
                do_confusion=True,
                do_viz=True,
            )

            # Export PR/mIoU vs threshold curves for this scale+repeat
            out_dir = os.path.join("analysis", f"scale_{pct}", f"rep_{rep}")
            curves = export_threshold_curves(
                model=model,
                processor=processor,
                device=device,
                test_img_dir=TEST_IMG_DIR,
                test_anno_file=TEST_ANNO_FILE,
                out_dir=out_dir,
                thresholds=CURVE_THRESHOLDS,   # or None for default
                iou_match_thresh=0.5,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
            )
            # Optionally log a few example points to W&B
            if curves and wandb.run is not None:
                # Log the last threshold snapshot as a quick table
                t_last = curves["thresholds"][-1]
                log_row = {f"curves/last_threshold": t_last}
                for cls, d in curves["per_class"].items():
                    log_row[f"curve_precision/{cls}"] = d["precision"][-1]
                    log_row[f"curve_recall/{cls}"] = d["recall"][-1]
                    log_row[f"curve_mIoU/{cls}"] = d["miou"][-1]
                wandb.log(log_row)

            wandb.finish()

    print("\nAll scales/repeats finished. Curve data saved under ./analysis/<scale>/rep_<n>/curves.{json,csv}")

if __name__ == "__main__":
    main()
