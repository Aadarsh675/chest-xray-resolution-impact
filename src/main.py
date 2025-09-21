import os
import json
import uuid
from zoneinfo import ZoneInfo
from datetime import datetime

import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from pycocotools.coco import COCO
import wandb

# -----------------------------
# Paths & constants
# -----------------------------
DATA_DIR = "data"
ANNO_DIR = os.path.join(DATA_DIR, "annotations")
IMAGE_DIR = os.path.join(DATA_DIR, "images")

TRAIN_ANNO_FILE = os.path.join(ANNO_DIR, "train_annotations_coco.json")
VAL_ANNO_FILE   = os.path.join(ANNO_DIR, "test_annotations_coco.json")   # using test as val if you don't have a val set
TEST_ANNO_FILE  = os.path.join(ANNO_DIR, "test_annotations_coco.json")

TRAIN_IMG_DIR   = os.path.join(IMAGE_DIR, "train")
VAL_IMG_DIR     = os.path.join(IMAGE_DIR, "test")
TEST_IMG_DIR    = os.path.join(IMAGE_DIR, "test")

WEIGHTS_DIR = "weights"
MODEL_NAME  = "facebook/detr-resnet-50"

BATCH_SIZE    = 2
NUM_WORKERS   = 0
NUM_EPOCHS    = 10
LEARNING_RATE = 1e-4

TOPK = 100
SCORE_THRESH = 0.0

# -----------------------------
# Orchestration
# -----------------------------
def main():
    print(f"cwd: {os.getcwd()}")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # ---- W&B init with Pacific time ----
    run_name = f"detr_run_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    wandb.init(
        project="detr-nih-chest-xray",
        name=run_name,
        config={
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "model": MODEL_NAME,
        }
    )
    print(f"W&B run name: {run_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Load classes
    with open(TRAIN_ANNO_FILE, "r") as f:
        train_coco_data = json.load(f)
    num_classes = len(train_coco_data["categories"])
    print(f"num_classes: {num_classes}")

    # Model & processor
    processor = DetrImageProcessor.from_pretrained(MODEL_NAME, size=800)
    model = DetrForObjectDetection.from_pretrained(
        MODEL_NAME, num_labels=num_classes, ignore_mismatched_sizes=True
    ).to(device)

    # ---- Train (with validation each epoch) ----
    from train import train_and_validate
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

    # ---- Final TEST (with best weights) ----
    import test as test_mod
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

    # Log test metrics to W&B
    wandb.log({f"test/{k}": v for k, v in test_metrics.items() if k != "per_class"})
    if test_metrics.get("per_class"):
        per_cls_log = {}
        for cls_name, cls_metrics in test_metrics["per_class"].items():
            per_cls_log[f"test/AP/{cls_name}"] = cls_metrics["AP"]
            per_cls_log[f"test/AP50/{cls_name}"] = cls_metrics["AP50"]
        wandb.log(per_cls_log)

    wandb.finish()

if __name__ == "__main__":
    main()
