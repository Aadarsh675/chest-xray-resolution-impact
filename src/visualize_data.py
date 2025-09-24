# visualize_data.py
import os
import json
import argparse
from typing import Dict, List, Tuple, Iterable
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# -----------------------------
# Drawing helpers
# -----------------------------
def _show_img_with_boxes(ax, img_path: str, boxes_xywh: Iterable[Tuple[float,float,float,float]], title: str):
    if not os.path.exists(img_path):
        ax.set_title(f"Missing: {os.path.basename(img_path)}", fontsize=9, color="red")
        ax.axis("off")
        return
    img = Image.open(img_path).convert("RGB")
    ax.imshow(img, cmap="gray")
    for (x, y, w, h) in boxes_xywh:
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _grid(n: int) -> Tuple[int, int]:
    # up to 5 images per class; make a nice grid
    if n <= 3: return (1, n)
    if n == 4: return (2, 2)
    return (1, 5)


# -----------------------------
# Mode 1: pre-split (CSV)
# -----------------------------
def mode_pre_split(csv_path: str, image_dir: str, per_class: int = 5):
    """
    Show up to 5 images per class BEFORE splitting, using NIH CSV with columns:
      'Image Index', 'Finding Label', 'Bbox [x', 'y', 'w', 'h]'
    Also plots a pie chart of unique images per label.
    """
    df = pd.read_csv(csv_path)

    # --- Pie chart (unique images per label) ---
    label_counts = df.groupby('Image Index')['Finding Label'].first().value_counts()
    plt.figure(figsize=(10, 8))
    plt.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Images by Disease Label (unique images)')
    plt.axis('equal')
    plt.show()

    # --- Sample up to per_class images per label (unique images per label) ---
    images_by_label: Dict[str, List[str]] = (
        df.groupby("Finding Label")["Image Index"].apply(lambda s: list(dict.fromkeys(s.tolist()))).to_dict()
    )

    for label, img_list in images_by_label.items():
        sample_imgs = img_list[:per_class]
        if len(sample_imgs) == 0:
            continue

        fig_rows, fig_cols = _grid(len(sample_imgs))
        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(4*fig_cols, 4*fig_rows))
        if not isinstance(axes, (list, tuple, np.ndarray)):
            axes = [axes]
        axes = axes.flatten() if hasattr(axes, "flatten") else axes

        for ax, img_name in zip(axes, sample_imgs):
            rows = df[df["Image Index"] == img_name]
            # handle multiple boxes per image if present
            boxes = []
            for _, r in rows.iterrows():
                x, y, w, h = r["Bbox [x"], r["y"], r["w"], r["h]"]
                boxes.append((float(x), float(y), float(w), float(h)))
            _show_img_with_boxes(
                ax,
                os.path.join(image_dir, img_name),
                boxes,
                title=f"{img_name} | {label}"
            )

        # hide extra axes (if any)
        for ax in axes[len(sample_imgs):]:
            ax.axis("off")

        fig.suptitle(f"[Pre-split] {label} — up to {per_class} examples", fontsize=12)
        plt.tight_layout()
        plt.show()


# -----------------------------
# COCO helpers (post-split)
# -----------------------------
def _load_coco(anno_path: str):
    with open(anno_path, "r") as f:
        data = json.load(f)
    img_by_id = {im["id"]: im for im in data.get("images", [])}
    cat_by_id = {c["id"]: c for c in data.get("categories", [])}
    anns_by_img: Dict[int, List[dict]] = {}
    for ann in data.get("annotations", []):
        anns_by_img.setdefault(ann["image_id"], []).append(ann)
    return data, img_by_id, cat_by_id, anns_by_img


def _sample_imgs_per_class_coco(
    cat_by_id: Dict[int, dict],
    anns_by_img: Dict[int, List[dict]],
    per_class: int
) -> Dict[str, List[int]]:
    """
    Return dict: class_name -> list of image_ids (up to per_class) that have at least one ann of that category.
    """
    by_class: Dict[str, List[int]] = {}
    # Build mapping from image -> set(categories)
    img_to_cats: Dict[int, set] = {}
    for img_id, anns in anns_by_img.items():
        img_to_cats[img_id] = set(a["category_id"] for a in anns)

    for cid, cinfo in cat_by_id.items():
        cname = cinfo["name"]
        hits = [img_id for img_id, s in img_to_cats.items() if cid in s]
        random.shuffle(hits)
        by_class[cname] = hits[:per_class]
    return by_class


# -----------------------------
# Mode 2: post-split (train/test)
# -----------------------------
def mode_post_split(train_json: str, test_json: str, train_img_dir: str, test_img_dir: str, per_class: int = 5):
    # Train
    _, train_img_by_id, train_cat_by_id, train_anns_by_img = _load_coco(train_json)
    # Test
    _, test_img_by_id, test_cat_by_id, test_anns_by_img = _load_coco(test_json)

    # Use train categories as reference (assume same set)
    class_to_imgs_train = _sample_imgs_per_class_coco(train_cat_by_id, train_anns_by_img, per_class)
    class_to_imgs_test  = _sample_imgs_per_class_coco(test_cat_by_id,  test_anns_by_img,  per_class)

    # Plot per class for TRAIN and TEST
    for cls in sorted(class_to_imgs_train.keys()):
        # TRAIN
        img_ids = class_to_imgs_train[cls]
        if img_ids:
            rows, cols = _grid(len(img_ids))
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            if not isinstance(axes, (list, tuple, np.ndarray)):
                axes = [axes]
            axes = axes.flatten() if hasattr(axes, "flatten") else axes

            for ax, img_id in zip(axes, img_ids):
                img_info = train_img_by_id[img_id]
                anns = train_anns_by_img.get(img_id, [])
                boxes = [tuple(a["bbox"]) for a in anns]
                _show_img_with_boxes(
                    ax,
                    os.path.join(train_img_dir, img_info["file_name"]),
                    boxes,
                    title=f"{img_info['file_name']} | {cls} (train)"
                )
            for ax in axes[len(img_ids):]:
                ax.axis("off")
            fig.suptitle(f"[Post-split TRAIN] {cls} — up to {per_class} examples", fontsize=12)
            plt.tight_layout()
            plt.show()

        # TEST
        img_ids = class_to_imgs_test.get(cls, [])
        if img_ids:
            rows, cols = _grid(len(img_ids))
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            if not isinstance(axes, (list, tuple, np.ndarray)):
                axes = [axes]
            axes = axes.flatten() if hasattr(axes, "flatten") else axes

            for ax, img_id in zip(axes, img_ids):
                img_info = test_img_by_id[img_id]
                anns = test_anns_by_img.get(img_id, [])
                boxes = [tuple(a["bbox"]) for a in anns]
                _show_img_with_boxes(
                    ax,
                    os.path.join(test_img_dir, img_info["file_name"]),
                    boxes,
                    title=f"{img_info['file_name']} | {cls} (test)"
                )
            for ax in axes[len(img_ids):]:
                ax.axis("off")
            fig.suptitle(f"[Post-split TEST] {cls} — up to {per_class} examples", fontsize=12)
            plt.tight_layout()
            plt.show()


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualize chest X-ray samples with bounding boxes and labels."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # Mode 1: pre-split
    p1 = sub.add_parser("pre_split", help="Show 5 images of each class before splitting (CSV-based) + pie chart.")
    p1.add_argument("--csv_path", required=True, type=str)
    p1.add_argument("--image_dir", required=True, type=str)
    p1.add_argument("--per_class", type=int, default=5)

    # Mode 2: post-split
    p2 = sub.add_parser("post_split", help="Show 5 images of each class after splitting (train/test COCO).")
    p2.add_argument("--train_json", required=True, type=str)
    p2.add_argument("--test_json",  required=True, type=str)
    p2.add_argument("--train_img_dir", required=True, type=str)
    p2.add_argument("--test_img_dir",  required=True, type=str)
    p2.add_argument("--per_class", type=int, default=5)

    args = parser.parse_args()

    if args.mode == "pre_split":
        mode_pre_split(csv_path=args.csv_path, image_dir=args.image_dir, per_class=args.per_class)
    elif args.mode == "post_split":
        mode_post_split(
            train_json=args.train_json,
            test_json=args.test_json,
            train_img_dir=args.train_img_dir,
            test_img_dir=args.test_img_dir,
            per_class=args.per_class,
        )


if __name__ == "__main__":
    main()
