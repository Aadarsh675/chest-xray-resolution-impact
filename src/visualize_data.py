# visualize_data.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from typing import List, Tuple, Iterable

# -----------------------------
# CONFIG (hard-coded paths)
# -----------------------------
TRAIN_LABELS_CSV = "/content/drive/MyDrive/vindr_pcxr/physionet.org/image_labels_train.csv"
TEST_LABELS_CSV  = "/content/drive/MyDrive/vindr_pcxr/physionet.org/image_labels_test.csv"
TRAIN_IMG_DIR    = "/content/drive/MyDrive/vindr_pcxr/physionet.org/train (python)"
TEST_IMG_DIR     = "/content/drive/MyDrive/vindr_pcxr/physionet.org/test (python)"
PER_CLASS        = 5
IMAGE_EXT        = ".png"

# -----------------------------
# Helpers
# -----------------------------
def _show_img(ax, img_path: str, title: str):
    if not os.path.exists(img_path):
        ax.set_title(f"Missing: {os.path.basename(img_path)}", fontsize=9, color="red")
        ax.axis("off")
        return
    img = Image.open(img_path).convert("RGB")
    ax.imshow(img, cmap="gray")
    ax.set_title(title, fontsize=9)
    ax.axis("off")

def _grid(n: int) -> Tuple[int, int]:
    if n <= 3: return (1, n)
    if n == 4: return (2, 2)
    return (1, 5)

def _prepare_labels(df: pd.DataFrame):
    """Collapse multi-label one-hot rows into image_id -> single label list."""
    classes = df.columns[2:]  # skip image_id, rad_ID
    img_to_labels = {}
    for _, row in df.iterrows():
        img_id = row["image_id"]
        # take all classes where value == 1
        labels = [cls for cls in classes if row[cls] == 1.0]
        if img_id not in img_to_labels:
            img_to_labels[img_id] = set(labels)
        else:
            img_to_labels[img_id].update(labels)
    return img_to_labels, list(classes)

# -----------------------------
# Visualization
# -----------------------------
def plot_pie(img_to_labels, classes, split_name: str):
    counts = {cls: 0 for cls in classes}
    for lbls in img_to_labels.values():
        for l in lbls:
            counts[l] += 1
    plt.figure(figsize=(10, 8))
    plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', startangle=90)
    plt.title(f'Distribution of {split_name} Images by Disease Label')
    plt.axis('equal')
    plt.show()

def plot_examples(img_to_labels, classes, img_dir, split_name: str, per_class: int):
    for cls in classes:
        img_ids = [img_id for img_id, lbls in img_to_labels.items() if cls in lbls]
        if not img_ids:
            continue
        sample_imgs = img_ids[:per_class]
        rows, cols = _grid(len(sample_imgs))
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if not isinstance(axes, (list, tuple, np.ndarray)):
            axes = [axes]
        axes = axes.flatten() if hasattr(axes, "flatten") else axes

        for ax, img_id in zip(axes, sample_imgs):
            img_path = os.path.join(img_dir, img_id + IMAGE_EXT)
            _show_img(ax, img_path, title=f"{img_id} | {cls}")
        for ax in axes[len(sample_imgs):]:
            ax.axis("off")

        fig.suptitle(f"[{split_name}] {cls} — up to {per_class} examples", fontsize=12)
        plt.tight_layout()
        plt.show()

# -----------------------------
# Main
# -----------------------------
def main():
    # Train
    df_train = pd.read_csv(TRAIN_LABELS_CSV, sep=None, engine="python")
    train_map, classes = _prepare_labels(df_train)
    # Test
    df_test = pd.read_csv(TEST_LABELS_CSV, sep=None, engine="python")
    test_map, _ = _prepare_labels(df_test)

    # Pie charts
    plot_pie(train_map, classes, "Train")
    plot_pie(test_map, classes, "Test")

    # Example images
    plot_examples(train_map, classes, TRAIN_IMG_DIR, "Train", PER_CLASS)
    plot_examples(test_map, classes, TEST_IMG_DIR, "Test", PER_CLASS)

if __name__ == "__main__":
    main()
