# visualize_data.py
import os
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# =============================
# CONFIG – edit paths if needed
# =============================
# Labels
TRAIN_LABELS_CSV = "/content/drive/MyDrive/vindr_pcxr/image_labels_train.csv"
TEST_LABELS_CSV  = "/content/drive/MyDrive/vindr_pcxr/image_labels_test.csv"

# Annotations (optional; boxes drawn if present)
TRAIN_ANN_CSV    = "/content/drive/MyDrive/vindr_pcxr/annotations_train.csv"
TEST_ANN_CSV     = "/content/drive/MyDrive/vindr_pcxr/annotations_test.csv"

# Images (point these to your PNGs)
TRAIN_IMG_DIR    = "/content/drive/MyDrive/vindr_pcxr/train"
TEST_IMG_DIR     = "/content/drive/MyDrive/vindr_pcxr/test"
IMAGE_EXT        = ".png"   # change to ".jpg" if needed

# How many examples per split per class
N_TRAIN_PER_CLASS = 5
N_TEST_PER_CLASS  = 5

# Figure sizing
TILE_SIZE         = 3.6     # inches per tile
PIE_FIGSIZE       = (9, 7)

# =============================
# Helpers
# =============================
def _read_csv_safely(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except FileNotFoundError:
        print(f"[WARN] CSV not found: {path}")
        return None
    except Exception as e:
        print(f"[WARN] Failed to read CSV {path}: {e}")
        return None

def _prepare_labels(df: pd.DataFrame) -> Tuple[Dict[str, set], List[str]]:
    """
    Collapse one-hot rows into image_id -> set(labels).
    Treat all columns except common metadata as label columns.
    """
    cols = list(df.columns)
    non_labels = {"image_id", "img_id", "image", "rad_id", "rad_ID", "dataset", "split"}
    classes = [c for c in cols if c not in non_labels]
    if len(classes) < 2 and len(cols) >= 3:
        classes = cols[2:]

    img_to_labels: Dict[str, set] = {}
    for _, row in df.iterrows():
        img_id = str(row.get("image_id", row.get("img_id", row.get("image", ""))))
        if not img_id:
            continue
        labels = [cls for cls in classes if str(row.get(cls, 0)) in ("1", "1.0") or row.get(cls, 0) in (1, 1.0)]
        img_to_labels.setdefault(img_id, set()).update(labels)
    return img_to_labels, classes

def _normalize_ann_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize to columns:
      image_id, x_min, y_min, x_max, y_max, [label?]
    Accepts (x,y,w,h) or (x_min,y_min,x_max,y_max).
    """
    cols_lower = {c.lower(): c for c in df.columns}
    img_col = None
    for k in ("image_id", "img_id", "image", "filename"):
        if k in cols_lower:
            img_col = cols_lower[k]; break
    if img_col is None:
        raise ValueError("No image_id/img_id/image/filename column in annotations CSV.")

    label_col = None
    for k in ("label", "class", "category", "finding_name", "disease"):
        if k in cols_lower:
            label_col = cols_lower[k]; break

    if all(k in cols_lower for k in ("x", "y", "w", "h")):
        x, y, w, h = (cols_lower["x"], cols_lower["y"], cols_lower["w"], cols_lower["h"])
        df["_x_min"] = df[x].astype(float)
        df["_y_min"] = df[y].astype(float)
        df["_x_max"] = df["_x_min"] + df[w].astype(float)
        df["_y_max"] = df["_y_min"] + df[h].astype(float)
    elif all(k in cols_lower for k in ("x_min", "y_min", "x_max", "y_max")):
        df["_x_min"] = df[cols_lower["x_min"]].astype(float)
        df["_y_min"] = df[cols_lower["y_min"]].astype(float)
        df["_x_max"] = df[cols_lower["x_max"]].astype(float)
        df["_y_max"] = df[cols_lower["y_max"]].astype(float)
    else:
        raise ValueError("Annotations require (x,y,w,h) or (x_min,y_min,x_max,y_max).")

    out = {
        "image_id": df[img_col].astype(str),
        "x_min": df["_x_min"], "y_min": df["_y_min"],
        "x_max": df["_x_max"], "y_max": df["_y_max"],
    }
    if label_col:
        out["label"] = df[label_col].astype(str)
    return pd.DataFrame(out)

def _build_ann_index(df_ann: Optional[pd.DataFrame]) -> Dict[str, List[dict]]:
    idx: Dict[str, List[dict]] = {}
    if df_ann is None or df_ann.empty:
        return idx
    try:
        norm = _normalize_ann_columns(df_ann)
    except Exception as e:
        print(f"[WARN] Could not normalize annotations: {e}")
        return idx
    for _, r in norm.iterrows():
        rec = {
            "x_min": float(r["x_min"]), "y_min": float(r["y_min"]),
            "x_max": float(r["x_max"]), "y_max": float(r["y_max"]),
        }
        if "label" in norm.columns:
            rec["label"] = r["label"]
        idx.setdefault(str(r["image_id"]), []).append(rec)
    return idx

def _grid(n: int) -> Tuple[int, int]:
    # up to 10 per class (5 train + 5 test) -> use 2 rows x 5 cols by default
    cols = min(5, max(1, n))
    rows = int(np.ceil(n / cols))
    return rows, cols

def _show_img(ax, img_path: str, title: str):
    if not os.path.exists(img_path):
        ax.set_title(f"Missing: {os.path.basename(img_path)}", fontsize=8, color="red")
        ax.axis("off"); return
    img = Image.open(img_path).convert("RGB")
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

def _draw_boxes(ax, boxes: List[dict]):
    # Draw ALL boxes (no class filter)
    if not boxes: return
    for b in boxes:
        x0, y0, x1, y1 = b["x_min"], b["y_min"], b["x_max"], b["y_max"]
        w, h = max(0.0, x1 - x0), max(0.0, y1 - y0)
        rect = patches.Rectangle((x0, y0), w, h, linewidth=1.5, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        if "label" in b:
            ax.text(x0, max(0, y0 - 5), str(b["label"]), fontsize=8,
                    bbox=dict(facecolor="black", alpha=0.4, edgecolor="none"), color="white")

# =============================
# Visualization
# =============================
def plot_pie(img_to_labels: Dict[str, set], classes: List[str], split_name: str):
    counts = {cls: 0 for cls in classes}
    for lbls in img_to_labels.values():
        for l in lbls:
            if l in counts:
                counts[l] += 1
    plt.figure(figsize=PIE_FIGSIZE)
    plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', startangle=90)
    plt.title(f'{split_name} label distribution')
    plt.axis('equal')
    plt.show()

def plot_examples_combined(
    train_map: Optional[Dict[str, set]],
    test_map: Optional[Dict[str, set]],
    classes: List[str],
    train_img_dir: str,
    test_img_dir: str,
    ann_train: Dict[str, List[dict]],
    ann_test: Dict[str, List[dict]],
    n_train: int,
    n_test: int
):
    train_map = train_map or {}
    test_map  = test_map or {}

    for cls in classes:
        # pick up to n_train/n_test ids for this class
        train_ids = [iid for iid, lbls in train_map.items() if cls in lbls][:n_train]
        test_ids  = [iid for iid, lbls in test_map.items()  if cls in lbls][:n_test]

        total = len(train_ids) + len(test_ids)
        if total == 0:
            continue

        rows, cols = _grid(total)
        fig, axes = plt.subplots(rows, cols, figsize=(TILE_SIZE*cols, TILE_SIZE*rows))
        if not isinstance(axes, (list, tuple, np.ndarray)):
            axes = [axes]
        axes = axes.flatten() if hasattr(axes, "flatten") else axes

        i = 0
        # First: TRAIN examples
        for img_id in train_ids:
            ax = axes[i]; i += 1
            img_path = os.path.join(train_img_dir, img_id + IMAGE_EXT)
            _show_img(ax, img_path, title=f"[Train] {img_id} | {cls}")
            _draw_boxes(ax, ann_train.get(img_id, []))

        # Then: TEST examples
        for img_id in test_ids:
            ax = axes[i]; i += 1
            img_path = os.path.join(test_img_dir, img_id + IMAGE_EXT)
            _show_img(ax, img_path, title=f"[Test] {img_id} | {cls}")
            _draw_boxes(ax, ann_test.get(img_id, []))

        # Turn off any leftover axes
        for ax in axes[i:]:
            ax.axis("off")

        fig.suptitle(f"{cls} — {len(train_ids)} train + {len(test_ids)} test", fontsize=12)
        plt.tight_layout()
        plt.show()

# =============================
# Main
# =============================
def main():
    # Read labels (skip gracefully if one side missing)
    df_train = _read_csv_safely(TRAIN_LABELS_CSV)
    df_test  = _read_csv_safely(TEST_LABELS_CSV)

    train_map = None
    test_map  = None
    classes: List[str] = []

    if df_train is not None:
        train_map, classes = _prepare_labels(df_train)
    if df_test is not None:
        test_map, classes_test = _prepare_labels(df_test)
        if not classes:  # if train missing, take classes from test
            classes = classes_test

    if not classes:
        print("[ERROR] Could not infer classes from the provided CSVs.")
        return

    # Build annotation indices (draw ALL boxes if present)
    ann_train = _build_ann_index(_read_csv_safely(TRAIN_ANN_CSV))
    ann_test  = _build_ann_index(_read_csv_safely(TEST_ANN_CSV))

    # Pies (only for splits we have)
    if train_map is not None:
        plot_pie(train_map, classes, "Train")
    if test_map is not None:
        plot_pie(test_map, classes, "Test")

    # Combined per-class panels: 5 from train + 5 from test (by default)
    plot_examples_combined(
        train_map, test_map, classes,
        TRAIN_IMG_DIR, TEST_IMG_DIR,
        ann_train, ann_test,
        n_train=N_TRAIN_PER_CLASS,
        n_test=N_TEST_PER_CLASS
    )

if __name__ == "__main__":
    main()
