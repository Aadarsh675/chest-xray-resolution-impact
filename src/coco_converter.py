# coco_converter.py
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image

# =========================
# CONFIG: ViNDr-PCXR paths
# =========================
ROOT = Path("/content/drive/MyDrive/vindr_pcxr/physionet.org")

TRAIN_CSV = ROOT / "annotations_train.csv"
TEST_CSV  = ROOT / "annotations_test.csv"

TRAIN_IMG_DIR = ROOT / "train (python)"   # expects <image_id>.png
TEST_IMG_DIR  = ROOT / "test (python)"

# Where to save COCO JSONs used by your pipeline
OUT_DIR = Path("data/annotations")
TRAIN_COCO_JSON = OUT_DIR / "train_annotations_coco.json"
TEST_COCO_JSON  = OUT_DIR / "test_annotations_coco.json"

# If you also want a combined file (optional):
COMBINED_COCO_JSON = OUT_DIR / "annotations_coco_all.json"


# =========================
# Helpers
# =========================
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_csv_any_delim(csv_path: Path) -> pd.DataFrame:
    """
    Read CSV or tabular file that may be comma- or tab-separated.
    Using engine='python' + sep=None lets pandas sniff the delimiter.
    """
    df = pd.read_csv(csv_path, sep=None, engine="python")
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    expected = ["image_id", "rad_ID", "class_name", "x_min", "y_min", "x_max", "y_max", "class_id"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}\nFound: {list(df.columns)}")
    return df


def _img_size_or_none(img_dir: Path, image_id: str, ext: str = ".png") -> Tuple[int, int]:
    """
    Return (width, height) if the PNG exists; otherwise (None, None).
    """
    p = img_dir / f"{image_id}{ext}"
    if p.exists():
        try:
            with Image.open(p) as im:
                return im.size  # (W, H)
        except Exception:
            return (None, None)
    return (None, None)


def _build_categories_union(train_df: pd.DataFrame, test_df: pd.DataFrame) -> List[Dict]:
    """
    Create a consistent categories list across both splits
    using the union of class_name values, sorted alphabetically (stable).
    """
    classes = sorted(set(train_df["class_name"].unique()) | set(test_df["class_name"].unique()))
    categories = [{"id": i + 1, "name": cls} for i, cls in enumerate(classes)]
    return categories


def _cat_id_lookup(categories: List[Dict]) -> Dict[str, int]:
    return {c["name"]: c["id"] for c in categories}


def _df_to_coco(
    df: pd.DataFrame,
    img_dir: Path,
    categories: List[Dict],
    image_ext: str = ".png"
) -> Dict:
    """
    Convert a split dataframe to COCO dict:
      - images: unique by image_id (filename = <image_id>.png)
      - annotations: bbox in [x,y,w,h], area computed, iscrowd=0
      - categories: provided (consistent IDs across splits)
    """
    cat_name_to_id = _cat_id_lookup(categories)

    # Images (unique by image_id)
    images = []
    image_id_map: Dict[str, int] = {}
    next_img_id = 1

    # Use unique image ids present in this split
    for img_id in df["image_id"].unique():
        W, H = _img_size_or_none(img_dir, img_id, ext=image_ext)
        image_id_map[img_id] = next_img_id
        images.append({
            "id": next_img_id,
            "file_name": f"{img_id}{image_ext}",
            "width": W,
            "height": H,
        })
        next_img_id += 1

    # Annotations
    anns = []
    ann_id = 1
    for _, r in df.iterrows():
        img_id_str = str(r["image_id"])
        if img_id_str not in image_id_map:
            # Should not happen, but guard anyway
            continue
        img_id = image_id_map[img_id_str]

        # Convert x_min, y_min, x_max, y_max to x, y, w, h
        try:
            x_min = float(r["x_min"]); y_min = float(r["y_min"])
            x_max = float(r["x_max"]); y_max = float(r["y_max"])
        except Exception:
            # skip bad rows
            continue

        w = max(0.0, x_max - x_min)
        h = max(0.0, y_max - y_min)

        # If width/height is zero or negative after safeguards, skip
        if w <= 0.0 or h <= 0.0:
            continue

        cname = str(r["class_name"]).strip()
        if cname not in cat_name_to_id:
            # Unknown category: skip or add dynamically (here we skip to keep IDs stable)
            continue
        cid = cat_name_to_id[cname]

        anns.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": cid,
            "bbox": [float(x_min), float(y_min), float(w), float(h)],
            "area": float(w * h),
            "iscrowd": 0,
        })
        ann_id += 1

    coco = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": anns,
        "categories": categories,
    }
    return coco


# =========================
# Main
# =========================
def main():
    print(f"[INFO] Reading train annotations: {TRAIN_CSV}")
    train_df = _read_csv_any_delim(TRAIN_CSV)

    print(f"[INFO] Reading test annotations:  {TEST_CSV}")
    test_df = _read_csv_any_delim(TEST_CSV)

    # Build a consistent categories list across both splits
    categories = _build_categories_union(train_df, test_df)
    print(f"[INFO] Found {len(categories)} categories.")
    # Optional: print categories for sanity
    print("       Categories:", [c["name"] for c in categories])

    # Convert each split
    print("[INFO] Building COCO for TRAIN …")
    coco_train = _df_to_coco(train_df, TRAIN_IMG_DIR, categories, image_ext=".png")

    print("[INFO] Building COCO for TEST …")
    coco_test = _df_to_coco(test_df, TEST_IMG_DIR, categories, image_ext=".png")

    # Save
    _ensure_dir(OUT_DIR)
    with open(TRAIN_COCO_JSON, "w") as f:
        json.dump(coco_train, f, indent=2)
    with open(TEST_COCO_JSON, "w") as f:
        json.dump(coco_test, f, indent=2)
    print(f"[OK] Wrote: {TRAIN_COCO_JSON}")
    print(f"[OK] Wrote: {TEST_COCO_JSON}")

    # (Optional) Combined file
    # This is helpful if you ever want a single annotations file for quick checks.
    combined = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }
    # Re-map image/ann IDs to be unique in the combined file
    img_id_offset = 0
    ann_id_offset = 0

    def _append_split(coco_split):
        nonlocal img_id_offset, ann_id_offset
        # map old image id -> new id
        img_map = {}
        for im in coco_split["images"]:
            new_id = len(combined["images"]) + 1
            img_map[im["id"]] = new_id
            im2 = dict(im)
            im2["id"] = new_id
            combined["images"].append(im2)
        for an in coco_split["annotations"]:
            an2 = dict(an)
            an2["id"] = len(combined["annotations"]) + 1
            an2["image_id"] = img_map[an["image_id"]]
            combined["annotations"].append(an2)

    _append_split(coco_train)
    _append_split(coco_test)

    with open(COMBINED_COCO_JSON, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"[OK] (optional) Wrote combined file: {COMBINED_COCO_JSON}")


if __name__ == "__main__":
    main()
