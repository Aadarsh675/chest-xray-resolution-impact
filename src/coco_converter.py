# coco_converter.py
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle

import pandas as pd
from PIL import Image
from tqdm import tqdm

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.environment import get_environment_config

# =========================
# CONFIG: ViNDr-PCXR paths (environment-aware)
# =========================
config = get_environment_config()

ROOT = Path(config['image_root'])

TRAIN_CSV = ROOT / "annotations_train.csv"
TEST_CSV  = ROOT / "annotations_test.csv"

# These should point to the PNGs your training code will load
TRAIN_IMG_DIR = ROOT / "train"   # expects <image_id>.png
TEST_IMG_DIR  = ROOT / "test"
IMAGE_EXT = ".png"

# Where to save COCO JSONs used by your pipeline
OUT_DIR = Path(config['anno_dir'])

# In Colab, also save to Google Drive for persistence
if config['is_colab']:
    DRIVE_OUT_DIR = Path('/content/drive/MyDrive/chest-xray-annotations')
else:
    DRIVE_OUT_DIR = None

TRAIN_COCO_JSON = OUT_DIR / "train_annotations_coco.json"
TEST_COCO_JSON  = OUT_DIR / "test_annotations_coco.json"
COMBINED_COCO_JSON = OUT_DIR / "annotations_coco_all.json"  # optional

# Cache for image dimensions
DIMENSION_CACHE_FILE = OUT_DIR / "image_dimensions_cache.pkl"

# =========================
# Helpers
# =========================
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _read_csv_any_delim(csv_path: Path) -> pd.DataFrame:
    """Read CSV/TSV with flexible columns; normalize names and bbox fields."""
    df = pd.read_csv(csv_path, sep=None, engine="python")
    # normalize header names
    rename_map = {c: c.strip() for c in df.columns}
    df.rename(columns=rename_map, inplace=True)
    cols = {c.lower(): c for c in df.columns}

    # image id column
    img_col = None
    for k in ("image_id", "img_id", "image", "filename"):
        if k in cols:
            img_col = cols[k]; break
    if img_col is None:
        raise ValueError(f"{csv_path}: missing an image id column")

    # class name column
    cls_col = None
    for k in ("class_name", "label", "class", "category", "finding_name", "disease"):
        if k in cols:
            cls_col = cols[k]; break
    if cls_col is None:
        raise ValueError(f"{csv_path}: missing a class/category column")

    # bbox columns: allow (x,y,w,h) or (x_min,y_min,x_max,y_max)
    if all(k in cols for k in ("x", "y", "w", "h")):
        x, y, w, h = (cols["x"], cols["y"], cols["w"], cols["h"])
        df["_x_min"] = df[x].astype(float)
        df["_y_min"] = df[y].astype(float)
        df["_x_max"] = df["_x_min"] + df[w].astype(float)
        df["_y_max"] = df["_y_min"] + df[h].astype(float)
    elif all(k in cols for k in ("x_min", "y_min", "x_max", "y_max")):
        df["_x_min"] = df[cols["x_min"]].astype(float)
        df["_y_min"] = df[cols["y_min"]].astype(float)
        df["_x_max"] = df[cols["x_max"]].astype(float)
        df["_y_max"] = df[cols["y_max"]].astype(float)
    else:
        raise ValueError(f"{csv_path}: bbox columns must be (x,y,w,h) or (x_min,y_min,x_max,y_max)")

    # keep only what we need in a normalized frame
    out = pd.DataFrame({
        "image_id": df[img_col].astype(str),
        "class_name": df[cls_col].astype(str),
        "x_min": df["_x_min"],
        "y_min": df["_y_min"],
        "x_max": df["_x_max"],
        "y_max": df["_y_max"],
    })
    return out

def _img_size(img_dir: Path, image_id: str, ext: str = IMAGE_EXT, cache: dict = None) -> Optional[Tuple[int, int]]:
    """Return (width, height) if the image exists and is readable; else None. Uses cache if provided."""
    if cache is not None and image_id in cache:
        return cache[image_id]
    
    p = img_dir / f"{image_id}{ext}"
    if not p.exists():
        return None
    try:
        with Image.open(p) as im:
            size = im.size  # (W, H)
            if cache is not None:
                cache[image_id] = size
            return size
    except Exception:
        return None

def _load_dimension_cache() -> dict:
    """Load cached image dimensions from disk."""
    if DIMENSION_CACHE_FILE.exists():
        print(f"[INFO] Loading image dimensions from cache: {DIMENSION_CACHE_FILE}")
        try:
            with open(DIMENSION_CACHE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[WARNING] Failed to load cache: {e}. Will rebuild cache.")
    return {}

def _save_dimension_cache(cache: dict):
    """Save image dimensions cache to disk."""
    _ensure_dir(DIMENSION_CACHE_FILE.parent)
    try:
        with open(DIMENSION_CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
        print(f"[INFO] Saved image dimensions cache: {DIMENSION_CACHE_FILE}")
    except Exception as e:
        print(f"[WARNING] Failed to save cache: {e}")

def _build_categories_union(train_df: pd.DataFrame, test_df: pd.DataFrame) -> List[Dict]:
    """Consistent categories across splits: union of class_name values, sorted."""
    classes = sorted(set(train_df["class_name"].unique()) | set(test_df["class_name"].unique()))
    return [{"id": i + 1, "name": cls} for i, cls in enumerate(classes)]

def _cat_id_lookup(categories: List[Dict]) -> Dict[str, int]:
    return {c["name"]: c["id"] for c in categories}

def _clip_bbox(x0: float, y0: float, x1: float, y1: float, W: int, H: int) -> Tuple[float, float, float, float]:
    x0 = max(0.0, min(x0, W))
    y0 = max(0.0, min(y0, H))
    x1 = max(0.0, min(x1, W))
    y1 = max(0.0, min(y1, H))
    # ensure x1>=x0, y1>=y0 after clipping
    if x1 < x0: x0, x1 = x1, x0
    if y1 < y0: y0, y1 = y1, y0
    return x0, y0, x1, y1

def _df_to_coco(
    df: pd.DataFrame,
    img_dir: Path,
    categories: List[Dict],
    image_ext: str = IMAGE_EXT,
    dimension_cache: dict = None
) -> Dict:
    """
    Convert one split to a COCO dict (detection):
      images: [{id, file_name, width, height}]
      annotations: [{id, image_id, category_id, bbox[x,y,w,h], area, iscrowd}]
      categories: provided list
    """
    cat_name_to_id = _cat_id_lookup(categories)

    # Build image entries with sizes; skip images we can't find
    images: List[Dict] = []
    image_id_map: Dict[str, int] = {}
    next_img_id = 1

    # unique images present in this split
    unique_images = df["image_id"].drop_duplicates().tolist()
    print(f"[INFO] Processing {len(unique_images)} unique images...")
    
    for img_id in tqdm(unique_images, desc="Processing images"):
        size = _img_size(img_dir, img_id, ext=image_ext, cache=dimension_cache)
        if size is None:
            # skip images not present; prevents invalid COCO
            continue
        W, H = size
        image_id_map[img_id] = next_img_id
        images.append({
            "id": next_img_id,
            "file_name": f"{img_id}{image_ext}",
            "width": int(W),
            "height": int(H),
        })
        next_img_id += 1

    # Annotations (clip to image boundaries; drop degenerate/empty)
    print(f"[INFO] Processing {len(df)} annotations...")
    anns: List[Dict] = []
    ann_id = 1
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Processing annotations"):
        img_id_str = str(r["image_id"])
        if img_id_str not in image_id_map:
            continue  # image was skipped (missing file)
        img_id = image_id_map[img_id_str]

        # fetch image size to clip bbox
        W = next(im["width"] for im in images if im["id"] == img_id)
        H = next(im["height"] for im in images if im["id"] == img_id)

        try:
            x0, y0 = float(r["x_min"]), float(r["y_min"])
            x1, y1 = float(r["x_max"]), float(r["y_max"])
        except Exception:
            continue

        x0, y0, x1, y1 = _clip_bbox(x0, y0, x1, y1, W, H)
        w, h = x1 - x0, y1 - y0
        if w <= 0.0 or h <= 0.0:
            continue

        cname = str(r["class_name"]).strip()
        if cname not in cat_name_to_id:
            continue
        cid = cat_name_to_id[cname]

        anns.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": cid,
            "bbox": [float(x0), float(y0), float(w), float(h)],  # COCO: [x,y,w,h] in pixels
            "area": float(w * h),
            "iscrowd": 0,
        })
        ann_id += 1

    return {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": anns,
        "categories": categories,
    }

def _copy_to_drive(output_file: Path):
    """Copy JSON file to Google Drive if in Colab."""
    if DRIVE_OUT_DIR and DRIVE_OUT_DIR != output_file.parent:
        try:
            _ensure_dir(DRIVE_OUT_DIR)
            drive_file = DRIVE_OUT_DIR / output_file.name
            # Read and write to copy
            with open(output_file, 'r') as src:
                data = json.load(src)
            with open(drive_file, 'w') as dst:
                json.dump(data, dst, indent=2)
            print(f"[OK] Copied to Google Drive: {drive_file}")
        except Exception as e:
            print(f"[WARNING] Failed to copy to Drive: {e}")

# =========================
# Main
# =========================
def main(skip_if_exists=True):
    """
    Main entry point for COCO converter.
    
    Args:
        skip_if_exists: If True, skip conversion if output files already exist
    """
    # Check if output files already exist
    if skip_if_exists and TRAIN_COCO_JSON.exists() and TEST_COCO_JSON.exists():
        print("[INFO] Output JSON files already exist!")
        print(f"  {TRAIN_COCO_JSON}")
        print(f"  {TEST_COCO_JSON}")
        print("[INFO] Skipping conversion. Using existing files.")
        print("[INFO] To force regeneration, call with skip_if_exists=False")
        return
    
    print(f"[INFO] Reading: {TRAIN_CSV}")
    if not TRAIN_CSV.exists():
        print(f"[ERROR] Train CSV not found: {TRAIN_CSV}")
        return
    train_df = _read_csv_any_delim(TRAIN_CSV)

    print(f"[INFO] Reading: {TEST_CSV}")
    if not TEST_CSV.exists():
        print(f"[ERROR] Test CSV not found: {TEST_CSV}")
        return
    test_df = _read_csv_any_delim(TEST_CSV)

    # Load dimension cache
    dimension_cache = _load_dimension_cache()
    print(f"[INFO] Cache contains {len(dimension_cache)} image dimensions")

    categories = _build_categories_union(train_df, test_df)
    print(f"[INFO] Categories ({len(categories)}): {[c['name'] for c in categories]}")

    print("[INFO] Building COCO for TRAIN …")
    coco_train = _df_to_coco(train_df, TRAIN_IMG_DIR, categories, image_ext=IMAGE_EXT, dimension_cache=dimension_cache)

    print("[INFO] Building COCO for TEST …")
    coco_test  = _df_to_coco(test_df,  TEST_IMG_DIR,  categories, image_ext=IMAGE_EXT, dimension_cache=dimension_cache)

    # Save dimension cache for next time
    _save_dimension_cache(dimension_cache)

    _ensure_dir(OUT_DIR)
    with open(TRAIN_COCO_JSON, "w") as f:
        json.dump(coco_train, f, indent=2)
    print(f"[OK] Wrote {TRAIN_COCO_JSON}")
    _copy_to_drive(TRAIN_COCO_JSON)
    
    with open(TEST_COCO_JSON, "w") as f:
        json.dump(coco_test, f, indent=2)
    print(f"[OK] Wrote {TEST_COCO_JSON}")
    _copy_to_drive(TEST_COCO_JSON)

    # Optional combined file with unique IDs
    print("[INFO] Building combined COCO file...")
    combined = {"info": {}, "licenses": [], "images": [], "annotations": [], "categories": categories}
    def _append_split(coco_split):
        img_id_map = {}
        for im in coco_split["images"]:
            new_id = len(combined["images"]) + 1
            img_id_map[im["id"]] = new_id
            im2 = dict(im); im2["id"] = new_id
            combined["images"].append(im2)
        for an in coco_split["annotations"]:
            an2 = dict(an); an2["id"] = len(combined["annotations"]) + 1
            an2["image_id"] = img_id_map.get(an["image_id"], an["image_id"])
            combined["annotations"].append(an2)

    _append_split(coco_train)
    _append_split(coco_test)
    with open(COMBINED_COCO_JSON, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"[OK] Wrote combined: {COMBINED_COCO_JSON}")
    _copy_to_drive(COMBINED_COCO_JSON)
    
    print("\n" + "="*60)
    print("COCO CONVERSION COMPLETE")
    print("="*60)
    print(f"Train images: {len(coco_train['images'])}, annotations: {len(coco_train['annotations'])}")
    print(f"Test images:  {len(coco_test['images'])}, annotations: {len(coco_test['annotations'])}")
    print(f"Categories:   {len(categories)}")
    print("="*60)

if __name__ == "__main__":
    main()
