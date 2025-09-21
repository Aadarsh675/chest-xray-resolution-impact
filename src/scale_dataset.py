import os
import json
from PIL import Image
from pathlib import Path
from typing import Optional, Dict

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _scale_box_xywh(box, sx: float, sy: float):
    x, y, w, h = box
    return [float(x * sx), float(y * sy), float(w * sx), float(h * sy)]

def _resize_image(src_path: str, dst_path: str, sx: float, sy: Optional[float] = None) -> tuple[int, int]:
    """Resize with independent x/y scales (sy defaults to sx). Returns (width, height) of saved image."""
    sy = sx if sy is None else sy
    img = Image.open(src_path).convert("RGB")
    W, H = img.size
    new_W = max(1, int(round(W * sx)))
    new_H = max(1, int(round(H * sy)))
    img = img.resize((new_W, new_H), Image.BICUBIC)
    _ensure_dir(os.path.dirname(dst_path))
    img.save(dst_path, quality=95)
    return new_W, new_H

def _scale_coco_json(in_json: str, out_json: str, img_src_dir: str, img_dst_dir: str, sx: float, sy: Optional[float] = None) -> None:
    """Read a COCO json, scale its images (writing to img_dst_dir) and bbox annotations, write a new json."""
    sy = sx if sy is None else sy
    with open(in_json, "r") as f:
        data = json.load(f)

    # Build map from image_id -> file_name for quick lookup
    id_to_fname: Dict[int, str] = {img["id"]: img["file_name"] for img in data["images"]}

    # Resize all images listed in the json and collect their new sizes
    new_sizes: Dict[int, tuple[int, int]] = {}
    for img in data["images"]:
        fname = img["file_name"]
        src = os.path.join(img_src_dir, fname)
        dst = os.path.join(img_dst_dir, fname)
        new_W, new_H = _resize_image(src, dst, sx, sy)
        img["width"] = new_W
        img["height"] = new_H
        new_sizes[img["id"]] = (new_W, new_H)

    # Scale all annotations’ bboxes
    for ann in data["annotations"]:
        ann["bbox"] = _scale_box_xywh(ann["bbox"], sx, sy)
        # Recompute area (still axis-aligned rectangle)
        x, y, w, h = ann["bbox"]
        ann["area"] = float(max(0.0, w) * max(0.0, h))

    # Write out the scaled JSON
    _ensure_dir(os.path.dirname(out_json))
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)

def prepare_scaled_split(
    split_name: str,
    base_images_dir: str,          # e.g., data/images/train
    base_ann_file: str,            # e.g., data/annotations/train_annotations_coco.json
    out_images_root: str,          # e.g., data/images_s50 (where subdir /train will be made)
    out_ann_file: str,             # e.g., data/annotations/train_annotations_coco_s50.json
    scale: float,
    overwrite: bool = False
) -> None:
    """
    Create a scaled copy of (images, annotations) for a single split (train/val/test).
    Images are written to {out_images_root}/{split_name}. Annotations to out_ann_file.
    """
    dst_img_dir = os.path.join(out_images_root, split_name)
    # Skip if already exists and not overwriting
    if (not overwrite) and os.path.exists(out_ann_file) and os.path.isdir(dst_img_dir):
        print(f"[scale:{split_name}] Reusing existing scaled data at {dst_img_dir} / {out_ann_file}")
        return

    print(f"[scale:{split_name}] Building scaled split at scale={scale} into {dst_img_dir}")
    _ensure_dir(dst_img_dir)
    _scale_coco_json(base_ann_file, out_ann_file, base_images_dir, dst_img_dir, scale, scale)

def prepare_all_scales(
    scales,                         # e.g., [1.0, 0.5, 0.25]
    base_image_root: str,           # e.g., data/images
    base_ann_dir: str,              # e.g., data/annotations
    out_image_root_fmt: str = "data/images_s{pct}",    # pct = int(scale*100)
    out_ann_name_fmt: str = "{split}_annotations_coco_s{pct}.json",
    splits=("train", "test"),       # include "val" if you have it
    overwrite=False
):
    """
    For each scale, create parallel image/annotation sets for each split.
    - Images: data/images_s{pct}/{split}/<fname>
    - Anns:   data/annotations/{split}_annotations_coco_s{pct}.json
    """
    for s in scales:
        pct = int(round(s * 100))
        out_img_root = out_image_root_fmt.format(pct=pct)
        print(f"\n=== Preparing scale={s} ({pct}%) ===")
        for split in splits:
            base_split_dir = os.path.join(base_image_root, split)
            base_ann_file  = os.path.join(base_ann_dir, f"{split}_annotations_coco.json")
            out_ann_file   = os.path.join(base_ann_dir, out_ann_name_fmt.format(split=split, pct=pct))

            if not (os.path.isdir(base_split_dir) and os.path.exists(base_ann_file)):
                print(f"[skip] Missing base {split} images or annotations. ({base_split_dir}, {base_ann_file})")
                continue

            prepare_scaled_split(
                split_name=split,
                base_images_dir=base_split_dir,
                base_ann_file=base_ann_file,
                out_images_root=out_img_root,
                out_ann_file=out_ann_file,
                scale=s,
                overwrite=overwrite
            )
