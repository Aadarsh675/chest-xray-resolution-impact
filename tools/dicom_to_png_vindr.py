# dicom_to_png_vindr.py
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut

# ---- CONFIG ----
ROOT = Path("/content/drive/MyDrive/vindr_pcxr/physionet.org")
SRC_TRAIN = ROOT / "train"
SRC_TEST  = ROOT / "test"
DST_TRAIN = ROOT / "train (python)"
DST_TEST  = ROOT / "test (python)"

VALID_EXTS = {".dcm", ".dicom", ""}  # some sets omit extension

# ---- Helpers ----
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _ds_to_uint8(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    """Convert a DICOM dataset to an 8-bit grayscale ndarray with proper windowing and MONOCHROME1 handling."""
    arr = ds.pixel_array

    # Apply Modality LUT (e.g., Rescale Slope/Intercept) if present
    try:
        arr = apply_modality_lut(arr, ds)
    except Exception:
        pass  # safe to ignore if not applicable

    # Apply VOI LUT / windowing if present
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        # fall back to simple min-max if no VOI LUT
        arr = arr.astype(np.float32)

    # Convert to float for normalization
    arr = arr.astype(np.float32)

    # Invert for MONOCHROME1 (black/white reversed)
    photometric = getattr(ds, "PhotometricInterpretation", "").upper()
    if photometric == "MONOCHROME1":
        arr = np.max(arr) - arr

    # Robust min-max scale to [0,255]
    vmin = np.percentile(arr, 0.5)
    vmax = np.percentile(arr, 99.5)
    if vmax <= vmin:
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
    if vmax > vmin:
        arr = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    arr = (arr * 255.0).round().astype(np.uint8)
    return arr

def convert_folder(src_dir: Path, dst_dir: Path) -> None:
    _ensure_dir(dst_dir)

    # Collect candidate files (recursive, typical ViNDr layout is flat)
    cand = []
    for p in src_dir.rglob("*"):
        if p.is_file():
            if p.suffix.lower() in VALID_EXTS or p.suffix.lower() in {".dcm", ".dicom"}:
                cand.append(p)

    if not cand:
        print(f"[WARN] No DICOM files found under {src_dir}")
        return

    print(f"[INFO] Converting {len(cand)} files from {src_dir} -> {dst_dir}")
    for src in tqdm(cand, ncols=80):
        # Output filename: keep stem, write PNG
        # If source has no extension, .png will just be appended.
        out_name = src.stem + ".png"
        dst = dst_dir / out_name

        # Skip if already exists
        if dst.exists():
            continue

        try:
            ds = pydicom.dcmread(str(src), force=True)
            # Some DICOMs are encapsulated; ensure pixel data exists
            _ = ds.pixel_array  # triggers decode
        except Exception as e:
            print(f"[skip] Cannot read {src}: {e}")
            continue

        try:
            img8 = _ds_to_uint8(ds)
            Image.fromarray(img8).save(str(dst), format="PNG")
        except Exception as e:
            print(f"[skip] Failed to convert {src}: {e}")

def main():
    # Basic checks
    for p in [ROOT, SRC_TRAIN, SRC_TEST]:
        if not p.exists():
            print(f"[ERROR] Missing path: {p}")
    _ensure_dir(DST_TRAIN)
    _ensure_dir(DST_TEST)

    convert_folder(SRC_TRAIN, DST_TRAIN)
    convert_folder(SRC_TEST,  DST_TEST)

    print("\n✅ Done. PNGs saved to:")
    print(f"   - {DST_TRAIN}")
    print(f"   - {DST_TEST}")

if __name__ == "__main__":
    # Allow running via: %run dicom_to_png_vindr.py
    main()
