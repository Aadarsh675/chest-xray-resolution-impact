# dicom_to_png_vindr.py
import os
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
from PIL import Image
from tqdm import tqdm

import pydicom
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut
import pydicom.pixel_data_handlers as pdh

# ---- CONFIG ----
ROOT = Path("/content/drive/MyDrive/vindr_pcxr/physionet.org")
SRC_TRAIN = ROOT / "train"
SRC_TEST  = ROOT / "test"
DST_TRAIN = ROOT / "train (python)"
DST_TEST  = ROOT / "test (python)"

VALID_EXTS = {".dcm", ".dicom", ""}  # some files may lack an extension

# ---- Helpers ----
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _check_handlers() -> None:
    """Print which pixel data handlers are available/enabled (for debugging JPEG Lossless issues)."""
    try:
        avail = [h.__name__ for h in pdh.available_handlers]
        enabled = [h.__name__ for h in pdh.image_handlers]
        print(f"[pydicom] Available handlers: {avail}")
        print(f"[pydicom] Enabled handlers:   {enabled}")
        if not any(n in avail for n in ("pylibjpeg_handler", "GDCMHandler")):
            print("[WARN] Neither pylibjpeg nor GDCM appears available. "
                  "Install: pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg gdcm")
    except Exception as e:
        print(f"[WARN] Could not query handlers: {e}")

def _ds_to_uint8(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Convert a DICOM dataset to an 8-bit grayscale ndarray with proper
    Modality LUT, VOI LUT, and MONOCHROME1 inversion.
    """
    # Trigger handler decode (pylibjpeg/gdcm) via pixel_array
    arr = ds.pixel_array

    # Modality LUT (rescale slope/intercept), safe if absent
    try:
        arr = apply_modality_lut(arr, ds)
    except Exception:
        pass

    # VOI LUT / Windowing if present, else raw
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        arr = arr.astype(np.float32)

    arr = arr.astype(np.float32)

    # Invert for MONOCHROME1 (display convention)
    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper()
    if photometric == "MONOCHROME1":
        arr = np.max(arr) - arr

    # Robust min/max scaling (clip 0.5–99.5 percentiles)
    vmin = np.percentile(arr, 0.5)
    vmax = np.percentile(arr, 99.5)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.min(arr)), float(np.max(arr))

    if vmax > vmin:
        arr = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    return (arr * 255.0).round().astype(np.uint8)

def _iter_dicom_files(src_dir: Path) -> Iterable[Path]:
    # Many ViNDr files use ".dicom"; include extensionless just in case
    for p in src_dir.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in VALID_EXTS or ext in {".dcm", ".dicom"}:
            yield p

def convert_folder(src_dir: Path, dst_dir: Path) -> None:
    _ensure_dir(dst_dir)
    files = list(_iter_dicom_files(src_dir))
    if not files:
        print(f"[WARN] No DICOM files found under {src_dir}")
        return

    print(f"[INFO] Converting {len(files)} files from {src_dir} -> {dst_dir}")
    for src in tqdm(files, ncols=80, desc=f"{src_dir.name}"):
        out_name = src.stem + ".png"
        dst = dst_dir / out_name
        if dst.exists():
            continue

        try:
            # force=True lets us read even with some minor tag issues
            ds = dcmread(str(src), force=True)
            # Accessing pixel_array triggers decode via available handler
            _ = ds.pixel_array
        except Exception as e:
            print(f"[skip] Cannot read/decode {src.name}: {e}")
            continue

        try:
            img8 = _ds_to_uint8(ds)
            Image.fromarray(img8).save(str(dst), format="PNG")
        except Exception as e:
            print(f"[skip] Failed to convert {src.name}: {e}")

def main():
    # Basic path checks
    missing = [p for p in [ROOT, SRC_TRAIN, SRC_TEST] if not p.exists()]
    if missing:
        for m in missing:
            print(f"[ERROR] Missing path: {m}")
        return

    _ensure_dir(DST_TRAIN)
    _ensure_dir(DST_TEST)

    _check_handlers()

    convert_folder(SRC_TRAIN, DST_TRAIN)
    convert_folder(SRC_TEST,  DST_TEST)

    print("\n✅ Done. PNGs saved to:")
    print(f"   - {DST_TRAIN}")
    print(f"   - {DST_TEST}")

if __name__ == "__main__":
    main()
