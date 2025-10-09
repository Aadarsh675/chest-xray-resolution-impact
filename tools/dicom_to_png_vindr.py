# dicom_to_png_vindr.py
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from PIL import Image
from tqdm import tqdm

import pydicom
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut
import pydicom.pixel_data_handlers as pdh

# ---- CONFIG ----
# Adjust ROOT if your mount is different
ROOT = Path("/content/drive/MyDrive/vindr_pcxr/physionet.org")
SRC_TRAIN = ROOT / "train"
SRC_TEST  = ROOT / "test"
DST_TRAIN = ROOT / "train (python)"
DST_TEST  = ROOT / "test (python)"

# If True, mirror the source tree under the destination (avoids name collisions)
PRESERVE_SUBDIRS = True

# How many parallel worker processes to use
WORKERS = 6  # try 4–8 on Colab depending on CPU

# Stage PNGs to fast local storage, then copy to Drive at the end
STAGE_LOCAL = True
LOCAL_STAGE_ROOT = Path("/content/tmp_pngs")

# Some ViNDr files use .dicom, and some may lack extension
VALID_EXTS = {".dcm", ".dicom", ""}


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
                  "For compressed DICOMs install: "
                  "pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg gdcm")
    except Exception as e:
        print(f"[WARN] Could not query handlers: {e}")

def _ds_to_uint8(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Convert a DICOM dataset to an 8-bit grayscale ndarray with proper
    Modality LUT, VOI LUT, and MONOCHROME1 inversion.
    """
    # Decode via pixel_array (this will engage pylibjpeg/gdcm if installed)
    arr = ds.pixel_array

    # Modality LUT (rescale slope/intercept), safe if absent
    try:
        arr = apply_modality_lut(arr, ds)
    except Exception:
        pass

    # VOI LUT / Windowing if present
    try:
        arr = apply_voi_lut(arr, ds).astype(np.float32)
    except Exception:
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
    for p in src_dir.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in VALID_EXTS:
            yield p

def _dst_path_for(src: Path, src_root: Path, dst_root: Path, preserve_tree: bool) -> Path:
    """
    Compute destination PNG path for a given source DICOM.
    If preserve_tree=True, mirror the relative folder tree from src_root under dst_root.
    """
    if preserve_tree:
        rel = src.parent.relative_to(src_root) if src.parent != src_root else Path()
        return (dst_root / rel / f"{src.stem}.png")
    else:
        return (dst_root / f"{src.stem}.png")

def _convert_one(src_path: str, dst_path: str) -> Tuple[str, str, Optional[str]]:
    """
    Convert one DICOM -> PNG.
    Returns (src_path, dst_path, error_msg_or_None).
    Designed to run safely inside a process pool.
    """
    try:
        ds = dcmread(src_path, force=True)
    except Exception as e:
        return (src_path, dst_path, f"read/decode error: {e}")
    try:
        img8 = _ds_to_uint8(ds)
        Image.fromarray(img8).save(dst_path, format="PNG")
        return (src_path, dst_path, None)
    except Exception as e:
        return (src_path, dst_path, f"convert error: {e}")

def convert_folder(
    src_dir: Path,
    dst_dir: Path,
    *,
    workers: int = WORKERS,
    stage_local: bool = STAGE_LOCAL,
    preserve_tree: bool = PRESERVE_SUBDIRS,
) -> None:
    """
    Convert all DICOMs under src_dir to PNGs in dst_dir.
    - Skips files that already have a corresponding PNG.
    - Optionally writes to local /content/tmp_pngs first for speed, then copies to Drive.
    - Optionally mirrors the source subdirectories under the destination to avoid collisions.
    """
    _ensure_dir(dst_dir)
    files = list(_iter_dicom_files(src_dir))
    if not files:
        print(f"[WARN] No DICOM files found under {src_dir}")
        return

    # Build work list, skipping outputs that already exist
    jobs = []
    for src in files:
        final_dst = _dst_path_for(src, src_root=src_dir, dst_root=dst_dir, preserve_tree=preserve_tree)
        if final_dst.exists():
            continue
        jobs.append((src, final_dst))

    if not jobs:
        print(f"[INFO] All PNGs already exist for {src_dir}")
        return

    # Choose temp (local) output root
    if stage_local:
        local_root = LOCAL_STAGE_ROOT / src_dir.name  # keep train/test separate
        _ensure_dir(local_root)
    else:
        local_root = dst_dir  # write directly to Drive (slower)

    print(f"[INFO] Converting {len(jobs)} files from {src_dir} -> {dst_dir} "
          f"{'(staging to local first)' if stage_local else '(writing directly)'}")

    # Prepare per-file temp destinations, mirroring subdirs if requested
    temp_pairs = []
    for src, final_dst in jobs:
        if stage_local:
            temp_dst = _dst_path_for(src, src_root=src_dir, dst_root=local_root, preserve_tree=preserve_tree)
        else:
            temp_dst = final_dst
        _ensure_dir(Path(temp_dst).parent)
        temp_pairs.append((src, final_dst, Path(temp_dst)))

    # Parallel convert
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = []
        for src, final_dst, temp_dst in temp_pairs:
            futures.append(ex.submit(_convert_one, str(src), str(temp_dst)))
        for fut in tqdm(futures, total=len(futures), ncols=80, desc=src_dir.name):
            src_path, dst_path, err = fut.result()
            if err is not None:
                print(f"[skip] {Path(src_path).name}: {err}")

    # Copy staged PNGs into Drive destination
    if stage_local:
        copied = 0
        for _, final_dst, temp_dst in temp_pairs:
            if temp_dst.exists():
                if not final_dst.exists():
                    _ensure_dir(final_dst.parent)
                    shutil.copy2(temp_dst, final_dst)
                copied += 1
        print(f"[INFO] Copied {copied} staged PNGs to {dst_dir}")

def main():
    # Sanity checks
    missing = [p for p in [ROOT, SRC_TRAIN, SRC_TEST] if not p.exists()]
    if missing:
        for m in missing:
            print(f"[ERROR] Missing path: {m}")
        return

    _ensure_dir(DST_TRAIN)
    _ensure_dir(DST_TEST)

    _check_handlers()

    convert_folder(SRC_TRAIN, DST_TRAIN, workers=WORKERS, stage_local=STAGE_LOCAL, preserve_tree=PRESERVE_SUBDIRS)
    convert_folder(SRC_TEST,  DST_TEST,  workers=WORKERS, stage_local=STAGE_LOCAL, preserve_tree=PRESERVE_SUBDIRS)

    print("\n✅ Done. PNGs saved to:")
    print(f"   - {DST_TRAIN}")
    print(f"   - {DST_TEST}")

if __name__ == "__main__":
    main()
