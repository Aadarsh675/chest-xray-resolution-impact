# plot_curves.py
import os
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# I/O helpers
# ----------------------------
def find_scales(analysis_root: str) -> List[str]:
    out = []
    if not os.path.isdir(analysis_root):
        return out
    for d in sorted(os.listdir(analysis_root)):
        full = os.path.join(analysis_root, d)
        if os.path.isdir(full) and d.startswith("scale_"):
            out.append(full)
    return out

def find_repeats(scale_dir: str) -> List[str]:
    reps = []
    for d in sorted(os.listdir(scale_dir)):
        full = os.path.join(scale_dir, d)
        if os.path.isdir(full) and d.startswith("rep_"):
            reps.append(full)
    return reps

def load_curves_json(curves_path: str) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    with open(curves_path, "r") as f:
        data = json.load(f)

    if "thresholds" in data:
        thresholds = np.array(data["thresholds"], dtype=float)
    else:
        raise ValueError(f"Unable to find 'thresholds' in {curves_path}")

    if "per_class" in data and isinstance(data["per_class"], dict):
        per_class = {}
        for cls, metrics in data["per_class"].items():
            per_class[cls] = {
                "precision": np.array(metrics.get("precision", []), dtype=float),
                "recall":    np.array(metrics.get("recall",    []), dtype=float),
                "miou":      np.array(metrics.get("miou",      []), dtype=float),
            }
        return thresholds, per_class

    if "classes" in data:
        per_class = {}
        classes = data["classes"]
        for cls in classes:
            per_class[cls] = {
                "precision": np.array(data.get("precision", {}).get(cls, []), dtype=float),
                "recall":    np.array(data.get("recall", {}).get(cls, []), dtype=float),
                "miou":      np.array(data.get("miou", {}).get(cls, []), dtype=float),
            }
        return thresholds, per_class

    raise ValueError(f"Unsupported curves.json format in {curves_path}")

# ----------------------------
# Aggregation
# ----------------------------
def aggregate_repeats(
    thresholds_list: List[np.ndarray],
    per_class_list: List[Dict[str, Dict[str, np.ndarray]]]
) -> Tuple[np.ndarray, Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    base = thresholds_list[0]
    for th in thresholds_list[1:]:
        if len(th) != len(base) or np.max(np.abs(th - base)) > 1e-9:
            raise ValueError("Threshold arrays differ across repeats; cannot aggregate safely.")

    from collections import defaultdict as _dd
    stash: Dict[str, Dict[str, List[np.ndarray]]] = _dd(lambda: _dd(list))
    for per_class in per_class_list:
        for cls, metrics in per_class.items():
            for mname, arr in metrics.items():
                if arr.size == 0:
                    continue
                stash[cls][mname].append(arr)

    agg: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for cls, m_dict in stash.items():
        agg[cls] = {}
        for mname, series in m_dict.items():
            M = np.stack(series, axis=0)  # repeats x T
            agg[cls][mname] = {
                "mean": np.nanmean(M, axis=0),
                "std":  np.nanstd(M, axis=0, ddof=0),
            }

    return base, agg

# ----------------------------
# Plotting
# ----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def plot_per_class_three_panel(
    thresholds: np.ndarray,
    agg_per_class: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    out_dir: str,
    dpi: int = 140,
    show: bool = False,   # <<< NEW
) -> None:
    """
    Save one 3-panel figure per class: Precision, Recall, mIoU vs threshold (mean ± std).
    If show=True, also display inline (useful in Google Colab).
    """
    ensure_dir(out_dir)
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("miou", "mIoU")]

    for cls in sorted(agg_per_class.keys()):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for ax, (mkey, mlabel) in zip(axes, metrics):
            if mkey not in agg_per_class[cls]:
                ax.set_title(f"{mlabel} (no data)")
                ax.set_xlabel("Confidence threshold")
                ax.set_ylabel(mlabel)
                ax.grid(True, alpha=0.3)
                continue

            mean = agg_per_class[cls][mkey]["mean"]
            std  = agg_per_class[cls][mkey]["std"]

            ax.plot(thresholds, mean, label=f"{mlabel}")
            ax.fill_between(thresholds, mean - std, mean + std, alpha=0.2)
            ax.set_title(f"{cls} — {mlabel}")
            ax.set_xlabel("Confidence threshold")
            ax.set_ylabel(mlabel)
            ax.set_xlim(thresholds.min(), thresholds.max())
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"{cls} — Precision / Recall / mIoU vs Confidence", y=1.04, fontsize=12)
        out_path = os.path.join(out_dir, f"{cls.replace(' ', '_')}_curves.png")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        if show:  # <<< NEW
            plt.show()
        plt.close(fig)

def plot_overview_per_metric(
    thresholds: np.ndarray,
    agg_per_class: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    out_dir: str,
    dpi: int = 140,
    show: bool = False,   # <<< NEW
) -> None:
    """
    Optional: Save one figure per metric overlaying all classes (mean curves).
    If show=True, also display inline (useful in Google Colab).
    """
    ensure_dir(out_dir)
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("miou", "mIoU")]
    for mkey, mlabel in metrics:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        for cls in sorted(agg_per_class.keys()):
            if mkey not in agg_per_class[cls]:
                continue
            mean = agg_per_class[cls][mkey]["mean"]
            ax.plot(thresholds, mean, label=cls)
        ax.set_title(f"{mlabel} vs Confidence — All Classes")
        ax.set_xlabel("Confidence threshold")
        ax.set_ylabel(mlabel)
        ax.set_xlim(thresholds.min(), thresholds.max())
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=2)
        out_path = os.path.join(out_dir, f"ALLCLASSES_{mkey}_overview.png")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        if show:  # <<< NEW
            plt.show()
        plt.close(fig)

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot Precision/Recall/mIoU vs confidence for each class per resolution.")
    parser.add_argument("--analysis_root", type=str, default="analysis",
                        help="Root directory containing scale_* directories.")
    parser.add_argument("--make_overview", action="store_true",
                        help="Also produce per-metric overview plots with all classes.")
    parser.add_argument("--dpi", type=int, default=140, help="Figure DPI.")
    parser.add_argument("--show", action="store_true",  # <<< NEW
                        help="Display plots inline (use in Google Colab / notebooks).")
    args = parser.parse_args()

    scale_dirs = find_scales(args.analysis_root)
    if not scale_dirs:
        print(f"[!] No scale_* directories found under {args.analysis_root}")
        return

    for scale_dir in scale_dirs:
        pct = os.path.basename(scale_dir).replace("scale_", "")  # e.g., "100", "50"
        rep_dirs = find_repeats(scale_dir)
        if not rep_dirs:
            print(f"[skip] No rep_* found in {scale_dir}")
            continue

        thresholds_list = []
        per_class_list = []
        for rep_dir in rep_dirs:
            curves_path = os.path.join(rep_dir, "curves.json")
            if not os.path.exists(curves_path):
                print(f"[warn] Missing {curves_path}, skipping this repeat.")
                continue
            try:
                th, pc = load_curves_json(curves_path)
                thresholds_list.append(th)
                per_class_list.append(pc)
            except Exception as e:
                print(f"[warn] Failed to read {curves_path}: {e}")

        if not per_class_list:
            print(f"[skip] No valid curves for {scale_dir}")
            continue

        thresholds, agg = aggregate_repeats(thresholds_list, per_class_list)

        out_dir = os.path.join(scale_dir, "plots")
        plot_per_class_three_panel(thresholds, agg, out_dir=out_dir, dpi=args.dpi, show=args.show)
        if args.make_overview:
            plot_overview_per_metric(thresholds, agg, out_dir=out_dir, dpi=args.dpi, show=args.show)

        print(f"[OK] Saved plots for scale={pct}% → {out_dir}")

if __name__ == "__main__":
    main()
