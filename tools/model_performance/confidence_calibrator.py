"""
confidence_calibrator.py

Calibrate a YOLOv8 model's confidence scores using temperature scaling,
evaluated against YOLO-format ground-truth labels.

CLI:
    python tools/model_performance/confidence_calibrator.py \
      --model models/baseline_v1/weights/best.pt \
      --val-dir data/yolo_format/val/images \
      --label-dir data/yolo_format/val/labels
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV_PATH = Path("/Users/kahlil/satellite-project/.env")
load_dotenv(dotenv_path=ENV_PATH)

import os  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, (xa2 - xa1) * (ya2 - ya1))
    area_b = max(0.0, (xb2 - xb1) * (yb2 - yb1))
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _yolo_to_xyxy(cx: float, cy: float, w: float, h: float,
                  img_w: int, img_h: int) -> list[float]:
    """Convert YOLO normalised [cx, cy, w, h] to pixel [x1, y1, x2, y2]."""
    px = cx * img_w
    py = cy * img_h
    pw = w * img_w
    ph = h * img_h
    return [px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2]


def _load_ground_truth(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    """Parse a YOLO label file into a list of {cls, box} dicts."""
    boxes: list[dict] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])
        box = _yolo_to_xyxy(cx, cy, bw, bh, img_w, img_h)
        boxes.append({"cls": cls, "box": box})
    return boxes


# ---------------------------------------------------------------------------
# Calibration maths
# ---------------------------------------------------------------------------

def _compute_ece(confs: np.ndarray, labels: np.ndarray,
                 n_bins: int = 10) -> float:
    """
    Expected Calibration Error across n_bins equal-width confidence bins.
    ECE = sum_b (|frac_positive_b - mean_conf_b| * frac_samples_b)
    """
    if len(confs) == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confs)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confs >= lo) & (confs < hi)
        if i == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        bin_confs = confs[mask]
        bin_labels = labels[mask]
        mean_conf = bin_confs.mean()
        frac_positive = bin_labels.mean()
        frac_in_bin = mask.sum() / n
        ece += abs(frac_positive - mean_conf) * frac_in_bin

    return float(ece)


def _bin_stats(confs: np.ndarray, labels: np.ndarray,
               n_bins: int = 10) -> tuple[list[float], list[float]]:
    """
    Returns (mean_conf_per_bin, accuracy_per_bin) for plotting.
    Bins with no samples are excluded.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mean_confs: list[float] = []
    accuracies: list[float] = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confs >= lo) & (confs < hi)
        if i == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        mean_confs.append(float(confs[mask].mean()))
        accuracies.append(float(labels[mask].mean()))

    return mean_confs, accuracies


def _find_optimal_temperature(confs: np.ndarray, labels: np.ndarray,
                               t_min: float = 0.1, t_max: float = 3.0,
                               t_step: float = 0.05,
                               n_bins: int = 10) -> tuple[float, float]:
    """
    Grid-search over temperatures in [t_min, t_max] with step t_step.
    Returns (T*, ECE_at_T*).
    """
    temperatures = np.arange(t_min, t_max + t_step / 2, t_step)
    best_T = 1.0
    best_ece = float("inf")

    for T in tqdm(temperatures, desc="Searching temperature", unit="T"):
        calibrated = np.clip(confs / T, 0.0, 1.0)
        ece = _compute_ece(calibrated, labels, n_bins)
        if ece < best_ece:
            best_ece = ece
            best_T = float(T)

    return best_T, best_ece


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_predictions(model_path: Path, val_dir: Path,
                         label_dir: Path,
                         iou_thresh: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference over all val images and return (confidences, is_correct) arrays.
    is_correct[i] = 1 if the i-th prediction has IoU >= iou_thresh with a GT box.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    try:
        import cv2  # noqa: F401
    except ImportError:
        print("❌ opencv-python is not installed. Run: pip install opencv-python")
        sys.exit(1)

    import cv2  # noqa: F811

    print(f"📊 Loading model from {model_path}")
    try:
        model = YOLO(str(model_path))
    except Exception as exc:
        print(f"❌ Failed to load model: {exc}")
        sys.exit(1)

    image_paths = sorted(
        p for p in val_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )
    if not image_paths:
        print(f"❌ No images found in {val_dir}")
        sys.exit(1)

    print(f"📊 Collecting predictions from {len(image_paths)} images")

    all_confs: list[float] = []
    all_correct: list[int] = []

    for img_path in tqdm(image_paths, desc="Collecting predictions", unit="img"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Could not read: {img_path.name}")
            continue
        img_h, img_w = img.shape[:2]

        label_path = label_dir / (img_path.stem + ".txt")
        gts = _load_ground_truth(label_path, img_w, img_h)

        try:
            results = model(str(img_path), verbose=False)
        except Exception as exc:
            print(f"⚠️ Inference failed for {img_path.name}: {exc}")
            continue

        matched_gt: set[int] = set()

        for r in results:
            if r.boxes is None:
                continue
            for box_data in r.boxes:
                x1, y1, x2, y2 = box_data.xyxy[0].tolist()
                conf = float(box_data.conf[0])
                cls = int(box_data.cls[0])

                best_iou = 0.0
                best_idx = -1
                for i, gt in enumerate(gts):
                    if i in matched_gt:
                        continue
                    if gt["cls"] != cls:
                        continue
                    iou = _compute_iou([x1, y1, x2, y2], gt["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i

                is_correct = 0
                if best_iou >= iou_thresh:
                    is_correct = 1
                    matched_gt.add(best_idx)

                all_confs.append(conf)
                all_correct.append(is_correct)

    if not all_confs:
        print("❌ No predictions collected – cannot calibrate.")
        sys.exit(1)

    print(f"📊 Collected {len(all_confs)} predictions "
          f"({sum(all_correct)} correct, {len(all_confs) - sum(all_correct)} incorrect)")

    return np.array(all_confs, dtype=np.float32), np.array(all_correct, dtype=np.float32)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _generate_calibration_curve(
    confs: np.ndarray,
    labels: np.ndarray,
    opt_T: float,
    output_path: Path,
    n_bins: int = 10,
) -> None:
    """
    Scatter plot of mean predicted confidence vs actual accuracy per bin,
    shown before and after temperature scaling, plus a perfect-calibration diagonal.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calibrated_confs = np.clip(confs / opt_T, 0.0, 1.0)

    mc_before, acc_before = _bin_stats(confs, labels, n_bins)
    mc_after, acc_after = _bin_stats(calibrated_confs, labels, n_bins)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")

    # Before calibration
    ax.scatter(mc_before, acc_before, s=80, color="tomato", zorder=5,
               label="Before calibration")
    ax.plot(mc_before, acc_before, color="tomato", linewidth=1.2, alpha=0.6)

    # After calibration
    ax.scatter(mc_after, acc_after, s=80, color="steelblue", marker="^", zorder=5,
               label=f"After calibration (T={opt_T:.2f})")
    ax.plot(mc_after, acc_after, color="steelblue", linewidth=1.2, alpha=0.6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Actual Accuracy (Fraction Correct)", fontsize=12)
    ax.set_title("Calibration Curve – Before vs After Temperature Scaling", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"✅ Saved calibration curve → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate YOLOv8 confidence scores via temperature scaling"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("MODEL_PATH", "models/baseline_v1/weights/best.pt")),
        help="Path to YOLOv8 .pt weights file",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        required=True,
        help="Directory containing validation images",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        required=True,
        help="Directory containing YOLO-format label .txt files",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.5,
        help="IoU threshold for correctness (default: 0.5)",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of calibration bins for ECE (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save output files (defaults to --val-dir parent)",
    )
    args = parser.parse_args()

    # Validate paths
    if not args.model.exists():
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)
    if not args.val_dir.is_dir():
        print(f"❌ Validation directory not found: {args.val_dir}")
        sys.exit(1)
    if not args.label_dir.is_dir():
        print(f"❌ Label directory not found: {args.label_dir}")
        sys.exit(1)

    output_dir = args.output_dir or args.val_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve models dir relative to model path for saving calibration JSON
    models_dir = args.model.parent.parent  # e.g. models/baseline_v1 -> models/
    models_dir.mkdir(parents=True, exist_ok=True)

    # Collect raw predictions
    confs, labels = collect_predictions(
        args.model, args.val_dir, args.label_dir, args.iou_thresh
    )

    # ECE before calibration (T=1)
    ece_before = _compute_ece(confs, labels, args.n_bins)
    print(f"📊 ECE before calibration: {ece_before:.4f}")

    # Find optimal temperature
    opt_T, ece_after = _find_optimal_temperature(
        confs, labels, t_min=0.1, t_max=3.0, t_step=0.05, n_bins=args.n_bins
    )
    print(f"✅ Optimal temperature T* = {opt_T:.4f}  ECE after = {ece_after:.4f}")

    # Calibration curve
    _generate_calibration_curve(
        confs, labels, opt_T,
        output_dir / "calibration_curve.png",
        n_bins=args.n_bins,
    )

    # Save calibration temperature
    calibration_data = {
        "temperature": round(opt_T, 4),
        "ece_before": round(ece_before, 6),
        "ece_after": round(ece_after, 6),
    }
    cal_json_path = models_dir / "calibration_temperature.json"
    cal_json_path.write_text(json.dumps(calibration_data, indent=2))
    print(f"✅ Saved calibration temperature → {cal_json_path}")

    # Recommendation
    improvement = ece_before - ece_after
    if improvement > 0.01 and opt_T != 1.0:
        recommendation = (
            f"Calibration recommended. Apply temperature T={opt_T:.2f} to "
            f"reduce ECE from {ece_before:.4f} to {ece_after:.4f} "
            f"(improvement: {improvement:.4f})."
        )
    elif abs(opt_T - 1.0) < 0.05:
        recommendation = (
            f"Model is already well-calibrated (ECE={ece_before:.4f}). "
            "Temperature scaling provides negligible benefit."
        )
    else:
        recommendation = (
            f"Marginal calibration benefit. T={opt_T:.2f} reduces ECE by "
            f"{improvement:.4f}. Apply if precision matters for downstream use."
        )

    # Summary printout
    print("\n" + "=" * 55)
    print("📊 Confidence Calibration Summary")
    print("=" * 55)
    print(f"  Optimal temperature T*:  {opt_T:.4f}")
    print(f"  ECE before calibration:  {ece_before:.6f}")
    print(f"  ECE after  calibration:  {ece_after:.6f}")
    print(f"  ECE improvement:         {improvement:.6f}")
    print(f"\n  Recommendation: {recommendation}")
    print("=" * 55)


if __name__ == "__main__":
    main()
