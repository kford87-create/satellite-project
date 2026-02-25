"""
tools/model_performance/confidence_calibrator.py

Applies temperature scaling to calibrate YOLOv8 confidence scores.
Raw confidence scores are not true probabilities — this fixes that.
A calibrated model's 0.7 confidence actually means ~70% correct.
Critical for the false negative quantification model's accuracy.

Usage:
  python tools/model_performance/confidence_calibrator.py \
    --model models/baseline_v1/weights/best.pt \
    --val-dir data/yolo_format/val
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import argparse
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))


class ConfidenceCalibrator:
    """
    Temperature scaling calibration for YOLOv8 confidence scores.
    Finds optimal temperature T such that confidence/T is well-calibrated.
    Think of temperature as a dial: T>1 softens overconfident scores,
    T<1 sharpens underconfident ones.
    """

    def __init__(self, model_path: str):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.temperature = 1.0  # Will be updated after calibration

    def _collect_predictions(self, img_dir: Path, label_dir: Path, conf: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect raw confidence scores and correctness labels from validation set.
        Returns (confidences, is_correct) arrays.
        """
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        all_confs, all_correct = [], []

        def load_labels(label_path):
            if not label_path.exists():
                return []
            boxes = []
            for line in label_path.read_text().strip().splitlines():
                if line.strip():
                    parts = list(map(float, line.split()))
                    boxes.append({"class_id": int(parts[0]), "bbox": parts[1:5]})
            return boxes

        def compute_iou(b1, b2):
            def corners(b):
                x, y, w, h = b
                return x-w/2, y-h/2, x+w/2, y+h/2
            ax1,ay1,ax2,ay2 = corners(b1)
            bx1,by1,bx2,by2 = corners(b2)
            ix = max(0, min(ax2,bx2) - max(ax1,bx1))
            iy = max(0, min(ay2,by2) - max(ay1,by1))
            inter = ix * iy
            union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
            return inter / max(union, 1e-6)

        for img_path in tqdm(images, desc="Collecting predictions"):
            label_path = label_dir / (img_path.stem + ".txt")
            gt_boxes = load_labels(label_path)
            results = self.model.predict(source=str(img_path), conf=conf, verbose=False)[0]

            if results.boxes is None:
                continue

            matched = set()
            for box in results.boxes:
                pred_conf = float(box.conf.item())
                pred_cls = int(box.cls.item())
                pred_bbox = box.xywhn.tolist()[0]
                correct = 0
                for i, gt in enumerate(gt_boxes):
                    if i in matched or gt["class_id"] != pred_cls:
                        continue
                    if compute_iou(pred_bbox, gt["bbox"]) >= 0.5:
                        correct = 1
                        matched.add(i)
                        break
                all_confs.append(pred_conf)
                all_correct.append(correct)

        return np.array(all_confs), np.array(all_correct)

    def calibrate(self, img_dir: Path, label_dir: Path, n_bins: int = 10) -> Dict:
        """Find optimal temperature and compute calibration metrics."""
        print(f"\n🌡️  Calibrating confidence scores...")
        confs, correct = self._collect_predictions(img_dir, label_dir)

        if len(confs) == 0:
            print("❌ No predictions collected")
            return {}

        print(f"   Collected {len(confs)} predictions ({correct.sum()} correct)")

        # Find optimal temperature via grid search
        best_t, best_ece = 1.0, float("inf")
        for t in np.arange(0.1, 5.0, 0.05):
            calibrated = np.clip(confs / t, 0, 1)
            ece = self._compute_ece(calibrated, correct, n_bins)
            if ece < best_ece:
                best_ece, best_t = ece, t

        self.temperature = best_t
        uncalibrated_ece = self._compute_ece(confs, correct, n_bins)
        calibrated_confs = np.clip(confs / best_t, 0, 1)
        calibrated_ece = self._compute_ece(calibrated_confs, correct, n_bins)

        print(f"\n📊 Calibration Results:")
        print(f"   Optimal temperature:   T = {best_t:.3f}")
        print(f"   ECE before:            {uncalibrated_ece:.4f}")
        print(f"   ECE after:             {calibrated_ece:.4f}")
        print(f"   Improvement:           {(uncalibrated_ece - calibrated_ece) / uncalibrated_ece:.1%}")

        if best_t > 1.0:
            print(f"   Interpretation: Model was OVERCONFIDENT — T={best_t:.2f} softens scores")
        else:
            print(f"   Interpretation: Model was UNDERCONFIDENT — T={best_t:.2f} sharpens scores")

        self._plot_calibration(confs, calibrated_confs, correct, n_bins, img_dir)

        result = {
            "optimal_temperature": round(best_t, 4),
            "ece_before": round(uncalibrated_ece, 4),
            "ece_after": round(calibrated_ece, 4),
            "n_predictions": len(confs),
            "interpretation": "overconfident" if best_t > 1.0 else "underconfident"
        }

        report_path = img_dir / "calibration_report.json"
        report_path.write_text(json.dumps(result, indent=2))

        # Save temperature for use by inference server
        temp_path = Path(MODELS_DIR) / "calibration_temperature.json"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(json.dumps({"temperature": best_t}, indent=2))
        print(f"\n💾 Temperature saved: {temp_path}")
        print(f"   Use in inference_server.py: conf / {best_t:.3f}")
        return result

    def _compute_ece(self, confs: np.ndarray, correct: np.ndarray, n_bins: int) -> float:
        """Expected Calibration Error — lower is better calibrated."""
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (confs >= bins[i]) & (confs < bins[i+1])
            if mask.sum() == 0:
                continue
            avg_conf = confs[mask].mean()
            avg_acc = correct[mask].mean()
            ece += (mask.sum() / len(confs)) * abs(avg_conf - avg_acc)
        return float(ece)

    def _plot_calibration(self, raw_confs: np.ndarray, cal_confs: np.ndarray,
                          correct: np.ndarray, n_bins: int, output_dir: Path):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Confidence Calibration", fontsize=14)

        for ax, confs, title in [(axes[0], raw_confs, "Before Calibration"), (axes[1], cal_confs, "After Calibration")]:
            bins = np.linspace(0, 1, n_bins + 1)
            bin_accs, bin_confs, bin_counts = [], [], []
            for i in range(n_bins):
                mask = (confs >= bins[i]) & (confs < bins[i+1])
                if mask.sum() > 0:
                    bin_accs.append(correct[mask].mean())
                    bin_confs.append(confs[mask].mean())
                    bin_counts.append(mask.sum())

            ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.5)
            ax.scatter(bin_confs, bin_accs, s=[c/5 for c in bin_counts], alpha=0.8, label="Actual")
            ax.plot(bin_confs, bin_accs, alpha=0.6)
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
            ax.set_title(title)
            ax.legend()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        plt.tight_layout()
        path = output_dir / "calibration_plot.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"📊 Calibration plot saved: {path}")
        plt.close()

    def apply(self, confidence: float) -> float:
        """Apply calibration to a single confidence score."""
        return float(np.clip(confidence / self.temperature, 0, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(MODELS_DIR / "baseline_v1/weights/best.pt"))
    parser.add_argument("--val-dir", default=str(DATA_DIR / "yolo_format/val/images"))
    parser.add_argument("--label-dir", default=str(DATA_DIR / "yolo_format/val/labels"))
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    cal = ConfidenceCalibrator(args.model)
    cal.calibrate(Path(args.val_dir), Path(args.label_dir), args.bins)
