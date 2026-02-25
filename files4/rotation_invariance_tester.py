"""
tools/model_performance/rotation_invariance_tester.py

Tests model performance across all orientations (0-360°).
Satellite imagery captures objects at any angle — vehicles, aircraft,
and ships are highly rotation-sensitive. This tool reveals exactly
where your model's orientation blind spots are.

Usage:
  python tools/model_performance/rotation_invariance_tester.py \
    --model models/baseline_v1/weights/best.pt \
    --test-dir data/yolo_format/val/images \
    --angles 0 45 90 135 180 225 270 315
"""

import os
import cv2
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


class RotationInvarianceTester:
    """
    Tests detection consistency across image rotations.
    A perfect model would detect the same objects regardless of orientation.
    Performance drop at specific angles reveals training data bias.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def rotate_image(self, img: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate image and return rotated image + rotation matrix."""
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)
        return rotated, M

    def detect_image(self, img: np.ndarray) -> List[Dict]:
        """Run inference and return detections."""
        results = self.model.predict(source=img, conf=self.conf_threshold, verbose=False)[0]
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                detections.append({
                    "class_id": int(box.cls.item()),
                    "confidence": float(box.conf.item()),
                    "bbox": box.xywhn.tolist()[0]
                })
        return detections

    def test_single_image(self, img_path: Path, angles: List[float]) -> Dict:
        """Test one image across all specified rotation angles."""
        img = cv2.imread(str(img_path))
        if img is None:
            return {}

        baseline = self.detect_image(img)
        baseline_count = len(baseline)
        baseline_conf = np.mean([d["confidence"] for d in baseline]) if baseline else 0.0

        angle_results = {"0": {"n_detections": baseline_count, "avg_confidence": round(baseline_conf, 4)}}

        for angle in angles:
            if angle == 0:
                continue
            rotated, _ = self.rotate_image(img, angle)
            dets = self.detect_image(rotated)
            n = len(dets)
            conf = np.mean([d["confidence"] for d in dets]) if dets else 0.0
            consistency = min(n, baseline_count) / max(n, baseline_count, 1)
            angle_results[str(angle)] = {
                "n_detections": n,
                "avg_confidence": round(conf, 4),
                "consistency_vs_0deg": round(consistency, 4)
            }

        return {
            "image": img_path.name,
            "baseline_detections": baseline_count,
            "baseline_confidence": round(baseline_conf, 4),
            "by_angle": angle_results
        }

    def test_dataset(self, img_dir: Path, angles: List[float] = None, max_images: int = 100) -> Dict:
        """Test rotation invariance across a dataset."""
        angles = angles or [0, 45, 90, 135, 180, 225, 270, 315]
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        images = images[:max_images]

        if not images:
            print(f"❌ No images found in {img_dir}")
            return {}

        print(f"\n🔄 Testing rotation invariance on {len(images)} images across {len(angles)} angles...")
        all_results = []

        for img_path in tqdm(images, desc="Testing rotations"):
            result = self.test_single_image(img_path, angles)
            if result:
                all_results.append(result)

        # Aggregate stats per angle
        angle_stats = {}
        for angle in angles:
            key = str(angle)
            confs = [r["by_angle"][key]["avg_confidence"] for r in all_results if key in r.get("by_angle", {})]
            dets = [r["by_angle"][key]["n_detections"] for r in all_results if key in r.get("by_angle", {})]
            consistencies = [r["by_angle"][key].get("consistency_vs_0deg", 1.0)
                             for r in all_results if key in r.get("by_angle", {})]
            angle_stats[key] = {
                "avg_confidence": round(float(np.mean(confs)) if confs else 0, 4),
                "avg_detections": round(float(np.mean(dets)) if dets else 0, 2),
                "avg_consistency": round(float(np.mean(consistencies)) if consistencies else 0, 4)
            }

        # Find worst angles
        worst_angle = min(angle_stats, key=lambda a: angle_stats[a]["avg_consistency"])
        best_angle = max(angle_stats, key=lambda a: angle_stats[a]["avg_consistency"])

        print(f"\n📊 Rotation Invariance Results:")
        print(f"   {'Angle':>6} | {'Avg Conf':>9} | {'Avg Dets':>9} | {'Consistency':>11}")
        print(f"   {'-'*45}")
        for angle in sorted(angles):
            s = angle_stats[str(angle)]
            print(f"   {angle:>6}° | {s['avg_confidence']:>9.4f} | {s['avg_detections']:>9.2f} | {s['avg_consistency']:>11.4f}")

        print(f"\n   ✅ Best angle:  {best_angle}° (consistency: {angle_stats[best_angle]['avg_consistency']:.4f})")
        print(f"   ⚠️  Worst angle: {worst_angle}° (consistency: {angle_stats[worst_angle]['avg_consistency']:.4f})")

        # Plot
        self._plot_results(angle_stats, img_dir)

        summary = {
            "n_images_tested": len(images),
            "angles_tested": angles,
            "angle_stats": angle_stats,
            "worst_angle": worst_angle,
            "best_angle": best_angle,
            "recommendation": f"Add more training data with objects rotated ~{worst_angle}° to improve invariance"
        }

        report_path = img_dir / "rotation_invariance_report.json"
        report_path.write_text(json.dumps(summary, indent=2))
        print(f"\n📋 Report saved: {report_path}")
        return summary

    def _plot_results(self, angle_stats: Dict, output_dir: Path):
        angles = sorted([int(a) for a in angle_stats.keys()])
        consistencies = [angle_stats[str(a)]["avg_consistency"] for a in angles]
        confidences = [angle_stats[str(a)]["avg_confidence"] for a in angles]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), subplot_kw=dict(polar=True))
        fig.suptitle("Rotation Invariance Analysis", fontsize=14)

        for ax, values, title, color in [
            (ax1, consistencies, "Detection Consistency", "blue"),
            (ax2, confidences, "Average Confidence", "green")
        ]:
            angles_rad = [np.deg2rad(a) for a in angles] + [np.deg2rad(angles[0])]
            values_plot = values + [values[0]]
            ax.plot(angles_rad, values_plot, color=color, linewidth=2)
            ax.fill(angles_rad, values_plot, color=color, alpha=0.2)
            ax.set_title(title, pad=15)
            ax.set_xticks([np.deg2rad(a) for a in angles])
            ax.set_xticklabels([f"{a}°" for a in angles])

        plt.tight_layout()
        path = output_dir / "rotation_invariance_plot.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"📊 Rotation plot saved: {path}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(MODELS_DIR / "baseline_v1/weights/best.pt"))
    parser.add_argument("--test-dir", default=str(DATA_DIR / "yolo_format/val/images"))
    parser.add_argument("--angles", nargs="+", type=float, default=[0, 45, 90, 135, 180, 225, 270, 315])
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    tester = RotationInvarianceTester(args.model, args.conf)
    tester.test_dataset(Path(args.test_dir), args.angles, args.max_images)
