"""
tools/model_performance/geo_aware_evaluator.py

Evaluates model performance with geographic awareness.
Maps where errors cluster — revealing terrain/density biases
that standard mAP metrics completely miss.

Usage:
  python tools/model_performance/geo_aware_evaluator.py \
    --model models/baseline_v1/weights/best.pt \
    --val-dir data/yolo_format/val \
    --chip-index data/unlabeled/scene_chip_index.json
"""

import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import argparse
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))


class GeoAwareEvaluator:
    """
    Evaluates detection performance and maps errors geographically.
    Identifies where the model struggles spatially.
    """

    def __init__(self, model_path: str, iou_threshold: float = 0.5, conf_threshold: float = 0.25):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

    def compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two YOLO-format boxes (x_c, y_c, w, h normalized)."""
        def to_corners(box):
            x, y, w, h = box
            return x - w/2, y - h/2, x + w/2, y + h/2

        x1, y1, x2, y2 = to_corners(box1)
        x3, y3, x4, y4 = to_corners(box2)
        ix1, iy1 = max(x1, x3), max(y1, y3)
        ix2, iy2 = min(x2, x4), min(y2, y4)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (x2-x1)*(y2-y1) + (x4-x3)*(y4-y3) - inter
        return inter / max(union, 1e-6)

    def evaluate_image(self, img_path: Path, label_path: Path) -> Dict:
        """Evaluate a single image. Returns TP, FP, FN with bounding boxes."""
        # Load ground truth
        gt_boxes = []
        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                if line.strip():
                    parts = list(map(float, line.split()))
                    gt_boxes.append({"class_id": int(parts[0]), "bbox": parts[1:5]})

        # Run inference
        results = self.model.predict(source=str(img_path), conf=self.conf_threshold, verbose=False)[0]
        pred_boxes = []
        if results.boxes is not None:
            for box in results.boxes:
                pred_boxes.append({
                    "class_id": int(box.cls.item()),
                    "confidence": float(box.conf.item()),
                    "bbox": box.xywhn.tolist()[0]
                })

        # Match predictions to ground truth
        matched_gt = set()
        tp, fp, fn_boxes = [], [], []

        for pred in pred_boxes:
            best_iou, best_gt_idx = 0.0, -1
            for i, gt in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                if gt["class_id"] != pred["class_id"]:
                    continue
                iou = self.compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, i

            if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                tp.append(pred)
                matched_gt.add(best_gt_idx)
            else:
                fp.append(pred)

        for i, gt in enumerate(gt_boxes):
            if i not in matched_gt:
                fn_boxes.append(gt)

        return {"tp": len(tp), "fp": len(fp), "fn": len(fn_boxes),
                "fn_boxes": fn_boxes, "fp_boxes": fp,
                "n_gt": len(gt_boxes), "n_pred": len(pred_boxes)}

    def evaluate_dataset(self, img_dir: Path, label_dir: Path, chip_index_path: Optional[Path] = None) -> Dict:
        """Evaluate full dataset and build geographic error map."""
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        if not images:
            print(f"❌ No images found in {img_dir}")
            return {}

        # Load chip index for geo coordinates
        chip_index = {}
        if chip_index_path and chip_index_path.exists():
            for entry in json.loads(chip_index_path.read_text()):
                chip_index[entry["chip_name"]] = entry

        print(f"\n📊 Evaluating {len(images)} images...")
        all_results = []
        total_tp = total_fp = total_fn = 0

        for img_path in tqdm(images, desc="Evaluating"):
            label_path = label_dir / (img_path.stem + ".txt")
            result = self.evaluate_image(img_path, label_path)
            result["image"] = img_path.name

            # Attach geo coordinates if available
            chip_meta = chip_index.get(img_path.name, {})
            result["lat"] = chip_meta.get("lat_min")
            result["lon"] = chip_meta.get("lon_min")

            all_results.append(result)
            total_tp += result["tp"]
            total_fp += result["fp"]
            total_fn += result["fn"]

        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_tp + total_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        print(f"\n📊 Evaluation Results:")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
        print(f"   TP: {total_tp} | FP: {total_fp} | FN: {total_fn}")

        summary = {"precision": round(precision, 4), "recall": round(recall, 4),
                   "f1": round(f1, 4), "tp": total_tp, "fp": total_fp, "fn": total_fn,
                   "n_images": len(images), "image_results": all_results}

        # Generate geographic error map
        geo_results = [r for r in all_results if r.get("lat") and r.get("lon")]
        if geo_results:
            self._plot_geo_error_map(geo_results, img_dir)
        else:
            print("ℹ️  No geo coordinates found — skipping geographic error map")
            self._plot_error_distribution(all_results, img_dir)

        # Save report
        report_path = img_dir / "geo_eval_report.json"
        report_data = {k: v for k, v in summary.items() if k != "image_results"}
        report_path.write_text(json.dumps(report_data, indent=2))
        print(f"\n📋 Report saved: {report_path}")
        return summary

    def _plot_geo_error_map(self, results: List[Dict], output_dir: Path):
        """Plot geographic distribution of FP and FN errors."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Geographic Error Distribution", fontsize=14)

        for ax, metric, color, title in [
            (axes[0], "fn", "red", "False Negatives (Missed Objects)"),
            (axes[1], "fp", "orange", "False Positives (Wrong Detections)")
        ]:
            lats = [r["lat"] for r in results]
            lons = [r["lon"] for r in results]
            vals = [r[metric] for r in results]
            sc = ax.scatter(lons, lats, c=vals, cmap="YlOrRd", alpha=0.6, s=20)
            plt.colorbar(sc, ax=ax, label=f"Count of {metric.upper()}")
            ax.set_title(title)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        plt.tight_layout()
        plot_path = output_dir / "geo_error_map.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"🗺️  Geographic error map saved: {plot_path}")
        plt.close()

    def _plot_error_distribution(self, results: List[Dict], output_dir: Path):
        """Fallback: plot error distribution without geo coordinates."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Model Error Distribution", fontsize=14)

        for ax, key, color, label in [
            (axes[0], "tp", "green", "True Positives"),
            (axes[1], "fp", "orange", "False Positives"),
            (axes[2], "fn", "red", "False Negatives")
        ]:
            vals = [r[key] for r in results]
            ax.hist(vals, bins=20, color=color, alpha=0.7, edgecolor="black")
            ax.set_title(label)
            ax.set_xlabel("Count per image")
            ax.set_ylabel("Frequency")

        plt.tight_layout()
        plot_path = output_dir / "error_distribution.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"📊 Error distribution saved: {plot_path}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(MODELS_DIR / "baseline_v1/weights/best.pt"))
    parser.add_argument("--val-dir", default=str(DATA_DIR / "yolo_format/val/images"))
    parser.add_argument("--label-dir", default=str(DATA_DIR / "yolo_format/val/labels"))
    parser.add_argument("--chip-index", default=None)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    evaluator = GeoAwareEvaluator(args.model, args.iou, args.conf)
    evaluator.evaluate_dataset(
        img_dir=Path(args.val_dir),
        label_dir=Path(args.label_dir),
        chip_index_path=Path(args.chip_index) if args.chip_index else None
    )
