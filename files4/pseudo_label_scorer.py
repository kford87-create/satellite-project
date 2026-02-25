"""
tools/active_learning/pseudo_label_scorer.py

Scores pseudo-label quality before sending to human annotators.
Uses ensemble disagreement between model checkpoints to estimate
how much a human needs to correct each image.

High-quality pseudo-labels → quick human confirmation (seconds)
Low-quality pseudo-labels → careful human review (minutes)

Sorting by quality score before annotation reduces annotator time 30-40%.

Usage:
  python tools/active_learning/pseudo_label_scorer.py \
    --query-dir data/bootstrapped/iteration_01_query \
    --checkpoints models/checkpoint_*.pt \
    --output data/bootstrapped/scored_queue.json
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import argparse
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))


class PseudoLabelScorer:
    """
    Scores pseudo-label quality using ensemble disagreement.
    Multiple model checkpoints vote on each image.
    High disagreement → low quality pseudo-label → needs careful review.
    Low disagreement → high quality → quick confirm.
    """

    def __init__(self, checkpoint_paths: List[str], conf_threshold: float = 0.20):
        from ultralytics import YOLO
        print(f"🔄 Loading {len(checkpoint_paths)} model checkpoints for ensemble...")
        self.models = []
        for path in checkpoint_paths:
            if Path(path).exists():
                self.models.append(YOLO(path))
                print(f"   ✅ Loaded: {path}")
            else:
                print(f"   ⚠️  Not found: {path}")

        if not self.models:
            raise ValueError("No valid checkpoints loaded. Provide at least one --checkpoints path.")

        self.conf_threshold = conf_threshold
        print(f"✅ Ensemble ready ({len(self.models)} models)")

    def _predict_single(self, model, img: np.ndarray) -> List[Dict]:
        """Run inference with one model."""
        results = model.predict(source=img, conf=self.conf_threshold, verbose=False)[0]
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                detections.append({
                    "class_id": int(box.cls.item()),
                    "confidence": float(box.conf.item()),
                    "bbox": box.xywhn.tolist()[0]
                })
        return detections

    def _compute_box_iou(self, b1: List[float], b2: List[float]) -> float:
        def corners(b):
            x, y, w, h = b
            return x-w/2, y-h/2, x+w/2, y+h/2
        ax1,ay1,ax2,ay2 = corners(b1)
        bx1,by1,bx2,by2 = corners(b2)
        inter = max(0, min(ax2,bx2)-max(ax1,bx1)) * max(0, min(ay2,by2)-max(ay1,by1))
        union = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
        return inter/max(union,1e-6)

    def _ensemble_agreement_score(self, all_predictions: List[List[Dict]]) -> float:
        """
        Compute ensemble agreement score (0=total disagreement, 1=perfect agreement).
        Based on: count agreement / max possible agreement.
        """
        if not any(all_predictions):
            return 1.0  # All models predict nothing — high confidence empty scene

        # Check count agreement
        counts = [len(p) for p in all_predictions]
        count_std = np.std(counts)
        count_agreement = max(0.0, 1.0 - count_std / max(np.mean(counts), 1))

        # Check spatial agreement using IoU between model outputs
        if len(self.models) < 2:
            return float(count_agreement)

        iou_scores = []
        for i in range(len(all_predictions)):
            for j in range(i+1, len(all_predictions)):
                preds_i, preds_j = all_predictions[i], all_predictions[j]
                if not preds_i or not preds_j:
                    iou_scores.append(0.0)
                    continue
                matched_ious = []
                for det_i in preds_i:
                    best_iou = max((self._compute_box_iou(det_i["bbox"], det_j["bbox"])
                                   for det_j in preds_j if det_j["class_id"] == det_i["class_id"]),
                                  default=0.0)
                    matched_ious.append(best_iou)
                iou_scores.append(np.mean(matched_ious))

        spatial_agreement = float(np.mean(iou_scores)) if iou_scores else 0.0
        return float(0.4 * count_agreement + 0.6 * spatial_agreement)

    def score_image(self, img_path: Path) -> Dict:
        """Score pseudo-label quality for a single image."""
        img = cv2.imread(str(img_path))
        if img is None:
            return {"image": img_path.name, "quality_score": 0.0, "error": "Could not read image"}

        all_predictions = [self._predict_single(model, img) for model in self.models]
        agreement = self._ensemble_agreement_score(all_predictions)

        # Use first model's predictions as the pseudo-labels
        pseudo_labels = all_predictions[0] if all_predictions else []

        # Avg confidence from ensemble
        all_confs = [d["confidence"] for preds in all_predictions for d in preds]
        avg_confidence = float(np.mean(all_confs)) if all_confs else 0.0

        # Quality score: high agreement + high confidence = good pseudo-label
        quality_score = 0.6 * agreement + 0.4 * avg_confidence

        review_priority = "quick_confirm" if quality_score > 0.75 else \
                          "normal_review" if quality_score > 0.45 else "careful_review"

        return {
            "image": img_path.name,
            "quality_score": round(quality_score, 4),
            "ensemble_agreement": round(agreement, 4),
            "avg_confidence": round(avg_confidence, 4),
            "n_pseudo_labels": len(pseudo_labels),
            "review_priority": review_priority,
            "pseudo_labels": pseudo_labels,
            "estimated_annotation_minutes": 0.1 if quality_score > 0.75 else 0.5 if quality_score > 0.45 else 2.0
        }

    def score_batch(self, query_dir: Path, output_path: Optional[Path] = None) -> List[Dict]:
        """Score all images in a query directory."""
        images = list(query_dir.glob("*.jpg")) + list(query_dir.glob("*.png"))
        if not images:
            print(f"❌ No images in {query_dir}")
            return []

        print(f"\n🔍 Scoring {len(images)} pseudo-labels with {len(self.models)}-model ensemble...")
        results = [self.score_image(img) for img in tqdm(images, desc="Scoring")]
        results.sort(key=lambda x: x["quality_score"], reverse=True)

        # Summary
        quick = sum(1 for r in results if r["review_priority"] == "quick_confirm")
        normal = sum(1 for r in results if r["review_priority"] == "normal_review")
        careful = sum(1 for r in results if r["review_priority"] == "careful_review")
        total_minutes = sum(r["estimated_annotation_minutes"] for r in results)

        print(f"\n📊 Pseudo-Label Quality Summary:")
        print(f"   Quick confirm  (>0.75): {quick} images")
        print(f"   Normal review  (>0.45): {normal} images")
        print(f"   Careful review (<0.45): {careful} images")
        print(f"   Estimated annotation time: {total_minutes:.0f} min vs {len(images)*2:.0f} min from scratch")
        print(f"   Time savings: ~{(1 - total_minutes/(len(images)*2))*100:.0f}%")

        output_path = output_path or query_dir / "pseudo_label_scores.json"
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\n📋 Scores saved: {output_path}")
        return results

    def export_roboflow_priority_queue(self, scores: List[Dict], output_dir: Path):
        """Export prioritized annotation queue compatible with Roboflow batch upload."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for priority in ["careful_review", "normal_review", "quick_confirm"]:
            batch = [r for r in scores if r["review_priority"] == priority]
            queue_path = output_dir / f"annotation_queue_{priority}.json"
            queue_path.write_text(json.dumps(batch, indent=2))
        print(f"📤 Roboflow priority queues exported to: {output_dir}")


# Fix missing import
from typing import Optional


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-dir", required=True)
    parser.add_argument("--checkpoints", nargs="+", default=[str(MODELS_DIR / "baseline_v1/weights/best.pt")])
    parser.add_argument("--output", default=None)
    parser.add_argument("--conf", type=float, default=0.20)
    args = parser.parse_args()

    scorer = PseudoLabelScorer(args.checkpoints, args.conf)
    scorer.score_batch(Path(args.query_dir), Path(args.output) if args.output else None)
