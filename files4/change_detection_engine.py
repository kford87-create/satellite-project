"""
tools/commercial/change_detection_engine.py

Compares the same location across two timestamps and flags what
appeared, disappeared, or moved. Highest commercial value feature
for insurance customers — answers "what changed between policy
issuance and claim filing?"

Usage:
  python tools/commercial/change_detection_engine.py \
    --before data/scene_2023_01.jpg \
    --after data/scene_2024_01.jpg \
    --model models/baseline_v1/weights/best.pt \
    --output data/change_report.json
"""

import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import argparse
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))


@dataclass
class DetectedObject:
    class_name: str
    class_id: int
    confidence: float
    bbox: List[float]  # [x_c, y_c, w, h] normalized

    @property
    def area(self):
        return self.bbox[2] * self.bbox[3]

    def iou(self, other: "DetectedObject") -> float:
        def corners(b):
            x, y, w, h = b
            return x-w/2, y-h/2, x+w/2, y+h/2
        ax1,ay1,ax2,ay2 = corners(self.bbox)
        bx1,by1,bx2,by2 = corners(other.bbox)
        inter = max(0, min(ax2,bx2)-max(ax1,bx1)) * max(0, min(ay2,by2)-max(ay1,by1))
        union = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
        return inter/max(union,1e-6)


@dataclass
class ChangeEvent:
    change_type: str       # "appeared", "disappeared", "moved", "unchanged"
    class_name: str
    confidence: float
    location_before: Optional[List[float]]
    location_after: Optional[List[float]]
    significance: str      # "high", "medium", "low"
    description: str


class ChangeDetectionEngine:
    """
    Detects changes between two satellite images of the same location.
    Classifies changes as: appeared, disappeared, moved, unchanged.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_match_threshold: float = 0.4):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_match_threshold = iou_match_threshold

        self.CLASS_NAMES = {0: "building", 1: "vehicle", 2: "aircraft", 3: "ship"}
        self.SIGNIFICANCE = {
            "building": {"appeared": "high", "disappeared": "high", "moved": "medium"},
            "vehicle": {"appeared": "low", "disappeared": "low", "moved": "low"},
            "aircraft": {"appeared": "medium", "disappeared": "medium", "moved": "low"},
            "ship": {"appeared": "medium", "disappeared": "medium", "moved": "low"},
        }

    def _detect(self, img: np.ndarray) -> List[DetectedObject]:
        results = self.model.predict(source=img, conf=self.conf_threshold, verbose=False)[0]
        objects = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls.item())
                objects.append(DetectedObject(
                    class_name=self.CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                    class_id=cls_id,
                    confidence=round(float(box.conf.item()), 4),
                    bbox=box.xywhn.tolist()[0]
                ))
        return objects

    def _align_images(self, img_before: np.ndarray, img_after: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align two images using feature matching (ORB + homography).
        Critical for accurate change detection — even small shifts
        will cause false positives without alignment.
        """
        if img_before.shape != img_after.shape:
            img_after = cv2.resize(img_after, (img_before.shape[1], img_before.shape[0]))

        gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=2000)
        kp1, desc1 = orb.detectAndCompute(gray_before, None)
        kp2, desc2 = orb.detectAndCompute(gray_after, None)

        if desc1 is None or desc2 is None or len(kp1) < 10 or len(kp2) < 10:
            return img_before, img_after

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)[:200]

        if len(matches) < 10:
            return img_before, img_after

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is None:
            return img_before, img_after

        h, w = img_before.shape[:2]
        aligned_after = cv2.warpPerspective(img_after, H, (w, h))
        return img_before, aligned_after

    def _match_objects(self, before: List[DetectedObject], after: List[DetectedObject]) -> List[ChangeEvent]:
        """Match objects between frames and classify changes."""
        changes = []
        matched_before = set()
        matched_after = set()

        # Find matches (unchanged / moved)
        for i, obj_b in enumerate(before):
            best_iou, best_j = 0.0, -1
            for j, obj_a in enumerate(after):
                if j in matched_after or obj_a.class_id != obj_b.class_id:
                    continue
                iou = obj_b.iou(obj_a)
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_iou >= self.iou_match_threshold:
                # Matched — unchanged
                matched_before.add(i)
                matched_after.add(best_j)
                changes.append(ChangeEvent(
                    change_type="unchanged",
                    class_name=obj_b.class_name,
                    confidence=round((obj_b.confidence + after[best_j].confidence)/2, 4),
                    location_before=obj_b.bbox,
                    location_after=after[best_j].bbox,
                    significance="low",
                    description=f"{obj_b.class_name} persists at same location"
                ))
            elif best_iou > 0.1:
                # Partial overlap — moved
                matched_before.add(i)
                matched_after.add(best_j)
                sig = self.SIGNIFICANCE.get(obj_b.class_name, {}).get("moved", "low")
                changes.append(ChangeEvent(
                    change_type="moved",
                    class_name=obj_b.class_name,
                    confidence=round((obj_b.confidence + after[best_j].confidence)/2, 4),
                    location_before=obj_b.bbox,
                    location_after=after[best_j].bbox,
                    significance=sig,
                    description=f"{obj_b.class_name} moved position"
                ))

        # Disappeared — in before but not matched
        for i, obj_b in enumerate(before):
            if i not in matched_before:
                sig = self.SIGNIFICANCE.get(obj_b.class_name, {}).get("disappeared", "medium")
                changes.append(ChangeEvent(
                    change_type="disappeared",
                    class_name=obj_b.class_name,
                    confidence=obj_b.confidence,
                    location_before=obj_b.bbox,
                    location_after=None,
                    significance=sig,
                    description=f"{obj_b.class_name} no longer present"
                ))

        # Appeared — in after but not matched
        for j, obj_a in enumerate(after):
            if j not in matched_after:
                sig = self.SIGNIFICANCE.get(obj_a.class_name, {}).get("appeared", "medium")
                changes.append(ChangeEvent(
                    change_type="appeared",
                    class_name=obj_a.class_name,
                    confidence=obj_a.confidence,
                    location_before=None,
                    location_after=obj_a.bbox,
                    significance=sig,
                    description=f"New {obj_a.class_name} detected"
                ))

        return changes

    def detect_changes(self, before_path: Path, after_path: Path, output_dir: Optional[Path] = None) -> Dict:
        """Full change detection pipeline between two images."""
        print(f"\n🔄 Change Detection")
        print(f"   Before: {before_path.name}")
        print(f"   After:  {after_path.name}")

        img_before = cv2.imread(str(before_path))
        img_after = cv2.imread(str(after_path))

        if img_before is None or img_after is None:
            return {"error": "Could not load images"}

        # Align images
        print("   Aligning images...")
        img_before, img_after = self._align_images(img_before, img_after)

        # Detect in both
        print("   Detecting objects...")
        objects_before = self._detect(img_before)
        objects_after = self._detect(img_after)

        # Match and classify
        changes = self._match_objects(objects_before, objects_after)

        # Summarize
        by_type = {}
        for c in changes:
            by_type[c.change_type] = by_type.get(c.change_type, 0) + 1

        high_sig = [c for c in changes if c.significance == "high"]

        print(f"\n📊 Change Summary:")
        for change_type, count in by_type.items():
            emoji = {"appeared": "🆕", "disappeared": "❌", "moved": "➡️", "unchanged": "✅"}.get(change_type, "•")
            print(f"   {emoji} {change_type}: {count}")
        print(f"   ⚠️  High significance: {len(high_sig)}")

        report = {
            "before_image": str(before_path),
            "after_image": str(after_path),
            "summary": {
                "total_changes": len([c for c in changes if c.change_type != "unchanged"]),
                "by_type": by_type,
                "high_significance_count": len(high_sig),
            },
            "changes": [asdict(c) for c in changes],
            "objects_before": len(objects_before),
            "objects_after": len(objects_after),
        }

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            report_path = output_dir / f"change_report_{before_path.stem}_vs_{after_path.stem}.json"
            report_path.write_text(json.dumps(report, indent=2))
            self._visualize(img_before, img_after, changes, output_dir, before_path.stem)
            print(f"\n📋 Report saved: {report_path}")

        return report

    def _visualize(self, img_before: np.ndarray, img_after: np.ndarray,
                   changes: List[ChangeEvent], output_dir: Path, stem: str):
        """Generate side-by-side annotated visualization."""
        COLORS = {"appeared": (0, 255, 0), "disappeared": (0, 0, 255),
                  "moved": (0, 165, 255), "unchanged": (128, 128, 128)}

        def draw_boxes(img, change_list, use_before=True):
            out = img.copy()
            h, w = out.shape[:2]
            for c in change_list:
                bbox = c.location_before if use_before else c.location_after
                if bbox is None:
                    continue
                x, y, bw, bh = bbox
                x1 = int((x - bw/2) * w); y1 = int((y - bh/2) * h)
                x2 = int((x + bw/2) * w); y2 = int((y + bh/2) * h)
                color = COLORS.get(c.change_type, (255, 255, 255))
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(out, c.change_type[:3], (x1, y1-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            return out

        annotated_before = draw_boxes(img_before, changes, use_before=True)
        annotated_after = draw_boxes(img_after, changes, use_before=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, img, title in [(axes[0], annotated_before, "Before"), (axes[1], annotated_after, "After")]:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=13)
            ax.axis("off")

        legend = [plt.Line2D([0], [0], color=np.array(c)/255, linewidth=3, label=t)
                  for t, c in COLORS.items()]
        fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=10)
        plt.suptitle("Change Detection Analysis", fontsize=14)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        path = output_dir / f"change_visualization_{stem}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"🖼️  Visualization saved: {path}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--model", default=str(MODELS_DIR / "baseline_v1/weights/best.pt"))
    parser.add_argument("--output", default=str(DATA_DIR / "change_reports"))
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    engine = ChangeDetectionEngine(args.model, args.conf)
    engine.detect_changes(Path(args.before), Path(args.after), Path(args.output))
