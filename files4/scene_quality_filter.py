"""
tools/data_acquisition/scene_quality_filter.py

Scores and filters satellite imagery before it enters the bootstrapping pipeline.
Rejects scenes with excessive cloud cover, NoData regions, or poor contrast.
Training on bad imagery actively degrades model performance.

Think of this as quality control at a factory entrance —
bad raw material never makes it to the production line.

Usage:
  python tools/data_acquisition/scene_quality_filter.py \
    --input-dir data/unlabeled \
    --max-cloud 20 \
    --min-contrast 0.1 \
    --move-rejected
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import argparse
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
REJECTED_DIR = DATA_DIR / "rejected"


class SceneQualityFilter:
    """
    Multi-metric quality scorer for satellite imagery chips.

    Metrics:
    1. Cloud cover estimate (brightness + variance analysis)
    2. NoData fraction (black pixel ratio)
    3. Contrast score (useful signal vs flat/saturated)
    4. Edge density (proxy for information content)
    """

    def __init__(
        self,
        max_cloud_fraction: float = 0.20,
        max_nodata_fraction: float = 0.10,
        min_contrast_score: float = 0.10,
        min_edge_density: float = 0.02
    ):
        self.max_cloud_fraction = max_cloud_fraction
        self.max_nodata_fraction = max_nodata_fraction
        self.min_contrast_score = min_contrast_score
        self.min_edge_density = min_edge_density

    def estimate_cloud_fraction(self, img: np.ndarray) -> float:
        """
        Estimate cloud fraction using brightness + low variance heuristic.
        Clouds are bright and spatially uniform (low local variance).
        Not as accurate as a dedicated cloud mask but fast and dependency-free.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        gray_float = gray.astype(np.float32) / 255.0

        # Bright pixels
        bright_mask = gray_float > 0.75

        # Low local variance (uniform = likely cloud)
        kernel_size = 15
        local_mean = cv2.blur(gray_float, (kernel_size, kernel_size))
        local_sq_mean = cv2.blur(gray_float ** 2, (kernel_size, kernel_size))
        local_var = local_sq_mean - local_mean ** 2
        low_var_mask = local_var < 0.005

        # Cloud pixels = bright AND low variance
        cloud_mask = bright_mask & low_var_mask
        return float(cloud_mask.mean())

    def estimate_nodata_fraction(self, img: np.ndarray) -> float:
        """
        Estimate fraction of NoData (completely black) pixels.
        Occurs at scene edges or when imagery is missing.
        """
        if len(img.shape) == 3:
            nodata_mask = np.all(img == 0, axis=2)
        else:
            nodata_mask = img == 0
        return float(nodata_mask.mean())

    def compute_contrast_score(self, img: np.ndarray) -> float:
        """
        Compute normalized contrast score using RMS contrast.
        Low contrast = washed out or uniform image = poor training data.
        Returns 0-1, higher is better.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        gray_float = gray.astype(np.float32) / 255.0
        rms_contrast = float(np.sqrt(np.mean((gray_float - gray_float.mean()) ** 2)))
        return min(rms_contrast * 4.0, 1.0)  # Normalize to 0-1

    def compute_edge_density(self, img: np.ndarray) -> float:
        """
        Compute edge density as proxy for information content.
        Images with many edges have more distinguishable features.
        Returns 0-1, higher means more structural content.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        edges = cv2.Canny(gray, threshold1=30, threshold2=100)
        return float(edges.mean() / 255.0)

    def score_image(self, img_path: Path) -> Dict:
        """
        Compute all quality metrics for a single image.
        Returns a score dict with pass/fail determination.
        """
        img = cv2.imread(str(img_path))
        if img is None:
            return {
                "path": str(img_path),
                "valid": False,
                "reject_reason": "Could not decode image",
                "cloud_fraction": 1.0,
                "nodata_fraction": 1.0,
                "contrast_score": 0.0,
                "edge_density": 0.0,
                "overall_score": 0.0,
            }

        cloud = self.estimate_cloud_fraction(img)
        nodata = self.estimate_nodata_fraction(img)
        contrast = self.compute_contrast_score(img)
        edges = self.compute_edge_density(img)

        # Determine rejection reason
        reject_reason = None
        if cloud > self.max_cloud_fraction:
            reject_reason = f"Cloud cover {cloud:.1%} > {self.max_cloud_fraction:.1%}"
        elif nodata > self.max_nodata_fraction:
            reject_reason = f"NoData {nodata:.1%} > {self.max_nodata_fraction:.1%}"
        elif contrast < self.min_contrast_score:
            reject_reason = f"Contrast {contrast:.3f} < {self.min_contrast_score}"
        elif edges < self.min_edge_density:
            reject_reason = f"Edge density {edges:.3f} < {self.min_edge_density}"

        # Overall quality score (0-1)
        overall = (
            (1 - cloud) * 0.4 +
            (1 - nodata) * 0.2 +
            contrast * 0.2 +
            edges * 0.2
        )

        return {
            "path": str(img_path),
            "stem": img_path.stem,
            "valid": reject_reason is None,
            "reject_reason": reject_reason,
            "cloud_fraction": round(cloud, 4),
            "nodata_fraction": round(nodata, 4),
            "contrast_score": round(contrast, 4),
            "edge_density": round(edges, 4),
            "overall_score": round(overall, 4),
        }

    def filter_directory(
        self,
        input_dir: Path,
        move_rejected: bool = False,
        save_report: bool = True
    ) -> Dict:
        """
        Score and filter all images in a directory.

        Args:
            input_dir: Directory containing satellite image chips
            move_rejected: Move rejected images to data/rejected/
            save_report: Save JSON quality report

        Returns:
            Summary dict with pass/fail counts and scores
        """
        images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
        if not images:
            print(f"❌ No images found in {input_dir}")
            return {}

        print(f"\n🔍 Scoring {len(images)} images for quality...")

        if move_rejected:
            REJECTED_DIR.mkdir(parents=True, exist_ok=True)

        results = []
        passed = 0
        failed = 0
        rejection_reasons = {}

        for img_path in tqdm(images, desc="Quality scoring"):
            score = self.score_image(img_path)
            results.append(score)

            if score["valid"]:
                passed += 1
            else:
                failed += 1
                reason = score["reject_reason"] or "Unknown"
                # Categorize reason
                category = reason.split(" ")[0]
                rejection_reasons[category] = rejection_reasons.get(category, 0) + 1

                if move_rejected and score["reject_reason"]:
                    import shutil
                    dest = REJECTED_DIR / img_path.name
                    shutil.move(str(img_path), str(dest))

                    # Move label too if exists
                    label = img_path.parent / "labels" / (img_path.stem + ".txt")
                    if label.exists():
                        shutil.move(str(label), str(REJECTED_DIR / label.name))

        # Summary stats
        scores = [r["overall_score"] for r in results]
        summary = {
            "total": len(images),
            "passed": passed,
            "rejected": failed,
            "pass_rate": round(passed / len(images), 3),
            "avg_quality_score": round(float(np.mean(scores)), 3),
            "rejection_breakdown": rejection_reasons,
            "top_quality_images": [
                r["stem"] for r in sorted(results, key=lambda x: x["overall_score"], reverse=True)[:10]
            ]
        }

        print(f"\n📊 Quality Filter Results:")
        print(f"   Total images:    {summary['total']}")
        print(f"   Passed:          {summary['passed']} ({summary['pass_rate']:.1%})")
        print(f"   Rejected:        {summary['rejected']}")
        print(f"   Avg quality:     {summary['avg_quality_score']:.3f}")
        if rejection_reasons:
            print(f"   Rejection breakdown:")
            for reason, count in rejection_reasons.items():
                print(f"     {reason}: {count}")

        if save_report:
            report_path = input_dir / "quality_report.json"
            report_path.write_text(json.dumps({
                "summary": summary,
                "images": results
            }, indent=2))
            print(f"\n📋 Full report saved: {report_path}")

        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter satellite imagery by quality")
    parser.add_argument("--input-dir", default=str(DATA_DIR / "unlabeled"))
    parser.add_argument("--max-cloud", type=float, default=0.20)
    parser.add_argument("--max-nodata", type=float, default=0.10)
    parser.add_argument("--min-contrast", type=float, default=0.10)
    parser.add_argument("--min-edges", type=float, default=0.02)
    parser.add_argument("--move-rejected", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    filt = SceneQualityFilter(
        max_cloud_fraction=args.max_cloud,
        max_nodata_fraction=args.max_nodata,
        min_contrast_score=args.min_contrast,
        min_edge_density=args.min_edges
    )
    filt.filter_directory(
        input_dir=Path(args.input_dir),
        move_rejected=args.move_rejected,
        save_report=not args.no_report
    )
