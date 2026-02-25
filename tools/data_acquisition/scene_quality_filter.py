"""
scene_quality_filter.py
-----------------------
Rejects low-quality imagery chips before they enter the active learning loop.

Usage:
    python tools/data_acquisition/scene_quality_filter.py \
        --input-dir data/unlabeled --max-cloud 0.20 --move-rejected
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV_PATH = Path("/Users/kahlil/satellite-project/.env")
load_dotenv(ENV_PATH)

# ---------------------------------------------------------------------------
# Supabase (non-fatal)
# ---------------------------------------------------------------------------
try:
    import os
    from supabase import create_client, Client as SupabaseClient

    _supa_url = os.environ.get("SUPABASE_URL", "")
    _supa_key = os.environ.get("SUPABASE_KEY", "")
    supabase: SupabaseClient | None = (
        create_client(_supa_url, _supa_key) if _supa_url and _supa_key else None
    )
except Exception:
    supabase = None


def _supabase_write_report(report: dict[str, Any]) -> None:
    """Non-fatal write of quality report summary to Supabase."""
    try:
        if supabase is None:
            return
        supabase.table("quality_reports").insert(
            {
                "total": report["summary"]["total_scanned"],
                "passed": report["summary"]["passed"],
                "rejected": report["summary"]["rejected"],
                "rejection_reasons": json.dumps(report["summary"]["rejection_reasons"]),
            }
        ).execute()
    except Exception as exc:
        print(f"⚠️  Supabase write skipped (falling back to local only): {exc}")


# ---------------------------------------------------------------------------
# Quality metric helpers
# ---------------------------------------------------------------------------

# Thresholds (also used as defaults for CLI)
NODATA_THRESHOLD = 0.05
CONTRAST_THRESHOLD = 15.0
ENTROPY_THRESHOLD = 3.5


def _cloud_fraction(rgb: np.ndarray) -> float:
    """Fraction of pixels where R>200, G>200, B>200 (bright white = cloud)."""
    mask = (rgb[:, :, 0] > 200) & (rgb[:, :, 1] > 200) & (rgb[:, :, 2] > 200)
    return float(mask.sum()) / float(mask.size)


def _nodata_fraction(rgb: np.ndarray) -> float:
    """Fraction of pixels where all channels < 5 (pure black = no-data)."""
    mask = (rgb[:, :, 0] < 5) & (rgb[:, :, 1] < 5) & (rgb[:, :, 2] < 5)
    return float(mask.sum()) / float(mask.size)


def _contrast_score(gray: np.ndarray) -> float:
    """Standard deviation of grayscale image. Low = flat / featureless."""
    return float(np.std(gray.astype(np.float32)))


def _information_score(gray: np.ndarray) -> float:
    """Shannon entropy of grayscale histogram. Low = low information content."""
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    total = hist.sum()
    if total == 0:
        return 0.0
    probs = hist[hist > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


def compute_metrics(img_path: Path) -> dict[str, float]:
    """Return a dict of all four quality metrics for an image file."""
    img = Image.open(img_path).convert("RGB")
    rgb = np.array(img)
    gray = np.array(img.convert("L"))

    return {
        "cloud_fraction": _cloud_fraction(rgb),
        "nodata_fraction": _nodata_fraction(rgb),
        "contrast_score": _contrast_score(gray),
        "information_score": _information_score(gray),
    }


def assess_quality(
    metrics: dict[str, float],
    max_cloud: float,
) -> tuple[bool, list[str]]:
    """
    Return (passes: bool, reasons: list[str]).
    reasons is empty when passes=True.
    """
    reasons: list[str] = []
    if metrics["cloud_fraction"] > max_cloud:
        reasons.append(
            f"cloud_fraction {metrics['cloud_fraction']:.3f} > {max_cloud:.3f}"
        )
    if metrics["nodata_fraction"] > NODATA_THRESHOLD:
        reasons.append(
            f"nodata_fraction {metrics['nodata_fraction']:.3f} > {NODATA_THRESHOLD:.2f}"
        )
    if metrics["contrast_score"] < CONTRAST_THRESHOLD:
        reasons.append(
            f"contrast_score {metrics['contrast_score']:.2f} < {CONTRAST_THRESHOLD:.1f}"
        )
    if metrics["information_score"] < ENTROPY_THRESHOLD:
        reasons.append(
            f"information_score {metrics['information_score']:.2f} < {ENTROPY_THRESHOLD:.1f}"
        )
    return (len(reasons) == 0, reasons)


# ---------------------------------------------------------------------------
# Main filter logic
# ---------------------------------------------------------------------------

def run_filter(
    input_dir: Path,
    max_cloud: float,
    move_rejected: bool,
) -> dict[str, Any]:
    rejected_dir = input_dir.parent / "rejected"

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    if not image_paths:
        print(f"⚠️  No .jpg/.png files found in {input_dir}")
        return {}

    print(f"📊 Scanning {len(image_paths)} image(s) in {input_dir}")

    per_image: dict[str, Any] = {}
    passed_count = 0
    rejected_count = 0
    reason_counts: dict[str, int] = defaultdict(int)

    for img_path in tqdm(image_paths, desc="Quality filtering", unit="img"):
        try:
            metrics = compute_metrics(img_path)
            passes, reasons = assess_quality(metrics, max_cloud)

            per_image[img_path.name] = {
                "metrics": metrics,
                "pass": passes,
                "rejection_reasons": reasons,
            }

            if passes:
                passed_count += 1
            else:
                rejected_count += 1
                for r in reasons:
                    # Bucket by metric name (first token before space)
                    metric_name = r.split()[0]
                    reason_counts[metric_name] += 1

                if move_rejected:
                    rejected_dir.mkdir(parents=True, exist_ok=True)
                    dest = rejected_dir / img_path.name
                    shutil.move(str(img_path), str(dest))

        except Exception as exc:
            print(f"❌ Error processing {img_path.name}: {exc}")
            per_image[img_path.name] = {
                "metrics": {},
                "pass": False,
                "rejection_reasons": [f"processing_error: {exc}"],
            }
            rejected_count += 1

    report: dict[str, Any] = {
        "input_dir": str(input_dir.resolve()),
        "max_cloud_threshold": max_cloud,
        "move_rejected": move_rejected,
        "summary": {
            "total_scanned": len(image_paths),
            "passed": passed_count,
            "rejected": rejected_count,
            "rejection_reasons": dict(reason_counts),
        },
        "images": per_image,
    }

    report_path = input_dir / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"✅ Quality report saved to {report_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter low-quality imagery chips from a directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory containing .jpg/.png chip files to evaluate",
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=0.20,
        metavar="FRAC",
        help="Maximum allowed cloud fraction (0.0–1.0)",
    )
    parser.add_argument(
        "--move-rejected",
        action="store_true",
        help="Move rejected images to data/rejected/ instead of leaving them in place",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    input_dir: Path = args.input_dir.resolve()

    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)

    if not (0.0 <= args.max_cloud <= 1.0):
        print(f"❌ --max-cloud must be between 0.0 and 1.0, got {args.max_cloud}")
        sys.exit(1)

    report = run_filter(input_dir, args.max_cloud, args.move_rejected)

    if not report:
        sys.exit(0)

    s = report["summary"]
    print(f"\n📊 Results:")
    print(f"   Total scanned : {s['total_scanned']}")
    print(f"   Passed        : {s['passed']}")
    print(f"   Rejected      : {s['rejected']}")

    if s["rejection_reasons"]:
        print(f"   Rejection reasons breakdown:")
        for metric, count in sorted(s["rejection_reasons"].items(), key=lambda x: -x[1]):
            print(f"     {metric}: {count} chip(s)")

    if args.move_rejected and s["rejected"] > 0:
        rejected_dir = input_dir.parent / "rejected"
        print(f"⚠️  {s['rejected']} chip(s) moved to {rejected_dir}")

    _supabase_write_report(report)


if __name__ == "__main__":
    main()
