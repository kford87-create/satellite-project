"""
pseudo_label_scorer.py
-----------------------
Runs one or more YOLOv8 checkpoints over a query image directory, collects
ensemble detection statistics, and assigns each image to an annotation tier
based on pseudo-label confidence.

Usage:
    # Single checkpoint
    python tools/active_learning/pseudo_label_scorer.py \
      --query-dir data/bootstrapped/iteration_01_query \
      --checkpoints models/baseline_v1/weights/best.pt \
      --output data/bootstrapped/scored_queue.json

    # Ensemble (multiple checkpoints)
    python tools/active_learning/pseudo_label_scorer.py \
      --query-dir data/bootstrapped/iteration_01_query \
      --checkpoints models/v1/weights/best.pt models/v2/weights/best.pt \
      --output data/bootstrapped/scored_queue.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV_PATH = Path("/Users/kahlil/satellite-project/.env")
load_dotenv(ENV_PATH)

import os  # noqa: E402 – after dotenv

# ---------------------------------------------------------------------------
# Supabase (non-fatal)
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://obdsgqjkjjmmtbcfjhnn.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

try:
    from supabase import create_client, Client as SupabaseClient

    _supabase: SupabaseClient | None = (
        create_client(SUPABASE_URL, SUPABASE_KEY)
        if SUPABASE_URL and SUPABASE_KEY
        else None
    )
except Exception:
    _supabase = None


def _supabase_write_scored_queue(output: dict[str, Any]) -> None:
    """Non-fatal write of the scored queue summary to Supabase."""
    try:
        if _supabase is None:
            return
        _supabase.table("pseudo_label_scored_queues").insert(
            {
                "scored_at": output["scored_at"],
                "total_images": output["total_images"],
                "stats": json.dumps(output["stats"]),
            }
        ).execute()
    except Exception as exc:
        print(f"⚠️  Supabase write skipped (falling back to local only): {exc}")


# ---------------------------------------------------------------------------
# YOLO inference helpers
# ---------------------------------------------------------------------------

def _load_yolo_models(checkpoint_paths: list[Path]) -> list[Any]:
    """Load YOLOv8 models from checkpoint paths. Exits on hard failure."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    models = []
    for ckpt in checkpoint_paths:
        if not ckpt.exists():
            print(f"❌ Checkpoint not found: {ckpt}")
            sys.exit(1)
        try:
            model = YOLO(str(ckpt))
            models.append(model)
            print(f"✅ Loaded checkpoint: {ckpt.name}")
        except Exception as exc:
            print(f"❌ Failed to load checkpoint {ckpt}: {exc}")
            sys.exit(1)

    return models


def _run_inference_single(model: Any, image_path: Path) -> list[dict[str, Any]]:
    """
    Run a single YOLO model on *image_path*.

    Returns a list of detection dicts:
        {"bbox": [x1, y1, x2, y2], "class_id": int, "confidence": float}
    """
    try:
        results = model.predict(str(image_path), verbose=False, conf=0.01)
        detections: list[dict[str, Any]] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                detections.append(
                    {
                        "bbox": [round(v, 2) for v in xyxy],
                        "class_id": cls,
                        "confidence": round(conf, 6),
                    }
                )
        return detections
    except Exception as exc:
        print(f"⚠️  Inference error on {image_path.name}: {exc}")
        return []


# ---------------------------------------------------------------------------
# Detection matching & ensemble aggregation
# ---------------------------------------------------------------------------

def _iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _match_detections_across_models(
    all_model_detections: list[list[dict[str, Any]]],
    iou_threshold: float = 0.45,
) -> list[dict[str, Any]]:
    """
    Given per-model detection lists, group detections that overlap (IoU >= threshold)
    across models and compute ensemble statistics.

    Returns a list of aggregated detection records:
        {
            "bbox": [...],          # from the model with highest confidence
            "class_id": int,
            "mean_conf": float,
            "std_conf": float,
            "pseudo_label_confidence": float,
            "n_models_detecting": int,
        }
    """
    n_models = len(all_model_detections)

    if n_models == 0:
        return []

    # Single model: trivial case
    if n_models == 1:
        results = []
        for det in all_model_detections[0]:
            conf = det["confidence"]
            results.append(
                {
                    "bbox": det["bbox"],
                    "class_id": det["class_id"],
                    "mean_conf": round(conf, 6),
                    "std_conf": 0.0,
                    "pseudo_label_confidence": round(conf, 6),
                    "n_models_detecting": 1,
                }
            )
        return results

    # Multi-model: match detections greedily by IoU
    # Build a flat list of (model_idx, det)
    flat: list[tuple[int, dict[str, Any]]] = []
    for m_idx, detections in enumerate(all_model_detections):
        for det in detections:
            flat.append((m_idx, det))

    # Group overlapping detections
    visited = [False] * len(flat)
    groups: list[list[tuple[int, dict[str, Any]]]] = []

    for i, (m_i, det_i) in enumerate(flat):
        if visited[i]:
            continue
        group = [(m_i, det_i)]
        visited[i] = True
        for j, (m_j, det_j) in enumerate(flat):
            if visited[j] or m_i == m_j:
                continue
            if _iou(det_i["bbox"], det_j["bbox"]) >= iou_threshold:
                group.append((m_j, det_j))
                visited[j] = True
        groups.append(group)

    # Aggregate each group
    results = []
    for group in groups:
        confs = [d["confidence"] for _, d in group]
        mean_conf = float(np.mean(confs))
        std_conf = float(np.std(confs))
        best_det = max(group, key=lambda x: x[1]["confidence"])[1]

        # Penalise detections seen by fewer models (lower agreement = lower confidence)
        detection_ratio = len(group) / n_models
        pseudo_confidence = mean_conf * detection_ratio

        results.append(
            {
                "bbox": best_det["bbox"],
                "class_id": best_det["class_id"],
                "mean_conf": round(mean_conf, 6),
                "std_conf": round(std_conf, 6),
                "pseudo_label_confidence": round(pseudo_confidence, 6),
                "n_models_detecting": len(group),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Tiering
# ---------------------------------------------------------------------------

# Annotation time estimates (minutes per detection)
_TIME_PER_DETECTION = {
    "quick_confirm": 0.25,  # ~15 s – just confirm the box
    "normal_review": 1.0,   # ~1 min – adjust or reject
    "careful_review": 3.0,  # ~3 min – draw from scratch likely
}

TIER_THRESHOLDS = {
    "quick_confirm": 0.80,
    "normal_review": 0.50,
}


def _assign_tier(pseudo_confidence: float) -> str:
    if pseudo_confidence >= TIER_THRESHOLDS["quick_confirm"]:
        return "quick_confirm"
    if pseudo_confidence >= TIER_THRESHOLDS["normal_review"]:
        return "normal_review"
    return "careful_review"


def _image_tier(detections: list[dict[str, Any]]) -> str:
    """
    Assign the most conservative tier that applies to any detection in the image.
    Images with zero detections get careful_review (annotator must confirm empty).
    """
    if not detections:
        return "careful_review"
    tiers = [_assign_tier(d["pseudo_label_confidence"]) for d in detections]
    # Order: careful < normal < quick
    tier_order = {"careful_review": 0, "normal_review": 1, "quick_confirm": 2}
    return min(tiers, key=lambda t: tier_order[t])


def _image_confidence(detections: list[dict[str, Any]]) -> float:
    """Aggregate pseudo-label confidence for a whole image (mean of detection confidences)."""
    if not detections:
        return 0.0
    return float(np.mean([d["pseudo_label_confidence"] for d in detections]))


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------

def run_scorer(
    query_dir: Path,
    checkpoint_paths: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    # ---- Load models --------------------------------------------------------
    print(f"📊 Loading {len(checkpoint_paths)} checkpoint(s)...")
    models = _load_yolo_models(checkpoint_paths)
    n_models = len(models)

    # ---- Discover images ----------------------------------------------------
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    images = sorted(
        p for p in query_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    if not images:
        print(f"❌ No images found in: {query_dir}")
        sys.exit(1)

    print(f"📊 {len(images):,} images found in {query_dir}")
    print(f"📊 Running ensemble inference ({n_models} model(s)) ...")

    # ---- Inference loop -----------------------------------------------------
    tiers: dict[str, list[dict[str, Any]]] = {
        "quick_confirm": [],
        "normal_review": [],
        "careful_review": [],
    }

    for img_path in tqdm(images, desc="Scoring images", unit="img"):
        # Run each model
        per_model_detections: list[list[dict[str, Any]]] = []
        for model in models:
            dets = _run_inference_single(model, img_path)
            per_model_detections.append(dets)

        # Aggregate
        aggregated = _match_detections_across_models(per_model_detections)
        image_conf = _image_confidence(aggregated)
        tier = _image_tier(aggregated)

        record: dict[str, Any] = {
            "filename": img_path.name,
            "detections": aggregated,
            "confidence": round(image_conf, 6),
            "n_detections": len(aggregated),
        }
        tiers[tier].append(record)

    # ---- Statistics ---------------------------------------------------------
    total = len(images)
    n_quick = len(tiers["quick_confirm"])
    n_normal = len(tiers["normal_review"])
    n_careful = len(tiers["careful_review"])

    stats = {
        "quick_confirm_pct": round(100.0 * n_quick / max(total, 1), 1),
        "normal_review_pct": round(100.0 * n_normal / max(total, 1), 1),
        "careful_review_pct": round(100.0 * n_careful / max(total, 1), 1),
    }

    # Estimate annotation time savings
    # Baseline: assume all careful_review (3 min per image, 1 det avg assumed)
    baseline_time = total * _TIME_PER_DETECTION["careful_review"]
    actual_time = (
        n_quick * _TIME_PER_DETECTION["quick_confirm"]
        + n_normal * _TIME_PER_DETECTION["normal_review"]
        + n_careful * _TIME_PER_DETECTION["careful_review"]
    )
    time_saved = baseline_time - actual_time
    time_saved_pct = 100.0 * time_saved / max(baseline_time, 1.0)

    # ---- Build output -------------------------------------------------------
    output: dict[str, Any] = {
        "scored_at": datetime.now(tz=timezone.utc).isoformat(),
        "total_images": total,
        "n_models": n_models,
        "checkpoints": [str(p) for p in checkpoint_paths],
        "tiers": tiers,
        "stats": stats,
    }

    # ---- Print summary ------------------------------------------------------
    print(f"\n📊 Scoring complete — {total:,} images:")
    print(f"   quick_confirm  (conf >= 0.80) : {n_quick:4d}  ({stats['quick_confirm_pct']:5.1f}%)")
    print(f"   normal_review  (conf  0.50-0.80): {n_normal:4d}  ({stats['normal_review_pct']:5.1f}%)")
    print(f"   careful_review (conf <  0.50) : {n_careful:4d}  ({stats['careful_review_pct']:5.1f}%)")
    print()
    print(f"📊 Estimated annotation time:")
    print(f"   Baseline (all careful): {baseline_time:.0f} min")
    print(f"   With pre-scoring      : {actual_time:.0f} min")
    print(f"   Time saved            : {time_saved:.0f} min  ({time_saved_pct:.1f}%)")

    # ---- Save output --------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\n✅ Scored queue saved → {output_path}")

    # ---- Supabase (non-fatal) -----------------------------------------------
    _supabase_write_scored_queue(output)

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score a query set of images using one or more YOLOv8 checkpoints. "
            "Assigns each image to an annotation tier based on ensemble confidence."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory of images to score (the active-learning query set)",
    )
    parser.add_argument(
        "--checkpoints",
        required=True,
        nargs="+",
        type=Path,
        metavar="PT",
        help="One or more YOLOv8 .pt checkpoint paths (space-separated for ensemble)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="JSON",
        help="Output path for the scored queue JSON file",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    query_dir: Path = args.query_dir.resolve()
    checkpoint_paths: list[Path] = [p.resolve() for p in args.checkpoints]
    output_path: Path = args.output.resolve()

    if not query_dir.is_dir():
        print(f"❌ --query-dir does not exist: {query_dir}")
        sys.exit(1)

    print(f"📊 Query dir   : {query_dir}")
    print(f"📊 Checkpoints : {[str(p) for p in checkpoint_paths]}")
    print(f"📊 Output      : {output_path}")
    print()

    run_scorer(
        query_dir=query_dir,
        checkpoint_paths=checkpoint_paths,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
