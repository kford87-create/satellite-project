"""
change_detection_engine.py
--------------------------
Detect changes between two satellite images using YOLOv8 inference and
ORB-based image alignment.

CLI:
    python tools/commercial/change_detection_engine.py \\
      --before data/scene_2023_01.jpg \\
      --after data/scene_2024_01.jpg \\
      --model models/baseline_v1/weights/best.pt \\
      --output data/change_reports
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
from PIL import Image, ImageDraw
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV_PATH = Path("/Users/kahlil/satellite-project/.env")
load_dotenv(dotenv_path=ENV_PATH)

import os  # noqa: E402 – after load_dotenv

# ---------------------------------------------------------------------------
# Supabase (non-fatal)
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://obdsgqjkjjmmtbcfjhnn.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

try:
    from supabase import create_client, Client as SupabaseClient

    supabase: SupabaseClient | None = (
        create_client(SUPABASE_URL, SUPABASE_KEY)
        if SUPABASE_URL and SUPABASE_KEY
        else None
    )
except Exception:
    supabase = None

# ---------------------------------------------------------------------------
# Optional OpenCV import
# ---------------------------------------------------------------------------
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IOU_UNCHANGED_THRESH = 0.5
IOU_MOVED_LOW_THRESH = 0.2

CHANGE_COLORS: dict[str, tuple[int, int, int]] = {
    "appeared": (59, 130, 246),   # blue
    "disappeared": (239, 68, 68),  # red
    "moved": (234, 179, 8),        # yellow
    "unchanged": (107, 114, 128),  # gray
    "before": (34, 197, 94),       # green (before image boxes)
}

HIGH_VALUE_ALERT_CLASSES: dict[str, str] = {
    "building": "disappeared",
    "vehicle": "appeared",
    "car": "appeared",
    "truck": "appeared",
}

MODEL_PATH_DEFAULT = os.environ.get(
    "MODEL_PATH", "models/baseline_v1/weights/best.pt"
)


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def _compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] pixel format."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    ix1 = max(xa1, xb1)
    iy1 = max(ya1, yb1)
    ix2 = min(xa2, xb2)
    iy2 = min(ya2, yb2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, (xa2 - xa1) * (ya2 - ya1))
    area_b = max(0.0, (xb2 - xb1) * (yb2 - yb1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


# ---------------------------------------------------------------------------
# Image alignment
# ---------------------------------------------------------------------------

def _align_images_orb(before_arr: np.ndarray, after_arr: np.ndarray) -> np.ndarray:
    """
    Align *after_arr* to *before_arr* using ORB feature matching + homography.
    Falls back to returning *after_arr* unchanged if alignment fails.
    Requires OpenCV.
    """
    gray_before = cv2.cvtColor(before_arr, cv2.COLOR_RGB2GRAY)
    gray_after = cv2.cvtColor(after_arr, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(gray_before, None)
    kp2, des2 = orb.detectAndCompute(gray_after, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("⚠️  ORB: insufficient keypoints – skipping alignment")
        return after_arr

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    if len(matches) < 4:
        print("⚠️  ORB: too few matches – skipping alignment")
        return after_arr

    top_n = max(10, len(matches) // 4)
    good = matches[:top_n]

    pts_before = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_after  = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_after, pts_before, cv2.RANSAC, 5.0)
    if H is None:
        print("⚠️  ORB: homography not found – skipping alignment")
        return after_arr

    h, w = before_arr.shape[:2]
    aligned = cv2.warpPerspective(after_arr, H, (w, h))
    return aligned


def _align_images_fallback(before_arr: np.ndarray, after_arr: np.ndarray) -> np.ndarray:
    """
    Homography-based alignment fallback when OpenCV is unavailable.
    Simply resizes the after image to match before image dimensions.
    """
    h, w = before_arr.shape[:2]
    after_pil = Image.fromarray(after_arr).resize((w, h), Image.LANCZOS)
    return np.array(after_pil)


def align_images(before_arr: np.ndarray, after_arr: np.ndarray) -> np.ndarray:
    """Align *after* to *before*. Uses ORB when OpenCV is available."""
    if CV2_AVAILABLE:
        try:
            return _align_images_orb(before_arr, after_arr)
        except Exception as exc:
            print(f"⚠️  ORB alignment error ({exc}) – using resize fallback")
    return _align_images_fallback(before_arr, after_arr)


# ---------------------------------------------------------------------------
# YOLOv8 inference
# ---------------------------------------------------------------------------

def _run_inference(model: Any, image_arr: np.ndarray) -> list[dict]:
    """
    Run YOLOv8 on a numpy RGB array.
    Returns list of {class_name, confidence, bbox: [x1, y1, x2, y2]}.
    """
    results = model(image_arr, verbose=False)
    detections: list[dict] = []
    for r in results:
        if r.boxes is None:
            continue
        names = r.names
        for box_data in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in box_data.xyxy[0].tolist()]
            conf = float(box_data.conf[0])
            cls_id = int(box_data.cls[0])
            class_name = names[cls_id] if names and cls_id in names else str(cls_id)
            detections.append(
                {
                    "class_name": class_name,
                    "class_id": cls_id,
                    "confidence": round(conf, 4),
                    "bbox_xyxy": [x1, y1, x2, y2],
                }
            )
    return detections


# ---------------------------------------------------------------------------
# Change classification
# ---------------------------------------------------------------------------

def classify_changes(
    before_dets: list[dict],
    after_dets: list[dict],
) -> list[dict]:
    """
    Match detections between before/after and classify each as:
      - unchanged  (IoU >= 0.5, same class)
      - moved      (0.2 <= IoU < 0.5, same class)
      - appeared   (in after, no match in before)
      - disappeared (in before, no match in after)
    Returns list of change records.
    """
    matched_before: set[int] = set()
    matched_after: set[int] = set()
    changes: list[dict] = []

    # For each after detection, find the best before match
    for ai, adet in enumerate(after_dets):
        best_iou = 0.0
        best_bi = -1
        for bi, bdet in enumerate(before_dets):
            if bi in matched_before:
                continue
            iou = _compute_iou(adet["bbox_xyxy"], bdet["bbox_xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_bi = bi

        x1, y1, x2, y2 = adet["bbox_xyxy"]
        w = x2 - x1
        h = y2 - y1
        bbox_dict = {
            "x_center": round((x1 + x2) / 2, 2),
            "y_center": round((y1 + y2) / 2, 2),
            "width": round(w, 2),
            "height": round(h, 2),
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2),
        }

        same_class = (
            best_bi >= 0
            and before_dets[best_bi]["class_name"] == adet["class_name"]
        )

        if best_iou >= IOU_UNCHANGED_THRESH and same_class:
            change_type = "unchanged"
            matched_before.add(best_bi)
            matched_after.add(ai)
        elif IOU_MOVED_LOW_THRESH <= best_iou < IOU_UNCHANGED_THRESH and same_class:
            change_type = "moved"
            matched_before.add(best_bi)
            matched_after.add(ai)
        else:
            change_type = "appeared"
            matched_after.add(ai)

        changes.append(
            {
                "class": adet["class_name"],
                "change_type": change_type,
                "confidence": adet["confidence"],
                "bbox": bbox_dict,
                "image": "after",
            }
        )

    # Remaining before detections with no after match → disappeared
    for bi, bdet in enumerate(before_dets):
        if bi in matched_before:
            continue
        x1, y1, x2, y2 = bdet["bbox_xyxy"]
        changes.append(
            {
                "class": bdet["class_name"],
                "change_type": "disappeared",
                "confidence": bdet["confidence"],
                "bbox": {
                    "x_center": round((x1 + x2) / 2, 2),
                    "y_center": round((y1 + y2) / 2, 2),
                    "width": round(x2 - x1, 2),
                    "height": round(y2 - y1, 2),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                },
                "image": "before",
            }
        )

    return changes


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _draw_boxes(
    img: Image.Image,
    detections: list[dict],
    color: tuple[int, int, int],
    label: bool = True,
) -> Image.Image:
    """Draw bounding boxes on a PIL image copy. Returns new image."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for det in detections:
        b = det["bbox_xyxy"]
        x1, y1, x2, y2 = b
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if label:
            text = f"{det['class_name']} {det['confidence']:.2f}"
            draw.text((x1 + 2, max(0, y1 - 12)), text, fill=color)
    return out


def _draw_changes(
    img: Image.Image,
    changes: list[dict],
    side: str,
) -> Image.Image:
    """
    Draw change boxes on the after image.
    *side* is either "before" (only draw disappeared) or "after" (all others).
    """
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for ch in changes:
        if side == "before" and ch["change_type"] != "disappeared":
            continue
        if side == "after" and ch["change_type"] == "disappeared":
            continue

        b = ch["bbox"]
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        color = CHANGE_COLORS.get(ch["change_type"], (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = f"{ch['class']} [{ch['change_type']}]"
        draw.text((x1 + 2, max(0, y1 - 12)), text, fill=color)
    return out


def generate_visualization(
    before_img: Image.Image,
    after_img: Image.Image,
    before_dets: list[dict],
    changes: list[dict],
    output_path: Path,
) -> None:
    """Save side-by-side PNG: left=before (green boxes), right=after (color-coded)."""
    # Draw before image with green boxes
    before_drawn = _draw_boxes(
        before_img,
        before_dets,
        color=CHANGE_COLORS["before"],
        label=True,
    )
    # Draw after image with change-coded boxes
    after_drawn = _draw_changes(after_img, changes, side="after")
    # Also mark disappeared on before panel
    before_drawn = _draw_changes(before_drawn, changes, side="before")

    # Ensure same height for side-by-side concat
    h = max(before_drawn.height, after_drawn.height)
    w_total = before_drawn.width + after_drawn.width + 10  # 10px gap

    canvas = Image.new("RGB", (w_total, h), color=(20, 20, 20))
    canvas.paste(before_drawn, (0, 0))
    canvas.paste(after_drawn, (before_drawn.width + 10, 0))

    # Legend
    draw = ImageDraw.Draw(canvas)
    legend_items = [
        ("Before detections", CHANGE_COLORS["before"]),
        ("Appeared", CHANGE_COLORS["appeared"]),
        ("Disappeared", CHANGE_COLORS["disappeared"]),
        ("Moved", CHANGE_COLORS["moved"]),
        ("Unchanged", CHANGE_COLORS["unchanged"]),
    ]
    lx = before_drawn.width + 15
    ly = h - len(legend_items) * 18 - 10
    for label_text, color in legend_items:
        draw.rectangle([lx, ly, lx + 14, ly + 14], fill=color)
        draw.text((lx + 18, ly), label_text, fill=(220, 220, 220))
        ly += 18

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"✅ Visualization saved → {output_path}")


# ---------------------------------------------------------------------------
# Supabase write (non-fatal)
# ---------------------------------------------------------------------------

def _supabase_save_report(report: dict) -> None:
    try:
        if supabase is None:
            return
        supabase.table("change_reports").insert(
            {
                "before_image": report["before_image"],
                "after_image": report["after_image"],
                "analysis_date": report["analysis_date"],
                "summary": json.dumps(report["summary"]),
                "changes_count": len(report["changes"]),
            }
        ).execute()
        print("✅ Report metadata written to Supabase")
    except Exception as exc:
        print(f"⚠️  Supabase write skipped (falling back to local only): {exc}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_change_detection(
    before_path: Path,
    after_path: Path,
    model_path: Path,
    output_dir: Path,
    conf_threshold: float = 0.25,
) -> dict:
    """Full change detection pipeline. Returns the JSON report dict."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load images
    # ------------------------------------------------------------------
    print(f"📊 Loading before image: {before_path}")
    try:
        before_img = Image.open(before_path).convert("RGB")
    except Exception as exc:
        print(f"❌ Cannot open before image: {exc}")
        sys.exit(1)

    print(f"📊 Loading after image: {after_path}")
    try:
        after_img = Image.open(after_path).convert("RGB")
    except Exception as exc:
        print(f"❌ Cannot open after image: {exc}")
        sys.exit(1)

    before_arr = np.array(before_img)
    after_arr  = np.array(after_img)

    # ------------------------------------------------------------------
    # Align after → before
    # ------------------------------------------------------------------
    print("📊 Aligning after image to before image...")
    after_aligned_arr = align_images(before_arr, after_arr)
    after_aligned_img = Image.fromarray(after_aligned_arr)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"📊 Loading YOLOv8 model: {model_path}")
    try:
        model = YOLO(str(model_path))
    except Exception as exc:
        print(f"❌ Failed to load model: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Run inference on both images (wrapped in tqdm for UX)
    # ------------------------------------------------------------------
    inference_steps = ["before", "after"]
    before_dets: list[dict] = []
    after_dets: list[dict] = []

    for step in tqdm(inference_steps, desc="Running inference", unit="image"):
        if step == "before":
            before_dets = _run_inference(model, before_arr)
        else:
            after_dets = _run_inference(model, after_aligned_arr)

    print(
        f"📊 Detections — before: {len(before_dets)}, after: {len(after_dets)}"
    )

    # ------------------------------------------------------------------
    # Classify changes
    # ------------------------------------------------------------------
    changes = classify_changes(before_dets, after_dets)

    summary: dict[str, int] = {
        "appeared": 0,
        "disappeared": 0,
        "moved": 0,
        "unchanged": 0,
    }
    for ch in changes:
        summary[ch["change_type"]] = summary.get(ch["change_type"], 0) + 1

    # ------------------------------------------------------------------
    # Build report
    # ------------------------------------------------------------------
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report: dict = {
        "before_image": str(before_path.resolve()),
        "after_image": str(after_path.resolve()),
        "analysis_date": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "changes": [
            {
                "class": ch["class"],
                "change_type": ch["change_type"],
                "confidence": ch["confidence"],
                "bbox": ch["bbox"],
            }
            for ch in changes
        ],
    }

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{timestamp}_change_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"✅ JSON report saved → {report_path}")

    # ------------------------------------------------------------------
    # Save visualization
    # ------------------------------------------------------------------
    viz_path = output_dir / f"{timestamp}_change_visualization.png"
    generate_visualization(
        before_img=before_img,
        after_img=after_aligned_img,
        before_dets=before_dets,
        changes=changes,
        output_path=viz_path,
    )

    # ------------------------------------------------------------------
    # Supabase (non-fatal)
    # ------------------------------------------------------------------
    _supabase_save_report(report)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("📊 Change Detection Summary")
    print("=" * 50)
    print(f"  Appeared:    {summary['appeared']}")
    print(f"  Disappeared: {summary['disappeared']}")
    print(f"  Moved:       {summary['moved']}")
    print(f"  Unchanged:   {summary['unchanged']}")
    print("=" * 50)

    # High-value alerts
    alerts: list[str] = []
    for ch in changes:
        cls = ch["class"].lower()
        ctype = ch["change_type"]
        for alert_cls, alert_type in HIGH_VALUE_ALERT_CLASSES.items():
            if alert_cls in cls and ctype == alert_type:
                alerts.append(
                    f"  ⚠️  HIGH-VALUE ALERT: {ch['class']} {ctype} "
                    f"(conf={ch['confidence']:.2f})"
                )

    if alerts:
        print("\n📊 High-Value Alerts:")
        for alert in alerts:
            print(alert)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YOLOv8 satellite image change detection engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--before",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to the 'before' satellite image",
    )
    parser.add_argument(
        "--after",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to the 'after' satellite image",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(MODEL_PATH_DEFAULT),
        metavar="PATH",
        help="Path to YOLOv8 .pt weights file",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="DIR",
        help="Output directory for reports and visualizations",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        metavar="FLOAT",
        help="YOLOv8 confidence threshold",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    before_path: Path = args.before.resolve()
    after_path: Path  = args.after.resolve()
    model_path: Path  = args.model.resolve()
    output_dir: Path  = args.output

    if not before_path.exists():
        print(f"❌ Before image not found: {before_path}")
        sys.exit(1)
    if not after_path.exists():
        print(f"❌ After image not found: {after_path}")
        sys.exit(1)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    run_change_detection(
        before_path=before_path,
        after_path=after_path,
        model_path=model_path,
        output_dir=output_dir,
        conf_threshold=args.conf,
    )


if __name__ == "__main__":
    main()
