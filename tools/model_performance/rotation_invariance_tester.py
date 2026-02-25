"""
rotation_invariance_tester.py

Test a YOLOv8 model's sensitivity to image rotation across 8 canonical angles.

CLI:
    python tools/model_performance/rotation_invariance_tester.py \
      --model models/baseline_v1/weights/best.pt \
      --test-dir data/yolo_format/val/images
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
# Constants
# ---------------------------------------------------------------------------
ANGLES: list[int] = [0, 45, 90, 135, 180, 225, 270, 315]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotate_image(img, angle: int):
    """Rotate a PIL Image by `angle` degrees counter-clockwise."""
    from PIL import Image  # local import to keep top-level clean

    if isinstance(img, np.ndarray):
        from PIL import Image as PILImage
        img = PILImage.fromarray(img)
    return img.rotate(angle, expand=True)


def _count_detections(model, img_path: Path, angle: int,
                      conf_thresh: float) -> int:
    """Rotate img_path by angle and return total detection count."""
    import tempfile
    from PIL import Image

    try:
        pil_img = Image.open(img_path).convert("RGB")
    except Exception as exc:
        print(f"⚠️ Could not open {img_path.name}: {exc}")
        return 0

    rotated = _rotate_image(pil_img, angle)

    # Save to a temp file so ultralytics can read it normally
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        rotated.save(tmp_path)

    try:
        results = model(str(tmp_path), verbose=False, conf=conf_thresh)
        count = sum(
            len(r.boxes) for r in results if r.boxes is not None
        )
    except Exception as exc:
        print(f"⚠️ Inference failed ({img_path.name}, {angle}°): {exc}")
        count = 0
    finally:
        tmp_path.unlink(missing_ok=True)

    return count


# ---------------------------------------------------------------------------
# Core test
# ---------------------------------------------------------------------------

def run_rotation_test(model_path: Path, test_dir: Path,
                      conf_thresh: float = 0.25,
                      max_images: int | None = None) -> dict:
    """
    For each angle in ANGLES, run inference on all test images and return
    normalised detection rates relative to the 0° baseline.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    print(f"📊 Loading model from {model_path}")
    try:
        model = YOLO(str(model_path))
    except Exception as exc:
        print(f"❌ Failed to load model: {exc}")
        sys.exit(1)

    image_paths = sorted(
        p for p in test_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )
    if not image_paths:
        print(f"❌ No images found in {test_dir}")
        sys.exit(1)

    if max_images is not None:
        image_paths = image_paths[:max_images]

    print(f"📊 Testing on {len(image_paths)} image(s) across {len(ANGLES)} angles")

    # angle -> total detections across all images
    angle_totals: dict[int, int] = {a: 0 for a in ANGLES}

    for img_path in tqdm(image_paths, desc="Processing images", unit="img"):
        for angle in tqdm(ANGLES, desc=f"  {img_path.name}", leave=False, unit="°"):
            count = _count_detections(model, img_path, angle, conf_thresh)
            angle_totals[angle] += count

    baseline = angle_totals[0]
    if baseline == 0:
        print("⚠️ Zero detections at 0° baseline – normalisation will use 1 to avoid division by zero")
        baseline = 1

    detection_rates: list[float] = [
        round(angle_totals[a] / baseline, 4) for a in ANGLES
    ]

    worst_idx = int(np.argmin(detection_rates))
    best_idx = int(np.argmax(detection_rates))

    worst_angle = ANGLES[worst_idx]
    worst_rate = detection_rates[worst_idx]
    best_angle = ANGLES[best_idx]
    best_rate = detection_rates[best_idx]

    # Rotation sensitivity score: std of detection rates (lower = more stable)
    sensitivity_score = round(float(np.std(detection_rates)), 4)

    # Recommendation
    if worst_rate < 0.75:
        recommendation = (
            "Model shows significant rotation sensitivity "
            f"(worst rate {worst_rate:.2f} at {worst_angle}°). "
            "Consider augmenting training data with rotated samples."
        )
    elif worst_rate < 0.90:
        recommendation = (
            "Model shows moderate rotation sensitivity. "
            "Consider augmenting training data with rotated samples."
        )
    else:
        recommendation = (
            "Model is reasonably rotation-invariant. "
            "Minor augmentation may still improve robustness."
        )

    return {
        "angles": ANGLES,
        "detection_rates": detection_rates,
        "angle_totals": {str(a): angle_totals[a] for a in ANGLES},
        "baseline_detections": angle_totals[0],
        "worst_angle": worst_angle,
        "worst_rate": worst_rate,
        "best_angle": best_angle,
        "best_rate": best_rate,
        "rotation_sensitivity_score": sensitivity_score,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _generate_polar_plot(results: dict, output_path: Path) -> None:
    """
    Polar chart: r = normalised detection rate, theta = rotation angle.
    - Area under curve filled in blue.
    - 0° baseline marked in red.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    angles_deg = results["angles"]
    rates = results["detection_rates"]

    # Convert degrees to radians; close the loop for fill
    theta = np.deg2rad(angles_deg + [angles_deg[0]])
    r = rates + [rates[0]]

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))

    # Filled area
    ax.fill(theta, r, color="steelblue", alpha=0.35)
    ax.plot(theta, r, color="steelblue", linewidth=2, label="Detection rate")

    # 0° baseline marker
    baseline_theta = 0.0  # 0° in radians
    ax.plot(
        [baseline_theta, baseline_theta],
        [0, 1.0],
        color="red",
        linewidth=2,
        linestyle="--",
        label="0° baseline (1.0)",
    )

    # Angle labels
    ax.set_thetagrids(angles_deg, labels=[f"{a}°" for a in angles_deg])
    ax.set_rlabel_position(45)
    ax.set_title(
        "Rotation Invariance – Normalised Detection Rate",
        va="bottom",
        pad=20,
    )
    ax.set_ylim(0, max(1.2, max(r) * 1.1))
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"✅ Saved polar plot → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test YOLOv8 model rotation invariance across 8 canonical angles"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("MODEL_PATH", "models/baseline_v1/weights/best.pt")),
        help="Path to YOLOv8 .pt weights file",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Directory containing test images",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for inference (default: 0.25)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of images to test (useful for quick runs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save output files (defaults to --test-dir parent)",
    )
    args = parser.parse_args()

    # Validate paths
    if not args.model.exists():
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)
    if not args.test_dir.is_dir():
        print(f"❌ Test directory not found: {args.test_dir}")
        sys.exit(1)

    output_dir = args.output_dir or args.test_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run test
    results = run_rotation_test(
        args.model, args.test_dir, args.conf, args.max_images
    )

    # Polar plot
    _generate_polar_plot(results, output_dir / "polar_plot.png")

    # Save report (subset matching documented schema)
    report = {
        "angles": results["angles"],
        "detection_rates": results["detection_rates"],
        "worst_angle": results["worst_angle"],
        "worst_rate": results["worst_rate"],
        "recommendation": results["recommendation"],
    }
    report_path = output_dir / "rotation_invariance_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"✅ Saved report → {report_path}")

    # Summary printout
    print("\n" + "=" * 55)
    print("📊 Rotation Invariance Summary")
    print("=" * 55)
    print(f"  Best angle:               {results['best_angle']}° "
          f"(rate: {results['best_rate']:.4f})")
    print(f"  Worst angle:              {results['worst_angle']}° "
          f"(rate: {results['worst_rate']:.4f})")
    print(f"  Rotation sensitivity std: {results['rotation_sensitivity_score']:.4f}")
    print(f"\n  Recommendation: {results['recommendation']}")

    print("\n  Per-angle detection rates:")
    for angle, rate in zip(results["angles"], results["detection_rates"]):
        bar = "#" * int(rate * 20)
        print(f"    {angle:>3}°  [{bar:<20}]  {rate:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
