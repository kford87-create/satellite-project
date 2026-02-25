"""
scripts/10_train_multiclass.py

Trains a 4-class YOLOv8 model fine-tuned from the baseline_v1 building detector.

Fine-tuning from baseline_v1 preserves satellite-specific low-level features
(rooftop textures, shadows, overhead perspectives) learned from SpaceNet,
while adapting to vehicles, aircraft, and ships from DOTA.

Classes:
    0: building  (SpaceNet Vegas)
    1: vehicle   (DOTA large-vehicle + small-vehicle)
    2: aircraft  (DOTA plane + helicopter)
    3: ship      (DOTA ship)

Run on Vast.ai:
  1. Rent an RTX 4090 instance (~$0.30-0.50/hr)
  2. tar the project and upload
  3. pip install -r requirements.txt
  4. python scripts/10_train_multiclass.py --smoke-test  # validate first
  5. python scripts/10_train_multiclass.py              # full run

Training time estimate:
  - RTX 4090: ~4-6 hours for 100 epochs on ~42k images
  - Cost: ~$2-3 total

Performance targets:
  - Overall mAP50 >= 0.45
  - Building  >= 0.55  (must not regress from baseline)
  - Vehicle   >= 0.40
  - Ship      >= 0.45
  - Aircraft  >= 0.30
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from dotenv import load_dotenv
import torch

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR   = Path(os.getenv("DATA_DIR",   "./data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_WEIGHTS = MODELS_DIR / "baseline_v1" / "weights" / "best.pt"

TRAINING_CONFIG = {
    # Fine-tune from baseline: preserves satellite-learned features
    "model": str(BASELINE_WEIGHTS),

    "data": str(DATA_DIR / "multiclass" / "dataset.yaml"),

    # Training hyperparameters
    "epochs":        100,       # More epochs for larger, multi-class dataset
    "imgsz":         640,
    "batch":         16,        # Adjusted by check_gpu()
    "patience":      20,        # Extra patience for multi-class convergence
    "lr0":           0.005,     # Lower LR for fine-tuning (vs 0.01 for scratch)
    "lrf":           0.0005,    # Proportionally lower final LR
    "momentum":      0.937,
    "weight_decay":  0.0005,

    # Augmentation — unchanged from baseline (satellite-appropriate)
    "degrees":       180,       # Full rotation (key for satellite imagery)
    "flipud":        0.5,
    "fliplr":        0.5,
    "mosaic":        1.0,
    "scale":         0.5,
    "copy_paste":    0.1,       # Helps minority class (aircraft) via augmentation

    # Output
    "project": str(MODELS_DIR),
    "name":    "multiclass_v1",
    "save":    True,
    "plots":   True,
}

# Smoke-test overrides: fast validation that the pipeline works
SMOKE_TEST_OVERRIDES = {
    "epochs":  2,
    "batch":   8,
    "workers": 0,
}

CLASS_NAMES = ["building", "vehicle", "aircraft", "ship"]
PERFORMANCE_TARGETS = {
    "overall":  0.45,
    "building": 0.55,
    "vehicle":  0.40,
    "ship":     0.45,
    "aircraft": 0.30,
}


def check_gpu() -> bool:
    """Verify GPU and adjust batch size."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU detected: {gpu_name} ({vram:.1f}GB VRAM)")

        if vram < 8:
            TRAINING_CONFIG["batch"] = 8
            print(f"   Adjusted batch size to 8 for {vram:.1f}GB VRAM")
        elif vram >= 24:
            TRAINING_CONFIG["batch"] = 32
            print(f"   Adjusted batch size to 32 for {vram:.1f}GB VRAM")
        return True
    else:
        print("⚠️  No GPU detected. Training on CPU will be very slow.")
        print("   Recommend running on Vast.ai GPU instance.")
        TRAINING_CONFIG["batch"] = 4
        return False


def validate_dataset(yaml_path: Path) -> bool:
    """Check dataset exists before training."""
    if not yaml_path.exists():
        print(f"❌ Dataset config not found: {yaml_path}")
        print("   Run scripts/09_merge_datasets.py first.")
        return False

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    data_path = Path(config["path"])
    train_imgs = data_path / config["train"]

    if not train_imgs.exists():
        print(f"❌ Training images not found: {train_imgs}")
        return False

    n_images = len(list(train_imgs.glob("*.jpg")))
    print(f"✅ Dataset found: {n_images:,} training images")

    # Sanity: warn if unexpectedly small
    if n_images < 1000:
        print(f"⚠️  Only {n_images} training images — expected ~42k.")
        print("   Check that scripts/08 and 09 completed successfully.")

    # Print class distribution from a sample
    lbl_dir = data_path / "train" / "labels"
    if lbl_dir.exists():
        from collections import Counter
        import random
        label_files = list(lbl_dir.glob("*.txt"))
        sample = random.sample(label_files, min(500, len(label_files)))
        counts: Counter = Counter()
        for lf in sample:
            for line in lf.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1
        print(f"   Class distribution (sampled from {len(sample)} files):")
        for cid, name in enumerate(CLASS_NAMES):
            print(f"     {cid} {name}: {counts[cid]:,}")

    return True


def patch_dataset_yaml_path(yaml_path: Path) -> None:
    """
    Ensure the 'path' field in dataset.yaml is absolute.
    Required on Vast.ai where DATA_DIR env var overrides the default path.
    """
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    expected_path = str((DATA_DIR / "multiclass").resolve())
    if config.get("path") != expected_path:
        config["path"] = expected_path
        yaml_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
        print(f"   Patched dataset.yaml path → {expected_path}")


def validate_baseline_weights() -> bool:
    """Check that fine-tuning weights exist."""
    if not BASELINE_WEIGHTS.exists():
        print(f"❌ Baseline weights not found: {BASELINE_WEIGHTS}")
        print("   Run scripts/04_train_baseline.py first.")
        return False
    print(f"✅ Fine-tuning from: {BASELINE_WEIGHTS}")
    return True


def run_rotation_invariance_test(model_path: Path) -> None:
    """
    Auto-run rotation invariance tester on the trained model.
    Critical for vehicle and aircraft which appear at all orientations.
    """
    tester_path = Path(__file__).parent.parent / "tools" / "model_performance" / "rotation_invariance_tester.py"
    if not tester_path.exists():
        print(f"⚠️  Rotation invariance tester not found: {tester_path}")
        return

    val_images = DATA_DIR / "multiclass" / "val" / "images"
    if not val_images.exists():
        print("⚠️  Val images not found, skipping rotation invariance test.")
        return

    print("\n🔄 Running rotation invariance test (vehicle, aircraft critical)...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(tester_path),
             "--images-dir", str(val_images),
             "--model", str(model_path),
             "--angles", "0", "45", "90", "135", "180", "225", "270", "315"],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"⚠️  Rotation tester exited {result.returncode}")
            print(result.stderr[-2000:] if result.stderr else "")
    except Exception as exc:
        print(f"⚠️  Rotation invariance test skipped: {exc}")


def print_performance_summary(results) -> None:
    """Print final metrics and compare against targets."""
    print(f"\n📊 Final Metrics:")
    overall_map = results.results_dict.get("metrics/mAP50(B)", 0)
    print(f"   mAP50:     {overall_map:.4f}  (target ≥ {PERFORMANCE_TARGETS['overall']})")
    print(f"   mAP50-95:  {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"   Precision: {results.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"   Recall:    {results.results_dict.get('metrics/recall(B)', 0):.4f}")

    print(f"\n   Performance vs targets:")
    for name, target in PERFORMANCE_TARGETS.items():
        if name == "overall":
            val = overall_map
        else:
            # YOLO per-class keys vary; log overall only with certainty
            val = None

        if val is not None:
            status = "✅" if val >= target else "❌"
            print(f"   {status} {name}: {val:.4f} (target ≥ {target})")

    if overall_map < PERFORMANCE_TARGETS["overall"]:
        print(f"\n⚠️  Overall mAP50 {overall_map:.3f} below target {PERFORMANCE_TARGETS['overall']}")
        print("   Consider: more epochs, lower lr0, or aircraft oversampling (script 09)")

    # Aircraft-specific reminder
    print("\n   If aircraft mAP50 < 0.25 after per-class eval:")
    print("   python scripts/09_merge_datasets.py --oversample-aircraft 3 && retrain")


def train(smoke_test: bool = False):
    """Run 4-class YOLOv8 fine-tuning."""
    from ultralytics import YOLO

    print("🛰️  Training 4-Class YOLOv8 (Multi-Class Expansion)")
    print("=" * 55)

    # Pre-flight checks
    check_gpu()

    if not validate_baseline_weights():
        return

    yaml_path = Path(TRAINING_CONFIG["data"])
    patch_dataset_yaml_path(yaml_path)

    if not validate_dataset(yaml_path):
        return

    # Apply smoke-test overrides
    if smoke_test:
        print("\n⚡ SMOKE TEST MODE: 2 epochs, 100 images")
        TRAINING_CONFIG.update(SMOKE_TEST_OVERRIDES)
        # Limit images for smoke test
        TRAINING_CONFIG["fraction"] = min(
            100 / max(len(list((DATA_DIR / "multiclass" / "train" / "images").glob("*.jpg"))), 1),
            1.0,
        )

    print(f"\n🚀 Starting training with config:")
    for k, v in TRAINING_CONFIG.items():
        print(f"   {k}: {v}")

    # Pop model key before passing to train()
    config = dict(TRAINING_CONFIG)
    model_path = config.pop("model")
    model = YOLO(model_path)

    # Train
    results = model.train(**config)

    best_model_path = MODELS_DIR / "multiclass_v1" / "weights" / "best.pt"
    print(f"\n✅ Training complete!")
    print_performance_summary(results)

    print(f"\n💾 Best model saved to: {best_model_path}")

    if not smoke_test:
        run_rotation_invariance_test(best_model_path)

        print(f"\n➡️  Next steps:")
        print(f"   1. python tools/model_performance/confidence_calibrator.py \\")
        print(f"         --model {best_model_path} \\")
        print(f"         --val-dir data/multiclass/val/images \\")
        print(f"         --label-dir data/multiclass/val/labels")
        print(f"   2. Upload {best_model_path} to HuggingFace Space")
        print(f"   3. Set MODEL_PATH secret to ./models/multiclass_v1/weights/best.pt")
    else:
        print("\n✅ Smoke test passed. Run without --smoke-test for full training.")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train 4-class YOLOv8 fine-tuned from baseline_v1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run 2 epochs on ~100 images to validate the pipeline before full training",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR,
        help="Root data directory (DATA_DIR env var or ./data)",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=MODELS_DIR,
        help="Models output directory",
    )
    args = parser.parse_args()

    global DATA_DIR, MODELS_DIR, BASELINE_WEIGHTS
    DATA_DIR   = Path(args.data_dir)
    MODELS_DIR = Path(args.models_dir)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_WEIGHTS = MODELS_DIR / "baseline_v1" / "weights" / "best.pt"

    TRAINING_CONFIG["data"]    = str(DATA_DIR / "multiclass" / "dataset.yaml")
    TRAINING_CONFIG["model"]   = str(BASELINE_WEIGHTS)
    TRAINING_CONFIG["project"] = str(MODELS_DIR)

    train(smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
