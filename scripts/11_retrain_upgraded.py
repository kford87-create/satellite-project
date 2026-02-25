"""
scripts/11_retrain_upgraded.py

Trains YOLOv8s (11.2M params) on satellite imagery. Supports two data configs:

  --data-config multiclass  (default)  4-class merged SpaceNet + DOTA dataset
  --data-config spacenet               1-class SpaceNet buildings only

For multiclass, uses pretrained COCO weights (yolov8s.pt) with a fresh 4-class
head — NOT fine-tuned from baseline_v1 since it only knows 1 class.

Usage:
  python scripts/11_retrain_upgraded.py --smoke-test   # validate pipeline (2 epochs)
  python scripts/11_retrain_upgraded.py                # full 4-class training (needs GPU)
  python scripts/11_retrain_upgraded.py --data-config spacenet  # 1-class buildings only

Output:
  multiclass:  models/multiclass_v2/weights/best.pt
  spacenet:    models/baseline_v2/weights/best.pt
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Data config presets: maps --data-config value to (yaml subpath, output name)
DATA_CONFIGS = {
    "multiclass": ("multiclass/dataset.yaml", "multiclass_v2"),
    "spacenet":   ("yolo_format/dataset.yaml", "baseline_v2"),
}

CLASS_NAMES = {
    "multiclass": ["building", "vehicle", "aircraft", "ship"],
    "spacenet":   ["building"],
}

PERFORMANCE_TARGETS = {
    "multiclass": {
        "overall":  0.50,
        "building": 0.70,
        "vehicle":  0.45,
        "ship":     0.45,
        "aircraft": 0.30,
    },
    "spacenet": {
        "overall": 0.90,
    },
}

TRAINING_CONFIG = {
    # Start from pretrained YOLOv8s (COCO weights) — fresh head for 4 classes
    "model": "yolov8s.pt",

    "data": str(DATA_DIR / "multiclass" / "dataset.yaml"),

    # Training hyperparameters — tuned for satellite fine-tuning
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "patience": 20,
    "lr0": 0.003,       # Lower than default (0.01) for fine-tuning
    "lrf": 0.0003,
    "momentum": 0.937,
    "weight_decay": 0.0005,

    # Augmentation — satellite-appropriate
    "degrees": 180,      # Full rotation (overhead imagery has no "up")
    "flipud": 0.5,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "scale": 0.5,
    "copy_paste": 0.15,  # Bumped from 0.1 — helps minority class (aircraft)
    "mixup": 0.15,       # Blends training images for better generalization

    # Output
    "project": str(MODELS_DIR),
    "name": "multiclass_v2",
    "save": True,
    "plots": True,
}

SMOKE_TEST_OVERRIDES = {
    "epochs": 2,
    "batch": 8,
    "workers": 0,
}

# Active data config — set by main()
_data_config = "multiclass"


def check_gpu() -> bool:
    """Verify GPU and adjust batch size."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detected: {gpu_name} ({vram:.1f}GB VRAM)")

        if vram < 8:
            TRAINING_CONFIG["batch"] = 8
            print(f"  Adjusted batch size to 8 for {vram:.1f}GB VRAM")
        elif vram >= 24:
            TRAINING_CONFIG["batch"] = 32
            print(f"  Adjusted batch size to 32 for {vram:.1f}GB VRAM")
        return True
    else:
        print("No GPU detected. Training on CPU will be very slow.")
        print("  Recommend running on Vast.ai or similar GPU instance.")
        TRAINING_CONFIG["batch"] = 4
        return False


def validate_dataset(yaml_path: Path) -> bool:
    """Check dataset exists and print class distribution."""
    import yaml

    if not yaml_path.exists():
        print(f"Dataset config not found: {yaml_path}")
        if _data_config == "multiclass":
            print("  Run scripts/08_preprocess_dota.py and scripts/09_merge_datasets.py first.")
        else:
            print("  Run training data preparation scripts first.")
        return False

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    data_path = Path(config["path"])
    train_imgs = data_path / config["train"]

    if not train_imgs.exists():
        print(f"Training images not found: {train_imgs}")
        return False

    n_images = len(list(train_imgs.glob("*.jpg")) + list(train_imgs.glob("*.png")))
    print(f"Dataset found: {n_images:,} training images")

    if _data_config == "multiclass" and n_images < 1000:
        print(f"  Only {n_images} training images — expected ~42k for multiclass.")
        print("  Check that scripts 08 and 09 completed successfully.")

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
        names = CLASS_NAMES[_data_config]
        print(f"  Class distribution (sampled from {len(sample)} files):")
        for cid, name in enumerate(names):
            print(f"    {cid} {name}: {counts[cid]:,}")

    return True


def patch_dataset_yaml_path(yaml_path: Path) -> None:
    """Ensure the 'path' field in dataset.yaml is absolute."""
    import yaml

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    yaml_subpath, _ = DATA_CONFIGS[_data_config]
    # The parent dir of the yaml file is the dataset root
    expected_path = str(yaml_path.parent.resolve())
    if config.get("path") != expected_path:
        config["path"] = expected_path
        yaml_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
        print(f"  Patched dataset.yaml path -> {expected_path}")


def print_performance_summary(results) -> None:
    """Print final metrics and compare against targets."""
    targets = PERFORMANCE_TARGETS[_data_config]
    names = CLASS_NAMES[_data_config]
    output_name = DATA_CONFIGS[_data_config][1]

    overall_map = results.results_dict.get("metrics/mAP50(B)", 0)
    map50_95 = results.results_dict.get("metrics/mAP50-95(B)", 0)
    precision = results.results_dict.get("metrics/precision(B)", 0)
    recall = results.results_dict.get("metrics/recall(B)", 0)

    print(f"\nFinal Metrics ({output_name} / YOLOv8s):")
    print(f"  mAP50:     {overall_map:.4f}  (target >= {targets['overall']})")
    print(f"  mAP50-95:  {map50_95:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")

    # Overall target check
    print(f"\n  Performance vs targets:")
    if overall_map >= targets["overall"]:
        print(f"    overall: {overall_map:.4f} >= {targets['overall']} ACHIEVED")
    else:
        gap = targets["overall"] - overall_map
        print(f"    overall: {overall_map:.4f} — {gap:.4f} below target {targets['overall']}")

    # Per-class targets (multiclass only)
    if _data_config == "multiclass":
        for name in ["building", "vehicle", "ship", "aircraft"]:
            target = targets.get(name)
            if target is not None:
                print(f"    {name}: target >= {target} (check per-class eval in results/)")

        if overall_map < targets["overall"]:
            print(f"\n  Overall mAP50 below target. Consider:")
            print(f"    - More epochs or lower lr0")
            print(f"    - Aircraft oversampling: python scripts/09_merge_datasets.py --oversample-aircraft 3")


def train(smoke_test: bool = False):
    """Run YOLOv8s training."""
    from ultralytics import YOLO

    output_name = DATA_CONFIGS[_data_config][1]
    names = CLASS_NAMES[_data_config]
    n_classes = len(names)

    print(f"Training YOLOv8s — {_data_config} ({n_classes} classes: {', '.join(names)})")
    print("=" * 55)

    check_gpu()

    yaml_path = Path(TRAINING_CONFIG["data"])
    patch_dataset_yaml_path(yaml_path)

    if not validate_dataset(yaml_path):
        return

    if smoke_test:
        print("\nSMOKE TEST: 2 epochs, small batch")
        TRAINING_CONFIG.update(SMOKE_TEST_OVERRIDES)

    print(f"\nStarting training:")
    for k, v in TRAINING_CONFIG.items():
        print(f"  {k}: {v}")

    config = dict(TRAINING_CONFIG)
    model_key = config.pop("model")
    model = YOLO(model_key)

    results = model.train(**config)

    best_path = MODELS_DIR / output_name / "weights" / "best.pt"
    print(f"\nTraining complete!")
    print(f"Best model: {best_path}")
    print_performance_summary(results)

    if not smoke_test:
        print(f"\nNext steps:")
        print(f"  1. Upload {best_path} to HuggingFace Space")
        if _data_config == "multiclass":
            print(f"  2. Update MODEL_PATH to ./models/multiclass_v2/weights/best.pt")
            print(f"  3. Update model_version in inference_server.py to yolov8s-satellite-multiclass-v2")
        else:
            print(f"  2. Update MODEL_PATH to ./models/baseline_v2/weights/best.pt")
    else:
        print("\nSmoke test passed. Run without --smoke-test for full training.")

    return results


def main():
    global DATA_DIR, MODELS_DIR, _data_config

    parser = argparse.ArgumentParser(
        description="Train YOLOv8s on satellite imagery (multiclass or SpaceNet-only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run 2 epochs to validate pipeline",
    )
    parser.add_argument(
        "--data-config", choices=["multiclass", "spacenet"], default="multiclass",
        help="Dataset to train on: multiclass (4-class) or spacenet (buildings only)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR,
        help="Root data directory",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=MODELS_DIR,
        help="Models output directory",
    )
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir)
    MODELS_DIR = Path(args.models_dir)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _data_config = args.data_config

    yaml_subpath, output_name = DATA_CONFIGS[_data_config]
    TRAINING_CONFIG["data"] = str(DATA_DIR / yaml_subpath)
    TRAINING_CONFIG["name"] = output_name
    TRAINING_CONFIG["project"] = str(MODELS_DIR)

    train(smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
