"""
scripts/04_train_baseline.py

Trains a YOLOv8 baseline model on the SpaceNet dataset.
This is your SEED MODEL - the starting point for the bootstrapping loop.

Run this on Vast.ai:
  1. Rent an RTX 4090 or A100 instance (~$0.30-0.50/hr)
  2. Upload this project folder
  3. pip install -r requirements.txt
  4. python scripts/04_train_baseline.py

Training time estimate:
  - RTX 4090: ~2-3 hours for 50 epochs on SpaceNet Vegas
  - Cost: ~$1.00-1.50 total
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
from dotenv import load_dotenv
import torch

load_dotenv()

# ─── Training Config ──────────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_CONFIG = {
    # Model: yolov8n = nano (fastest), yolov8s = small, yolov8m = medium
    # Start with nano to validate pipeline, upgrade for production
    "model": "yolov8n.pt",

    "data": str(DATA_DIR / "yolo_format" / "dataset.yaml"),

    # Training hyperparameters
    "epochs": 50,           # Start low to validate, increase to 100-200 for production
    "imgsz": 640,           # Image size - satellite tiles are typically 400-650px
    "batch": 16,            # Adjust based on GPU VRAM (16 for RTX 4090, 8 for smaller)
    "patience": 15,         # Early stopping - stops if no improvement for 15 epochs
    "lr0": 0.01,            # Initial learning rate
    "lrf": 0.001,           # Final learning rate
    "momentum": 0.937,
    "weight_decay": 0.0005,

    # Augmentation - critical for satellite imagery
    # Satellites capture from above, so objects can appear at any rotation
    "degrees": 180,         # Full rotation augmentation (key for satellite imagery!)
    "flipud": 0.5,          # Vertical flip probability
    "fliplr": 0.5,          # Horizontal flip probability
    "mosaic": 1.0,          # Mosaic augmentation
    "scale": 0.5,           # Scale augmentation

    # Output
    "project": str(MODELS_DIR),
    "name": "baseline_v1",
    "save": True,
    "plots": True,
}


def check_gpu():
    """Verify GPU is available for training."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU detected: {gpu_name} ({vram:.1f}GB VRAM)")

        # Adjust batch size based on VRAM
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


def validate_dataset():
    """Check dataset exists before training."""
    yaml_path = Path(TRAINING_CONFIG["data"])
    if not yaml_path.exists():
        print(f"❌ Dataset config not found: {yaml_path}")
        print("   Run scripts 01, 02, and 03 first to prepare the dataset.")
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
    return True


def train():
    """Run YOLOv8 baseline training."""
    print("🛰️  Training Baseline YOLOv8 Model")
    print("=" * 50)

    # Pre-flight checks
    check_gpu()
    if not validate_dataset():
        return

    print(f"\n🚀 Starting training with config:")
    for k, v in TRAINING_CONFIG.items():
        print(f"   {k}: {v}")

    # Load pretrained YOLOv8 (transfer learning from COCO)
    # Even though satellite imagery is different from COCO,
    # the low-level feature detectors (edges, textures) transfer well
    model = YOLO(TRAINING_CONFIG.pop("model"))

    # Train
    results = model.train(**TRAINING_CONFIG)

    # Print results summary
    print(f"\n✅ Training complete!")
    print(f"\n📊 Final Metrics:")
    print(f"   mAP50:    {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"   mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"   Precision: {results.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"   Recall:    {results.results_dict.get('metrics/recall(B)', 0):.4f}")

    best_model_path = MODELS_DIR / "baseline_v1" / "weights" / "best.pt"
    print(f"\n💾 Best model saved to: {best_model_path}")
    print(f"\n➡️  Next step: Run scripts/05_active_learning_loop.py")

    return results


if __name__ == "__main__":
    train()
