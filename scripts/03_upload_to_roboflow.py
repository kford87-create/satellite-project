"""
scripts/03_upload_to_roboflow.py

Uploads the YOLO-formatted SpaceNet dataset to Roboflow.
Roboflow will act as our:
  - Dataset version control system
  - Augmentation pipeline
  - Active learning management platform

Think of Roboflow as "GitHub for your labeled datasets"
- you can version, branch, and augment datasets just like code.
"""

import os
import roboflow
from pathlib import Path
from dotenv import load_dotenv
import yaml

load_dotenv()

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
YOLO_DIR = DATA_DIR / "yolo_format"

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT", "satellite-object-detection")

CLASS_NAMES = ["building"]  # Expand as you add new object classes


def create_dataset_yaml():
    """
    Create a YOLO dataset.yaml config file.
    This tells YOLOv8 where to find images/labels and what classes exist.
    """
    dataset_config = {
        "path": str(YOLO_DIR.absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }

    yaml_path = YOLO_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    print(f"✅ Created dataset.yaml at: {yaml_path}")
    return yaml_path


def split_dataset(train_ratio=0.8, val_ratio=0.15):
    """
    Split dataset into train/val/test sets.
    80% train, 15% val, 5% test — standard split for object detection.
    """
    import shutil
    import random

    all_images = list((YOLO_DIR / "train" / "images").glob("*.jpg"))
    random.shuffle(all_images)

    n_total = len(all_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        "train": all_images[:n_train],
        "val": all_images[n_train:n_train + n_val],
        "test": all_images[n_train + n_val:],
    }

    print(f"\n📂 Splitting {n_total} images:")
    print(f"   Train: {len(splits['train'])} ({train_ratio*100:.0f}%)")
    print(f"   Val:   {len(splits['val'])} ({val_ratio*100:.0f}%)")
    print(f"   Test:  {n_total - n_train - n_val} (remaining)")

    for split_name, images in splits.items():
        split_img_dir = YOLO_DIR / split_name / "images"
        split_lbl_dir = YOLO_DIR / split_name / "labels"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        if split_name == "train":
            continue  # Already in train/

        for img_path in images:
            # Move image
            shutil.move(str(img_path), str(split_img_dir / img_path.name))

            # Move corresponding label
            label_path = YOLO_DIR / "train" / "labels" / (img_path.stem + ".txt")
            if label_path.exists():
                shutil.move(str(label_path), str(split_lbl_dir / label_path.name))

    print("✅ Dataset split complete")


def upload_to_roboflow():
    """
    Upload dataset to Roboflow for version control and management.
    """
    if not ROBOFLOW_API_KEY:
        print("❌ ROBOFLOW_API_KEY not set in .env file")
        print("   Get your key at: https://app.roboflow.com/settings/api")
        return

    print(f"\n🚀 Uploading to Roboflow workspace: {ROBOFLOW_WORKSPACE}")
    print(f"   Project: {ROBOFLOW_PROJECT}\n")

    rf = roboflow.Roboflow(api_key=ROBOFLOW_API_KEY)
    workspace = rf.workspace(ROBOFLOW_WORKSPACE)

    # Create project if it doesn't exist
    try:
        project = workspace.project(ROBOFLOW_PROJECT)
        print(f"✅ Found existing project: {ROBOFLOW_PROJECT}")
    except Exception:
        print(f"📝 Creating new project: {ROBOFLOW_PROJECT}")
        project = workspace.create_project(
            project_name=ROBOFLOW_PROJECT,
            project_type="object-detection",
            license="MIT",
            project_description="Satellite imagery object detection - bootstrapping pipeline"
        )

    # Upload images and labels
    for split in ["train", "val", "test"]:
        img_dir = YOLO_DIR / split / "images"
        lbl_dir = YOLO_DIR / split / "labels"

        if not img_dir.exists():
            continue

        images = list(img_dir.glob("*.jpg"))
        print(f"\n📤 Uploading {len(images)} {split} images...")

        for img_path in images:
            label_path = lbl_dir / (img_path.stem + ".txt")

            annotation_path = str(label_path) if label_path.exists() else None

            project.upload(
                image_path=str(img_path),
                annotation_path=annotation_path,
                annotation_labelmap={str(i): name for i, name in enumerate(CLASS_NAMES)},
                split=split,
                num_retry_uploads=3,
                batch_name=f"spacenet-{split}-v1"
            )

    print(f"\n✅ Upload complete!")
    print(f"   View your dataset at: https://app.roboflow.com/{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}")


if __name__ == "__main__":
    print("🛰️  Upload SpaceNet Dataset to Roboflow")
    print("=" * 50)

    # Step 1: Split dataset
    print("\n[1/3] Splitting dataset...")
    split_dataset()

    # Step 2: Create YAML config
    print("\n[2/3] Creating dataset config...")
    create_dataset_yaml()

    # Step 3: Upload to Roboflow
    print("\n[3/3] Uploading to Roboflow...")
    upload_to_roboflow()
