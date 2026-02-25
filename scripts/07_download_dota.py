"""
scripts/07_download_dota.py

Checks for DOTA-v1.0 dataset directory structure and prints download
instructions if the data is not present.

DOTA-v1.0 is not freely downloadable via script — registration is required.
This script validates your local copy and prints a class inventory.

Usage:
  python scripts/07_download_dota.py
  python scripts/07_download_dota.py --data-dir /path/to/custom/data

Download instructions (if you haven't already):
  1. Register at https://captain-bench.github.io/dota/
  2. Download DOTA-v1.0_train.zip and DOTA-v1.0_val.zip
  3. Extract to data/DOTA-v1.0/ so the structure matches:
       data/DOTA-v1.0/
         images/train/   (800-4000px JPEGs)
         images/val/
         labelTxt/train/ (.txt files, one per image)
         labelTxt/val/
"""

import os
import random
import argparse
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))

# DOTA classes we care about (others are skipped during preprocessing)
DOTA_CLASS_MAP = {
    "plane":         2,  # aircraft
    "helicopter":    2,  # aircraft
    "large-vehicle": 1,  # vehicle
    "small-vehicle": 1,  # vehicle
    "ship":          3,  # ship
}

# All DOTA-v1.0 classes (for inventory)
ALL_DOTA_CLASSES = {
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool",
}

EXPECTED_DIRS = [
    "images/train",
    "images/val",
    "labelTxt/train",
    "labelTxt/val",
]


def check_structure(dota_dir: Path) -> bool:
    """Return True if all expected subdirectories exist."""
    all_ok = True
    for subdir in EXPECTED_DIRS:
        p = dota_dir / subdir
        if p.exists():
            n = len(list(p.iterdir()))
            print(f"  ✅  {p}  ({n:,} files)")
        else:
            print(f"  ❌  {p}  — MISSING")
            all_ok = False
    return all_ok


def sample_labels(label_dir: Path, n_samples: int = 200) -> Counter:
    """Read up to n_samples label files and count class occurrences."""
    label_files = list(label_dir.glob("*.txt"))
    if not label_files:
        return Counter()
    sample = random.sample(label_files, min(n_samples, len(label_files)))
    counts: Counter = Counter()
    for lf in sample:
        for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            cls = parts[8].strip().lower()
            counts[cls] += 1
    return counts


def print_inventory(dota_dir: Path):
    """Print annotation counts from a random sample of label files."""
    print("\n📊 Class inventory (sampled from 200 random label files per split):")
    for split in ("train", "val"):
        label_dir = dota_dir / "labelTxt" / split
        if not label_dir.exists():
            continue
        counts = sample_labels(label_dir)
        total_files = len(list(label_dir.glob("*.txt")))
        print(f"\n  {split.upper()} ({total_files:,} label files):")
        for cls in sorted(counts, key=lambda c: -counts[c]):
            mapped = DOTA_CLASS_MAP.get(cls, None)
            tag = f" → class {mapped}" if mapped is not None else " (skipped)"
            print(f"    {cls:<25} {counts[cls]:>6,}{tag}")


def print_sample_lines(dota_dir: Path, n: int = 3):
    """Show n example OBB annotation lines."""
    label_dir = dota_dir / "labelTxt" / "train"
    if not label_dir.exists():
        return
    files = list(label_dir.glob("*.txt"))
    if not files:
        return
    sample_file = random.choice(files)
    lines = [l for l in sample_file.read_text(errors="ignore").splitlines() if l.strip()]
    print(f"\n🔍 Sample OBB lines from {sample_file.name}:")
    for line in lines[:n]:
        print(f"   {line}")
    print("   Format: x1 y1 x2 y2 x3 y3 x4 y4 category difficulty")


def print_download_instructions(dota_dir: Path):
    print("\n" + "=" * 60)
    print("📥  DOTA-v1.0 NOT FOUND")
    print("=" * 60)
    print(f"\n  Expected location: {dota_dir.resolve()}")
    print("""
  Download steps (via Kaggle):
    Dataset: https://www.kaggle.com/datasets/chandlertimm/dota-data

    Option A — Kaggle CLI (fastest):
      pip install kaggle
      # Put your kaggle.json API token in ~/.kaggle/kaggle.json
      # Get it from: https://www.kaggle.com/settings → API → Create New Token
      kaggle datasets download -d chandlertimm/dota-data -p data/DOTA-v1.0 --unzip

    Option B — Browser:
      1. Log in to Kaggle, go to the dataset page above
      2. Click "Download" (zip, ~3.7 GB)
      3. Unzip to data/DOTA-v1.0/

    After extracting, ensure the directory tree matches:
         data/DOTA-v1.0/
           images/
             train/   ← ~1,411 JPEG/PNG scenes
             val/     ← ~458 JPEG/PNG scenes
           labelTxt/
             train/   ← one .txt per scene
             val/
    (Rename subdirs if needed to match this structure.)

    4. Re-run this script to confirm structure.

  After downloading, run:
    python scripts/08_preprocess_dota.py --max-scenes 5   # spot-check
    python scripts/08_preprocess_dota.py                  # full run
""")


def main():
    parser = argparse.ArgumentParser(
        description="Verify DOTA-v1.0 dataset structure and print download instructions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR,
        help="Root data directory (DATA_DIR env var or ./data)",
    )
    args = parser.parse_args()

    dota_dir = Path(args.data_dir) / "DOTA-v1.0"

    print("🛰️  DOTA-v1.0 Dataset Check")
    print("=" * 50)
    print(f"\nLooking for: {dota_dir.resolve()}\n")

    structure_ok = check_structure(dota_dir)

    if not structure_ok:
        print_download_instructions(dota_dir)
        return

    print("\n✅ All expected directories found.")
    print_sample_lines(dota_dir)
    print_inventory(dota_dir)

    # Count total images
    for split in ("train", "val"):
        img_dir = dota_dir / "images" / split
        lbl_dir = dota_dir / "labelTxt" / split
        n_img = len(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
        n_lbl = len(list(lbl_dir.glob("*.txt")))
        print(f"\n  {split.upper()}: {n_img:,} images, {n_lbl:,} label files")

    print("\n➡️  Ready for preprocessing. Run:")
    print("   python scripts/08_preprocess_dota.py --max-scenes 5   # spot-check")
    print("   python scripts/08_preprocess_dota.py                  # full run (~2-4 hrs)")


if __name__ == "__main__":
    main()
