"""
scripts/09_merge_datasets.py

Merges SpaceNet (class 0 — buildings) and DOTA chips (classes 1,2,3) into a
unified 4-class dataset at data/multiclass/.

No class remapping is needed: SpaceNet labels are already class 0,
and DOTA chips from script 08 are already classes 1/2/3.

Usage:
    python scripts/09_merge_datasets.py            # symlink mode (fast, local)
    python scripts/09_merge_datasets.py --copy     # copy mode (Vast.ai, cross-device)

Output:
    data/multiclass/train/images/
    data/multiclass/train/labels/
    data/multiclass/val/images/
    data/multiclass/val/labels/
    data/multiclass/dataset.yaml

Expected dataset size (approximate):
    ~42k train images (1,971 SpaceNet + ~40k DOTA chips)
    ~12k val images
    vehicle ~202k, building ~144k, ship ~43k, aircraft ~15k annotations
"""

import os
import sys
import shutil
import argparse
import yaml
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
SPACENET_YOLO_DIR = DATA_DIR / "yolo_format"
DOTA_CHIPS_DIR = DATA_DIR / "multiclass" / "dota_chips"
OUTPUT_DIR = DATA_DIR / "multiclass"

CLASS_NAMES = ["building", "vehicle", "aircraft", "ship"]


def link_or_copy(src: Path, dst: Path, use_copy: bool) -> None:
    """Create a symlink (or copy) from src to dst, skipping if dst exists."""
    if dst.exists():
        return
    if use_copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def merge_split(
    split: str,
    use_copy: bool,
    stats: dict,
) -> None:
    out_img_dir = OUTPUT_DIR / split / "images"
    out_lbl_dir = OUTPUT_DIR / split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    class_counts: Counter = Counter()
    n_linked = {"spacenet": 0, "dota": 0}

    # ── SpaceNet (class 0 buildings) ──────────────────────────────────────────
    sn_img_dir = SPACENET_YOLO_DIR / split / "images"
    sn_lbl_dir = SPACENET_YOLO_DIR / split / "labels"

    if not sn_img_dir.exists():
        print(f"  ⚠️  SpaceNet {split} images not found: {sn_img_dir}")
        print("      Run scripts 01-03 to prepare SpaceNet data.")
    else:
        sn_images = sorted(sn_img_dir.glob("*.jpg"))
        for img_path in tqdm(sn_images, desc=f"  SpaceNet {split}", unit="img"):
            lbl_path = sn_lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            link_or_copy(img_path, out_img_dir / img_path.name, use_copy)
            link_or_copy(lbl_path, out_lbl_dir / lbl_path.name, use_copy)
            # Count class 0 annotations
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    class_counts[int(parts[0])] += 1
            n_linked["spacenet"] += 1

    # ── DOTA chips (classes 1/2/3) ────────────────────────────────────────────
    dota_img_dir = DOTA_CHIPS_DIR / split / "images"
    dota_lbl_dir = DOTA_CHIPS_DIR / split / "labels"

    if not dota_img_dir.exists():
        print(f"  ⚠️  DOTA chips {split} not found: {dota_img_dir}")
        print("      Run scripts/08_preprocess_dota.py first.")
    else:
        dota_images = sorted(dota_img_dir.glob("*.jpg"))
        for img_path in tqdm(dota_images, desc=f"  DOTA chips {split}", unit="img"):
            lbl_path = dota_lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            link_or_copy(img_path, out_img_dir / img_path.name, use_copy)
            link_or_copy(lbl_path, out_lbl_dir / lbl_path.name, use_copy)
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    class_counts[int(parts[0])] += 1
            n_linked["dota"] += 1

    total_images = n_linked["spacenet"] + n_linked["dota"]
    print(f"\n  {split.upper()}: {total_images:,} total images "
          f"({n_linked['spacenet']:,} SpaceNet + {n_linked['dota']:,} DOTA)")

    print(f"  Annotation counts:")
    for cid, name in enumerate(CLASS_NAMES):
        print(f"    class {cid} ({name}): {class_counts[cid]:,}")

    stats[split] = {
        "total_images": total_images,
        "spacenet_images": n_linked["spacenet"],
        "dota_images": n_linked["dota"],
        "class_counts": {str(k): class_counts[k] for k in range(4)},
    }


def write_dataset_yaml(output_dir: Path) -> Path:
    """Write the 4-class dataset.yaml for YOLO training."""
    yaml_path = output_dir / "dataset.yaml"
    config = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 4,
        "names": CLASS_NAMES,
    }
    yaml_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
    return yaml_path


def check_aircraft_warning(stats: dict) -> None:
    """Warn if aircraft annotations are below the recommended threshold."""
    THRESHOLD = 10_000
    for split, s in stats.items():
        aircraft_n = s.get("class_counts", {}).get("2", 0)
        if aircraft_n < THRESHOLD:
            print(f"\n  ⚠️  WARNING: Only {aircraft_n:,} aircraft annotations in {split} "
                  f"(target >= {THRESHOLD:,})")
            print("     If aircraft mAP50 < 0.25 after training, rerun script 09 with:")
            print("       python scripts/09_merge_datasets.py --oversample-aircraft 3")


def main():
    global DATA_DIR, SPACENET_YOLO_DIR, DOTA_CHIPS_DIR, OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="Merge SpaceNet + DOTA chips into a unified 4-class dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy files instead of symlinking (required on Vast.ai / cross-device)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR,
        help="Root data directory",
    )
    parser.add_argument(
        "--oversample-aircraft", type=int, default=1, metavar="N",
        help="Duplicate DOTA aircraft chips N times (use if aircraft mAP50 < 0.25)",
    )
    args = parser.parse_args()
    DATA_DIR = Path(args.data_dir)
    SPACENET_YOLO_DIR = DATA_DIR / "yolo_format"
    DOTA_CHIPS_DIR = DATA_DIR / "multiclass" / "dota_chips"
    OUTPUT_DIR = DATA_DIR / "multiclass"

    mode = "copy" if args.copy else "symlink"
    print("🛰️  Multi-Class Dataset Merger")
    print("=" * 50)
    print(f"  SpaceNet source : {SPACENET_YOLO_DIR.resolve()}")
    print(f"  DOTA chips      : {DOTA_CHIPS_DIR.resolve()}")
    print(f"  Output          : {OUTPUT_DIR.resolve()}")
    print(f"  Mode            : {mode}")

    stats: dict = {}
    for split in ("train", "val"):
        print(f"\n{'─'*40}")
        print(f"  Processing: {split.upper()}")
        merge_split(split, args.copy, stats)

    # Handle aircraft oversampling if requested
    if args.oversample_aircraft > 1:
        _oversample_aircraft(args.oversample_aircraft, stats)

    # Write dataset.yaml
    yaml_path = write_dataset_yaml(OUTPUT_DIR)
    print(f"\n📄 Dataset config written to: {yaml_path}")
    print(f"   Path field: {OUTPUT_DIR.resolve()}")

    check_aircraft_warning(stats)

    print("\n✅ Merge complete!")
    print("\n➡️  Next steps:")
    print("   python scripts/10_train_multiclass.py --smoke-test   # validate pipeline")
    print("   python scripts/10_train_multiclass.py                # full training")


def _oversample_aircraft(factor: int, stats: dict) -> None:
    """Duplicate aircraft chips in train split by `factor`."""
    print(f"\n🔁 Oversampling aircraft chips ×{factor} in train split...")
    train_img_dir = OUTPUT_DIR / "train" / "images"
    train_lbl_dir = OUTPUT_DIR / "train" / "labels"

    aircraft_chips: list[Path] = []
    for lbl_path in train_lbl_dir.glob("*.txt"):
        for line in lbl_path.read_text().splitlines():
            parts = line.strip().split()
            if parts and int(parts[0]) == 2:
                aircraft_chips.append(lbl_path)
                break

    print(f"  Found {len(aircraft_chips):,} aircraft chips in train")

    added = 0
    for lbl_path in aircraft_chips:
        stem = lbl_path.stem
        img_path = train_img_dir / (stem + ".jpg")
        if not img_path.exists():
            continue
        for i in range(1, factor):
            new_stem = f"{stem}_os{i}"
            dst_img = train_img_dir / f"{new_stem}.jpg"
            dst_lbl = train_lbl_dir / f"{new_stem}.txt"
            if not dst_img.exists():
                shutil.copy2(img_path.resolve() if img_path.is_symlink() else img_path,
                             dst_img)
            if not dst_lbl.exists():
                shutil.copy2(lbl_path.resolve() if lbl_path.is_symlink() else lbl_path,
                             dst_lbl)
            added += 1

    print(f"  Added {added:,} oversampled aircraft chips")
    if "train" in stats:
        stats["train"]["aircraft_oversampled"] = added


if __name__ == "__main__":
    main()
