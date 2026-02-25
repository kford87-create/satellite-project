"""
scripts/08_preprocess_dota.py

Converts DOTA-v1.0 OBB (oriented bounding box) annotations into YOLO format
and chips large aerial scenes into 640×640 tiles.

Class mapping (only these 5 DOTA classes are kept):
    plane / helicopter  → 2  (aircraft)
    large-vehicle / small-vehicle → 1  (vehicle)
    ship                → 3  (ship)

OBB → YOLO conversion:
    DOTA stores 4-corner polygon coords in scene pixels.
    We compute an axis-aligned bounding box, chip the scene with 640px windows
    (overlap=64), clip each annotation to its chip, and normalize to [0,1].

Usage:
    python scripts/08_preprocess_dota.py                  # full run
    python scripts/08_preprocess_dota.py --max-scenes 5   # spot-check 5 scenes
    python scripts/08_preprocess_dota.py --splits train   # train only

Output:
    data/multiclass/dota_chips/{train,val}/{images,labels}/
    data/multiclass/dota_chips/preprocessing_stats.json
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DOTA_DIR = DATA_DIR / "DOTA-v1.0"
OUTPUT_DIR = DATA_DIR / "multiclass" / "dota_chips"

# ─── Constants ────────────────────────────────────────────────────────────────
CHIP_SIZE = 640
OVERLAP = 64
MIN_VISIBILITY = 0.3   # fraction of bbox area that must fall inside chip

DOTA_CLASS_MAP: dict[str, int] = {
    "plane":         2,  # aircraft
    "helicopter":    2,  # aircraft
    "large-vehicle": 1,  # vehicle
    "small-vehicle": 1,  # vehicle
    "ship":          3,  # ship
}


# ─── Import grid helpers from chip_generator ─────────────────────────────────
# We reuse the tested grid logic rather than reimplementing it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.data_acquisition.chip_generator import _compute_grid, _pad_array


# ─── OBB annotation parsing ───────────────────────────────────────────────────

def parse_dota_label_file(label_path: Path) -> list[dict]:
    """
    Parse a DOTA .txt label file.

    Returns a list of dicts:
        {"class_id": int, "x_min": float, "y_min": float,
         "x_max": float, "y_max": float}
    Only includes classes in DOTA_CLASS_MAP; skips difficulty==2.
    """
    annotations = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("imagesource") or line.startswith("gsd"):
            continue  # DOTA header lines

        parts = line.split()
        if len(parts) < 10:
            continue  # need 8 coords + category + difficulty

        try:
            coords = [float(p) for p in parts[:8]]
            category = parts[8].strip().lower()
            difficulty = int(parts[9])
        except (ValueError, IndexError):
            continue

        if difficulty == 2:
            continue  # skip hard instances

        class_id = DOTA_CLASS_MAP.get(category)
        if class_id is None:
            continue  # class not in our target set

        xs = coords[0::2]  # x1, x2, x3, x4
        ys = coords[1::2]  # y1, y2, y3, y4

        annotations.append({
            "class_id": class_id,
            "x_min": min(xs),
            "y_min": min(ys),
            "x_max": max(xs),
            "y_max": max(ys),
        })

    return annotations


# ─── Chip annotation projection ───────────────────────────────────────────────

def project_annotations_to_chip(
    annotations: list[dict],
    x_off: int,
    y_off: int,
    chip_size: int = CHIP_SIZE,
    min_visibility: float = MIN_VISIBILITY,
) -> list[str]:
    """
    Clip scene-level annotations to a chip window and return YOLO label lines.

    Args:
        annotations: list from parse_dota_label_file()
        x_off, y_off: top-left corner of chip in scene pixels
        chip_size: chip width/height in pixels
        min_visibility: minimum clip_area/orig_area to keep annotation

    Returns:
        List of YOLO label strings "class_id cx cy w h" (normalized to [0,1])
    """
    chip_right = x_off + chip_size
    chip_bottom = y_off + chip_size

    yolo_lines = []
    for ann in annotations:
        x_min, y_min = ann["x_min"], ann["y_min"]
        x_max, y_max = ann["x_max"], ann["y_max"]

        orig_area = max((x_max - x_min) * (y_max - y_min), 1e-6)

        # Clip to chip window
        cx_min = max(x_min, x_off)
        cy_min = max(y_min, y_off)
        cx_max = min(x_max, chip_right)
        cy_max = min(y_max, chip_bottom)

        if cx_max <= cx_min or cy_max <= cy_min:
            continue  # no overlap

        clip_area = (cx_max - cx_min) * (cy_max - cy_min)
        if clip_area / orig_area < min_visibility:
            continue

        # Shift to chip-local coords
        lx_min = cx_min - x_off
        ly_min = cy_min - y_off
        lx_max = cx_max - x_off
        ly_max = cy_max - y_off

        # Clamp to [0, chip_size]
        lx_min = max(0.0, min(lx_min, chip_size))
        ly_min = max(0.0, min(ly_min, chip_size))
        lx_max = max(0.0, min(lx_max, chip_size))
        ly_max = max(0.0, min(ly_max, chip_size))

        # YOLO normalize
        cx = (lx_min + lx_max) / 2.0 / chip_size
        cy = (ly_min + ly_max) / 2.0 / chip_size
        w  = (lx_max - lx_min) / chip_size
        h  = (ly_max - ly_min) / chip_size

        # Final clamp to [0,1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w  = max(0.0, min(1.0, w))
        h  = max(0.0, min(1.0, h))

        if w < 1e-4 or h < 1e-4:
            continue

        yolo_lines.append(f"{ann['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return yolo_lines


# ─── Scene processing ─────────────────────────────────────────────────────────

def process_scene(
    img_path: Path,
    label_path: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
) -> dict[int, int]:
    """
    Chip one scene and write chip images + YOLO labels.

    Returns per-class annotation counts for this scene.
    """
    class_counts: dict[int, int] = {1: 0, 2: 0, 3: 0}

    # Load image
    img = Image.open(img_path).convert("RGB")
    width, height = img.size
    img_array = np.array(img)

    # Parse annotations
    annotations = parse_dota_label_file(label_path)

    # Compute chip grid
    row_offsets = _compute_grid(height, CHIP_SIZE, OVERLAP)
    col_offsets = _compute_grid(width, CHIP_SIZE, OVERLAP)

    scene_stem = img_path.stem

    for ri, y_off in enumerate(row_offsets):
        for ci, x_off in enumerate(col_offsets):
            y_end = min(y_off + CHIP_SIZE, height)
            x_end = min(x_off + CHIP_SIZE, width)

            # Extract and pad chip
            chip_data = img_array[y_off:y_end, x_off:x_end, :]
            chip_h, chip_w = chip_data.shape[:2]
            if chip_h < CHIP_SIZE or chip_w < CHIP_SIZE:
                chip_data = _pad_array(chip_data, CHIP_SIZE, CHIP_SIZE)

            # Project annotations
            yolo_lines = project_annotations_to_chip(annotations, x_off, y_off)

            # Save chip image (always, even if empty — YOLO needs background chips)
            chip_stem = f"{scene_stem}_{ri:04d}_{ci:04d}"
            chip_img = Image.fromarray(chip_data, mode="RGB")
            chip_img.save(out_img_dir / f"{chip_stem}.jpg", quality=95)

            # Save label file
            label_text = "\n".join(yolo_lines)
            (out_lbl_dir / f"{chip_stem}.txt").write_text(label_text)

            # Accumulate counts
            for line in yolo_lines:
                cid = int(line.split()[0])
                class_counts[cid] = class_counts.get(cid, 0) + 1

    return class_counts


# ─── Split processing ─────────────────────────────────────────────────────────

def process_split(
    split: str,
    max_scenes: Optional[int],
    stats: dict,
) -> None:
    img_dir = DOTA_DIR / "images" / split
    lbl_dir = DOTA_DIR / "labelTxt" / split

    if not img_dir.exists():
        print(f"❌ Image directory not found: {img_dir}")
        print("   Run scripts/07_download_dota.py for download instructions.")
        return

    if not lbl_dir.exists():
        print(f"❌ Label directory not found: {lbl_dir}")
        return

    out_img_dir = OUTPUT_DIR / split / "images"
    out_lbl_dir = OUTPUT_DIR / split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Collect scenes that have both an image and a label
    img_exts = {".jpg", ".jpeg", ".png"}
    scene_pairs: list[tuple[Path, Path]] = []

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in img_exts:
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            scene_pairs.append((img_path, lbl_path))

    if max_scenes:
        scene_pairs = scene_pairs[:max_scenes]

    print(f"\n🔄 {split.upper()}: processing {len(scene_pairs):,} scenes "
          f"{'(limited)' if max_scenes else ''}...")

    split_counts: dict[int, int] = {}
    total_chips = 0
    errors = 0

    for img_path, lbl_path in tqdm(scene_pairs, desc=f"  {split}", unit="scene"):
        try:
            counts = process_scene(img_path, lbl_path, out_img_dir, out_lbl_dir)
            for cid, n in counts.items():
                split_counts[cid] = split_counts.get(cid, 0) + n
        except Exception as exc:
            print(f"\n  ⚠️  Error processing {img_path.name}: {exc}")
            errors += 1
            continue

    total_chips = len(list(out_img_dir.glob("*.jpg")))

    CLASS_NAMES = {1: "vehicle", 2: "aircraft", 3: "ship"}
    print(f"\n  ✅ {total_chips:,} chips written to {out_img_dir.parent}")
    print(f"     Annotation counts:")
    for cid in sorted(split_counts):
        print(f"       class {cid} ({CLASS_NAMES.get(cid,'?')}): {split_counts[cid]:,}")
    if errors:
        print(f"     Errors: {errors}")

    stats[split] = {
        "scenes_processed": len(scene_pairs),
        "chips_written": total_chips,
        "errors": errors,
        "class_counts": {str(k): v for k, v in split_counts.items()},
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    global DATA_DIR, DOTA_DIR, OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="Convert DOTA-v1.0 OBB labels to YOLO chips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max-scenes", type=int, default=None, metavar="N",
        help="Limit to first N scenes per split (for spot-checking)",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"],
        choices=["train", "val"],
        help="Which splits to process",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR,
        help="Root data directory",
    )
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir)
    DOTA_DIR = DATA_DIR / "DOTA-v1.0"
    OUTPUT_DIR = DATA_DIR / "multiclass" / "dota_chips"

    print("🛰️  DOTA-v1.0 → YOLO Chip Preprocessor")
    print("=" * 50)
    print(f"  DOTA source : {DOTA_DIR.resolve()}")
    print(f"  Output      : {OUTPUT_DIR.resolve()}")
    print(f"  Chip size   : {CHIP_SIZE}px  Overlap: {OVERLAP}px")
    print(f"  Min visibility: {MIN_VISIBILITY}")
    if args.max_scenes:
        print(f"  ⚠️  Limiting to {args.max_scenes} scenes per split (spot-check mode)")

    stats: dict = {}
    for split in args.splits:
        process_split(split, args.max_scenes, stats)

    # Write stats JSON
    stats_path = OUTPUT_DIR / "preprocessing_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"\n📄 Stats written to {stats_path}")

    total_chips = sum(s.get("chips_written", 0) for s in stats.values())
    print(f"\n✅ Total chips produced: {total_chips:,}")
    print("\n➡️  Next step: python scripts/09_merge_datasets.py")


if __name__ == "__main__":
    main()
