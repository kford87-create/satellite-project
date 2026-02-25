"""
scripts/02_preprocess_spacenet.py

Converts SpaceNet GeoJSON building footprint labels into YOLO format.
SpaceNet stores labels as polygon coordinates in GeoJSON.
YOLO needs bounding boxes in normalized (x_center, y_center, width, height) format.

Think of this as translating from "architect blueprints" (GeoJSON polygons)
to "simple bounding boxes" (YOLO format) that the detector can train on.
"""

import os
import json
import numpy as np
import rasterio
from rasterio.transform import rowcol
from pathlib import Path
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import shape
import cv2
from dotenv import load_dotenv

load_dotenv()


# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
SPACENET_DIR = DATA_DIR / "sn2_vegas"
OUTPUT_DIR = DATA_DIR / "yolo_format"

# YOLO class mapping - expandable as you add new object classes
CLASS_MAP = {
    "building": 0,
    # Future classes (Phase I expansion):
    # "vehicle": 1,
    # "aircraft": 2,
    # "ship": 3,
}


def geojson_to_yolo(geojson_path: Path, image_path: Path, output_label_path: Path):
    """
    Convert a single GeoJSON label file to YOLO format.

    Args:
        geojson_path: Path to the GeoJSON file with building polygons
        image_path: Corresponding satellite image (GeoTIFF)
        output_label_path: Where to save the YOLO .txt label file
    """
    # Read image dimensions
    with rasterio.open(image_path) as src:
        img_width = src.width
        img_height = src.height
        transform = src.transform
        crs = src.crs

    # Read GeoJSON labels
    gdf = gpd.read_file(geojson_path)

    if gdf.empty:
        # Create empty label file (valid for images with no objects)
        output_label_path.write_text("")
        return 0

    yolo_labels = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # Get bounding box in geographic coordinates
        minx, miny, maxx, maxy = geom.bounds

        # Convert geo coordinates to pixel coordinates
        # rasterio uses (row, col) = (y, x)
        row_min, col_min = rowcol(transform, minx, maxy)  # top-left
        row_max, col_max = rowcol(transform, maxx, miny)  # bottom-right

        # Clamp to image bounds
        col_min = max(0, min(col_min, img_width))
        col_max = max(0, min(col_max, img_width))
        row_min = max(0, min(row_min, img_height))
        row_max = max(0, min(row_max, img_height))

        # Skip degenerate boxes
        if col_max <= col_min or row_max <= row_min:
            continue

        # Convert to YOLO normalized format: x_center, y_center, width, height
        x_center = ((col_min + col_max) / 2) / img_width
        y_center = ((row_min + row_max) / 2) / img_height
        width = (col_max - col_min) / img_width
        height = (row_max - row_min) / img_height

        class_id = CLASS_MAP["building"]
        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Write YOLO label file
    output_label_path.write_text("\n".join(yolo_labels))
    return len(yolo_labels)


def convert_rgb_to_jpeg(tiff_path: Path, output_path: Path):
    """
    Convert GeoTIFF to JPEG for YOLO training.
    SpaceNet images are 3-band GeoTIFFs — we extract RGB for YOLOv8.
    """
    with rasterio.open(tiff_path) as src:
        # Read first 3 bands as RGB
        bands = min(3, src.count)
        img = src.read(list(range(1, bands + 1)))

        # Rearrange to HWC format
        img = np.transpose(img, (1, 2, 0))

        # Normalize to 0-255
        img = img.astype(np.float32)
        for i in range(img.shape[2]):
            band = img[:, :, i]
            min_val, max_val = band.min(), band.max()
            if max_val > min_val:
                img[:, :, i] = (band - min_val) / (max_val - min_val) * 255

        img = img.astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(output_path), img)


def process_spacenet_dataset(split: str = "train"):
    """
    Process the full SpaceNet Vegas dataset.
    Creates YOLO-compatible images/ and labels/ directories.
    """
    # SpaceNet 2 Vegas directory structure
    img_dir = SPACENET_DIR / f"AOI_2_Vegas_Train" / "RGB-PanSharpen"
    geojson_dir = SPACENET_DIR / f"AOI_2_Vegas_Train" / "geojson" / "buildings"

    if not img_dir.exists():
        print(f"❌ Image directory not found: {img_dir}")
        print("   Make sure you've run 01_download_spacenet.py first")
        return

    # Create YOLO output directories
    yolo_img_dir = OUTPUT_DIR / split / "images"
    yolo_label_dir = OUTPUT_DIR / split / "labels"
    yolo_img_dir.mkdir(parents=True, exist_ok=True)
    yolo_label_dir.mkdir(parents=True, exist_ok=True)

    tiff_files = list(img_dir.glob("*.tif"))
    print(f"\n🔄 Processing {len(tiff_files)} images from SpaceNet Vegas...\n")

    total_buildings = 0
    processed = 0

    for tiff_path in tqdm(tiff_files, desc="Converting to YOLO format"):
        stem = tiff_path.stem

        # Find corresponding GeoJSON label
        base_stem = stem.replace("RGB-PanSharpen_", "")
        geojson_path = geojson_dir / f"buildings_{base_stem}.geojson"
        if not geojson_path.exists():
            continue

        # Output paths
        out_img = yolo_img_dir / f"{stem}.jpg"
        out_label = yolo_label_dir / f"{stem}.txt"

        try:
            # Convert image
            convert_rgb_to_jpeg(tiff_path, out_img)

            # Convert labels
            n_buildings = geojson_to_yolo(geojson_path, tiff_path, out_label)
            total_buildings += n_buildings
            processed += 1

        except Exception as e:
            print(f"\n⚠️  Error processing {stem}: {e}")
            continue

    print(f"\n✅ Processed {processed}/{len(tiff_files)} images")
    print(f"   Total building annotations: {total_buildings:,}")
    print(f"   Output saved to: {OUTPUT_DIR}")

    # Generate dataset stats
    generate_stats(yolo_label_dir)


def generate_stats(label_dir: Path):
    """Generate and print dataset statistics."""
    label_files = list(label_dir.glob("*.txt"))
    total_objects = 0
    empty_images = 0

    for f in label_files:
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        if not lines:
            empty_images += 1
        total_objects += len(lines)

    print(f"\n📊 Dataset Statistics:")
    print(f"   Total images:        {len(label_files):,}")
    print(f"   Images with objects: {len(label_files) - empty_images:,}")
    print(f"   Empty images:        {empty_images:,}")
    print(f"   Total annotations:   {total_objects:,}")
    print(f"   Avg per image:       {total_objects/max(len(label_files),1):.1f}")


if __name__ == "__main__":
    print("🛰️  SpaceNet → YOLO Format Converter")
    print("=" * 50)
    process_spacenet_dataset(split="train")

    # Upload to Supabase after preprocessing
    print("\n☁️  Uploading processed dataset to Supabase Storage...")
    try:
        from supabase_storage import SupabaseStorageManager
        manager = SupabaseStorageManager()
        manager.upload_dataset_batch(
            image_dir=OUTPUT_DIR / "train" / "images",
            label_dir=OUTPUT_DIR / "train" / "labels",
            split="train",
            prefix="spacenet"
        )
        print("✅ Dataset synced to Supabase")
    except Exception as e:
        print(f"⚠️  Supabase upload skipped: {e}")
        print("   Run scripts/supabase_storage.py to set up Supabase first")
