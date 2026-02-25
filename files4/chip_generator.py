"""
tools/data_acquisition/chip_generator.py

Tiles large satellite scenes into overlapping 640x640 chips for YOLOv8.
Preserves geospatial metadata so detections can be mapped back to real-world coordinates.

Usage:
  python tools/data_acquisition/chip_generator.py \
    --input path/to/scene.tif \
    --output data/unlabeled \
    --chip-size 640 \
    --overlap 64
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import argparse
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))


class ChipGenerator:
    """
    Tiles satellite scenes into overlapping chips.
    Each chip carries geospatial metadata for coordinate reconstruction.
    """

    def __init__(self, chip_size: int = 640, overlap: int = 64, min_valid_fraction: float = 0.5):
        self.chip_size = chip_size
        self.overlap = overlap
        self.stride = chip_size - overlap
        self.min_valid_fraction = min_valid_fraction

    def chip_scene(self, scene_path: Path, output_dir: Path, scene_id: Optional[str] = None, save_metadata: bool = True) -> int:
        """
        Tile a single scene into chips. Returns number of chips generated.
        Supports both GeoTIFF (rasterio) and plain images (OpenCV).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        scene_id = scene_id or Path(scene_path).stem

        geo_transform = None
        geo_crs = None

        # Try rasterio first (GeoTIFF with geospatial metadata)
        try:
            import rasterio
            with rasterio.open(scene_path) as src:
                bands = min(3, src.count)
                img = src.read(list(range(1, bands + 1)))
                img = np.transpose(img, (1, 2, 0)).astype(np.float32)
                # Normalize to 0-255
                for i in range(img.shape[2]):
                    band = img[:, :, i]
                    mn, mx = band.min(), band.max()
                    if mx > mn:
                        img[:, :, i] = (band - mn) / (mx - mn) * 255
                img = img.astype(np.uint8)
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                geo_transform = src.transform
                geo_crs = str(src.crs)
        except Exception:
            img = cv2.imread(str(scene_path))
            if img is None:
                print(f"❌ Could not read: {scene_path}")
                return 0

        h, w = img.shape[:2]
        chip_metadata = []
        chip_count = 0

        row_starts = list(range(0, max(1, h - self.chip_size + 1), self.stride))
        col_starts = list(range(0, max(1, w - self.chip_size + 1), self.stride))

        for row in row_starts:
            for col in col_starts:
                row_end = min(row + self.chip_size, h)
                col_end = min(col + self.chip_size, w)
                chip = img[row:row_end, col:col_end]

                # Pad if needed
                if chip.shape[0] < self.chip_size or chip.shape[1] < self.chip_size:
                    padded = np.zeros((self.chip_size, self.chip_size, 3), dtype=np.uint8)
                    padded[:chip.shape[0], :chip.shape[1]] = chip
                    chip = padded

                # Reject mostly-black chips
                valid_fraction = float(np.any(chip > 5, axis=2).mean())
                if valid_fraction < self.min_valid_fraction:
                    continue

                chip_name = f"{scene_id}_r{row:05d}_c{col:05d}.jpg"
                cv2.imwrite(str(output_dir / chip_name), chip, [cv2.IMWRITE_JPEG_QUALITY, 92])

                meta = {
                    "chip_name": chip_name,
                    "scene_id": scene_id,
                    "pixel_row": row, "pixel_col": col,
                    "pixel_row_end": row_end, "pixel_col_end": col_end,
                    "chip_size": self.chip_size,
                    "valid_fraction": round(valid_fraction, 3),
                }
                if geo_transform:
                    try:
                        from rasterio.transform import xy
                        lon_min, lat_max = xy(geo_transform, row, col)
                        lon_max, lat_min = xy(geo_transform, row_end, col_end)
                        meta.update({"lon_min": lon_min, "lat_min": lat_min, "lon_max": lon_max, "lat_max": lat_max, "crs": geo_crs})
                    except Exception:
                        pass

                chip_metadata.append(meta)
                chip_count += 1

        if save_metadata and chip_metadata:
            meta_path = output_dir / f"{scene_id}_chip_index.json"
            meta_path.write_text(json.dumps(chip_metadata, indent=2))

        return chip_count

    def chip_directory(self, input_dir: Path, output_dir: Path, extensions: List[str] = None) -> Dict:
        """Chip all scenes in a directory."""
        extensions = extensions or [".tif", ".tiff", ".jp2", ".jpg", ".png"]
        scenes = [f for f in Path(input_dir).iterdir() if f.suffix.lower() in extensions]

        if not scenes:
            print(f"❌ No scenes found in {input_dir}")
            return {}

        print(f"\n✂️  Chipping {len(scenes)} scenes (chip={self.chip_size}px, overlap={self.overlap}px)...")
        total_chips = 0

        for scene in tqdm(scenes, desc="Chipping scenes"):
            n = self.chip_scene(scene, output_dir)
            total_chips += n
            tqdm.write(f"  {scene.name}: {n} chips")

        summary = {"scenes_processed": len(scenes), "total_chips": total_chips, "output_dir": str(output_dir)}
        print(f"\n✅ Generated {total_chips} chips from {len(scenes)} scenes → {output_dir}")
        return summary

    def reconstruct_scene_detections(self, chip_detections: List[Dict], scene_id: str, chip_index_path: Path) -> List[Dict]:
        """
        Reconstruct full-scene detections from chip-level detections.
        Converts chip pixel coordinates back to scene pixel coordinates,
        then to geographic coordinates if available.
        """
        index = json.loads(chip_index_path.read_text())
        chip_map = {c["chip_name"]: c for c in index}
        scene_detections = []

        for det in chip_detections:
            chip_name = det.get("chip_name")
            if chip_name not in chip_map:
                continue
            chip_meta = chip_map[chip_name]
            row_offset = chip_meta["pixel_row"]
            col_offset = chip_meta["pixel_col"]
            x_c, y_c, bw, bh = det["bbox_xywh_norm"]
            chip_x = x_c * self.chip_size + col_offset
            chip_y = y_c * self.chip_size + row_offset
            scene_det = {**det, "scene_pixel_x": chip_x, "scene_pixel_y": chip_y}
            if "lon_min" in chip_meta:
                lon_range = chip_meta["lon_max"] - chip_meta["lon_min"]
                lat_range = chip_meta["lat_max"] - chip_meta["lat_min"]
                scene_det["longitude"] = chip_meta["lon_min"] + x_c * lon_range
                scene_det["latitude"] = chip_meta["lat_max"] - y_c * lat_range
            scene_detections.append(scene_det)

        return scene_detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chip satellite scenes for YOLOv8")
    parser.add_argument("--input", required=True, help="Scene file or directory")
    parser.add_argument("--output", default=str(DATA_DIR / "unlabeled"))
    parser.add_argument("--chip-size", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--min-valid", type=float, default=0.5)
    args = parser.parse_args()

    gen = ChipGenerator(args.chip_size, args.overlap, args.min_valid)
    input_path = Path(args.input)

    if input_path.is_dir():
        gen.chip_directory(input_path, Path(args.output))
    else:
        n = gen.chip_scene(input_path, Path(args.output))
        print(f"✅ Generated {n} chips")
