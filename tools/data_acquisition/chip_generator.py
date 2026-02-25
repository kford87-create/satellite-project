"""
chip_generator.py
-----------------
Tiles large GeoTIFF or JPEG/PNG scenes into overlapping 640x640 chips.

Usage:
    python tools/data_acquisition/chip_generator.py \
        --input path/to/scene.tif --output data/unlabeled \
        --chip-size 640 --overlap 64
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV_PATH = Path("/Users/kahlil/satellite-project/.env")
load_dotenv(ENV_PATH)

# ---------------------------------------------------------------------------
# Optional rasterio import
# ---------------------------------------------------------------------------
try:
    import rasterio
    from rasterio.transform import Affine
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

# ---------------------------------------------------------------------------
# Supabase (non-fatal)
# ---------------------------------------------------------------------------
try:
    import os
    from supabase import create_client, Client as SupabaseClient

    _supa_url = os.environ.get("SUPABASE_URL", "")
    _supa_key = os.environ.get("SUPABASE_KEY", "")
    supabase: SupabaseClient | None = (
        create_client(_supa_url, _supa_key) if _supa_url and _supa_key else None
    )
except Exception:
    supabase = None


def _supabase_log_chip_index(scene_id: str, chip_index: dict[str, Any]) -> None:
    """Non-fatal write of chip index metadata to Supabase."""
    try:
        if supabase is None:
            return
        supabase.table("chip_index").upsert(
            {"scene_id": scene_id, "metadata": json.dumps(chip_index)}
        ).execute()
    except Exception as exc:
        print(f"⚠️  Supabase write skipped (falling back to local only): {exc}")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _scene_id_from_path(scene_path: Path) -> str:
    return scene_path.stem


def _compute_grid(dim: int, chip_size: int, overlap: int) -> list[int]:
    """Return list of top-left offsets for chips along one dimension."""
    step = chip_size - overlap
    if step <= 0:
        raise ValueError(f"overlap ({overlap}) must be less than chip_size ({chip_size})")
    offsets: list[int] = []
    pos = 0
    while pos < dim:
        offsets.append(pos)
        pos += step
    return offsets


def _pad_array(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Zero-pad a (H, W, C) or (H, W) array to target dimensions."""
    if arr.ndim == 2:
        h, w = arr.shape
        if h == target_h and w == target_w:
            return arr
        padded = np.zeros((target_h, target_w), dtype=arr.dtype)
        padded[:h, :w] = arr
        return padded
    h, w, c = arr.shape
    if h == target_h and w == target_w:
        return arr
    padded = np.zeros((target_h, target_w, c), dtype=arr.dtype)
    padded[:h, :w, :] = arr
    return padded


def _pixel_to_geo(transform: Any, col: int, row: int) -> tuple[float, float]:
    """Convert pixel (col, row) to (lon, lat) using an Affine transform."""
    lon = transform.c + col * transform.a + row * transform.b
    lat = transform.f + col * transform.d + row * transform.e
    return lon, lat


# ---------------------------------------------------------------------------
# GeoTIFF chipping
# ---------------------------------------------------------------------------

def chip_geotiff(
    scene_path: Path,
    output_dir: Path,
    chip_size: int,
    overlap: int,
) -> dict[str, Any]:
    if not RASTERIO_AVAILABLE:
        print("⚠️  rasterio not installed; falling back to PIL for GeoTIFF (no geo metadata).")
        return chip_pil(scene_path, output_dir, chip_size, overlap)

    output_dir.mkdir(parents=True, exist_ok=True)
    scene_id = _scene_id_from_path(scene_path)

    with rasterio.open(scene_path) as src:
        width = src.width
        height = src.height
        crs_str = src.crs.to_string() if src.crs else None
        transform: Affine = src.transform

        band_count = src.count
        # Read all bands as (bands, H, W) then transpose to (H, W, bands)
        data_full = src.read()  # (bands, H, W)

    print(f"📊 Scene dimensions: {width}w x {height}h, {band_count} band(s), CRS: {crs_str}")

    row_offsets = _compute_grid(height, chip_size, overlap)
    col_offsets = _compute_grid(width, chip_size, overlap)
    total_chips = len(row_offsets) * len(col_offsets)
    print(f"📊 Grid: {len(row_offsets)} rows x {len(col_offsets)} cols = {total_chips} chips")

    chips_meta: list[dict[str, Any]] = []
    chip_index = 0

    with tqdm(total=total_chips, desc="Chipping GeoTIFF", unit="chip") as pbar:
        for ri, y_off in enumerate(row_offsets):
            for ci, x_off in enumerate(col_offsets):
                y_end = min(y_off + chip_size, height)
                x_end = min(x_off + chip_size, width)

                # Slice: data_full is (bands, H, W)
                chip_data = data_full[:, y_off:y_end, x_off:x_end]  # (bands, h, w)
                chip_data = np.transpose(chip_data, (1, 2, 0))       # (h, w, bands)

                chip_h, chip_w = chip_data.shape[:2]
                if chip_h < chip_size or chip_w < chip_size:
                    chip_data = _pad_array(chip_data, chip_size, chip_size)

                # Normalise to uint8 if needed
                if chip_data.dtype != np.uint8:
                    band_min = chip_data.min()
                    band_max = chip_data.max()
                    if band_max > band_min:
                        chip_data = ((chip_data - band_min) / (band_max - band_min) * 255).astype(np.uint8)
                    else:
                        chip_data = np.zeros_like(chip_data, dtype=np.uint8)

                # Convert to RGB PIL image
                if chip_data.shape[2] >= 3:
                    img = Image.fromarray(chip_data[:, :, :3], mode="RGB")
                elif chip_data.shape[2] == 1:
                    img = Image.fromarray(chip_data[:, :, 0], mode="L").convert("RGB")
                else:
                    img = Image.fromarray(chip_data[:, :, :2]).convert("RGB")

                filename = f"{scene_id}_{ri:04d}_{ci:04d}.jpg"
                out_path = output_dir / filename
                img.save(out_path, quality=95)

                # Geo bounds
                west, north = _pixel_to_geo(transform, x_off, y_off)
                east, south = _pixel_to_geo(transform, x_end, y_end)

                chips_meta.append(
                    {
                        "filename": filename,
                        "row": ri,
                        "col": ci,
                        "x_offset": x_off,
                        "y_offset": y_off,
                        "geo_bounds": {
                            "west": west,
                            "east": east,
                            "north": north,
                            "south": south,
                        },
                    }
                )
                chip_index += 1
                pbar.update(1)

    chip_index_data: dict[str, Any] = {
        "scene_id": scene_id,
        "scene_path": str(scene_path.resolve()),
        "chip_size": chip_size,
        "overlap": overlap,
        "crs": crs_str,
        "chips": chips_meta,
    }

    index_path = output_dir / f"{scene_id}_chip_index.json"
    index_path.write_text(json.dumps(chip_index_data, indent=2))

    return chip_index_data


# ---------------------------------------------------------------------------
# PIL (JPEG/PNG + fallback GeoTIFF) chipping
# ---------------------------------------------------------------------------

def chip_pil(
    scene_path: Path,
    output_dir: Path,
    chip_size: int,
    overlap: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_id = _scene_id_from_path(scene_path)

    img = Image.open(scene_path).convert("RGB")
    width, height = img.size
    img_array = np.array(img)

    print(f"📊 Scene dimensions: {width}w x {height}h (PIL, no geo metadata)")

    row_offsets = _compute_grid(height, chip_size, overlap)
    col_offsets = _compute_grid(width, chip_size, overlap)
    total_chips = len(row_offsets) * len(col_offsets)
    print(f"📊 Grid: {len(row_offsets)} rows x {len(col_offsets)} cols = {total_chips} chips")

    chips_meta: list[dict[str, Any]] = []

    with tqdm(total=total_chips, desc="Chipping image", unit="chip") as pbar:
        for ri, y_off in enumerate(row_offsets):
            for ci, x_off in enumerate(col_offsets):
                y_end = min(y_off + chip_size, height)
                x_end = min(x_off + chip_size, width)

                chip_data = img_array[y_off:y_end, x_off:x_end, :]
                chip_h, chip_w = chip_data.shape[:2]
                if chip_h < chip_size or chip_w < chip_size:
                    chip_data = _pad_array(chip_data, chip_size, chip_size)

                chip_img = Image.fromarray(chip_data, mode="RGB")
                filename = f"{scene_id}_{ri:04d}_{ci:04d}.jpg"
                out_path = output_dir / filename
                chip_img.save(out_path, quality=95)

                chips_meta.append(
                    {
                        "filename": filename,
                        "row": ri,
                        "col": ci,
                        "x_offset": x_off,
                        "y_offset": y_off,
                        "geo_bounds": None,
                    }
                )
                pbar.update(1)

    chip_index_data: dict[str, Any] = {
        "scene_id": scene_id,
        "scene_path": str(scene_path.resolve()),
        "chip_size": chip_size,
        "overlap": overlap,
        "crs": None,
        "chips": chips_meta,
    }

    index_path = output_dir / f"{scene_id}_chip_index.json"
    index_path.write_text(json.dumps(chip_index_data, indent=2))

    return chip_index_data


# ---------------------------------------------------------------------------
# Public entry point (used by sentinel2_fetcher as a library call too)
# ---------------------------------------------------------------------------

def generate_chips(
    scene_path: Path,
    output_dir: Path,
    chip_size: int = 640,
    overlap: int = 64,
) -> dict[str, Any]:
    """
    Tile *scene_path* into chips and return the chip-index dict.
    Writes chips to *output_dir* and logs to Supabase (non-fatal).
    """
    suffix = scene_path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        chip_index = chip_geotiff(scene_path, output_dir, chip_size, overlap)
    elif suffix in (".jpg", ".jpeg", ".png"):
        chip_index = chip_pil(scene_path, output_dir, chip_size, overlap)
    else:
        print(f"❌ Unsupported file type: {suffix}")
        sys.exit(1)

    n_chips = len(chip_index["chips"])
    print(f"✅ {n_chips} chips written to {output_dir}")

    _supabase_log_chip_index(chip_index["scene_id"], chip_index)

    return chip_index


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tile a GeoTIFF or JPEG/PNG scene into overlapping chips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to input scene (.tif, .tiff, .jpg, .png)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="DIR",
        help="Output directory for chips",
    )
    parser.add_argument(
        "--chip-size",
        type=int,
        default=640,
        metavar="PX",
        help="Chip width and height in pixels",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        metavar="PX",
        help="Overlap between adjacent chips in pixels",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    scene_path: Path = args.input.resolve()
    output_dir: Path = args.output

    if not scene_path.exists():
        print(f"❌ Input file not found: {scene_path}")
        sys.exit(1)

    if args.overlap >= args.chip_size:
        print(f"❌ --overlap ({args.overlap}) must be less than --chip-size ({args.chip_size})")
        sys.exit(1)

    print(f"📊 Input : {scene_path}")
    print(f"📊 Output: {output_dir}")
    print(f"📊 Chip size: {args.chip_size}px  Overlap: {args.overlap}px")

    generate_chips(scene_path, output_dir, args.chip_size, args.overlap)


if __name__ == "__main__":
    main()
