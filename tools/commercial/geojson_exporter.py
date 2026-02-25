"""
geojson_exporter.py
-------------------
Convert YOLOv8 detection results + chip geo-bounds into a GeoJSON
FeatureCollection for downstream GIS workflows.

Supports both standard detection format and change-detection report format.

CLI:
    python tools/commercial/geojson_exporter.py \\
      --detections data/detections.json \\
      --chip-index data/unlabeled/scene_chip_index.json \\
      --output data/exports/detections.geojson
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV_PATH = Path("/Users/kahlil/satellite-project/.env")
load_dotenv(dotenv_path=ENV_PATH)

import os  # noqa: E402 – after load_dotenv

# ---------------------------------------------------------------------------
# Supabase (non-fatal)
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://obdsgqjkjjmmtbcfjhnn.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

try:
    from supabase import create_client, Client as SupabaseClient

    supabase: SupabaseClient | None = (
        create_client(SUPABASE_URL, SUPABASE_KEY)
        if SUPABASE_URL and SUPABASE_KEY
        else None
    )
except Exception:
    supabase = None


# ---------------------------------------------------------------------------
# Coordinate conversion helpers
# ---------------------------------------------------------------------------

def _normalized_bbox_to_polygon_coords(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    geo_bounds: dict[str, float],
) -> list[list[float]]:
    """
    Convert a normalized YOLO bbox to a GeoJSON Polygon coordinate ring.

    *geo_bounds* must contain keys: west, east, north, south (all floats,
    in decimal degrees).

    x_center, y_center, width, height are all in [0, 1] normalized space
    where (0,0) is the top-left corner of the chip.

    Returns a closed ring: [[lon, lat], ..., [lon, lat]] (5 points).
    """
    west  = float(geo_bounds["west"])
    east  = float(geo_bounds["east"])
    north = float(geo_bounds["north"])
    south = float(geo_bounds["south"])

    lon_range = east - west
    lat_range = south - north  # negative because south < north

    half_w = width  / 2.0
    half_h = height / 2.0

    x_min = x_center - half_w
    x_max = x_center + half_w
    y_min = y_center - half_h
    y_max = y_center + half_h

    def to_lon(x: float) -> float:
        return west + x * lon_range

    def to_lat(y: float) -> float:
        # y=0 → north, y=1 → south
        return north + y * lat_range

    # Polygon corners: top-left, top-right, bottom-right, bottom-left, close
    ring: list[list[float]] = [
        [to_lon(x_min), to_lat(y_min)],
        [to_lon(x_max), to_lat(y_min)],
        [to_lon(x_max), to_lat(y_max)],
        [to_lon(x_min), to_lat(y_max)],
        [to_lon(x_min), to_lat(y_min)],  # close ring
    ]
    return ring


# ---------------------------------------------------------------------------
# Detection format detection
# ---------------------------------------------------------------------------

def _is_change_report(data: Any) -> bool:
    """Return True if *data* looks like a change detection report."""
    if isinstance(data, dict):
        return "changes" in data and "summary" in data
    return False


def _normalize_detections(
    data: Any,
) -> list[dict[str, Any]]:
    """
    Normalise varied input formats into a flat list of records:
      {filename, class_name, confidence, bbox, change_type (optional)}

    Supported input shapes:
      1. Standard:  list of {filename, detections: [{class_name, confidence,
                    bbox: {x_center, y_center, width, height}}]}
      2. Change report: {before_image, after_image, changes: [{class,
                    change_type, confidence, bbox: {x_center, …}}]}
    """
    records: list[dict] = []

    if _is_change_report(data):
        # Change detection report
        after_image = data.get("after_image", "unknown")
        filename = Path(after_image).name
        for ch in data.get("changes", []):
            bbox = ch.get("bbox", {})
            # bbox may have x_center/y_center/width/height OR x1/y1/x2/y2
            if "x_center" not in bbox and "x1" in bbox:
                img_w = 1.0  # pixel values; will be treated as raw later
                x1 = float(bbox["x1"])
                y1 = float(bbox["y1"])
                x2 = float(bbox["x2"])
                y2 = float(bbox["y2"])
                # Store raw pixel coords; geo conversion will be skipped
                # because we cannot know image dims here.  We store
                # normalized placeholders that the caller must handle.
                bbox = {
                    "x_center": (x1 + x2) / 2,
                    "y_center": (y1 + y2) / 2,
                    "width":    x2 - x1,
                    "height":   y2 - y1,
                    "_pixel_coords": True,
                }
            records.append(
                {
                    "filename": filename,
                    "class_name": ch.get("class", "unknown"),
                    "confidence": ch.get("confidence", 0.0),
                    "bbox": bbox,
                    "change_type": ch.get("change_type"),
                }
            )
        return records

    # Standard list format
    if isinstance(data, list):
        for item in data:
            filename = item.get("filename", "unknown")
            for det in item.get("detections", []):
                records.append(
                    {
                        "filename": filename,
                        "class_name": det.get("class_name", "unknown"),
                        "confidence": det.get("confidence", 0.0),
                        "bbox": det.get("bbox", {}),
                        "change_type": None,
                    }
                )
        return records

    print("❌ Unrecognized detections format (expected list or change report dict)")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Chip index lookup helpers
# ---------------------------------------------------------------------------

def _build_chip_lookup(
    chip_index_data: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Build a filename → chip_meta dict from chip_index JSON.

    The chip_index produced by chip_generator looks like:
      {chips: [{filename, geo_bounds, ...}, ...]}

    Also handles a flat {filename: {...}} dict style.
    """
    lookup: dict[str, dict] = {}

    if isinstance(chip_index_data, dict):
        chips = chip_index_data.get("chips")
        if isinstance(chips, list):
            for chip in chips:
                fn = chip.get("filename")
                if fn:
                    lookup[fn] = chip
            return lookup
        # Flat dict style: {filename: meta}
        for key, val in chip_index_data.items():
            if isinstance(val, dict):
                lookup[key] = val
        return lookup

    if isinstance(chip_index_data, list):
        for chip in chip_index_data:
            fn = chip.get("filename")
            if fn:
                lookup[fn] = chip
        return lookup

    return lookup


# ---------------------------------------------------------------------------
# Supabase write (non-fatal)
# ---------------------------------------------------------------------------

def _supabase_log_export(output_path: Path, feature_count: int) -> None:
    try:
        if supabase is None:
            return
        supabase.table("geojson_exports").insert(
            {
                "output_path": str(output_path),
                "feature_count": feature_count,
            }
        ).execute()
        print("✅ Export metadata written to Supabase")
    except Exception as exc:
        print(f"⚠️  Supabase write skipped (falling back to local only): {exc}")


# ---------------------------------------------------------------------------
# Core export
# ---------------------------------------------------------------------------

def export_geojson(
    detections_path: Path,
    chip_index_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """
    Load detections + chip index and write a GeoJSON FeatureCollection.
    Returns the FeatureCollection dict.
    """
    # Load detections
    try:
        raw_data: Any = json.loads(detections_path.read_text())
    except Exception as exc:
        print(f"❌ Cannot read detections file: {exc}")
        sys.exit(1)

    records = _normalize_detections(raw_data)
    print(f"📊 Normalized {len(records)} detection records")

    # Load chip index
    try:
        chip_index_data: Any = json.loads(chip_index_path.read_text())
    except Exception as exc:
        print(f"❌ Cannot read chip index file: {exc}")
        sys.exit(1)

    chip_lookup = _build_chip_lookup(chip_index_data)
    print(f"📊 Chip index loaded: {len(chip_lookup)} chips")

    # Build features
    features: list[dict] = []
    skipped_no_geo = 0
    skipped_pixel_coords = 0
    class_counts: dict[str, int] = defaultdict(int)

    # Track bounding box of entire export
    all_lons: list[float] = []
    all_lats: list[float] = []

    for record in tqdm(records, desc="Building GeoJSON features", unit="det"):
        filename = record["filename"]
        bbox = record.get("bbox", {})

        # Skip detections stored as raw pixel coords (change report xyxy without
        # image dimensions context)
        if bbox.get("_pixel_coords"):
            skipped_pixel_coords += 1
            continue

        chip_meta = chip_lookup.get(filename)
        if chip_meta is None:
            # Try stem match (filename without extension)
            stem = Path(filename).stem
            chip_meta = chip_lookup.get(stem)

        if chip_meta is None:
            skipped_no_geo += 1
            continue

        geo_bounds = chip_meta.get("geo_bounds")
        if not geo_bounds:
            skipped_no_geo += 1
            continue

        # Extract normalized bbox values
        try:
            x_center = float(bbox["x_center"])
            y_center = float(bbox["y_center"])
            width    = float(bbox["width"])
            height   = float(bbox["height"])
        except (KeyError, TypeError, ValueError) as exc:
            print(f"⚠️  Skipping record with malformed bbox ({filename}): {exc}")
            continue

        # Clamp to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width    = max(0.0, min(1.0, width))
        height   = max(0.0, min(1.0, height))

        ring = _normalized_bbox_to_polygon_coords(
            x_center, y_center, width, height, geo_bounds
        )

        # Accumulate extent
        for lon, lat in ring:
            all_lons.append(lon)
            all_lats.append(lat)

        class_name = record["class_name"]
        class_counts[class_name] += 1

        properties: dict[str, Any] = {
            "class_name": class_name,
            "confidence": record["confidence"],
            "chip_filename": filename,
            "detection_id": str(uuid.uuid4()),
        }
        if record.get("change_type") is not None:
            properties["change_type"] = record["change_type"]

        feature: dict[str, Any] = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring],
            },
            "properties": properties,
        }
        features.append(feature)

    if skipped_no_geo > 0:
        print(
            f"⚠️  Skipped {skipped_no_geo} detection(s) with no geo_bounds in chip index"
        )
    if skipped_pixel_coords > 0:
        print(
            f"⚠️  Skipped {skipped_pixel_coords} detection(s) stored in pixel coords "
            f"(no image dimensions available for normalization)"
        )

    feature_collection: dict[str, Any] = {
        "type": "FeatureCollection",
        "features": features,
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(feature_collection, indent=2))
    print(f"✅ GeoJSON written → {output_path}")

    # Summary stats
    print("\n" + "=" * 50)
    print("📊 Export Summary")
    print("=" * 50)
    print(f"  Total features exported: {len(features)}")

    if class_counts:
        print("  Classes breakdown:")
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"    {cls}: {cnt}")

    if all_lons and all_lats:
        bbox_export = [
            round(min(all_lons), 6),
            round(min(all_lats), 6),
            round(max(all_lons), 6),
            round(max(all_lats), 6),
        ]
        print(
            f"  Export bounding box (W, S, E, N): "
            f"{bbox_export[0]}, {bbox_export[1]}, {bbox_export[2]}, {bbox_export[3]}"
        )
    else:
        print("  Export bounding box: N/A (no georeferenced features)")

    print("=" * 50)

    _supabase_log_export(output_path, len(features))

    return feature_collection


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 detections + chip geo-bounds as GeoJSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--detections",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to detections JSON (list or change report)",
    )
    parser.add_argument(
        "--chip-index",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to chip_index JSON (from chip_generator)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="PATH",
        help="Output path for the GeoJSON FeatureCollection",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    detections_path: Path = args.detections.resolve()
    chip_index_path: Path = args.chip_index.resolve()
    output_path: Path     = args.output

    if not detections_path.exists():
        print(f"❌ Detections file not found: {detections_path}")
        sys.exit(1)
    if not chip_index_path.exists():
        print(f"❌ Chip index file not found: {chip_index_path}")
        sys.exit(1)

    export_geojson(
        detections_path=detections_path,
        chip_index_path=chip_index_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
