"""
tools/commercial/geojson_exporter.py

Converts pixel-space YOLO detections to GeoJSON with real-world coordinates.
Output loads directly into ArcGIS, QGIS, Google Earth, or any mapping tool.
This is what makes your output usable to paying customers today.

Usage:
  python tools/commercial/geojson_exporter.py \
    --detections data/detections.json \
    --chip-index data/unlabeled/scene_chip_index.json \
    --output data/detections.geojson
"""

import os
import json
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import argparse
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))


CLASS_NAMES = {0: "building", 1: "vehicle", 2: "aircraft", 3: "ship"}

INSURANCE_RISK_LEVELS = {
    "building": {"appeared": "review_required", "disappeared": "high_priority", "unchanged": "low"},
    "vehicle": {"appeared": "low", "disappeared": "low", "unchanged": "low"},
    "aircraft": {"appeared": "medium", "disappeared": "medium", "unchanged": "low"},
    "ship": {"appeared": "medium", "disappeared": "medium", "unchanged": "low"},
}


class GeoJSONExporter:
    """
    Converts YOLO detections to GeoJSON FeatureCollection.
    Handles both chip-level detections (with chip index) and
    direct scene-level detections (with embedded geo metadata).
    """

    def __init__(self, chip_index_path: Optional[Path] = None):
        self.chip_index = {}
        if chip_index_path and Path(chip_index_path).exists():
            for entry in json.loads(Path(chip_index_path).read_text()):
                self.chip_index[entry["chip_name"]] = entry
            print(f"✅ Loaded chip index: {len(self.chip_index)} chips")
        else:
            print("ℹ️  No chip index — exporting pixel coordinates only")

    def _bbox_to_geo_polygon(self, bbox: List[float], chip_meta: Dict) -> Optional[List]:
        """
        Convert normalized YOLO bbox to geographic polygon coordinates.
        Returns [[lon,lat], ...] list (GeoJSON format: lon before lat).
        """
        if not chip_meta or "lon_min" not in chip_meta:
            return None

        x_c, y_c, bw, bh = bbox
        lon_min, lon_max = chip_meta["lon_min"], chip_meta["lon_max"]
        lat_min, lat_max = chip_meta["lat_min"], chip_meta["lat_max"]

        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min

        # Convert bbox center+size to corner coordinates
        obj_lon_min = lon_min + (x_c - bw/2) * lon_range
        obj_lon_max = lon_min + (x_c + bw/2) * lon_range
        obj_lat_min = lat_max - (y_c + bh/2) * lat_range  # Note: lat flipped (y=0 is top)
        obj_lat_max = lat_max - (y_c - bh/2) * lat_range

        # GeoJSON polygon: [lon, lat] pairs, closed ring
        return [[
            [obj_lon_min, obj_lat_max],
            [obj_lon_max, obj_lat_max],
            [obj_lon_max, obj_lat_min],
            [obj_lon_min, obj_lat_min],
            [obj_lon_min, obj_lat_max]  # Close ring
        ]]

    def _estimate_area_sqm(self, bbox: List[float], chip_meta: Dict) -> Optional[float]:
        """Estimate real-world area of detection in square meters."""
        if not chip_meta or "lon_min" not in chip_meta:
            return None

        lon_range = chip_meta["lon_max"] - chip_meta["lon_min"]
        lat_range = chip_meta["lat_max"] - chip_meta["lat_min"]
        center_lat = (chip_meta["lat_min"] + chip_meta["lat_max"]) / 2

        # Approximate meters per degree
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))

        obj_width_m = bbox[2] * lon_range * meters_per_deg_lon
        obj_height_m = bbox[3] * lat_range * meters_per_deg_lat
        return round(obj_width_m * obj_height_m, 2)

    def detection_to_feature(self, detection: Dict, chip_name: str, metadata: Dict = None) -> Dict:
        """Convert a single detection to a GeoJSON Feature."""
        chip_meta = self.chip_index.get(chip_name, {})
        bbox = detection.get("bbox", [0.5, 0.5, 0.1, 0.1])
        class_name = detection.get("class_name", CLASS_NAMES.get(detection.get("class_id", 0), "unknown"))
        confidence = detection.get("confidence", 0.0)
        change_type = detection.get("change_type", "detected")

        geo_polygon = self._bbox_to_geo_polygon(bbox, chip_meta)
        area_sqm = self._estimate_area_sqm(bbox, chip_meta)

        properties = {
            "class": class_name,
            "confidence": round(confidence, 4),
            "chip_source": chip_name,
            "change_type": change_type,
            "area_sqm": area_sqm,
            "bbox_normalized": bbox,
            "risk_level": INSURANCE_RISK_LEVELS.get(class_name, {}).get(change_type, "unknown"),
            "detected_at": datetime.utcnow().isoformat() + "Z",
        }

        if chip_meta:
            properties["scene_id"] = chip_meta.get("scene_id")
            properties["image_date"] = chip_meta.get("date")

        if metadata:
            properties.update(metadata)

        if geo_polygon:
            geometry = {"type": "Polygon", "coordinates": geo_polygon}
        else:
            # Fallback to pixel centroid as Point
            geometry = {
                "type": "Point",
                "coordinates": [
                    chip_meta.get("pixel_col", 0) + bbox[0] * 640,
                    chip_meta.get("pixel_row", 0) + bbox[1] * 640
                ]
            }

        return {"type": "Feature", "geometry": geometry, "properties": properties}

    def export_detections(
        self,
        detections_input: List[Dict],
        output_path: Path,
        collection_metadata: Dict = None
    ) -> Dict:
        """
        Export a list of detection dicts to GeoJSON FeatureCollection.

        detections_input format:
        [{"chip_name": "scene_r001_c002.jpg", "class_name": "building",
          "confidence": 0.87, "bbox": [0.5,0.5,0.1,0.1]}, ...]
        """
        features = []
        geo_count = pixel_count = 0

        for det in detections_input:
            chip_name = det.get("chip_name", det.get("image", "unknown"))
            feature = self.detection_to_feature(det, chip_name)
            features.append(feature)
            if feature["geometry"]["type"] == "Polygon":
                geo_count += 1
            else:
                pixel_count += 1

        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "total_detections": len(features),
                "geo_referenced": geo_count,
                "pixel_only": pixel_count,
                "crs": "EPSG:4326",
                **(collection_metadata or {})
            }
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(geojson, indent=2))

        print(f"\n✅ GeoJSON Export Complete")
        print(f"   Total features:    {len(features)}")
        print(f"   Geo-referenced:    {geo_count} (load in ArcGIS/QGIS)")
        print(f"   Pixel-only:        {pixel_count} (no chip index match)")
        print(f"   Output:            {output_path}")

        # Print class breakdown
        by_class = {}
        for f in features:
            cls = f["properties"]["class"]
            by_class[cls] = by_class.get(cls, 0) + 1
        print(f"\n   By class:")
        for cls, count in sorted(by_class.items()):
            print(f"     {cls}: {count}")

        return geojson

    def export_change_report(self, change_report: Dict, output_path: Path) -> Dict:
        """
        Export a change detection report to GeoJSON.
        Specifically structured for insurance claim review workflows.
        """
        detections = []
        for change in change_report.get("changes", []):
            if change["change_type"] == "unchanged":
                continue
            bbox = change.get("location_after") or change.get("location_before", [0.5, 0.5, 0.1, 0.1])
            chip_name = change_report.get("after_image", "unknown")
            detections.append({
                "chip_name": Path(chip_name).name,
                "class_name": change["class_name"],
                "confidence": change["confidence"],
                "bbox": bbox,
                "change_type": change["change_type"],
                "significance": change["significance"],
            })

        return self.export_detections(
            detections,
            output_path,
            collection_metadata={
                "report_type": "change_detection",
                "before_image": change_report.get("before_image"),
                "after_image": change_report.get("after_image"),
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True, help="JSON file with detections list")
    parser.add_argument("--chip-index", default=None)
    parser.add_argument("--output", default=str(DATA_DIR / "exports" / "detections.geojson"))
    args = parser.parse_args()

    exporter = GeoJSONExporter(chip_index_path=Path(args.chip_index) if args.chip_index else None)
    detections = json.loads(Path(args.detections).read_text())
    exporter.export_detections(detections, Path(args.output))
