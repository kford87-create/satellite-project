"""
tools/data_acquisition/sentinel2_fetcher.py

Automatically fetches Sentinel-2 EO imagery from ESA Copernicus
by bounding box and date range. Filters by cloud cover and drops
clean scenes directly into data/unlabeled/ for the bootstrapping loop.

Usage:
  python tools/data_acquisition/sentinel2_fetcher.py \
    --bbox -87.7 41.8 -87.5 42.0 \
    --start 2024-01-01 \
    --end 2024-06-01 \
    --max-cloud 20 \
    --limit 50

Requirements:
  pip install sentinelsat
  Free ESA account at: https://dataspace.copernicus.eu
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
from dotenv import load_dotenv
from tqdm import tqdm
import argparse

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
COPERNICUS_USER = os.getenv("COPERNICUS_USER")
COPERNICUS_PASSWORD = os.getenv("COPERNICUS_PASSWORD")
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
UNLABELED_DIR = DATA_DIR / "unlabeled"
SENTINEL_DIR = DATA_DIR / "sentinel2_raw"

# Sentinel-2 band config
# B04=Red, B03=Green, B02=Blue → natural color RGB
# B08=NIR → useful for vegetation/change detection
TARGET_BANDS = ["B04", "B03", "B02", "B08"]
RESOLUTION = 10  # meters per pixel (10m is highest Sentinel-2 resolution)


class Sentinel2Fetcher:
    """
    Fetches Sentinel-2 scenes from ESA Copernicus Data Space.
    Filters by cloud cover, downloads, and prepares for the bootstrapping pipeline.
    """

    def __init__(self):
        self._check_dependencies()
        self.api = self._connect()
        UNLABELED_DIR.mkdir(parents=True, exist_ok=True)
        SENTINEL_DIR.mkdir(parents=True, exist_ok=True)

    def _check_dependencies(self):
        try:
            import sentinelsat
        except ImportError:
            raise ImportError(
                "sentinelsat not installed.\n"
                "Run: pip install sentinelsat\n"
                "Then create a free account at: https://dataspace.copernicus.eu"
            )

    def _connect(self):
        from sentinelsat import SentinelAPI
        if not COPERNICUS_USER or not COPERNICUS_PASSWORD:
            raise ValueError(
                "Missing Copernicus credentials.\n"
                "Set COPERNICUS_USER and COPERNICUS_PASSWORD in .env\n"
                "Free account at: https://dataspace.copernicus.eu"
            )
        print("🛰️  Connecting to ESA Copernicus Data Space...")
        api = SentinelAPI(
            COPERNICUS_USER,
            COPERNICUS_PASSWORD,
            "https://dataspace.copernicus.eu/resto/api/collections/Sentinel2/search.json"
        )
        print("✅ Connected to Copernicus")
        return api

    def search_scenes(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        max_cloud_cover: float = 20.0,
        limit: int = 50,
        platform: str = "Sentinel-2"
    ) -> List[dict]:
        """
        Search for available Sentinel-2 scenes.

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            start_date: 'YYYY-MM-DD'
            end_date: 'YYYY-MM-DD'
            max_cloud_cover: Maximum cloud cover % (0-100)
            limit: Max scenes to return
            platform: Sentinel-2 platform name

        Returns:
            List of scene metadata dicts
        """
        from sentinelsat import geojson_to_wkt
        from shapely.geometry import box

        min_lon, min_lat, max_lon, max_lat = bbox
        footprint = geojson_to_wkt({
            "type": "Polygon",
            "coordinates": [[
                [min_lon, min_lat], [max_lon, min_lat],
                [max_lon, max_lat], [min_lon, max_lat],
                [min_lon, min_lat]
            ]]
        })

        print(f"\n🔍 Searching Sentinel-2 scenes...")
        print(f"   BBox: {bbox}")
        print(f"   Date range: {start_date} → {end_date}")
        print(f"   Max cloud cover: {max_cloud_cover}%")

        products = self.api.query(
            footprint,
            date=(start_date.replace("-", ""), end_date.replace("-", "")),
            platformname=platform,
            cloudcoverpercentage=(0, max_cloud_cover),
            producttype="S2MSI2A",  # Level-2A = surface reflectance (atmospherically corrected)
        )

        scenes = []
        for product_id, product_info in list(products.items())[:limit]:
            scenes.append({
                "id": product_id,
                "title": product_info.get("title", ""),
                "cloud_cover": product_info.get("cloudcoverpercentage", 0),
                "date": product_info.get("beginposition", ""),
                "size_mb": product_info.get("size", "unknown"),
                "footprint": product_info.get("footprint", ""),
            })

        scenes.sort(key=lambda x: x["cloud_cover"])
        print(f"\n✅ Found {len(scenes)} scenes (sorted by cloud cover)")

        for i, s in enumerate(scenes[:5]):
            print(f"   [{i+1}] {s['title'][:50]} — {s['cloud_cover']:.1f}% cloud")

        return scenes

    def download_scenes(
        self,
        scenes: List[dict],
        max_downloads: int = 10
    ) -> List[Path]:
        """Download scenes and return list of downloaded paths."""
        downloaded = []
        scenes_to_download = scenes[:max_downloads]

        print(f"\n📥 Downloading {len(scenes_to_download)} scenes...")

        for scene in tqdm(scenes_to_download, desc="Downloading"):
            product_id = scene["id"]
            output_dir = SENTINEL_DIR / scene["title"]

            if output_dir.exists():
                print(f"   ⏭️  Already downloaded: {scene['title'][:40]}")
                downloaded.append(output_dir)
                continue

            try:
                self.api.download(product_id, directory_path=SENTINEL_DIR)
                downloaded.append(output_dir)
            except Exception as e:
                print(f"   ⚠️  Failed to download {scene['title'][:40]}: {e}")

        print(f"\n✅ Downloaded {len(downloaded)} scenes to {SENTINEL_DIR}")
        return downloaded

    def process_to_chips(
        self,
        scene_paths: List[Path],
        chip_size: int = 640,
        overlap: int = 64
    ) -> int:
        """
        Process downloaded scenes into 640x640 chips for the bootstrapping pipeline.
        Delegates to the ChipGenerator tool for actual chipping.
        """
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from chip_generator import ChipGenerator

            generator = ChipGenerator(chip_size=chip_size, overlap=overlap)
            total_chips = 0

            for scene_path in scene_paths:
                # Find RGB bands in the scene
                tif_files = list(scene_path.rglob("*B04_10m.jp2"))  # Red band
                if not tif_files:
                    tif_files = list(scene_path.rglob("*.tif"))

                for tif in tif_files[:1]:  # Process first matching file
                    n_chips = generator.chip_scene(
                        scene_path=tif,
                        output_dir=UNLABELED_DIR,
                        scene_id=scene_path.name
                    )
                    total_chips += n_chips

            return total_chips

        except ImportError:
            print("⚠️  ChipGenerator not available — scenes saved to SENTINEL_DIR")
            print("   Run tools/data_acquisition/chip_generator.py separately")
            return 0

    def save_scene_index(self, scenes: List[dict]):
        """Save scene metadata index for reproducibility."""
        index_path = DATA_DIR / "sentinel2_scene_index.json"
        existing = []
        if index_path.exists():
            existing = json.loads(index_path.read_text())

        # Add new scenes (avoid duplicates)
        existing_ids = {s["id"] for s in existing}
        new_scenes = [s for s in scenes if s["id"] not in existing_ids]
        existing.extend(new_scenes)

        index_path.write_text(json.dumps(existing, indent=2, default=str))
        print(f"📋 Scene index saved: {index_path} ({len(existing)} total scenes)")


def fetch_scenes_for_region(
    bbox: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    max_cloud: float = 20.0,
    limit: int = 50,
    download: bool = True,
    chip: bool = True
):
    """High-level convenience function for the full fetch pipeline."""
    fetcher = Sentinel2Fetcher()

    # Search
    scenes = fetcher.search_scenes(bbox, start_date, end_date, max_cloud, limit)
    if not scenes:
        print("❌ No scenes found matching criteria")
        return

    # Save index
    fetcher.save_scene_index(scenes)

    if not download:
        print("\nSearch complete. Run with --download to fetch scenes.")
        return

    # Download
    downloaded = fetcher.download_scenes(scenes, max_downloads=min(limit, 10))

    if chip and downloaded:
        print("\n✂️  Chipping scenes for bootstrapping pipeline...")
        n_chips = fetcher.process_to_chips(downloaded)
        print(f"✅ Generated {n_chips} chips → {UNLABELED_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Sentinel-2 imagery from ESA Copernicus")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                        default=[-87.7, 41.8, -87.5, 42.0], help="Bounding box (default: Chicago)")
    parser.add_argument("--start", default=(datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d"))
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--max-cloud", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--search-only", action="store_true", help="Search without downloading")
    parser.add_argument("--no-chip", action="store_true", help="Download but don't chip")
    args = parser.parse_args()

    fetch_scenes_for_region(
        bbox=tuple(args.bbox),
        start_date=args.start,
        end_date=args.end,
        max_cloud=args.max_cloud,
        limit=args.limit,
        download=not args.search_only,
        chip=not args.no_chip
    )
