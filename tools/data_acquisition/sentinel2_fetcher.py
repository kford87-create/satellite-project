"""
sentinel2_fetcher.py
--------------------
Fetches Sentinel-2 L2A imagery from ESA Copernicus Data Space, then
chips each downloaded scene into 640x640 tiles ready for labelling.

Usage:
    python tools/data_acquisition/sentinel2_fetcher.py \
        --bbox -87.7 41.8 -87.5 42.0 \
        --start 2024-01-01 --end 2024-06-01 \
        --max-cloud 20 --limit 50
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV_PATH = Path("/Users/kahlil/satellite-project/.env")
load_dotenv(ENV_PATH)

import os

COPERNICUS_USER = os.environ.get("COPERNICUS_USER", "")
COPERNICUS_PASSWORD = os.environ.get("COPERNICUS_PASSWORD", "")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/Users/kahlil/satellite-project")
RAW_DIR = PROJECT_ROOT / "data" / "sentinel2_raw"
UNLABELED_DIR = PROJECT_ROOT / "data" / "unlabeled"

# ---------------------------------------------------------------------------
# Supabase (non-fatal)
# ---------------------------------------------------------------------------
try:
    from supabase import create_client, Client as SupabaseClient

    _supa_url = os.environ.get("SUPABASE_URL", "")
    _supa_key = os.environ.get("SUPABASE_KEY", "")
    supabase: SupabaseClient | None = (
        create_client(_supa_url, _supa_key) if _supa_url and _supa_key else None
    )
except Exception:
    supabase = None


def _supabase_log_scene(product_id: str, scene_meta: dict[str, Any]) -> None:
    """Non-fatal write of scene download metadata to Supabase."""
    try:
        if supabase is None:
            return
        supabase.table("sentinel2_scenes").upsert(
            {"product_id": product_id, "metadata": str(scene_meta)}
        ).execute()
    except Exception as exc:
        print(f"⚠️  Supabase write skipped (falling back to local only): {exc}")


# ---------------------------------------------------------------------------
# Chip generator integration
# ---------------------------------------------------------------------------

def _chip_scene(scene_path: Path, chip_size: int = 640, overlap: int = 64) -> int:
    """
    Import and call chip_generator.generate_chips.
    Returns the number of chips produced, or 0 on error.
    """
    try:
        # Allow running from any cwd by inserting the tools package root
        tools_root = Path(__file__).resolve().parent.parent.parent
        if str(tools_root) not in sys.path:
            sys.path.insert(0, str(tools_root))

        from tools.data_acquisition.chip_generator import generate_chips  # type: ignore

        chip_index = generate_chips(scene_path, UNLABELED_DIR, chip_size, overlap)
        return len(chip_index.get("chips", []))
    except Exception as exc:
        print(f"⚠️  Chipping failed for {scene_path.name}: {exc}")
        return 0


# ---------------------------------------------------------------------------
# Copernicus OAuth2 authentication
# ---------------------------------------------------------------------------

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
    "/protocol/openid-connect/token"
)
CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_BASE = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"


def get_access_token(username: str, password: str) -> str:
    """Obtain a short-lived OAuth2 bearer token from Copernicus Data Space."""
    if not username or not password:
        print(
            "❌ COPERNICUS_USER and COPERNICUS_PASSWORD must be set in the .env file.\n"
            "   Register at https://dataspace.copernicus.eu/"
        )
        sys.exit(1)

    payload = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": username,
        "password": password,
    }
    try:
        resp = requests.post(TOKEN_URL, data=payload, timeout=30)
        resp.raise_for_status()
        token: str = resp.json()["access_token"]
        return token
    except requests.HTTPError as exc:
        print(f"❌ Authentication failed: {exc}")
        print(f"   Response: {exc.response.text[:400]}")
        sys.exit(1)
    except Exception as exc:
        print(f"❌ Authentication error: {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# OData search query builder
# ---------------------------------------------------------------------------

def _bbox_to_wkt_polygon(west: float, south: float, east: float, north: float) -> str:
    """Return a WKT POLYGON string from a bounding box (minx, miny, maxx, maxy)."""
    return (
        f"{west} {south},{east} {south},"
        f"{east} {north},{west} {north},{west} {south}"
    )


def search_products(
    bbox: tuple[float, float, float, float],
    start: str,
    end: str,
    max_cloud: float,
    limit: int,
    token: str,
) -> list[dict[str, Any]]:
    """
    Search the Copernicus OData catalogue and return up to *limit* product dicts.
    bbox: (west, south, east, north)
    """
    west, south, east, north = bbox
    bbox_wkt = _bbox_to_wkt_polygon(west, south, east, north)

    filter_clause = (
        f"Collection/Name eq 'SENTINEL-2' and "
        f"Attributes/OData.CSC.DoubleAttribute/any("
        f"att:att/Name eq 'cloudCover' and "
        f"att/OData.CSC.DoubleAttribute/Value le {max_cloud}) and "
        f"ContentDate/Start gt {start}T00:00:00.000Z and "
        f"ContentDate/Start lt {end}T23:59:59.999Z and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({bbox_wkt}))')"
    )

    params: dict[str, Any] = {
        "$filter": filter_clause,
        "$orderby": "ContentDate/Start desc",
        "$top": min(limit, 100),  # API page size cap
        "$expand": "Attributes",
        "$count": "true",
    }

    headers = {"Authorization": f"Bearer {token}"}

    all_products: list[dict[str, Any]] = []
    url: str | None = CATALOGUE_URL

    print(f"📊 Querying Copernicus catalogue (bbox={bbox}, cloud<={max_cloud}%, limit={limit}) …")

    while url and len(all_products) < limit:
        try:
            resp = requests.get(url, params=params if url == CATALOGUE_URL else None,
                                headers=headers, timeout=60)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            print(f"❌ Catalogue query failed: {exc}")
            print(f"   Response: {exc.response.text[:400]}")
            sys.exit(1)
        except Exception as exc:
            print(f"❌ Catalogue request error: {exc}")
            sys.exit(1)

        body = resp.json()
        page_products: list[dict[str, Any]] = body.get("value", [])
        all_products.extend(page_products)

        # OData next-page link
        url = body.get("@odata.nextLink")
        params = {}  # params already encoded in nextLink URL

    all_products = all_products[:limit]
    print(f"📊 Found {len(all_products)} product(s) matching search criteria.")
    return all_products


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_with_progress(url: str, dest_path: Path, token: str) -> bool:
    """
    Stream-download *url* to *dest_path*, showing a tqdm progress bar.
    Returns True on success.
    """
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(dest_path, "wb") as fh, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest_path.name,
                leave=False,
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    pbar.update(len(chunk))
        return True
    except requests.HTTPError as exc:
        print(f"❌ Download failed [{exc.response.status_code}]: {url}")
        return False
    except Exception as exc:
        print(f"❌ Download error: {exc}")
        return False


def _unzip_to_dir(zip_path: Path, dest_dir: Path) -> list[Path]:
    """Unzip *zip_path* into *dest_dir*; return list of extracted .tif files."""
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist()]
        for member in tqdm(members, desc=f"Unzipping {zip_path.name}", unit="file", leave=False):
            zf.extract(member, dest_dir)
            extracted_path = dest_dir / member
            if extracted_path.suffix.lower() in (".tif", ".tiff", ".jp2"):
                extracted.append(extracted_path)
    return extracted


def _find_best_tif(extracted_files: list[Path]) -> Path | None:
    """
    From a list of extracted paths, prefer a TCI (True Colour Image) or
    the largest .tif/.jp2 file as the scene representative.
    """
    # Prefer TCI (RGB composite) if present
    for p in extracted_files:
        if "TCI" in p.name.upper():
            return p
    # Fall back to the largest file
    tifs = [p for p in extracted_files if p.exists()]
    if not tifs:
        return None
    return max(tifs, key=lambda p: p.stat().st_size)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_fetch(
    bbox: tuple[float, float, float, float],
    start: str,
    end: str,
    max_cloud: float,
    limit: int,
) -> tuple[int, int]:
    """
    Full fetch-and-chip pipeline.
    Returns (scenes_downloaded, total_chips_created).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    UNLABELED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Authenticate
    print("🔐 Authenticating with Copernicus Data Space …")
    token = get_access_token(COPERNICUS_USER, COPERNICUS_PASSWORD)
    print("✅ Authentication successful.")

    # 2. Search
    products = search_products(bbox, start, end, max_cloud, limit, token)

    if not products:
        print("⚠️  No products found for the given criteria.")
        return 0, 0

    scenes_downloaded = 0
    total_chips = 0

    # 3. Download + chip each product
    for product in tqdm(products, desc="Downloading scenes", unit="scene"):
        product_id: str = product.get("Id", "")
        product_name: str = product.get("Name", product_id)

        if not product_id:
            print(f"⚠️  Skipping product with missing Id: {product_name}")
            continue

        zip_dest = RAW_DIR / f"{product_name}.zip"

        # Skip already-downloaded zips
        if zip_dest.exists():
            print(f"⚠️  Already exists, skipping download: {zip_dest.name}")
        else:
            download_url = f"{DOWNLOAD_BASE}({product_id})/$value"
            print(f"⬇️  Downloading {product_name} …")
            success = _download_with_progress(download_url, zip_dest, token)
            if not success:
                if zip_dest.exists():
                    zip_dest.unlink()
                continue

        scenes_downloaded += 1

        # Log to Supabase (non-fatal)
        _supabase_log_scene(product_id, {
            "name": product_name,
            "zip_path": str(zip_dest),
        })

        # 4. Unzip
        extract_dir = RAW_DIR / product_name
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            extracted_files = _unzip_to_dir(zip_dest, extract_dir)
        except zipfile.BadZipFile as exc:
            print(f"❌ Bad zip file {zip_dest.name}: {exc}")
            zip_dest.unlink(missing_ok=True)
            continue
        except Exception as exc:
            print(f"❌ Unzip error for {zip_dest.name}: {exc}")
            continue

        # 5. Find representative scene file
        scene_file = _find_best_tif(extracted_files)
        if scene_file is None:
            # Try JP2 files as well
            jp2_files = list(extract_dir.rglob("*TCI*.jp2"))
            if jp2_files:
                scene_file = max(jp2_files, key=lambda p: p.stat().st_size)

        if scene_file is None:
            print(f"⚠️  No usable scene file found in {extract_dir.name}; skipping chipping.")
            continue

        print(f"✅ Scene extracted: {scene_file.name}")

        # 6. Chip
        n_chips = _chip_scene(scene_file)
        total_chips += n_chips
        print(f"✅ {n_chips} chip(s) created from {scene_file.name}")

        # Refresh token every 5 scenes (tokens expire after ~10 min)
        if scenes_downloaded % 5 == 0:
            try:
                token = get_access_token(COPERNICUS_USER, COPERNICUS_PASSWORD)
            except SystemExit:
                print("⚠️  Token refresh failed; continuing with existing token.")

    return scenes_downloaded, total_chips


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Sentinel-2 L2A imagery from Copernicus Data Space "
            "and chip into 640x640 tiles."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Bounding box as: west south east north (decimal degrees)",
    )
    parser.add_argument(
        "--start",
        required=True,
        metavar="YYYY-MM-DD",
        help="Start date for scene search (inclusive)",
    )
    parser.add_argument(
        "--end",
        required=True,
        metavar="YYYY-MM-DD",
        help="End date for scene search (inclusive)",
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=20.0,
        metavar="PCT",
        help="Maximum cloud cover percentage (0–100)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        metavar="N",
        help="Maximum number of scenes to download",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    west, south, east, north = args.bbox
    if west >= east:
        print(f"❌ WEST ({west}) must be less than EAST ({east})")
        sys.exit(1)
    if south >= north:
        print(f"❌ SOUTH ({south}) must be less than NORTH ({north})")
        sys.exit(1)
    if not (0 <= args.max_cloud <= 100):
        print(f"❌ --max-cloud must be between 0 and 100, got {args.max_cloud}")
        sys.exit(1)
    if args.limit <= 0:
        print(f"❌ --limit must be a positive integer, got {args.limit}")
        sys.exit(1)

    print(f"📊 Sentinel-2 Fetcher")
    print(f"   BBox       : {west},{south} → {east},{north}")
    print(f"   Date range : {args.start} → {args.end}")
    print(f"   Max cloud  : {args.max_cloud}%")
    print(f"   Limit      : {args.limit} scene(s)")
    print(f"   Raw output : {RAW_DIR}")
    print(f"   Chips to   : {UNLABELED_DIR}")
    print()

    scenes, chips = run_fetch(
        bbox=(west, south, east, north),
        start=args.start,
        end=args.end,
        max_cloud=args.max_cloud,
        limit=args.limit,
    )

    print()
    print(f"📊 Summary:")
    print(f"   Scenes downloaded : {scenes}")
    print(f"   Chips created     : {chips}")
    if scenes > 0:
        print(f"✅ Done. Chips are in {UNLABELED_DIR}")
    else:
        print("⚠️  No scenes were downloaded.")


if __name__ == "__main__":
    main()
