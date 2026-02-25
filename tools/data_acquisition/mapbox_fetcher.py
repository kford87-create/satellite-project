"""
tools/data_acquisition/mapbox_fetcher.py

Fetches high-resolution satellite tiles from the Mapbox Static Images API.
Enables an "address -> detection" workflow so customers don't need their own imagery.

Mapbox free tier: 200k static image requests/month.

Usage:
    from tools.data_acquisition.mapbox_fetcher import fetch_satellite_tile

    # By coordinates
    img = fetch_satellite_tile(lat=38.8977, lng=-77.0365, zoom=18)

    # By address (geocodes automatically)
    img = fetch_satellite_tile(address="1600 Pennsylvania Ave, Washington DC")
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import numpy as np

from dotenv import load_dotenv

ENV_PATH = Path("/Users/kahlil/satellite-project/.env")
load_dotenv(ENV_PATH)

MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN", "")
DEFAULT_ZOOM = 18          # ~0.6m/pixel — good for building/vehicle detection
DEFAULT_SIZE = "640x640"   # Matches YOLO input size
MAPBOX_STYLE = "mapbox/satellite-v9"


class MapboxError(Exception):
    """Raised when a Mapbox API call fails."""
    pass


def _get_token() -> str:
    """Return the Mapbox access token, raising a clear error if missing."""
    token = MAPBOX_ACCESS_TOKEN or os.getenv("MAPBOX_ACCESS_TOKEN", "")
    if not token:
        raise MapboxError(
            "MAPBOX_ACCESS_TOKEN not set. "
            "Get a free token at https://account.mapbox.com/access-tokens/ "
            "and add it to your .env file."
        )
    return token


def geocode_address(address: str) -> tuple[float, float]:
    """
    Geocode an address string to (lat, lng) using the Mapbox Geocoding API.

    Returns:
        Tuple of (latitude, longitude).

    Raises:
        MapboxError: If geocoding fails or returns no results.
    """
    token = _get_token()
    url = "https://api.mapbox.com/search/geocode/v6/forward"
    params = {
        "q": address,
        "access_token": token,
        "limit": 1,
    }

    with httpx.Client(timeout=15.0) as client:
        resp = client.get(url, params=params)

    if resp.status_code != 200:
        raise MapboxError(f"Geocoding failed (HTTP {resp.status_code}): {resp.text[:200]}")

    data = resp.json()
    features = data.get("features", [])
    if not features:
        raise MapboxError(f"No results found for address: {address!r}")

    coords = features[0]["geometry"]["coordinates"]  # [lng, lat]
    return coords[1], coords[0]


def fetch_satellite_tile(
    *,
    lat: float | None = None,
    lng: float | None = None,
    address: str | None = None,
    zoom: int = DEFAULT_ZOOM,
    size: str = DEFAULT_SIZE,
) -> np.ndarray:
    """
    Fetch a satellite image tile from Mapbox.

    Provide either (lat, lng) or address. If address is given, it will be
    geocoded first.

    Args:
        lat: Latitude (-90 to 90).
        lng: Longitude (-180 to 180).
        address: Street address to geocode.
        zoom: Map zoom level (0-22). 18 is ~0.6m/pixel.
        size: Image size as "WxH" string. Default "640x640".

    Returns:
        numpy array (H, W, 3) in BGR format (OpenCV-compatible, ready for YOLO).

    Raises:
        MapboxError: If the API call fails.
        ValueError: If neither coordinates nor address provided.
    """
    import cv2

    if address and (lat is None or lng is None):
        lat, lng = geocode_address(address)

    if lat is None or lng is None:
        raise ValueError("Provide either (lat, lng) or address.")

    token = _get_token()

    # Mapbox Static Images API
    url = (
        f"https://api.mapbox.com/styles/v1/{MAPBOX_STYLE}/static/"
        f"{lng},{lat},{zoom},0/{size}@2x"
        f"?access_token={token}"
        f"&attribution=false&logo=false"
    )

    with httpx.Client(timeout=30.0) as client:
        resp = client.get(url)

    if resp.status_code != 200:
        raise MapboxError(f"Satellite tile fetch failed (HTTP {resp.status_code}): {resp.text[:200]}")

    # Decode image bytes to numpy array
    img_array = np.frombuffer(resp.content, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise MapboxError("Failed to decode satellite tile image.")

    return img


async def fetch_satellite_tile_async(
    *,
    lat: float | None = None,
    lng: float | None = None,
    address: str | None = None,
    zoom: int = DEFAULT_ZOOM,
    size: str = DEFAULT_SIZE,
) -> np.ndarray:
    """
    Async version of fetch_satellite_tile for use in FastAPI endpoints.

    Same interface as fetch_satellite_tile but uses async HTTP client.
    """
    import cv2

    token = _get_token()

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Geocode if needed
        if address and (lat is None or lng is None):
            url = "https://api.mapbox.com/search/geocode/v6/forward"
            params = {"q": address, "access_token": token, "limit": 1}
            resp = await client.get(url, params=params)

            if resp.status_code != 200:
                raise MapboxError(f"Geocoding failed (HTTP {resp.status_code}): {resp.text[:200]}")

            data = resp.json()
            features = data.get("features", [])
            if not features:
                raise MapboxError(f"No results found for address: {address!r}")

            coords = features[0]["geometry"]["coordinates"]
            lat, lng = coords[1], coords[0]

        if lat is None or lng is None:
            raise ValueError("Provide either (lat, lng) or address.")

        # Fetch satellite tile
        tile_url = (
            f"https://api.mapbox.com/styles/v1/{MAPBOX_STYLE}/static/"
            f"{lng},{lat},{zoom},0/{size}@2x"
            f"?access_token={token}"
            f"&attribution=false&logo=false"
        )

        resp = await client.get(tile_url)

    if resp.status_code != 200:
        raise MapboxError(f"Satellite tile fetch failed (HTTP {resp.status_code}): {resp.text[:200]}")

    img_array = np.frombuffer(resp.content, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise MapboxError("Failed to decode satellite tile image.")

    return img
