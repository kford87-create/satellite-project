"""
scripts/inference_server.py

Lightweight FastAPI inference server.
This is what the Supabase Edge Function calls to run actual ML inference.

Deploy this on:
  - Hugging Face Spaces (free tier) — best for demo
  - RunPod serverless (~$0.00015/inference) — best for production
  - Any VPS or cloud VM

The Edge Function acts as the front door (auth, rate limiting, logging).
This server is the back room where the actual ML work happens.

Start locally:
  uvicorn scripts.inference_server:app --host 0.0.0.0 --port 8000

Test:
  curl -X POST http://localhost:8000/detect \
    -H "X-API-Key: your-key" \
    -H "Content-Type: application/json" \
    -d '{"image_url": "https://example.com/satellite.jpg"}'
"""

import os
import time
import base64
import hashlib
import hmac
import ipaddress
import httpx
import numpy as np
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import cv2
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "./models/multiclass_v2/weights/best.pt")
INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY", "change-this-key")
MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN", "")
CLASS_NAMES = ["building", "vehicle", "aircraft", "ship"]  # Update as classes expand
MAX_IMAGE_SIZE = 1280   # Max image dimension for inference

# Quality gate thresholds — reject images too poor for reliable detection
QUALITY_CONTRAST_MIN = 15.0    # Minimum grayscale std dev
QUALITY_NODATA_MAX = 0.05      # Maximum fraction of black/no-data pixels

app = FastAPI(
    title="Satellite Object Detection API",
    description="YOLOv8-based object detection for EO satellite imagery",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Model Loading ─────────────────────────────────────────────────────────────
# Load model once at startup — not on every request
# Like hiring a specialist who's always on-call rather than hiring a new one per job

_model: Optional[YOLO] = None

def get_model() -> YOLO:
    global _model
    if _model is None:
        if not Path(MODEL_PATH).exists():
            raise RuntimeError(
                f"Model not found at {MODEL_PATH}. "
                "Run scripts/04_train_baseline.py first, then set MODEL_PATH in .env"
            )
        print(f"🔄 Loading model from {MODEL_PATH}...")
        _model = YOLO(MODEL_PATH)
        print("✅ Model loaded and ready")
    return _model


# ── Request/Response Models ───────────────────────────────────────────────────

class DetectRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    confidence_threshold: float = 0.25
    classes: Optional[List[str]] = None
    return_image: bool = False
    use_tta: bool = False


class AddressDetectRequest(BaseModel):
    address: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    zoom: int = 18
    confidence_threshold: float = 0.25
    classes: Optional[List[str]] = None
    return_image: bool = False
    use_tta: bool = False


class BBox(BaseModel):
    x_center: float
    y_center: float
    width: float
    height: float


class BBoxPixels(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class Detection(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    bbox: BBox
    bbox_pixels: BBoxPixels


class DetectResponse(BaseModel):
    success: bool
    model_version: str
    processing_time_ms: float
    image_width: int
    image_height: int
    detections: List[Detection]
    annotated_image_base64: Optional[str] = None


# ── Auth ──────────────────────────────────────────────────────────────────────

async def verify_api_key(x_api_key: str = Header(...)):
    """Simple API key auth. The Edge Function passes this key server-to-server.
    Uses constant-time comparison to prevent timing attacks.
    """
    # hmac.compare_digest prevents timing attacks by always taking the same amount
    # of time regardless of where the strings first differ
    if not hmac.compare_digest(x_api_key, INFERENCE_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# ── Image Loading ─────────────────────────────────────────────────────────────

_PRIVATE_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),  # link-local
    ipaddress.ip_network("::1/128"),          # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),         # IPv6 private
]


def validate_image_url(url: str) -> None:
    """
    Guard against SSRF attacks by rejecting URLs that point to private/internal
    network addresses. Think of it as a bouncer who checks IDs at the door and
    refuses entry to anyone who looks like an insider trying to sneak in.
    Raises HTTPException 400 if the URL is invalid or targets a private address.
    """
    from urllib.parse import urlparse
    import socket

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(400, f"Image URL must use http or https, got: {parsed.scheme!r}")

    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(400, "Image URL has no hostname")

    try:
        resolved_ip = socket.gethostbyname(hostname)
        ip = ipaddress.ip_address(resolved_ip)
    except (socket.gaierror, ValueError) as exc:
        raise HTTPException(400, f"Cannot resolve image URL hostname: {exc}")

    for private_range in _PRIVATE_RANGES:
        if ip in private_range:
            raise HTTPException(400, "Image URL resolves to a private/internal address")


async def load_image(request: DetectRequest) -> np.ndarray:
    """Load image from URL or base64 string into numpy array."""
    if request.image_url:
        validate_image_url(request.image_url)  # SSRF guard
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(request.image_url)
            if response.status_code != 200:
                raise HTTPException(400, f"Failed to fetch image: {response.status_code}")
            img_bytes = response.content

    elif request.image_base64:
        # Strip data URI prefix if present (e.g., "data:image/jpeg;base64,...")
        b64_data = request.image_base64
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_data)

    else:
        raise HTTPException(400, "Provide either image_url or image_base64")

    # Decode to numpy array
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(400, "Could not decode image — check format (JPG/PNG supported)")

    return img


def resize_if_needed(img: np.ndarray, max_dim: int = MAX_IMAGE_SIZE) -> np.ndarray:
    """Resize large images to prevent OOM on inference server."""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return img


def annotate_image(img: np.ndarray, detections: List[Detection]) -> str:
    """Draw bounding boxes on image and return as base64."""
    CLASS_COLORS = {
        "building": (0, 255, 0),    # Green
        "vehicle": (255, 165, 0),   # Orange
        "aircraft": (0, 165, 255),  # Blue
        "ship": (255, 0, 255),      # Magenta
    }

    annotated = img.copy()
    for det in detections:
        color = CLASS_COLORS.get(det.class_name, (255, 255, 255))
        x1, y1, x2, y2 = det.bbox_pixels.x1, det.bbox_pixels.y1, det.bbox_pixels.x2, det.bbox_pixels.y2

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


# ── Detection Parsing ─────────────────────────────────────────────────────────

def _parse_detections(results) -> List[Detection]:
    """Parse YOLO results into Detection objects."""
    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x_center, y_center, bw, bh = box.xywhn.tolist()[0]
            x1, y1, x2, y2 = box.xyxy.tolist()[0]

            detections.append(Detection(
                class_name=CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}",
                class_id=class_id,
                confidence=round(confidence, 4),
                bbox=BBox(
                    x_center=round(x_center, 6),
                    y_center=round(y_center, 6),
                    width=round(bw, 6),
                    height=round(bh, 6),
                ),
                bbox_pixels=BBoxPixels(
                    x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2)
                )
            ))
    return detections


# ── Quality Gate ──────────────────────────────────────────────────────────────

def check_image_quality(img: np.ndarray) -> None:
    """
    Reject images too poor for reliable detection.
    Checks contrast and no-data fraction before wasting inference cycles.
    Raises HTTPException with a clear message if quality is too low.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast: standard deviation of grayscale values
    contrast = float(np.std(gray.astype(np.float32)))
    if contrast < QUALITY_CONTRAST_MIN:
        raise HTTPException(
            422,
            f"Image quality too low for reliable detection. "
            f"Contrast score {contrast:.1f} is below minimum {QUALITY_CONTRAST_MIN}. "
            f"The image may be too dark, overexposed, or featureless."
        )

    # No-data: fraction of near-black pixels (all channels < 5)
    nodata_mask = (img[:, :, 0] < 5) & (img[:, :, 1] < 5) & (img[:, :, 2] < 5)
    nodata_frac = float(nodata_mask.sum()) / float(nodata_mask.size)
    if nodata_frac > QUALITY_NODATA_MAX:
        raise HTTPException(
            422,
            f"Image has too much missing data ({nodata_frac:.0%} black pixels). "
            f"Maximum allowed is {QUALITY_NODATA_MAX:.0%}. "
            f"Try a different image or location."
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "message": "Satellite Object Detection API"}
@app.get("/health")
async def health():
    """Health check endpoint."""
    model_loaded = _model is not None
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH,
        "classes": CLASS_NAMES,
    }


@app.post("/detect", response_model=DetectResponse)
async def detect(
    request: DetectRequest,
    _api_key: str = Depends(verify_api_key)
):
    """
    Run object detection on a satellite image.
    Called by the Supabase Edge Function — not directly by end users.
    """
    start_time = time.time()
    model = get_model()

    # Load and preprocess image
    img = await load_image(request)
    img = resize_if_needed(img)

    # Quality gate — reject bad imagery before wasting inference cycles
    check_image_quality(img)

    h, w = img.shape[:2]

    # Filter class IDs if requested
    class_filter = None
    if request.classes:
        unknown = [c for c in request.classes if c not in CLASS_NAMES]
        if unknown:
            raise HTTPException(
                400,
                f"Unknown class(es): {unknown}. Valid classes: {CLASS_NAMES}"
            )
        class_filter = [CLASS_NAMES.index(c) for c in request.classes]
        # Note: an empty list here would tell YOLO to detect nothing — the check
        # above ensures we only reach this point with at least one valid class.

    # Run inference (with optional TTA for higher accuracy at ~3x speed cost)
    results = model.predict(
        source=img,
        conf=request.confidence_threshold,
        classes=class_filter,
        augment=request.use_tta,
        verbose=False
    )[0]

    # Parse detections
    detections = _parse_detections(results)

    processing_ms = (time.time() - start_time) * 1000

    model_version = "yolov8s-satellite-multiclass-v2"
    if request.use_tta:
        model_version += "+tta"

    response = DetectResponse(
        success=True,
        model_version=model_version,
        processing_time_ms=round(processing_ms, 2),
        image_width=w,
        image_height=h,
        detections=detections,
    )

    if request.return_image:
        response.annotated_image_base64 = annotate_image(img, detections)

    return response


@app.post("/detect-address", response_model=DetectResponse)
async def detect_address(
    request: AddressDetectRequest,
    _api_key: str = Depends(verify_api_key)
):
    """
    Detect objects at a street address or coordinates.
    Fetches a satellite tile from Mapbox, then runs detection.
    No need for customers to supply their own imagery.
    """
    from tools.data_acquisition.mapbox_fetcher import (
        fetch_satellite_tile_async, MapboxError
    )

    if not MAPBOX_ACCESS_TOKEN:
        raise HTTPException(
            503,
            "Address-based detection is not configured. "
            "Set the MAPBOX_ACCESS_TOKEN environment variable to enable this feature."
        )

    if not request.address and (request.lat is None or request.lng is None):
        raise HTTPException(400, "Provide either an address or both lat and lng.")

    start_time = time.time()
    model = get_model()

    # Fetch satellite tile
    try:
        img = await fetch_satellite_tile_async(
            address=request.address,
            lat=request.lat,
            lng=request.lng,
            zoom=request.zoom,
        )
    except MapboxError as e:
        raise HTTPException(502, f"Could not fetch satellite imagery: {e}")

    img = resize_if_needed(img)
    check_image_quality(img)
    h, w = img.shape[:2]

    # Filter class IDs if requested
    class_filter = None
    if request.classes:
        unknown = [c for c in request.classes if c not in CLASS_NAMES]
        if unknown:
            raise HTTPException(
                400,
                f"Unknown class(es): {unknown}. Valid classes: {CLASS_NAMES}"
            )
        class_filter = [CLASS_NAMES.index(c) for c in request.classes]

    results = model.predict(
        source=img,
        conf=request.confidence_threshold,
        classes=class_filter,
        augment=request.use_tta,
        verbose=False
    )[0]

    detections = _parse_detections(results)
    processing_ms = (time.time() - start_time) * 1000

    model_version = "yolov8s-satellite-multiclass-v2"
    if request.use_tta:
        model_version += "+tta"

    response = DetectResponse(
        success=True,
        model_version=model_version,
        processing_time_ms=round(processing_ms, 2),
        image_width=w,
        image_height=h,
        detections=detections,
    )

    if request.return_image:
        response.annotated_image_base64 = annotate_image(img, detections)

    return response


@app.get("/model-info")
async def model_info(_api_key: str = Depends(verify_api_key)):
    """Return model metadata."""
    return {
        "model_path": MODEL_PATH,
        "classes": CLASS_NAMES,
        "n_classes": len(CLASS_NAMES),
        "input_size": MAX_IMAGE_SIZE,
        "framework": "YOLOv8",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
