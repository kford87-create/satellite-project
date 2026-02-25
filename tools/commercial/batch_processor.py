"""
batch_processor.py
------------------
Submit, process, and monitor batch YOLOv8 inference jobs over customer
image portfolios.

CLI:
    # Submit a job
    python tools/commercial/batch_processor.py submit \\
      --images-dir data/customer_portfolio --client-id abc123

    # Run the worker (polls for queued jobs)
    python tools/commercial/batch_processor.py worker

    # Check job status
    python tools/commercial/batch_processor.py status --job-id job_abc123
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
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
# Config
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://obdsgqjkjjmmtbcfjhnn.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

MODEL_PATH_DEFAULT = os.environ.get(
    "MODEL_PATH", "models/baseline_v1/weights/best.pt"
)

BATCH_JOBS_DIR    = Path("data/batch_jobs")
BATCH_RESULTS_DIR = Path("data/batch_results")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Estimated seconds per image (used for ETA in submit)
SECONDS_PER_IMAGE_ESTIMATE = 2.0
WORKER_POLL_INTERVAL = 10  # seconds

# ---------------------------------------------------------------------------
# Supabase (non-fatal)
# ---------------------------------------------------------------------------
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
# Local job store helpers
# ---------------------------------------------------------------------------

def _local_job_path(job_id: str) -> Path:
    return BATCH_JOBS_DIR / f"{job_id}.json"


def _save_job_local(job: dict) -> Path:
    BATCH_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    path = _local_job_path(job["job_id"])
    path.write_text(json.dumps(job, indent=2))
    return path


def _load_job_local(job_id: str) -> dict | None:
    path = _local_job_path(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"⚠️  Could not parse local job file {path}: {exc}")
        return None


def _update_job_local(job_id: str, updates: dict) -> None:
    job = _load_job_local(job_id)
    if job is None:
        return
    job.update(updates)
    _save_job_local(job)


def _list_queued_jobs_local() -> list[dict]:
    if not BATCH_JOBS_DIR.exists():
        return []
    jobs: list[dict] = []
    for path in sorted(BATCH_JOBS_DIR.glob("*.json")):
        try:
            job = json.loads(path.read_text())
            if job.get("status") == "queued":
                jobs.append(job)
        except Exception:
            continue
    return jobs


# ---------------------------------------------------------------------------
# Supabase job store helpers (non-fatal)
# ---------------------------------------------------------------------------

def _supabase_insert_job(job: dict) -> bool:
    try:
        if supabase is None:
            return False
        supabase.table("batch_jobs").insert(
            {
                "job_id":      job["job_id"],
                "client_id":   job["client_id"],
                "status":      job["status"],
                "image_count": job["image_count"],
                "created_at":  job["created_at"],
                "image_paths": json.dumps(job["image_paths"]),
            }
        ).execute()
        return True
    except Exception as exc:
        print(f"⚠️  Supabase insert skipped (falling back to local): {exc}")
        return False


def _supabase_update_job(job_id: str, updates: dict) -> None:
    try:
        if supabase is None:
            return
        supabase.table("batch_jobs").update(updates).eq("job_id", job_id).execute()
    except Exception as exc:
        print(f"⚠️  Supabase update skipped (falling back to local): {exc}")


def _supabase_get_job(job_id: str) -> dict | None:
    try:
        if supabase is None:
            return None
        resp = supabase.table("batch_jobs").select("*").eq("job_id", job_id).execute()
        rows = resp.data
        if rows:
            row = rows[0]
            if isinstance(row.get("image_paths"), str):
                try:
                    row["image_paths"] = json.loads(row["image_paths"])
                except Exception:
                    pass
            return row
        return None
    except Exception as exc:
        print(f"⚠️  Supabase query failed: {exc}")
        return None


def _supabase_list_queued_jobs() -> list[dict]:
    try:
        if supabase is None:
            return []
        resp = (
            supabase.table("batch_jobs")
            .select("*")
            .eq("status", "queued")
            .execute()
        )
        rows = resp.data or []
        for row in rows:
            if isinstance(row.get("image_paths"), str):
                try:
                    row["image_paths"] = json.loads(row["image_paths"])
                except Exception:
                    pass
        return rows
    except Exception as exc:
        print(f"⚠️  Supabase query failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Unified job helpers
# ---------------------------------------------------------------------------

def _update_job(job_id: str, updates: dict) -> None:
    """Update job in both Supabase and local store."""
    _supabase_update_job(job_id, updates)
    _update_job_local(job_id, updates)


def _get_job(job_id: str) -> dict | None:
    """Fetch job from Supabase first, fall back to local."""
    job = _supabase_get_job(job_id)
    if job is None:
        job = _load_job_local(job_id)
    return job


def _list_queued_jobs() -> list[dict]:
    """Return queued jobs from Supabase or local store."""
    jobs = _supabase_list_queued_jobs()
    if not jobs:
        jobs = _list_queued_jobs_local()
    return jobs


# ---------------------------------------------------------------------------
# SUBMIT subcommand
# ---------------------------------------------------------------------------

def cmd_submit(args: argparse.Namespace) -> None:
    images_dir: Path = Path(args.images_dir).resolve()
    client_id: str   = args.client_id

    if not images_dir.is_dir():
        print(f"❌ Images directory not found: {images_dir}")
        sys.exit(1)

    image_paths = sorted(
        str(p) for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"❌ No image files found in {images_dir}")
        sys.exit(1)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job_id = f"job_{client_id}_{timestamp}"

    job: dict[str, Any] = {
        "job_id":       job_id,
        "client_id":    client_id,
        "status":       "queued",
        "image_count":  len(image_paths),
        "image_paths":  image_paths,
        "created_at":   datetime.now(timezone.utc).isoformat(),
        "started_at":   None,
        "completed_at": None,
        "processed":    0,
        "result_path":  None,
        "error":        None,
    }

    # Try Supabase first; fall back to local
    saved_to_supabase = _supabase_insert_job(job)
    local_path = _save_job_local(job)

    if saved_to_supabase:
        print(f"✅ Job inserted into Supabase and saved locally → {local_path}")
    else:
        print(f"✅ Job saved locally → {local_path}")

    eta_seconds = len(image_paths) * SECONDS_PER_IMAGE_ESTIMATE
    eta_minutes = eta_seconds / 60.0

    print("\n" + "=" * 50)
    print("📊 Job Submitted")
    print("=" * 50)
    print(f"  Job ID:           {job_id}")
    print(f"  Client ID:        {client_id}")
    print(f"  Image count:      {len(image_paths)}")
    print(f"  Estimated time:   {eta_minutes:.1f} min ({eta_seconds:.0f}s)")
    print(f"  Status:           queued")
    print("=" * 50)
    print(f"\nRun worker:  python tools/commercial/batch_processor.py worker")
    print(f"Check status: python tools/commercial/batch_processor.py status --job-id {job_id}")


# ---------------------------------------------------------------------------
# WORKER subcommand
# ---------------------------------------------------------------------------

def _run_inference_on_image(model: Any, image_path: str) -> list[dict]:
    """Run YOLOv8 on a single image path. Returns list of detection dicts."""
    try:
        results = model(image_path, verbose=False)
    except Exception as exc:
        print(f"⚠️  Inference failed for {Path(image_path).name}: {exc}")
        return []

    detections: list[dict] = []
    for r in results:
        if r.boxes is None:
            continue
        names = r.names
        for box_data in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in box_data.xyxy[0].tolist()]
            conf = float(box_data.conf[0])
            cls_id = int(box_data.cls[0])
            class_name = names[cls_id] if names and cls_id in names else str(cls_id)
            w_img = float(r.orig_shape[1]) if hasattr(r, "orig_shape") else 640.0
            h_img = float(r.orig_shape[0]) if hasattr(r, "orig_shape") else 640.0
            cx = ((x1 + x2) / 2.0) / w_img
            cy = ((y1 + y2) / 2.0) / h_img
            bw = (x2 - x1) / w_img
            bh = (y2 - y1) / h_img
            detections.append(
                {
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                    "bbox": {
                        "x_center": round(cx, 6),
                        "y_center": round(cy, 6),
                        "width":    round(bw, 6),
                        "height":   round(bh, 6),
                    },
                }
            )
    return detections


def _process_job(job: dict, model: Any) -> None:
    """Process a single job: run inference, save results, update status."""
    job_id = job["job_id"]
    image_paths: list[str] = job.get("image_paths", [])

    print(f"\n📊 Processing job {job_id} ({len(image_paths)} images)")

    started_at = datetime.now(timezone.utc).isoformat()
    _update_job(job_id, {"status": "processing", "started_at": started_at})

    result_dir = BATCH_RESULTS_DIR / job_id
    result_dir.mkdir(parents=True, exist_ok=True)

    all_detections: list[dict] = []
    total_detections = 0

    for img_path_str in tqdm(
        image_paths,
        desc=f"Job {job_id}",
        unit="img",
    ):
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"⚠️  Image not found, skipping: {img_path}")
            continue

        dets = _run_inference_on_image(model, img_path_str)
        total_detections += len(dets)
        all_detections.append(
            {
                "filename": img_path.name,
                "detections": dets,
            }
        )

        # Update progress in both stores
        processed = len(all_detections)
        _update_job(job_id, {"processed": processed})

    # Save detections JSON
    detections_path = result_dir / "detections.json"
    detections_path.write_text(json.dumps(all_detections, indent=2))
    print(f"✅ Detections saved → {detections_path}")

    # Attempt GeoJSON export if chip_index available
    chip_index_candidates = [
        Path("data/unlabeled") / "scene_chip_index.json",
        Path("data/chip_index.json"),
    ]
    chip_index_path: Path | None = None
    for candidate in chip_index_candidates:
        if candidate.exists():
            chip_index_path = candidate
            break

    if chip_index_path:
        try:
            from tools.commercial.geojson_exporter import export_geojson  # type: ignore

            geojson_path = result_dir / "detections.geojson"
            export_geojson(
                detections_path=detections_path,
                chip_index_path=chip_index_path,
                output_path=geojson_path,
            )
        except Exception as exc:
            print(f"⚠️  GeoJSON export failed (non-fatal): {exc}")
    else:
        print("⚠️  No chip_index found – skipping GeoJSON export")

    completed_at = datetime.now(timezone.utc).isoformat()
    _update_job(
        job_id,
        {
            "status":       "completed",
            "completed_at": completed_at,
            "result_path":  str(detections_path),
            "processed":    len(image_paths),
        },
    )

    print(f"\n✅ Job {job_id} completed")
    print("=" * 50)
    print("📊 Completion Stats")
    print("=" * 50)
    print(f"  Images processed: {len(all_detections)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Results path:     {result_dir}")
    print("=" * 50)


def cmd_worker(args: argparse.Namespace) -> None:
    model_path = Path(
        getattr(args, "model", None) or MODEL_PATH_DEFAULT
    ).resolve()

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print(f"📊 Loading model: {model_path}")
    try:
        model = YOLO(str(model_path))
    except Exception as exc:
        print(f"❌ Failed to load model: {exc}")
        sys.exit(1)

    print(f"✅ Worker started – polling every {WORKER_POLL_INTERVAL}s for queued jobs")
    print("   Press Ctrl+C to stop.\n")

    try:
        while True:
            queued = _list_queued_jobs()
            if queued:
                for job in queued:
                    _process_job(job, model)
            else:
                print(
                    f"📊 No queued jobs found. Sleeping {WORKER_POLL_INTERVAL}s…",
                    end="\r",
                    flush=True,
                )
                time.sleep(WORKER_POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\n⚠️  Worker stopped by user")


# ---------------------------------------------------------------------------
# STATUS subcommand
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> None:
    job_id: str = args.job_id

    job = _get_job(job_id)
    if job is None:
        print(f"❌ Job not found: {job_id}")
        sys.exit(1)

    status    = job.get("status", "unknown")
    processed = job.get("processed", 0)
    total     = job.get("image_count", 0)
    created   = job.get("created_at", "N/A")
    started   = job.get("started_at", "N/A")
    completed = job.get("completed_at")
    result    = job.get("result_path")
    error     = job.get("error")

    # Elapsed time
    elapsed_str = "N/A"
    if started and started != "N/A":
        try:
            t_start = datetime.fromisoformat(started)
            t_end = (
                datetime.fromisoformat(completed)
                if completed
                else datetime.now(timezone.utc)
            )
            delta = t_end - t_start
            elapsed_str = f"{int(delta.total_seconds())}s"
        except Exception:
            pass

    print("\n" + "=" * 50)
    print("📊 Job Status")
    print("=" * 50)
    print(f"  Job ID:      {job_id}")
    print(f"  Status:      {status}")
    print(f"  Progress:    {processed}/{total} images")
    print(f"  Created at:  {created}")
    print(f"  Started at:  {started}")
    print(f"  Elapsed:     {elapsed_str}")
    if completed:
        print(f"  Completed at: {completed}")
    if result:
        print(f"  Result path: {result}")
    if error:
        print(f"  ❌ Error:    {error}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch YOLOv8 inference job manager.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- submit ----
    submit_p = sub.add_parser(
        "submit",
        help="Submit a new batch inference job",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    submit_p.add_argument(
        "--images-dir",
        required=True,
        metavar="DIR",
        help="Directory containing .jpg/.png images to process",
    )
    submit_p.add_argument(
        "--client-id",
        required=True,
        metavar="ID",
        help="Client identifier (used in job_id)",
    )

    # ---- worker ----
    worker_p = sub.add_parser(
        "worker",
        help="Run the worker process (polls for queued jobs)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    worker_p.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH_DEFAULT,
        metavar="PATH",
        help="Path to YOLOv8 .pt weights file",
    )

    # ---- status ----
    status_p = sub.add_parser(
        "status",
        help="Check status of a job",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    status_p.add_argument(
        "--job-id",
        required=True,
        metavar="ID",
        help="Job ID to query",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "submit":
        cmd_submit(args)
    elif args.command == "worker":
        cmd_worker(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
