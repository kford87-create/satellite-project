"""
tools/commercial/batch_processor.py

Accepts batch jobs for processing large property portfolios overnight.
Insurance companies need to process thousands of images — not one at a time.
Uses Supabase as the job queue. Processes async, notifies on completion.

Usage:
  # Submit a batch job
  python tools/commercial/batch_processor.py submit \
    --images-dir data/customer_portfolio \
    --client-id abc123 \
    --job-name "Q4 Portfolio Review"

  # Run the worker (processes queued jobs)
  python tools/commercial/batch_processor.py worker

  # Check job status
  python tools/commercial/batch_processor.py status --job-id JOB_ID
"""

import os
import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
import argparse

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")


class BatchProcessor:
    """
    Manages batch detection jobs via Supabase queue.
    Worker polls for pending jobs and processes them sequentially.
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path or str(MODELS_DIR / "baseline_v1/weights/best.pt")
        self._model = None
        self._db = None

    def _get_db(self):
        if self._db is None:
            from supabase import create_client
            self._db = create_client(SUPABASE_URL, SUPABASE_KEY)
        return self._db

    def _get_model(self):
        if self._model is None:
            from ultralytics import YOLO
            print(f"🔄 Loading model: {self.model_path}")
            self._model = YOLO(self.model_path)
            print("✅ Model ready")
        return self._model

    # ── Job Submission ──────────────────────────────────────────────────────

    def submit_job(
        self,
        image_paths: List[Path],
        client_id: str,
        job_name: str,
        conf_threshold: float = 0.25,
        return_geojson: bool = True,
        notify_email: str = None
    ) -> str:
        """Submit a batch job to the queue. Returns job ID."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = {
            "job_id": job_id,
            "client_id": client_id,
            "job_name": job_name,
            "status": "pending",
            "n_images": len(image_paths),
            "n_processed": 0,
            "conf_threshold": conf_threshold,
            "return_geojson": return_geojson,
            "notify_email": notify_email,
            "image_paths": [str(p) for p in image_paths],
            "created_at": datetime.utcnow().isoformat() + "Z",
            "started_at": None,
            "completed_at": None,
            "output_path": None,
            "error": None,
        }

        # Save locally as fallback
        jobs_dir = DATA_DIR / "batch_jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        (jobs_dir / f"{job_id}.json").write_text(json.dumps(job, indent=2))

        # Also push to Supabase if available
        try:
            db = self._get_db()
            db.table("batch_jobs").insert({
                "job_id": job_id,
                "client_id": client_id,
                "job_name": job_name,
                "status": "pending",
                "n_images": len(image_paths),
                "config": json.dumps({"conf_threshold": conf_threshold, "return_geojson": return_geojson}),
                "image_paths": json.dumps([str(p) for p in image_paths]),
            }).execute()
            print(f"✅ Job submitted to Supabase queue: {job_id}")
        except Exception as e:
            print(f"⚠️  Supabase unavailable — job saved locally: {e}")

        print(f"\n📋 Job Created:")
        print(f"   ID:     {job_id}")
        print(f"   Name:   {job_name}")
        print(f"   Images: {len(image_paths)}")
        return job_id

    # ── Job Processing ──────────────────────────────────────────────────────

    def process_job(self, job: Dict) -> Dict:
        """Process a single batch job end-to-end."""
        from tqdm import tqdm
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from geojson_exporter import GeoJSONExporter

        job_id = job["job_id"]
        image_paths = [Path(p) for p in job["image_paths"]]
        conf = job.get("conf_threshold", 0.25)

        print(f"\n🚀 Processing job: {job_id} ({len(image_paths)} images)")
        model = self._get_model()
        exporter = GeoJSONExporter()

        all_detections = []
        n_processed = 0
        errors = []

        for img_path in tqdm(image_paths, desc="Processing"):
            if not img_path.exists():
                errors.append(f"Not found: {img_path}")
                continue
            try:
                results = model.predict(source=str(img_path), conf=conf, verbose=False)[0]
                if results.boxes is not None:
                    for box in results.boxes:
                        all_detections.append({
                            "chip_name": img_path.name,
                            "class_name": {0:"building",1:"vehicle",2:"aircraft",3:"ship"}.get(int(box.cls.item()), "unknown"),
                            "class_id": int(box.cls.item()),
                            "confidence": round(float(box.conf.item()), 4),
                            "bbox": box.xywhn.tolist()[0],
                        })
                n_processed += 1
            except Exception as e:
                errors.append(f"{img_path.name}: {e}")

        # Export
        output_dir = DATA_DIR / "batch_results" / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        detections_path = output_dir / "detections.json"
        detections_path.write_text(json.dumps(all_detections, indent=2))

        geojson_path = None
        if job.get("return_geojson", True):
            geojson_path = output_dir / "detections.geojson"
            exporter.export_detections(all_detections, geojson_path)

        summary = {
            "job_id": job_id,
            "status": "completed",
            "n_images": len(image_paths),
            "n_processed": n_processed,
            "n_detections": len(all_detections),
            "n_errors": len(errors),
            "errors": errors[:10],
            "output_dir": str(output_dir),
            "detections_path": str(detections_path),
            "geojson_path": str(geojson_path) if geojson_path else None,
            "completed_at": datetime.utcnow().isoformat() + "Z"
        }

        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        print(f"\n✅ Job complete: {job_id}")
        print(f"   Processed:   {n_processed}/{len(image_paths)}")
        print(f"   Detections:  {len(all_detections)}")
        print(f"   Errors:      {len(errors)}")
        print(f"   Output:      {output_dir}")
        return summary

    # ── Worker ──────────────────────────────────────────────────────────────

    def run_worker(self, poll_interval: int = 10):
        """Poll for pending jobs and process them. Run continuously."""
        print(f"\n👷 Batch Worker Started (polling every {poll_interval}s)")
        print("   Press Ctrl+C to stop\n")

        while True:
            job = self._get_next_pending_job()
            if job:
                self.process_job(job)
            else:
                print(f"   [{datetime.now().strftime('%H:%M:%S')}] No pending jobs — waiting...")
                time.sleep(poll_interval)

    def _get_next_pending_job(self) -> Optional[Dict]:
        """Get the next pending job from queue (local or Supabase)."""
        # Try Supabase first
        try:
            db = self._get_db()
            result = db.table("batch_jobs").select("*").eq("status", "pending").order("created_at").limit(1).execute()
            if result.data:
                row = result.data[0]
                return {
                    "job_id": row["job_id"],
                    "client_id": row["client_id"],
                    "job_name": row["job_name"],
                    "conf_threshold": json.loads(row.get("config", "{}")).get("conf_threshold", 0.25),
                    "return_geojson": json.loads(row.get("config", "{}")).get("return_geojson", True),
                    "image_paths": json.loads(row.get("image_paths", "[]")),
                }
        except Exception:
            pass

        # Fallback to local job files
        jobs_dir = DATA_DIR / "batch_jobs"
        if jobs_dir.exists():
            for job_file in sorted(jobs_dir.glob("*.json")):
                job = json.loads(job_file.read_text())
                if job.get("status") == "pending":
                    return job
        return None

    def get_status(self, job_id: str) -> Dict:
        """Get status of a specific job."""
        result_path = DATA_DIR / "batch_results" / job_id / "summary.json"
        if result_path.exists():
            return json.loads(result_path.read_text())
        job_path = DATA_DIR / "batch_jobs" / f"{job_id}.json"
        if job_path.exists():
            return json.loads(job_path.read_text())
        return {"job_id": job_id, "status": "not_found"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    submit = sub.add_parser("submit")
    submit.add_argument("--images-dir", required=True)
    submit.add_argument("--client-id", default="test_client")
    submit.add_argument("--job-name", default="Batch Job")
    submit.add_argument("--conf", type=float, default=0.25)

    worker = sub.add_parser("worker")
    worker.add_argument("--interval", type=int, default=10)
    worker.add_argument("--model", default=None)

    status = sub.add_parser("status")
    status.add_argument("--job-id", required=True)

    args = parser.parse_args()

    if args.command == "submit":
        images = list(Path(args.images_dir).glob("*.jpg")) + list(Path(args.images_dir).glob("*.png"))
        proc = BatchProcessor()
        proc.submit_job(images, args.client_id, args.job_name, args.conf)
    elif args.command == "worker":
        proc = BatchProcessor(args.model)
        proc.run_worker()
    elif args.command == "status":
        proc = BatchProcessor()
        status_data = proc.get_status(args.job_id)
        print(json.dumps(status_data, indent=2))
    else:
        parser.print_help()
