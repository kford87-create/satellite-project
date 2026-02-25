"""
scripts/supabase_storage.py

Central Supabase integration module.
Handles all storage, database, and metadata operations.

Bucket structure:
  satellite-images/
    ├── spacenet/train/images/     # Seed labeled images
    ├── spacenet/train/labels/     # YOLO label files
    ├── unlabeled/                 # Sentinel-2 unlabeled imagery
    └── bootstrapped/              # Active learning outputs

  satellite-models/
    └── baseline_v1/weights/       # Trained model weights

Database tables:
  - dataset_images     (tracks every image + metadata)
  - bootstrap_iterations (tracks each AL loop iteration)
  - fn_reports         (false negative quantification results)
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict
from supabase import create_client, Client
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ─── Supabase Config ──────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Storage bucket names
BUCKET_IMAGES = "satellite-images"
BUCKET_MODELS = "satellite-models"


def get_client() -> Client:
    """Initialize and return Supabase client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError(
            "Missing Supabase credentials.\n"
            "Set SUPABASE_URL and SUPABASE_KEY in your .env file.\n"
            "Get these from: https://app.supabase.com → Project Settings → API"
        )
    return create_client(SUPABASE_URL, SUPABASE_KEY)


class SupabaseStorageManager:
    """
    Manages all file storage and metadata operations via Supabase.
    Replaces local file system storage for images, labels, and reports.
    """

    def __init__(self):
        self.client = get_client()
        self._ensure_buckets()

    def _ensure_buckets(self):
        """Create storage buckets if they don't exist."""
        existing = [b.name for b in self.client.storage.list_buckets()]

        for bucket in [BUCKET_IMAGES, BUCKET_MODELS]:
            if bucket not in existing:
                self.client.storage.create_bucket(
                    bucket,
                    options={"public": False}  # Private by default
                )
                print(f"✅ Created bucket: {bucket}")
            else:
                print(f"✅ Bucket exists: {bucket}")

    # ─── Upload Methods ───────────────────────────────────────────────────────

    def upload_image(
        self,
        local_path: Path,
        remote_path: str,
        upsert: bool = False
    ) -> Optional[str]:
        """
        Upload a single image to Supabase Storage.

        Args:
            local_path: Local file path
            remote_path: Path in Supabase bucket (e.g., 'spacenet/train/images/img001.jpg')
            upsert: Overwrite if exists

        Returns:
            Public URL of uploaded file, or None on failure
        """
        try:
            with open(local_path, "rb") as f:
                self.client.storage.from_(BUCKET_IMAGES).upload(
                    path=remote_path,
                    file=f,
                    file_options={
                        "content-type": "image/jpeg",
                        "upsert": str(upsert).lower()
                    }
                )
            return remote_path

        except Exception as e:
            if "already exists" in str(e) and not upsert:
                return remote_path  # Already uploaded, skip
            print(f"⚠️  Upload failed for {local_path.name}: {e}")
            return None

    def upload_label(
        self,
        local_path: Path,
        remote_path: str,
        upsert: bool = False
    ) -> Optional[str]:
        """Upload a YOLO label .txt file to Supabase Storage."""
        try:
            with open(local_path, "rb") as f:
                self.client.storage.from_(BUCKET_IMAGES).upload(
                    path=remote_path,
                    file=f,
                    file_options={
                        "content-type": "text/plain",
                        "upsert": str(upsert).lower()
                    }
                )
            return remote_path

        except Exception as e:
            if "already exists" in str(e) and not upsert:
                return remote_path
            print(f"⚠️  Label upload failed for {local_path.name}: {e}")
            return None

    def upload_model_weights(self, local_path: Path, version: str = "baseline_v1") -> Optional[str]:
        """Upload trained model weights (.pt file) to Supabase."""
        remote_path = f"{version}/weights/{local_path.name}"
        try:
            with open(local_path, "rb") as f:
                self.client.storage.from_(BUCKET_MODELS).upload(
                    path=remote_path,
                    file=f,
                    file_options={
                        "content-type": "application/octet-stream",
                        "upsert": "true"
                    }
                )
            print(f"✅ Model weights uploaded: {remote_path}")
            return remote_path

        except Exception as e:
            print(f"❌ Model upload failed: {e}")
            return None

    def upload_dataset_batch(
        self,
        image_dir: Path,
        label_dir: Path,
        split: str = "train",
        prefix: str = "spacenet"
    ) -> Dict:
        """
        Batch upload an entire dataset split (images + labels) to Supabase.

        Args:
            image_dir: Local directory containing .jpg images
            label_dir: Local directory containing .txt YOLO labels
            split: 'train', 'val', or 'test'
            prefix: Bucket path prefix (e.g., 'spacenet', 'bootstrapped')

        Returns:
            Upload summary stats
        """
        images = list(image_dir.glob("*.jpg"))
        print(f"\n📤 Uploading {len(images)} {split} images to Supabase...")

        uploaded_images = 0
        uploaded_labels = 0
        failed = 0

        for img_path in tqdm(images, desc=f"Uploading {split}"):
            # Upload image
            remote_img = f"{prefix}/{split}/images/{img_path.name}"
            result = self.upload_image(img_path, remote_img)
            if result:
                uploaded_images += 1
            else:
                failed += 1

            # Upload corresponding label
            label_path = label_dir / (img_path.stem + ".txt")
            if label_path.exists():
                remote_lbl = f"{prefix}/{split}/labels/{label_path.name}"
                result = self.upload_label(label_path, remote_lbl)
                if result:
                    uploaded_labels += 1

            # Register in database
            self._register_image(
                stem=img_path.stem,
                split=split,
                prefix=prefix,
                image_path=remote_img,
                label_path=f"{prefix}/{split}/labels/{img_path.stem}.txt"
            )

        summary = {
            "uploaded_images": uploaded_images,
            "uploaded_labels": uploaded_labels,
            "failed": failed,
            "total": len(images)
        }

        print(f"\n✅ Upload complete:")
        print(f"   Images: {uploaded_images}/{len(images)}")
        print(f"   Labels: {uploaded_labels}/{len(images)}")
        if failed:
            print(f"   Failed: {failed}")

        return summary

    # ─── Download Methods ─────────────────────────────────────────────────────

    def download_image(self, remote_path: str, local_path: Path) -> bool:
        """Download a single image from Supabase Storage."""
        try:
            data = self.client.storage.from_(BUCKET_IMAGES).download(remote_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)
            return True
        except Exception as e:
            print(f"⚠️  Download failed for {remote_path}: {e}")
            return False

    def download_model_weights(self, version: str, local_dir: Path) -> Optional[Path]:
        """Download model weights from Supabase for inference/training."""
        remote_path = f"{version}/weights/best.pt"
        local_path = local_dir / version / "weights" / "best.pt"

        if self.download_image.__wrapped__ if hasattr(self.download_image, '__wrapped__') else True:
            try:
                data = self.client.storage.from_(BUCKET_MODELS).download(remote_path)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(data)
                print(f"✅ Downloaded model weights to: {local_path}")
                return local_path
            except Exception as e:
                print(f"❌ Model download failed: {e}")
                return None

    # ─── Database Methods ─────────────────────────────────────────────────────

    def _register_image(
        self,
        stem: str,
        split: str,
        prefix: str,
        image_path: str,
        label_path: str,
        is_labeled: bool = True,
        iteration: int = 0,
        uncertainty_score: float = None
    ):
        """Register an image in the dataset_images database table."""
        try:
            self.client.table("dataset_images").upsert({
                "stem": stem,
                "split": split,
                "prefix": prefix,
                "image_path": image_path,
                "label_path": label_path,
                "is_labeled": is_labeled,
                "bootstrap_iteration": iteration,
                "uncertainty_score": uncertainty_score,
            }).execute()
        except Exception as e:
            # Table may not exist yet - that's ok for first run
            pass

    def save_bootstrap_iteration(self, iteration_data: Dict):
        """Save bootstrapping iteration metrics to Supabase database."""
        try:
            self.client.table("bootstrap_iterations").insert(iteration_data).execute()
            print(f"✅ Iteration {iteration_data.get('iteration')} saved to database")
        except Exception as e:
            print(f"⚠️  Could not save iteration to database: {e}")
            # Fall back to local JSON
            fallback_path = Path("./data/bootstrapped/iterations_fallback.json")
            existing = json.loads(fallback_path.read_text()) if fallback_path.exists() else []
            existing.append(iteration_data)
            fallback_path.write_text(json.dumps(existing, indent=2))

    def save_fn_report(self, report: Dict):
        """Save false negative quantification report to Supabase database."""
        try:
            self.client.table("fn_reports").insert({
                "report_json": json.dumps(report),
                "created_at": "now()"
            }).execute()
            print(f"✅ FN report saved to database")
        except Exception as e:
            print(f"⚠️  Could not save FN report to database: {e}")

    def list_unlabeled_images(self) -> List[str]:
        """Retrieve list of unlabeled images from database."""
        try:
            result = self.client.table("dataset_images")\
                .select("stem, image_path")\
                .eq("is_labeled", False)\
                .execute()
            return result.data
        except Exception:
            return []

    def get_dataset_stats(self) -> Dict:
        """Get overall dataset statistics from database."""
        try:
            total = self.client.table("dataset_images").select("stem", count="exact").execute()
            labeled = self.client.table("dataset_images").select("stem", count="exact")\
                .eq("is_labeled", True).execute()
            return {
                "total_images": total.count,
                "labeled_images": labeled.count,
                "unlabeled_images": total.count - labeled.count,
            }
        except Exception as e:
            return {"error": str(e)}


def setup_database_schema(client: Client):
    """
    Print the SQL to create required database tables.
    Run this in your Supabase SQL Editor (Dashboard → SQL Editor).
    """
    schema_sql = """
-- Run this in your Supabase SQL Editor
-- Dashboard → SQL Editor → New Query

-- Track every image in the dataset
CREATE TABLE IF NOT EXISTS dataset_images (
    id              BIGSERIAL PRIMARY KEY,
    stem            TEXT UNIQUE NOT NULL,
    split           TEXT NOT NULL,           -- 'train', 'val', 'test'
    prefix          TEXT NOT NULL,           -- 'spacenet', 'bootstrapped'
    image_path      TEXT NOT NULL,           -- Path in Supabase storage
    label_path      TEXT,                    -- Path to YOLO label file
    is_labeled      BOOLEAN DEFAULT TRUE,
    bootstrap_iteration INT DEFAULT 0,
    uncertainty_score   FLOAT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Track each bootstrapping iteration
CREATE TABLE IF NOT EXISTS bootstrap_iterations (
    id              BIGSERIAL PRIMARY KEY,
    iteration       INT NOT NULL,
    n_images_labeled INT,
    cumulative_labels INT,
    map50           FLOAT,
    map50_95        FLOAT,
    precision_score FLOAT,
    recall_score    FLOAT,
    fn_rate         FLOAT,
    map_gain_per_label FLOAT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Store false negative quantification reports
CREATE TABLE IF NOT EXISTS fn_reports (
    id              BIGSERIAL PRIMARY KEY,
    report_json     JSONB NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_dataset_images_labeled ON dataset_images(is_labeled);
CREATE INDEX IF NOT EXISTS idx_dataset_images_iteration ON dataset_images(bootstrap_iteration);
"""
    print("\n📋 Run this SQL in your Supabase Dashboard:")
    print("   https://app.supabase.com → SQL Editor → New Query\n")
    print(schema_sql)

    # Save to file for reference
    schema_path = Path("./configs/supabase_schema.sql")
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(schema_sql)
    print(f"✅ Schema also saved to: {schema_path}")


if __name__ == "__main__":
    print("🗄️  Supabase Storage Setup")
    print("=" * 50)

    # Print schema for user to run manually
    client = get_client()
    setup_database_schema(client)

    # Test connection and create buckets
    print("\n🔧 Initializing storage buckets...")
    manager = SupabaseStorageManager()

    # Test stats
    stats = manager.get_dataset_stats()
    print(f"\n📊 Current dataset stats: {stats}")

    print("\n✅ Supabase setup complete!")
    print("   Next: Run python scripts/03_upload_to_roboflow.py")
    print("   (Updated to also sync to Supabase)")
