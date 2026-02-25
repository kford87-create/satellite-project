"""
scripts/01_download_spacenet.py

Downloads a manageable subset of SpaceNet data from AWS S3.
We start with SpaceNet 2 (Vegas) - buildings dataset.
~3.5GB for the small AOI, good for initial bootstrapping.

SpaceNet S3 bucket: s3://spacenet-dataset (requester pays)
Note: You need a free AWS account. Data transfer costs ~$0.09/GB.
For Vegas subset: estimated cost ~$0.30
"""

import os
import subprocess
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── SpaceNet Dataset Paths ───────────────────────────────────────────────────
# SpaceNet 2: Building Detection - Las Vegas (smallest AOI, good starting point)
SPACENET_BUCKET = "spacenet-dataset"

DATASETS = {
    "sn2_vegas": {
        "description": "SpaceNet 2 - Las Vegas Buildings (~3.5GB)",
        "s3_path": "spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_2_Vegas.tar.gz",
        "labels": "spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_2_Vegas.tar.gz",
    },
    "sn7_multitemporal": {
        "description": "SpaceNet 7 - Multi-Temporal Urban Development (~8GB) - Best for change detection",
        "s3_path": "spacenet/SN7_buildings/tarballs/SN7_buildings_train.tar.gz",
        "labels": "spacenet/SN7_buildings/tarballs/SN7_buildings_train.tar.gz",
    },
}


def check_aws_cli():
    """Verify AWS CLI is configured."""
    result = subprocess.run(
        ["aws", "sts", "get-caller-identity"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("❌ AWS CLI not configured. Run: aws configure")
        print("   You need a free AWS account at aws.amazon.com")
        return False
    print("✅ AWS CLI configured successfully")
    return True


def download_dataset(dataset_key: str, output_dir: str):
    """
    Download a SpaceNet dataset from S3.
    Uses --request-payer requester (small cost ~$0.30 for Vegas subset).
    """
    dataset = DATASETS[dataset_key]
    output_path = Path(output_dir) / dataset_key
    output_path.mkdir(parents=True, exist_ok=True)

    s3_uri = f"s3://{SPACENET_BUCKET}/{dataset['s3_path']}"
    local_file = output_path / Path(dataset['s3_path']).name

    print(f"\n📥 Downloading: {dataset['description']}")
    print(f"   From: {s3_uri}")
    print(f"   To:   {local_file}")
    print(f"   Note: Small AWS data transfer cost applies (~$0.30)\n")

    cmd = [
        "aws", "s3", "cp",
        s3_uri,
        str(local_file),
        "--request-payer", "requester",
        "--no-sign-request"  # Remove this line if using paid account
    ]

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"\n✅ Download complete: {local_file}")
        extract_tarball(local_file, output_path)
    else:
        print(f"\n❌ Download failed. Check AWS credentials and bucket access.")


def extract_tarball(tarball_path: Path, output_dir: Path):
    """Extract downloaded tarball."""
    print(f"\n📦 Extracting {tarball_path.name}...")
    cmd = ["tar", "-xzf", str(tarball_path), "-C", str(output_dir)]
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"✅ Extracted to: {output_dir}")
        # Remove tarball to save space
        os.remove(tarball_path)
        print(f"🗑️  Removed tarball to save disk space")
    else:
        print(f"❌ Extraction failed")


def list_available_datasets():
    """List available SpaceNet datasets without downloading."""
    print("\n📋 Available SpaceNet Datasets:\n")
    for key, info in DATASETS.items():
        print(f"  [{key}]")
        print(f"    {info['description']}")
        print(f"    S3 Path: s3://{SPACENET_BUCKET}/{info['s3_path']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SpaceNet datasets from AWS")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="sn2_vegas",
        help="Which SpaceNet dataset to download (default: sn2_vegas)"
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("DATA_DIR", "./data"),
        help="Directory to save downloaded data"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets without downloading"
    )

    args = parser.parse_args()

    if args.list:
        list_available_datasets()
    else:
        if check_aws_cli():
            download_dataset(args.dataset, args.output_dir)
        else:
            print("\nSetup instructions:")
            print("1. Create a free AWS account at https://aws.amazon.com")
            print("2. Run: aws configure")
            print("3. Enter your Access Key ID and Secret Access Key")
            print("4. Re-run this script")
