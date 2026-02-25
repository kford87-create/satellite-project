"""
scripts/api_client.py

Python client for calling the Satellite Detection API.
Use this for:
  - Testing your Edge Function
  - Building demo scripts for commercial prospects
  - Batch processing imagery in your pipeline

Usage:
  python scripts/api_client.py --image-url https://example.com/satellite.jpg
  python scripts/api_client.py --image-path ./data/test_image.jpg --return-image
"""

import os
import json
import base64
import argparse
import time
from pathlib import Path
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_USER_EMAIL = os.getenv("SUPABASE_USER_EMAIL")   # Your account email
SUPABASE_USER_PASSWORD = os.getenv("SUPABASE_USER_PASSWORD")  # Your account password


class SatelliteDetectionClient:
    """
    Client for the Satellite Object Detection API.
    Handles auth, image encoding, and response parsing.
    """

    def __init__(self):
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        self._auth_token = None
        self._function_url = f"{SUPABASE_URL}/functions/v1/detect-objects"

    def authenticate(self, email: str = None, password: str = None) -> bool:
        """Authenticate with Supabase to get JWT token."""
        email = email or SUPABASE_USER_EMAIL
        password = password or SUPABASE_USER_PASSWORD

        if not email or not password:
            print("❌ Set SUPABASE_USER_EMAIL and SUPABASE_USER_PASSWORD in .env")
            return False

        try:
            result = self.supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            self._auth_token = result.session.access_token
            print(f"✅ Authenticated as: {email}")
            return True
        except Exception as e:
            print(f"❌ Auth failed: {e}")
            return False

    def detect(
        self,
        image_url: str = None,
        image_path: Path = None,
        confidence_threshold: float = 0.25,
        classes: list = None,
        return_image: bool = False,
    ) -> dict:
        """
        Run object detection on a satellite image.

        Args:
            image_url: Public URL of satellite image
            image_path: Local path to image file (will be base64-encoded)
            confidence_threshold: Min detection confidence (0-1)
            classes: List of class names to detect (None = all)
            return_image: Return annotated image in response

        Returns:
            Detection results dict
        """
        if not self._auth_token:
            raise RuntimeError("Call authenticate() first")

        # Build request body
        body = {
            "confidence_threshold": confidence_threshold,
            "classes": classes,
            "return_image": return_image,
        }

        if image_url:
            body["image_url"] = image_url

        elif image_path:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            body["image_base64"] = base64.b64encode(img_bytes).decode("utf-8")

        else:
            raise ValueError("Provide either image_url or image_path")

       # Call Edge Function directly (bypasses SDK auth header conflicts)
        import httpx as _httpx
        start = time.time()
        with _httpx.Client(timeout=60.0) as http:
            r = http.post(
                self._function_url,
                json=body,
                headers={
                    "Authorization": f"Bearer {self._auth_token}",
                    "Content-Type": "application/json",
                }
            )
            r.raise_for_status()
        elapsed = (time.time() - start) * 1000
        response = r.json()
        response["client_elapsed_ms"] = round(elapsed, 2)

        return response

    def batch_detect(
        self,
        image_paths: list,
        confidence_threshold: float = 0.25,
        **kwargs
    ) -> list:
        """
        Run detection on a batch of local images.
        Useful for processing an entire dataset directory.
        """
        results = []
        for i, path in enumerate(image_paths):
            print(f"[{i+1}/{len(image_paths)}] Processing: {Path(path).name}")
            try:
                result = self.detect(
                    image_path=path,
                    confidence_threshold=confidence_threshold,
                    **kwargs
                )
                result["image_path"] = str(path)
                results.append(result)

                # Print summary
                if result.get("success"):
                    summary = result.get("summary", {})
                    print(f"  ✅ {summary.get('total_objects', 0)} objects detected "
                          f"({result.get('processing_time_ms', 0):.0f}ms)")
                else:
                    print(f"  ❌ Error: {result.get('error')}")

            except Exception as e:
                print(f"  ❌ Exception: {e}")
                results.append({"image_path": str(path), "success": False, "error": str(e)})

        return results

    def save_annotated_image(self, response: dict, output_path: Path):
        """Save annotated image from response to disk."""
        b64_img = response.get("annotated_image_base64")
        if not b64_img:
            print("No annotated image in response (set return_image=True)")
            return

        img_bytes = base64.b64decode(b64_img)
        output_path.write_bytes(img_bytes)
        print(f"✅ Annotated image saved to: {output_path}")

    def print_results(self, response: dict):
        """Pretty-print detection results."""
        if not response.get("success"):
            print(f"❌ Detection failed: {response.get('error')}")
            return

        print(f"\n{'='*50}")
        print(f"🛰️  Detection Results")
        print(f"{'='*50}")
        print(f"Request ID:       {response.get('request_id', 'N/A')}")
        print(f"Model:            {response.get('model_version', 'N/A')}")
        print(f"Processing time:  {response.get('processing_time_ms', 0):.1f}ms")
        print(f"Image size:       {response.get('image_width')}x{response.get('image_height')}px")

        summary = response.get("summary", {})
        print(f"\n📊 Summary:")
        print(f"  Total objects:    {summary.get('total_objects', 0)}")
        print(f"  High confidence:  {summary.get('high_confidence', 0)} (≥0.7)")
        print(f"  Low confidence:   {summary.get('low_confidence', 0)} (<0.7)")

        by_class = summary.get("by_class", {})
        if by_class:
            print(f"\n  By class:")
            for cls, count in by_class.items():
                print(f"    {cls}: {count}")

        detections = response.get("detections", [])
        if detections:
            print(f"\n🔍 Detections ({len(detections)}):")
            for i, det in enumerate(detections[:10]):  # Show first 10
                print(f"  [{i+1}] {det['class_name']}: {det['confidence']:.3f} confidence")
            if len(detections) > 10:
                print(f"  ... and {len(detections) - 10} more")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Satellite Object Detection API Client")
    parser.add_argument("--image-url", help="URL of satellite image")
    parser.add_argument("--image-path", help="Local path to satellite image")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--classes", nargs="+", help="Object classes to detect")
    parser.add_argument("--return-image", action="store_true", help="Return annotated image")
    parser.add_argument("--save-image", help="Path to save annotated image")
    parser.add_argument("--output-json", help="Save results to JSON file")
    args = parser.parse_args()

    if not args.image_url and not args.image_path:
        parser.error("Provide --image-url or --image-path")

    # Run detection
    client = SatelliteDetectionClient()

    if not client.authenticate():
        exit(1)

    response = client.detect(
        image_url=args.image_url,
        image_path=Path(args.image_path) if args.image_path else None,
        confidence_threshold=args.confidence,
        classes=args.classes,
        return_image=args.return_image or bool(args.save_image),
    )

    client.print_results(response)

    if args.save_image:
        client.save_annotated_image(response, Path(args.save_image))

    if args.output_json:
        # Remove base64 image from JSON output to keep file small
        output = {k: v for k, v in response.items() if k != "annotated_image_base64"}
        Path(args.output_json).write_text(json.dumps(output, indent=2))
        print(f"✅ Results saved to: {args.output_json}")
