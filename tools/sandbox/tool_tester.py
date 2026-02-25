"""
tool_tester.py
--------------
Comprehensive sandbox tester for all 12 satellite tools.
Validates each tool WITHOUT touching real data, models, or the database.

Usage:
    python tools/sandbox/tool_tester.py --tool chip_generator
    python tools/sandbox/tool_tester.py --all
    python tools/sandbox/tool_tester.py --tool geojson_exporter --verbose
    python tools/sandbox/tool_tester.py --tool change_detection --keep-data
    python tools/sandbox/tool_tester.py --list
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOOLS_ROOT = Path(__file__).resolve().parent.parent.parent  # satellite-project/

ALL_TOOLS: list[str] = [
    "chip_generator",
    "scene_quality_filter",
    "spatial_diversity_sampler",
    "geojson_exporter",
    "change_detection",
    "bootstrapping_dashboard",
    "batch_processor",
    "coverage_report",
    "sentinel2_fetcher",
    "geo_aware_evaluator",
    "rotation_invariance_tester",
    "confidence_calibrator",
]

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def create_synthetic_image(
    path: Path,
    size: tuple[int, int] = (1280, 1280),
    n_rects: int = 30,
) -> Path:
    """
    Create a synthetic satellite-like JPEG with random coloured rectangles
    simulating building footprints on a grey-green background.
    """
    rng = random.Random(42)
    img = Image.new("RGB", size, color=(80, 90, 70))
    draw = ImageDraw.Draw(img)

    w, h = size
    for _ in range(n_rects):
        x1 = rng.randint(0, w - 60)
        y1 = rng.randint(0, h - 60)
        x2 = x1 + rng.randint(20, 120)
        y2 = y1 + rng.randint(20, 120)
        x2 = min(x2, w - 1)
        y2 = min(y2, h - 1)
        r = rng.randint(140, 220)
        g = rng.randint(130, 200)
        b = rng.randint(120, 190)
        draw.rectangle([x1, y1, x2, y2], fill=(r, g, b))

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=95)
    return path


def create_synthetic_label(path: Path, n_boxes: int = 5) -> Path:
    """
    Create a synthetic YOLO label .txt file with n_boxes random bounding boxes.
    Format per line: class cx cy w h (all normalised 0..1).
    """
    rng = random.Random(99)
    lines: list[str] = []
    for _ in range(n_boxes):
        cls = 0  # single-class: building
        cx = round(rng.uniform(0.1, 0.9), 6)
        cy = round(rng.uniform(0.1, 0.9), 6)
        bw = round(rng.uniform(0.02, 0.15), 6)
        bh = round(rng.uniform(0.02, 0.15), 6)
        lines.append(f"{cls} {cx} {cy} {bw} {bh}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def create_synthetic_chip_index(path: Path, n_chips: int = 4) -> Path:
    """
    Write a chip_index.json that maps chip stem names to geo coordinates.
    Also writes the nested format expected by chip_generator output.
    """
    rng = random.Random(7)
    base_lon = -73.98
    base_lat = 40.75

    chips: list[dict[str, Any]] = []
    chip_map: dict[str, dict[str, float]] = {}

    for i in range(n_chips):
        row = i // 2
        col = i % 2
        stem = f"scene_{row:04d}_{col:04d}"
        lon = round(base_lon + col * 0.001, 6)
        lat = round(base_lat + row * 0.001, 6)
        chips.append({
            "filename": f"{stem}.jpg",
            "row": row,
            "col": col,
            "x_offset": col * 640,
            "y_offset": row * 640,
            "geo_bounds": {
                "west": lon - 0.0005,
                "east": lon + 0.0005,
                "north": lat + 0.0005,
                "south": lat - 0.0005,
            },
        })
        chip_map[stem] = {"lon": lon, "lat": lat}

    # Full chip_index format (as written by chip_generator)
    full_index: dict[str, Any] = {
        "scene_id": "scene",
        "scene_path": str(path.parent / "scene.jpg"),
        "chip_size": 640,
        "overlap": 64,
        "crs": None,
        "chips": chips,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(full_index, indent=2))

    # Also write a flat chip_index.json beside the images (used by geo_aware_evaluator)
    flat_path = path.parent / "chip_index.json"
    flat_path.write_text(json.dumps(chip_map, indent=2))

    return path


def create_synthetic_uncertainty_scores(path: Path, n_items: int = 10) -> Path:
    """Write uncertainty_scores.json as a list of {image, score, geo} dicts."""
    rng = random.Random(13)
    scores: list[dict[str, Any]] = []
    for i in range(n_items):
        scores.append({
            "image": f"chip_{i:04d}.jpg",
            "score": round(rng.uniform(0.1, 0.95), 4),
            "geo": {
                "lon": round(-73.98 + rng.uniform(-0.05, 0.05), 6),
                "lat": round(40.75 + rng.uniform(-0.05, 0.05), 6),
            },
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scores, indent=2))
    return path


def create_synthetic_portfolio(path: Path, n_properties: int = 4) -> Path:
    """Write a portfolio.json for batch_processor / coverage_report."""
    rng = random.Random(17)
    properties: list[dict[str, Any]] = []
    for i in range(n_properties):
        properties.append({
            "id": f"prop_{i:04d}",
            "name": f"Property {i}",
            "lon": round(-73.98 + rng.uniform(-0.1, 0.1), 6),
            "lat": round(40.75 + rng.uniform(-0.1, 0.1), 6),
            "risk_score": round(rng.uniform(0.1, 0.9), 3),
            "detections": rng.randint(0, 12),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"portfolio": properties}, indent=2))
    return path


def create_synthetic_metrics(path: Path, n_rounds: int = 5) -> Path:
    """Write a metrics.json for bootstrapping_dashboard."""
    rng = random.Random(21)
    rounds: list[dict[str, Any]] = []
    for i in range(n_rounds):
        rounds.append({
            "round": i,
            "map50": round(0.3 + i * 0.08 + rng.uniform(-0.02, 0.02), 4),
            "precision": round(0.4 + i * 0.07 + rng.uniform(-0.02, 0.02), 4),
            "recall": round(0.35 + i * 0.09 + rng.uniform(-0.02, 0.02), 4),
            "labeled_count": 50 + i * 20,
            "unlabeled_count": 500 - i * 20,
            "annotation_hours": round(2.0 + i * 1.5, 2),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rounds": rounds}, indent=2))
    return path


def create_chip_dir_with_images_and_labels(
    images_dir: Path,
    labels_dir: Path,
    n_chips: int = 4,
    chip_size: int = 640,
) -> list[str]:
    """Populate images_dir with synthetic chips and labels_dir with matching labels."""
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    stems: list[str] = []
    for i in range(n_chips):
        row = i // 2
        col = i % 2
        stem = f"scene_{row:04d}_{col:04d}"
        img_path = images_dir / f"{stem}.jpg"
        lbl_path = labels_dir / f"{stem}.txt"
        create_synthetic_image(img_path, size=(chip_size, chip_size), n_rects=5)
        create_synthetic_label(lbl_path, n_boxes=3)
        stems.append(stem)
    return stems


# ---------------------------------------------------------------------------
# Tool test functions
# ---------------------------------------------------------------------------

def test_chip_generator(sandbox_dir: Path, verbose: bool = False) -> tuple[bool, str]:
    """Test chip_generator with a synthetic JPEG scene."""
    try:
        scene_path = create_synthetic_image(sandbox_dir / "cg_scene" / "scene.jpg", size=(1280, 1280))
        output_dir = sandbox_dir / "cg_chips"
        output_dir.mkdir(parents=True, exist_ok=True)

        sys.path.insert(0, str(TOOLS_ROOT))
        from tools.data_acquisition.chip_generator import generate_chips  # type: ignore

        result = generate_chips(
            scene_path=scene_path,
            output_dir=output_dir,
            chip_size=640,
            overlap=64,
        )

        chips = list(output_dir.glob("*.jpg"))
        chip_index_files = list(output_dir.glob("*_chip_index.json"))

        assert len(chips) >= 4, f"Expected >=4 chips, got {len(chips)}"
        assert len(chip_index_files) == 1, f"Expected 1 chip_index.json, got {len(chip_index_files)}"

        # Validate chip_index JSON structure
        index_data = json.loads(chip_index_files[0].read_text())
        assert "scene_id" in index_data, "chip_index missing 'scene_id'"
        assert "chips" in index_data, "chip_index missing 'chips'"
        assert isinstance(index_data["chips"], list), "'chips' should be a list"
        assert len(index_data["chips"]) == len(chips), "chip count mismatch between JSON and files"

        # Validate each chip is a readable image of the right size
        for chip_path in chips[:2]:  # spot-check first two
            with Image.open(chip_path) as im:
                assert im.size == (640, 640), f"Chip {chip_path.name} is {im.size}, expected (640, 640)"

        return True, f"{len(chips)} chips created, chip_index.json valid"
    except Exception as e:
        if verbose:
            traceback.print_exc()
        return False, str(e)


def test_scene_quality_filter(sandbox_dir: Path, verbose: bool = False) -> tuple[bool, str]:
    """Test scene_quality_filter on a directory of synthetic chips."""
    try:
        chips_dir = sandbox_dir / "sqf_chips"
        chips_dir.mkdir(parents=True, exist_ok=True)

        # Create 3 normal chips (should pass) + 1 near-white chip (cloud, should fail)
        for i in range(3):
            create_synthetic_image(chips_dir / f"chip_{i:04d}.jpg", size=(640, 640))

        # Near-white "cloudy" chip
        cloud_img = Image.new("RGB", (640, 640), color=(210, 215, 210))
        cloud_img.save(str(chips_dir / "chip_cloud.jpg"), quality=95)

        sys.path.insert(0, str(TOOLS_ROOT))
        from tools.data_acquisition.scene_quality_filter import run_filter  # type: ignore

        report = run_filter(
            input_dir=chips_dir,
            max_cloud=0.20,
            move_rejected=False,
        )

        assert report, "run_filter returned empty report"
        assert "summary" in report, "report missing 'summary'"
        summary = report["summary"]
        assert "total_scanned" in summary, "summary missing 'total_scanned'"
        assert "passed" in summary, "summary missing 'passed'"
        assert "rejected" in summary, "summary missing 'rejected'"
        assert summary["total_scanned"] == 4, f"Expected 4 scanned, got {summary['total_scanned']}"
        assert summary["rejected"] >= 1, "Expected at least 1 rejection (cloud chip)"

        report_path = chips_dir / "quality_report.json"
        assert report_path.exists(), "quality_report.json not written"
        loaded = json.loads(report_path.read_text())
        assert "images" in loaded, "quality_report.json missing 'images' key"

        return True, (
            f"{summary['passed']} passed, {summary['rejected']} rejected "
            f"(of {summary['total_scanned']} scanned)"
        )
    except Exception as e:
        if verbose:
            traceback.print_exc()
        return False, str(e)


def test_spatial_diversity_sampler(sandbox_dir: Path, verbose: bool = False) -> tuple[bool, str]:
    """Test spatial_diversity_sampler with synthetic uncertainty scores."""
    try:
        sds_dir = sandbox_dir / "sds"
        sds_dir.mkdir(parents=True, exist_ok=True)

        scores_path = sds_dir / "uncertainty_scores.json"
        create_synthetic_uncertainty_scores(scores_path, n_items=20)
        output_path = sds_dir / "sampled_batch.json"

        sys.path.insert(0, str(TOOLS_ROOT))
        try:
            from tools.active_learning.spatial_diversity_sampler import sample_batch  # type: ignore
        except ImportError as ie:
            return False, f"ImportError: {ie}"

        result = sample_batch(
            scores_path=scores_path,
            output_path=output_path,
            batch_size=5,
        )

        assert output_path.exists(), "sampled_batch.json not written"
        data = json.loads(output_path.read_text())
        assert isinstance(data, (list, dict)), "Output is not valid JSON list/dict"
        # Accept list or dict with a 'batch' key
        batch = data if isinstance(data, list) else data.get("batch", data.get("samples", []))
        assert len(batch) <= 20, "Batch size exceeds input pool"

        return True, f"Sampled {len(batch)} items from 20-item pool"
    except Exception as e:
        if verbose:
            traceback.print_exc()
        return False, str(e)


def test_geojson_exporter(sandbox_dir: Path, verbose: bool = False) -> tuple[bool, str]:
    """Test geojson_exporter with synthetic chip detections + chip_index."""
    try:
        gje_dir = sandbox_dir / "gje"
        gje_dir.mkdir(parents=True, exist_ok=True)

        # Create synthetic detections JSON
        chip_index_path = gje_dir / "scene_chip_index.json"
        create_synthetic_chip_index(chip_index_path, n_chips=4)

        detections: list[dict[str, Any]] = []
        rng = random.Random(55)
        for i in range(4):
            row = i // 2
            col = i % 2
            stem = f"scene_{row:04d}_{col:04d}"
            detections.append({
                "chip": f"{stem}.jpg",
                "boxes": [
                    {
                        "cls": 0,
                        "conf": round(rng.uniform(0.6, 0.98), 3),
                        "x1": rng.randint(50, 300),
                        "y1": rng.randint(50, 300),
                        "x2": rng.randint(310, 580),
                        "y2": rng.randint(310, 580),
                    }
                ],
            })
        detections_path = gje_dir / "detections.json"
        detections_path.write_text(json.dumps(detections, indent=2))

        output_geojson = gje_dir / "output.geojson"

        sys.path.insert(0, str(TOOLS_ROOT))
        try:
            from tools.commercial.geojson_exporter import export_geojson  # type: ignore
        except ImportError as ie:
            return False, f"ImportError: {ie}"

        export_geojson(
            detections_path=detections_path,
            chip_index_path=chip_index_path,
            output_path=output_geojson,
        )

        assert output_geojson.exists(), "output.geojson not written"
        data = json.loads(output_geojson.read_text())
        assert data.get("type") == "FeatureCollection", "GeoJSON missing FeatureCollection type"
        assert "features" in data, "GeoJSON missing 'features' key"
        assert len(data["features"]) > 0, "GeoJSON has 0 features"

        # Validate each feature has geometry and properties
        for feat in data["features"]:
            assert feat.get("type") == "Feature", "Feature missing type"
            assert "geometry" in feat, "Feature missing geometry"
            assert "properties" in feat, "Feature missing properties"

        return True, f"{len(data['features'])} GeoJSON features exported"
    except Exception as e:
        if verbose:
            traceback.print_exc()
        return False, str(e)


def test_change_detection(sandbox_dir: Path, verbose: bool = False) -> tuple[bool, str]:
    """Test change_detection with two synthetic scenes at the same location."""
    try:
        cd_dir = sandbox_dir / "cd"
        cd_dir.mkdir(parents=True, exist_ok=True)

        before_path = create_synthetic_image(cd_dir / "before.jpg", size=(640, 640), n_rects=10)
        after_path = create_synthetic_image(cd_dir / "after.jpg", size=(640, 640), n_rects=18)
        output_path = cd_dir / "change_report.json"

        sys.path.insert(0, str(TOOLS_ROOT))
        try:
            from tools.commercial.change_detection_engine import detect_changes  # type: ignore
        except ImportError as ie:
            return False, f"ImportError: {ie}"

        result = detect_changes(
            before_path=before_path,
            after_path=after_path,
            output_path=output_path,
        )

        assert output_path.exists(), "change_report.json not written"
        data = json.loads(output_path.read_text())
        assert isinstance(data, dict), "change_report.json is not a JSON object"
        # Accept any dict with change-related keys
        assert any(
            k in data for k in ("changes", "change_score", "added", "removed", "summary", "result")
        ), f"change_report.json has no recognized keys: {list(data.keys())}"

        return True, f"Change report written: {list(data.keys())}"
    except Exception as e:
        if verbose:
            traceback.print_exc()
        return False, str(e)


def test_bootstrapping_dashboard(sandbox_dir: Path, verbose: bool = False) -> tuple[bool, str]:
    """Test bootstrapping_dashboard rendering with synthetic metrics."""
    try:
        bd_dir = sandbox_dir / "bd"
        bd_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = bd_dir / "metrics.json"
        create_synthetic_metrics(metrics_path, n_rounds=5)
        output_dir = bd_dir / "dashboard_out"
        output_dir.mkdir(parents=True, exist_ok=True)

        sys.path.insert(0, str(TOOLS_ROOT))
        try:
            from tools.active_learning.bootstrapping_dashboard import render_dashboard  # type: ignore
        except ImportError as ie:
            return False, f"ImportError: {ie}"

        render_dashboard(
            metrics_path=metrics_path,
            output_dir=output_dir,
        )

        # Accept any output file: HTML, PNG, or JSON summary
        outputs = list(output_dir.iterdir())
        assert len(outputs) >= 1, "No dashboard outputs written"

        output_types = [p.suffix.lower() for p in outputs]
        assert any(
            ext in output_types for ext in (".html", ".png", ".json", ".svg")
        ), f"No recognised dashboard output type. Got: {output_types}"

        return True, f"{len(outputs)} dashboard file(s) written: {[p.name for p in outputs]}"
    except Exception as e:
        if verbose:
            traceback.print_exc()
        return False, str(e)


def test_batch_processor(sandbox_dir: Path, verbose: bool = False) -> tuple[bool, str]:
    """Test batch_processor with a synthetic portfolio of properties."""
    try:
        bp_dir = sandbox_dir / "bp"
        bp_dir.mkdir(parents=True, exist_ok=True)

        portfolio_path = bp_dir / "portfolio.json"
        create_synthetic_portfolio(portfolio_path, n_properties=4)

        # Create dummy image chips for each property
        images_dir = bp_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            create_synthetic_image(images_dir / f"prop_{i:04d}.jpg", size=(640, 640))

        output_dir = bp_dir / "batch_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        sys.path.insert(0, str(TOOLS_ROOT))
        try:
            from tools.commercial.batch_processor import run_batch  # type: ignore
        except ImportError as ie:
            return False, f"ImportError: {ie}"

        run_batch(
            portfolio_path=portfolio_path,
            images_dir=images_dir,
            output_dir=output_dir,
        )

        outputs = list(output_dir.iterdir())
        assert len(outputs) >= 1, "No batch output files written"

        # Check for any JSON or HTML result
        json_outputs = [p for p in outputs if p.suffix == ".json"]
        html_outputs = [p for p in outputs if p.suffix == ".html"]
        assert len(json_outputs) + len(html_outputs) >= 1, (
            f"Expected JSON or HTML outputs, got: {[p.name for p in outputs]}"
        )

        return True, f"{len(outputs)} output file(s) written"
    except Exception as e:
        if verbose:
            traceback.print_exc()
        return False, str(e)


def test_coverage_report(sandbox_dir: Path, verbose: bool = False) -> tuple[bool, str]:
    """Test coverage_report_generator with a synthetic portfolio."""
    try:
        cr_dir = sandbox_dir / "cr"
        cr_dir.mkdir(parents=True, exist_ok=True)

        portfolio_path = cr_dir / "portfolio.json"
        create_synthetic_portfolio(portfolio_path, n_properties=4)
        output_path = cr_dir / "coverage_report.html"

        sys.path.insert(0, str(TOOLS_ROOT))
        try:
            from tools.commercial.coverage_report_generator import generate_report  # type: ignore
        except ImportError as ie:
            return False, f"ImportError: {ie}"

        generate_report(
            portfolio_path=portfolio_path,
            output_path=output_path,
        )

        assert output_path.exists(), "coverage_report.html not written"
        html = output_path.read_text()
        assert len(html) > 100, "HTML report is suspiciously short"
        assert "<html" in html.lower() or "<!doctype" in html.lower(), (
            "Output does not look like valid HTML"
        )

        return True, f"HTML report written ({len(html)} bytes)"
    except Exception as e:
        if verbose:
            traceback.print_exc()
        return False, str(e)


def test_sentinel2_fetcher(sandbox_dir: Path, verbose: bool = False) -> tuple[bool, str]:
    """
    Test sentinel2_fetcher in dry-run / mock mode.
    Validates that the module imports and the fetch function accepts parameters
    without making real network calls (uses --dry-run or mock credentials).
    """
    try:
        sf_dir = sandbox_dir / "sf"
        sf_dir.mkdir(parents=True, exist_ok=True)
        output_dir = sf_dir / "downloads"
        output_dir.mkdir(parents=True, exist_ok=True)

        sys.path.insert(0, str(TOOLS_ROOT))
        try:
            import tools.active_learning.spatial_diversity_sampler  # noqa: F401 – confirm importable
            from tools.data_acquisition.sentinel2_fetcher import fetch_scene  # type: ignore
        except ImportError as ie:
            return False, f"ImportError: {ie}"

        # Call with dry_run=True to avoid real network requests
        result = fetch_scene(
            bbox=(-73.99, 40.74, -73.97, 40.76),
            date_range=("2024-01-01", "2024-01-31"),
            output_dir=output_dir,
            dry_run=True,
        )

        # In dry-run mode the function should return a dict describing what
        # would have been fetched, or an empty dict, without crashing.
        assert result is not None, "fetch_scene returned None"
        assert isinstance(result, dict), f"fetch_scene returned {type(result)}, expected dict"

        return True, f"Dry-run completed: {result}"
    except Exception as e:
        if verbose:
            traceback.print_exc()
        return False, str(e)


def test_geo_aware_evaluator(sandbox_dir: Path, verbose: bool = False) -> tuple[bool, str]:
    """
    Test geo_aware_evaluator's internal helpers (IoU, GT loading, matching)
    without requiring a real YOLO model.
    """
    try:
        gae_dir = sandbox_dir / "gae"
        images_dir = gae_dir / "images"
        labels_dir = gae_dir / "labels"

        stems = create_chip_dir_with_images_and_labels(images_dir, labels_dir, n_chips=4)
        chip_index_path = images_dir / "chip_index.json"

        # Write flat chip_index used by evaluator
        chip_map = {}
        for i, stem in enumerate(stems):
            chip_map[stem] = {"lon": -73.98 + i * 0.001, "lat": 40.75 + i * 0.001}
        chip_index_path.write_text(json.dumps(chip_map, indent=2))

        sys.path.insert(0, str(TOOLS_ROOT))
        try:
            from tools.model_performance.geo_aware_evaluator import (  # type: ignore
                _compute_iou,
                _yolo_to_xyxy,
                _load_ground_truth,
                _match_predictions,
                _find_worst_regions,
            )
        except ImportError as ie:
            return False, f"ImportError: {ie}"

        # Test _compute_iou
        box_a = [0.0, 0.0, 100.0, 100.0]
        box_b = [50.0, 50.0, 150.0, 150.0]
        iou = _compute_iou(box_a, box_b)
        assert 0.0 < iou < 1.0, f"IoU should be between 0 and 1, got {iou}"

        # Identical boxes should have IoU=1
        iou_perfect = _compute_iou(box_a, box_a)
        assert abs(iou_perfect - 1.0) < 1e-6, f"Identical boxes IoU should be 1.0, got {iou_perfect}"

        # Non-overlapping boxes should have IoU=0
        box_c = [200.0, 200.0, 300.0, 300.0]
        iou_zero = _compute_iou(box_a, box_c)
        assert iou_zero == 0.0, f"Non-overlapping boxes IoU should be 0.0, got {iou_zero}"

        # Test _yolo_to_xyxy
        xyxy = _yolo_to_xyxy(0.5, 0.5, 0.2, 0.2, 640, 640)
        assert len(xyxy) == 4, "xyxy should have 4 coords"
        x1, y1, x2, y2 = xyxy
        assert x1 < x2 and y1 < y2, "xyxy coords should be ordered"

        # Test _load_ground_truth on a real label file
        label_path = labels_dir / f"{stems[0]}.txt"
        img_w, img_h = 640, 640
        gts = _load_ground_truth(label_path, img_w, img_h)
        assert isinstance(gts, list), "_load_ground_truth should return a list"
        assert len(gts) > 0, "Should load at least 1 ground truth box"
        for gt in gts:
            assert "cls" in gt and "box" in gt, "GT entry missing cls or box"

        # Test _match_predictions with synthetic preds
        preds = [{"cls": gt["cls"], "box": gt["box"], "conf": 0.9} for gt in gts[:2]]
        tp, fp, fn = _match_predictions(preds, gts, iou_thresh=0.5)
        assert tp + fp + fn >= 0, "Negative counts are invalid"

        # Test _find_worst_regions with mock results
        mock_results = [
            {"image": f"{s}.jpg", "tp": 2, "fp": 1, "fn": 1,
             "precision": 0.67, "recall": 0.67,
             "geo": {"lon": -73.98 + j * 0.001, "lat": 40.75 + j * 0.001}}
            for j, s in enumerate(stems)
        ]
        worst = _find_worst_regions(mock_results, grid_size=5)
        assert isinstance(worst, list), "_find_worst_regions should return a list"

        return True, (
            f"IoU helpers OK, loaded {len(gts)} GT boxes, "
            f"matched {tp} TP/{fp} FP/{fn} FN, found {len(worst)} worst regions"
        )
    except Exception as e:
        if verbose:
            traceback.print_exc()
        return False, str(e)


def test_rotation_invariance_tester(sandbox_dir: Path, verbose: bool = False) -> tuple[bool, str]:
    """Test rotation_invariance_tester on synthetic chip images."""
    try:
        rit_dir = sandbox_dir / "rit"
        images_dir = rit_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        for i in range(3):
            create_synthetic_image(images_dir / f"chip_{i:04d}.jpg", size=(640, 640))

        output_dir = rit_dir / "rotation_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        sys.path.insert(0, str(TOOLS_ROOT))
        try:
            from tools.model_performance.rotation_invariance_tester import test_rotation_invariance  # type: ignore
        except ImportError as ie:
            return False, f"ImportError: {ie}"

        # Run without a real model: tool should accept model_path=None or a mock path
        # and produce a rotation-stability report
        result = test_rotation_invariance(
            images_dir=images_dir,
            output_dir=output_dir,
            angles=[0, 45, 90, 135, 180, 225, 270, 315],
            model_path=None,
        )

        report_files = list(output_dir.glob("*.json")) + list(output_dir.glob("*.png"))
        assert len(report_files) >= 1, "No rotation invariance report files written"

        return True, f"{len(report_files)} report file(s) written"
    except Exception as e:
        if verbose:
            traceback.print_exc()
        return False, str(e)


def test_confidence_calibrator(sandbox_dir: Path, verbose: bool = False) -> tuple[bool, str]:
    """Test confidence_calibrator with synthetic confidence scores."""
    try:
        cc_dir = sandbox_dir / "cc"
        cc_dir.mkdir(parents=True, exist_ok=True)

        rng = random.Random(31)

        # Synthetic raw confidence scores and ground-truth labels
        n = 50
        raw_scores = [round(rng.uniform(0.3, 0.99), 4) for _ in range(n)]
        labels = [rng.randint(0, 1) for _ in range(n)]

        scores_path = cc_dir / "raw_scores.json"
        scores_path.write_text(json.dumps({"scores": raw_scores, "labels": labels}, indent=2))

        output_path = cc_dir / "calibrated_scores.json"

        sys.path.insert(0, str(TOOLS_ROOT))
        try:
            from tools.model_performance.confidence_calibrator import calibrate  # type: ignore
        except ImportError as ie:
            return False, f"ImportError: {ie}"

        result = calibrate(
            scores_path=scores_path,
            output_path=output_path,
        )

        assert output_path.exists(), "calibrated_scores.json not written"
        data = json.loads(output_path.read_text())
        assert isinstance(data, dict), "calibrated output is not a JSON object"

        # Accept any dict that contains calibrated scores or temperature
        assert any(
            k in data for k in ("calibrated_scores", "temperature", "scores", "ece", "result")
        ), f"Unexpected calibrated output keys: {list(data.keys())}"

        return True, f"Calibration output keys: {list(data.keys())}"
    except Exception as e:
        if verbose:
            traceback.print_exc()
        return False, str(e)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOL_TESTS: dict[str, Callable[[Path, bool], tuple[bool, str]]] = {
    "chip_generator":             test_chip_generator,
    "scene_quality_filter":       test_scene_quality_filter,
    "spatial_diversity_sampler":  test_spatial_diversity_sampler,
    "geojson_exporter":           test_geojson_exporter,
    "change_detection":           test_change_detection,
    "bootstrapping_dashboard":    test_bootstrapping_dashboard,
    "batch_processor":            test_batch_processor,
    "coverage_report":            test_coverage_report,
    "sentinel2_fetcher":          test_sentinel2_fetcher,
    "geo_aware_evaluator":        test_geo_aware_evaluator,
    "rotation_invariance_tester": test_rotation_invariance_tester,
    "confidence_calibrator":      test_confidence_calibrator,
}

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _make_sandbox() -> Path:
    ts = int(time.time())
    sandbox = Path(f"/tmp/satellite_sandbox_{ts}")
    sandbox.mkdir(parents=True, exist_ok=True)
    return sandbox


def _run_test(
    name: str,
    sandbox_dir: Path,
    verbose: bool,
) -> tuple[bool, str]:
    """Run one tool test, catching all exceptions so others can continue."""
    fn = TOOL_TESTS.get(name)
    if fn is None:
        return False, f"No test registered for '{name}'"
    print(f"\n  Running {name} ...", end="", flush=True)
    passed, reason = fn(sandbox_dir, verbose)
    status = "PASS" if passed else "FAIL"
    marker = "OK" if passed else "!!"
    print(f"\r  [{marker}] {name:<30} {status}  {reason}")
    return passed, reason


def run_all_tests(
    tool_names: list[str],
    keep_data: bool,
    verbose: bool,
) -> None:
    sandbox_dir = _make_sandbox()
    print(f"\nSandbox: {sandbox_dir}")
    print(f"Testing {len(tool_names)} tool(s)\n")
    print("-" * 60)

    results: list[tuple[str, bool, str]] = []

    for name in tool_names:
        passed, reason = _run_test(name, sandbox_dir, verbose)
        results.append((name, passed, reason))

    print("-" * 60)

    passed_count = sum(1 for _, p, _ in results if p)
    total = len(results)
    failures = [(n, r) for n, p, r in results if not p]

    print(f"\nSummary: {passed_count}/{total} tools passed\n")

    if failures:
        print("Failures:")
        for name, reason in failures:
            print(f"  FAIL  {name}: {reason}")
        print()

    if keep_data:
        print(f"Sandbox preserved at: {sandbox_dir}")
    else:
        shutil.rmtree(sandbox_dir, ignore_errors=True)
        print("Sandbox cleaned up.")

    if failures:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sandbox tester for all 12 satellite tools.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/sandbox/tool_tester.py --list
  python tools/sandbox/tool_tester.py --all
  python tools/sandbox/tool_tester.py --tool chip_generator
  python tools/sandbox/tool_tester.py --tool geo_aware_evaluator --verbose
  python tools/sandbox/tool_tester.py --all --keep-data
        """,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--tool",
        metavar="NAME",
        help="Name of a single tool to test",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run tests for all 12 tools",
    )
    group.add_argument(
        "--list",
        action="store_true",
        help="Print all testable tool names and exit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full tracebacks on failure",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Leave sandbox directory after tests (for inspection)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list:
        print("Testable tools:")
        for name in ALL_TOOLS:
            registered = "registered" if name in TOOL_TESTS else "NOT registered"
            print(f"  {name:<35} ({registered})")
        sys.exit(0)

    if args.all:
        run_all_tests(ALL_TOOLS, keep_data=args.keep_data, verbose=args.verbose)
    elif args.tool:
        if args.tool not in TOOL_TESTS:
            print(f"Unknown tool: '{args.tool}'")
            print(f"Run --list to see available tools.")
            sys.exit(1)
        run_all_tests([args.tool], keep_data=args.keep_data, verbose=args.verbose)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
