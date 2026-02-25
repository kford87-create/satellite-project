"""
tools/sandbox/tool_tester.py

Sandbox environment for testing and debugging any tool before production deployment.
Generates synthetic test data, runs each tool in isolation, validates outputs,
and produces a pass/fail report with detailed diagnostics.

Think of this as a flight simulator — you can crash here without consequences.
Every tool gets tested against realistic but fake data before it touches real data.

Usage:
  # Test a specific tool
  python tools/sandbox/tool_tester.py --tool chip_generator
  python tools/sandbox/tool_tester.py --tool scene_quality_filter
  python tools/sandbox/tool_tester.py --tool change_detection

  # Test all tools
  python tools/sandbox/tool_tester.py --all

  # Test with verbose output
  python tools/sandbox/tool_tester.py --all --verbose

  # Keep sandbox data after test (for manual inspection)
  python tools/sandbox/tool_tester.py --tool geojson_exporter --keep-data
"""

import os
import sys
import json
import time
import shutil
import traceback
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict, field
from dotenv import load_dotenv
import argparse

load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "data_acquisition"))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "model_performance"))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "active_learning"))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "commercial"))

SANDBOX_DIR = Path(os.getenv("DATA_DIR", "./data")) / "sandbox"


# ── Result Types ──────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    tool_name: str
    test_name: str
    passed: bool
    duration_seconds: float
    message: str = ""
    warnings: List[str] = field(default_factory=list)
    output_summary: Dict = field(default_factory=dict)
    error: str = ""
    traceback: str = ""


@dataclass
class ToolReport:
    tool_name: str
    total_tests: int
    passed: int
    failed: int
    warnings: int
    duration_seconds: float
    results: List[TestResult] = field(default_factory=list)
    sandbox_dir: str = ""

    @property
    def success_rate(self):
        return self.passed / max(self.total_tests, 1)

    @property
    def status(self):
        if self.failed == 0:
            return "✅ PASS"
        elif self.passed > 0:
            return "⚠️  PARTIAL"
        return "❌ FAIL"


# ── Synthetic Data Generators ─────────────────────────────────────────────────

class SyntheticDataFactory:
    """
    Generates realistic synthetic data for testing.
    No real satellite data needed — tests work completely offline.
    """

    @staticmethod
    def make_satellite_chip(w: int = 640, h: int = 640, n_buildings: int = 5) -> np.ndarray:
        """Generate a synthetic satellite-like image with fake buildings."""
        # Dark base (terrain)
        img = np.random.randint(40, 90, (h, w, 3), dtype=np.uint8)
        # Add roads (lighter lines)
        cv2.line(img, (0, h//2), (w, h//2), (100, 100, 100), 8)
        cv2.line(img, (w//2, 0), (w//2, h), (100, 100, 100), 8)
        # Add buildings (bright rectangles)
        for _ in range(n_buildings):
            x, y = np.random.randint(50, w-100), np.random.randint(50, h-100)
            bw, bh = np.random.randint(30, 80), np.random.randint(30, 80)
            color = tuple(int(c) for c in np.random.randint(150, 220, 3).tolist())
            cv2.rectangle(img, (x, y), (x+bw, y+bh), color, -1)
            cv2.rectangle(img, (x, y), (x+bw, y+bh), (50,50,50), 1)
        # Add vehicles (small dots)
        for _ in range(np.random.randint(3, 10)):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            cv2.circle(img, (x, y), 5, (180, 140, 60), -1)
        return img

    @staticmethod
    def make_cloudy_chip(w: int = 640, h: int = 640) -> np.ndarray:
        """Generate a chip with heavy cloud cover (should be rejected by quality filter)."""
        img = np.ones((h, w, 3), dtype=np.uint8) * 220
        noise = np.random.randint(-20, 20, (h, w, 3), dtype=np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def make_nodata_chip(w: int = 640, h: int = 640) -> np.ndarray:
        """Generate a mostly black (NoData) chip."""
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (w//4, h//4), (80, 80, 80), -1)
        return img

    @staticmethod
    def make_yolo_label(img_w: int = 640, img_h: int = 640, n_objects: int = 3) -> str:
        """Generate synthetic YOLO format labels."""
        lines = []
        for _ in range(n_objects):
            cls = np.random.randint(0, 4)
            x_c = np.random.uniform(0.1, 0.9)
            y_c = np.random.uniform(0.1, 0.9)
            bw = np.random.uniform(0.05, 0.2)
            bh = np.random.uniform(0.05, 0.2)
            lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")
        return "\n".join(lines)

    @staticmethod
    def make_uncertainty_scores(n: int = 100) -> List[Dict]:
        """Generate fake uncertainty scores for active learning tests."""
        return [{"image": f"chip_r{i:05d}_c{j:05d}.jpg",
                 "uncertainty_score": float(np.random.beta(2, 2)),
                 "lat": 40.0 + np.random.uniform(-1, 1),
                 "lon": -87.0 + np.random.uniform(-1, 1)}
                for i, j in zip(np.random.randint(0, 10000, n), np.random.randint(0, 10000, n))]

    @staticmethod
    def make_chip_index(n: int = 20, scene_id: str = "test_scene") -> List[Dict]:
        """Generate a fake chip index with geo coordinates."""
        chips = []
        for i in range(n):
            row = (i // 5) * 576
            col = (i % 5) * 576
            lat_base = 41.8 + i * 0.001
            lon_base = -87.6 + i * 0.001
            chips.append({
                "chip_name": f"{scene_id}_r{row:05d}_c{col:05d}.jpg",
                "scene_id": scene_id,
                "pixel_row": row, "pixel_col": col,
                "pixel_row_end": row + 640, "pixel_col_end": col + 640,
                "lat_min": lat_base, "lat_max": lat_base + 0.005,
                "lon_min": lon_base, "lon_max": lon_base + 0.005,
                "crs": "EPSG:4326"
            })
        return chips

    @staticmethod
    def make_detections_list(n: int = 20) -> List[Dict]:
        """Generate fake detection results."""
        CLASS_NAMES = ["building", "vehicle", "aircraft", "ship"]
        return [{"chip_name": f"test_scene_r{i:05d}_c{j:05d}.jpg",
                 "class_name": CLASS_NAMES[np.random.randint(0, 4)],
                 "class_id": int(np.random.randint(0, 4)),
                 "confidence": round(float(np.random.uniform(0.3, 0.99)), 3),
                 "bbox": [round(float(v), 4) for v in [np.random.uniform(0.1, 0.9),
                          np.random.uniform(0.1, 0.9), np.random.uniform(0.05, 0.2),
                          np.random.uniform(0.05, 0.2)]]}
                for i, j in zip(np.random.randint(0, 9999, n), np.random.randint(0, 9999, n))]

    @staticmethod
    def make_portfolio(n: int = 5, sandbox_dir: Path = None) -> List[Dict]:
        """Generate fake insurance portfolio with synthetic images."""
        portfolio = []
        for i in range(n):
            if sandbox_dir:
                img = SyntheticDataFactory.make_satellite_chip(n_buildings=np.random.randint(1, 8))
                img_path = sandbox_dir / f"property_{i:03d}.jpg"
                cv2.imwrite(str(img_path), img)
            portfolio.append({
                "id": f"PROP-{1000+i}",
                "address": f"{100+i*10} Main St, Chicago IL 6060{i}",
                "policy_number": f"POL-{2024*100+i}",
                "insured_value": 250000 + i * 50000,
                "image_path": str(sandbox_dir / f"property_{i:03d}.jpg") if sandbox_dir else None
            })
        return portfolio


# ── Individual Tool Tests ─────────────────────────────────────────────────────

class ToolTests:
    """One test method per tool. Each is independent and self-contained."""

    def __init__(self, sandbox_dir: Path, verbose: bool = False):
        self.sandbox = sandbox_dir
        self.verbose = verbose
        self.factory = SyntheticDataFactory()

    def _log(self, msg: str):
        if self.verbose:
            print(f"     {msg}")

    def _run(self, test_name: str, fn: Callable, **kwargs) -> TestResult:
        """Execute a single test function and capture result."""
        start = time.time()
        try:
            output = fn(**kwargs)
            duration = time.time() - start
            return TestResult(
                tool_name="", test_name=test_name,
                passed=True, duration_seconds=round(duration, 3),
                output_summary=output if isinstance(output, dict) else {"result": str(output)[:200]}
            )
        except Exception as e:
            duration = time.time() - start
            return TestResult(
                tool_name="", test_name=test_name,
                passed=False, duration_seconds=round(duration, 3),
                error=str(e), traceback=traceback.format_exc()
            )

    # ── Data Acquisition ────────────────────────────────────────────────────

    def test_chip_generator(self) -> List[TestResult]:
        from chip_generator import ChipGenerator
        results = []
        chip_dir = self.sandbox / "chip_test"
        chip_dir.mkdir(parents=True, exist_ok=True)

        # Create synthetic scene
        scene = self.factory.make_satellite_chip(w=1280, h=1280)
        scene_path = chip_dir / "test_scene.jpg"
        cv2.imwrite(str(scene_path), scene)
        self._log(f"Created synthetic scene: {scene_path}")

        def test_basic_chipping():
            gen = ChipGenerator(chip_size=640, overlap=64)
            n = gen.chip_scene(scene_path, chip_dir / "chips")
            assert n > 0, f"Expected chips, got {n}"
            chips = list((chip_dir / "chips").glob("*.jpg"))
            assert len(chips) > 0, "No chip files created"
            assert (chip_dir / "chips" / "test_scene_chip_index.json").exists(), "No chip index created"
            return {"n_chips": n, "files_created": len(chips)}

        def test_small_scene():
            gen = ChipGenerator(chip_size=640, overlap=64)
            small = self.factory.make_satellite_chip(w=320, h=320)
            small_path = chip_dir / "small_scene.jpg"
            cv2.imwrite(str(small_path), small)
            n = gen.chip_scene(small_path, chip_dir / "small_chips")
            assert n >= 1, "Should generate at least 1 chip for small scene"
            return {"n_chips": n}

        def test_reconstruction():
            gen = ChipGenerator(chip_size=640, overlap=64)
            gen.chip_scene(scene_path, chip_dir / "recon_chips")
            index_path = chip_dir / "recon_chips" / "test_scene_chip_index.json"
            fake_dets = [{"chip_name": "test_scene_r00000_c00000.jpg",
                          "bbox_xywh_norm": [0.5, 0.5, 0.1, 0.1]}]
            scene_dets = gen.reconstruct_scene_detections(fake_dets, "test_scene", index_path)
            assert isinstance(scene_dets, list)
            return {"reconstructed": len(scene_dets)}

        for name, fn in [("basic_chipping", test_basic_chipping),
                         ("small_scene", test_small_scene),
                         ("reconstruction", test_reconstruction)]:
            r = self._run(name, fn)
            self._log(f"{'✅' if r.passed else '❌'} {name}: {r.message or r.error[:80] if not r.passed else r.output_summary}")
            results.append(r)
        return results

    def test_scene_quality_filter(self) -> List[TestResult]:
        from scene_quality_filter import SceneQualityFilter
        results = []
        filter_dir = self.sandbox / "quality_test"
        filter_dir.mkdir(parents=True, exist_ok=True)

        # Create test images
        good_img = self.factory.make_satellite_chip()
        cloudy_img = self.factory.make_cloudy_chip()
        nodata_img = self.factory.make_nodata_chip()
        cv2.imwrite(str(filter_dir / "good.jpg"), good_img)
        cv2.imwrite(str(filter_dir / "cloudy.jpg"), cloudy_img)
        cv2.imwrite(str(filter_dir / "nodata.jpg"), nodata_img)

        def test_score_good():
            f = SceneQualityFilter()
            score = f.score_image(filter_dir / "good.jpg")
            assert score["valid"], f"Good image failed: {score['reject_reason']}"
            assert score["overall_score"] > 0.3
            return score

        def test_reject_cloudy():
            f = SceneQualityFilter(max_cloud_fraction=0.15)
            score = f.score_image(filter_dir / "cloudy.jpg")
            assert not score["valid"] or score["cloud_fraction"] > 0.1, "Cloudy image should be flagged"
            return {"cloud_fraction": score["cloud_fraction"], "valid": score["valid"]}

        def test_reject_nodata():
            f = SceneQualityFilter(max_nodata_fraction=0.05)
            score = f.score_image(filter_dir / "nodata.jpg")
            assert not score["valid"] or score["nodata_fraction"] > 0.3, "NoData image should be flagged"
            return {"nodata_fraction": score["nodata_fraction"]}

        def test_filter_directory():
            f = SceneQualityFilter()
            summary = f.filter_directory(filter_dir, move_rejected=False, save_report=True)
            assert summary["total"] >= 3
            assert (filter_dir / "quality_report.json").exists()
            return summary

        for name, fn in [("score_good_image", test_score_good), ("reject_cloudy", test_reject_cloudy),
                         ("reject_nodata", test_reject_nodata), ("filter_directory", test_filter_directory)]:
            r = self._run(name, fn)
            self._log(f"{'✅' if r.passed else '❌'} {name}")
            results.append(r)
        return results

    def test_spatial_diversity_sampler(self) -> List[TestResult]:
        from spatial_diversity_sampler import SpatialDiversitySampler
        results = []
        sampler_dir = self.sandbox / "sampler_test"
        sampler_dir.mkdir(parents=True, exist_ok=True)

        scores = self.factory.make_uncertainty_scores(100)
        scores_path = sampler_dir / "uncertainty_scores.json"
        scores_path.write_text(json.dumps(scores))

        chip_index = self.factory.make_chip_index(50)
        index_path = sampler_dir / "chip_index.json"
        index_path.write_text(json.dumps(chip_index))

        def test_basic_selection():
            sampler = SpatialDiversitySampler(alpha=0.6)
            score_dict = {s["image"]: s["uncertainty_score"] for s in scores}
            coords = {s["image"]: (s["lat"], s["lon"]) for s in scores}
            selected = sampler.select_diverse_batch(score_dict, coords, budget=20)
            assert len(selected) == 20, f"Expected 20, got {len(selected)}"
            assert all("image" in s for s in selected)
            return {"n_selected": len(selected), "avg_uncertainty": round(np.mean([s["uncertainty_score"] for s in selected]), 4)}

        def test_pure_uncertainty():
            sampler = SpatialDiversitySampler(alpha=1.0)
            score_dict = {s["image"]: s["uncertainty_score"] for s in scores}
            selected = sampler.select_diverse_batch(score_dict, None, budget=10)
            assert len(selected) == 10
            return {"n_selected": len(selected)}

        def test_small_budget():
            sampler = SpatialDiversitySampler()
            score_dict = {s["image"]: s["uncertainty_score"] for s in scores[:5]}
            selected = sampler.select_diverse_batch(score_dict, None, budget=10)
            assert len(selected) <= 5
            return {"n_selected": len(selected)}

        for name, fn in [("basic_selection", test_basic_selection),
                         ("pure_uncertainty_mode", test_pure_uncertainty),
                         ("small_budget", test_small_budget)]:
            r = self._run(name, fn)
            self._log(f"{'✅' if r.passed else '❌'} {name}")
            results.append(r)
        return results

    def test_geojson_exporter(self) -> List[TestResult]:
        from geojson_exporter import GeoJSONExporter
        results = []
        export_dir = self.sandbox / "geojson_test"
        export_dir.mkdir(parents=True, exist_ok=True)

        chip_index = self.factory.make_chip_index(20)
        index_path = export_dir / "chip_index.json"
        index_path.write_text(json.dumps(chip_index))
        detections = self.factory.make_detections_list(30)
        # Align chip names to index
        for i, det in enumerate(detections[:10]):
            det["chip_name"] = chip_index[i % len(chip_index)]["chip_name"]

        def test_basic_export():
            exp = GeoJSONExporter(index_path)
            gj = exp.export_detections(detections, export_dir / "test.geojson")
            assert gj["type"] == "FeatureCollection"
            assert len(gj["features"]) == len(detections)
            assert (export_dir / "test.geojson").exists()
            return {"n_features": len(gj["features"]), "geo_referenced": gj["metadata"]["geo_referenced"]}

        def test_no_chip_index():
            exp = GeoJSONExporter(None)
            gj = exp.export_detections(detections[:5], export_dir / "no_index.geojson")
            assert gj["type"] == "FeatureCollection"
            return {"n_features": len(gj["features"])}

        def test_valid_geojson_schema():
            exp = GeoJSONExporter(index_path)
            gj = exp.export_detections(detections[:3], export_dir / "schema_test.geojson")
            for f in gj["features"]:
                assert "type" in f
                assert "geometry" in f
                assert "properties" in f
                assert f["properties"]["class"] in ["building", "vehicle", "aircraft", "ship", "unknown"]
            return {"schema_valid": True}

        for name, fn in [("basic_export", test_basic_export),
                         ("no_chip_index", test_no_chip_index),
                         ("valid_geojson_schema", test_valid_geojson_schema)]:
            r = self._run(name, fn)
            self._log(f"{'✅' if r.passed else '❌'} {name}")
            results.append(r)
        return results

    def test_change_detection(self) -> List[TestResult]:
        from change_detection_engine import ChangeDetectionEngine
        results = []
        change_dir = self.sandbox / "change_test"
        change_dir.mkdir(parents=True, exist_ok=True)

        before = self.factory.make_satellite_chip(n_buildings=4)
        after = self.factory.make_satellite_chip(n_buildings=6)
        cv2.imwrite(str(change_dir / "before.jpg"), before)
        cv2.imwrite(str(change_dir / "after.jpg"), after)

        def test_image_alignment():
            engine = ChangeDetectionEngine.__new__(ChangeDetectionEngine)
            engine.conf_threshold = 0.25
            engine.iou_match_threshold = 0.4
            engine.CLASS_NAMES = {0:"building",1:"vehicle",2:"aircraft",3:"ship"}
            engine.SIGNIFICANCE = {"building":{"appeared":"high","disappeared":"high","moved":"medium"},
                                   "vehicle":{"appeared":"low","disappeared":"low","moved":"low"}}
            engine._model = None
            aligned_b, aligned_a = engine._align_images(before.copy(), after.copy())
            assert aligned_b.shape == before.shape
            assert aligned_a.shape == before.shape
            return {"aligned": True, "shape": list(aligned_b.shape)}

        def test_object_matching():
            from change_detection_engine import DetectedObject, ChangeEvent
            engine = ChangeDetectionEngine.__new__(ChangeDetectionEngine)
            engine.conf_threshold = 0.25
            engine.iou_match_threshold = 0.4
            engine.CLASS_NAMES = {0:"building",1:"vehicle"}
            engine.SIGNIFICANCE = {"building":{"appeared":"high","disappeared":"high","moved":"medium"},
                                   "vehicle":{"appeared":"low","disappeared":"low","moved":"low"}}
            before_objs = [DetectedObject("building", 0, 0.9, [0.3, 0.3, 0.1, 0.1]),
                           DetectedObject("building", 0, 0.85, [0.7, 0.7, 0.1, 0.1])]
            after_objs = [DetectedObject("building", 0, 0.88, [0.3, 0.3, 0.1, 0.1]),
                          DetectedObject("building", 0, 0.91, [0.5, 0.5, 0.1, 0.1])]
            changes = engine._match_objects(before_objs, after_objs)
            change_types = [c.change_type for c in changes]
            assert "unchanged" in change_types or "appeared" in change_types
            return {"n_changes": len(changes), "types": list(set(change_types))}

        def test_report_structure():
            # Test without real model — just structure validation
            engine = ChangeDetectionEngine.__new__(ChangeDetectionEngine)
            engine.conf_threshold = 0.25
            engine.iou_match_threshold = 0.4
            engine.CLASS_NAMES = {0:"building",1:"vehicle",2:"aircraft",3:"ship"}
            engine.SIGNIFICANCE = {"building":{"appeared":"high","disappeared":"high","moved":"medium"},
                                   "vehicle":{"appeared":"low","disappeared":"low","moved":"low"}}
            engine._model = None

            # Mock predict
            from change_detection_engine import DetectedObject
            def mock_detect(img): return []
            engine._detect = mock_detect

            changes = engine._match_objects([], [])
            assert isinstance(changes, list)
            return {"test_passed": True}

        for name, fn in [("image_alignment", test_image_alignment),
                         ("object_matching", test_object_matching),
                         ("report_structure", test_report_structure)]:
            r = self._run(name, fn)
            self._log(f"{'✅' if r.passed else '❌'} {name}")
            results.append(r)
        return results

    def test_bootstrapping_dashboard(self) -> List[TestResult]:
        from bootstrapping_dashboard import BootstrappingDashboard
        results = []
        dash_dir = self.sandbox / "dashboard_test"
        dash_dir.mkdir(parents=True, exist_ok=True)

        def test_demo_data_generation():
            dash = BootstrappingDashboard()
            assert len(dash.iterations) > 0
            for it in dash.iterations:
                assert "iteration" in it
                assert "map50" in it
            return {"n_iterations": len(dash.iterations)}

        def test_render_export():
            dash = BootstrappingDashboard()
            export_path = dash_dir / "test_dashboard.png"
            dash.render(export_path=export_path)
            assert export_path.exists()
            assert export_path.stat().st_size > 10000
            return {"file_size_kb": round(export_path.stat().st_size / 1024, 1)}

        for name, fn in [("demo_data_generation", test_demo_data_generation),
                         ("render_to_file", test_render_export)]:
            r = self._run(name, fn)
            self._log(f"{'✅' if r.passed else '❌'} {name}")
            results.append(r)
        return results

    def test_batch_processor(self) -> List[TestResult]:
        from batch_processor import BatchProcessor
        results = []
        batch_dir = self.sandbox / "batch_test"
        batch_dir.mkdir(parents=True, exist_ok=True)

        # Create test images
        img_dir = batch_dir / "images"
        img_dir.mkdir(exist_ok=True)
        for i in range(5):
            img = self.factory.make_satellite_chip(n_buildings=np.random.randint(1, 6))
            cv2.imwrite(str(img_dir / f"test_{i:03d}.jpg"), img)

        def test_job_submission():
            images = list(img_dir.glob("*.jpg"))
            proc = BatchProcessor.__new__(BatchProcessor)
            proc.model_path = "fake_path.pt"
            proc._model = None
            proc._db = None
            # Override db to avoid Supabase connection
            proc._get_db = lambda: (_ for _ in ()).throw(Exception("No DB in sandbox"))
            job_id = proc.submit_job(images, "test_client", "Sandbox Test Job")
            assert job_id.startswith("job_")
            job_path = (Path(os.getenv("DATA_DIR","./data")) / "batch_jobs" / f"{job_id}.json")
            assert job_path.exists()
            return {"job_id": job_id, "n_images": len(images)}

        def test_status_check():
            proc = BatchProcessor.__new__(BatchProcessor)
            proc.model_path = "fake_path.pt"
            proc._model = None
            proc._db = None
            status = proc.get_status("nonexistent_job_id")
            assert status["status"] == "not_found"
            return {"status": status["status"]}

        for name, fn in [("job_submission", test_job_submission),
                         ("status_check", test_status_check)]:
            r = self._run(name, fn)
            self._log(f"{'✅' if r.passed else '❌'} {name}")
            results.append(r)
        return results

    def test_coverage_report_generator(self) -> List[TestResult]:
        from coverage_report_generator import CoverageReportGenerator
        results = []
        report_dir = self.sandbox / "report_test"
        report_dir.mkdir(parents=True, exist_ok=True)

        portfolio = self.factory.make_portfolio(5, report_dir)

        def test_html_generation():
            gen = CoverageReportGenerator(model_path=None)
            out = report_dir / "test_report.html"
            gen.generate_report(portfolio, "Test Insurance Co", out)
            assert out.exists()
            content = out.read_text()
            assert "Test Insurance Co" in content
            assert "FeatureCollection" not in content  # Should be HTML, not GeoJSON
            assert "property-card" in content
            return {"file_size_kb": round(out.stat().st_size/1024, 1)}

        def test_empty_portfolio():
            gen = CoverageReportGenerator(model_path=None)
            out = report_dir / "empty_report.html"
            gen.generate_report([], "Empty Co", out)
            assert out.exists()
            return {"generated": True}

        for name, fn in [("html_generation", test_html_generation),
                         ("empty_portfolio", test_empty_portfolio)]:
            r = self._run(name, fn)
            self._log(f"{'✅' if r.passed else '❌'} {name}")
            results.append(r)
        return results


# ── Sandbox Runner ────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "chip_generator": "test_chip_generator",
    "scene_quality_filter": "test_scene_quality_filter",
    "spatial_diversity_sampler": "test_spatial_diversity_sampler",
    "geojson_exporter": "test_geojson_exporter",
    "change_detection": "test_change_detection",
    "bootstrapping_dashboard": "test_bootstrapping_dashboard",
    "batch_processor": "test_batch_processor",
    "coverage_report": "test_coverage_report_generator",
}


def run_tool_tests(tool_name: str, verbose: bool = False, keep_data: bool = False) -> ToolReport:
    """Run all tests for a specific tool in an isolated sandbox."""
    sandbox_dir = SANDBOX_DIR / f"{tool_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🧪 SANDBOX TEST: {tool_name}")
    print(f"   Directory: {sandbox_dir}")
    print(f"{'='*60}")

    tester = ToolTests(sandbox_dir, verbose=verbose)
    test_method_name = TOOL_REGISTRY.get(tool_name)

    if not test_method_name or not hasattr(tester, test_method_name):
        print(f"❌ Unknown tool: {tool_name}")
        print(f"   Available: {', '.join(TOOL_REGISTRY.keys())}")
        return ToolReport(tool_name=tool_name, total_tests=0, passed=0, failed=1,
                         warnings=0, duration_seconds=0)

    start = time.time()
    try:
        results = getattr(tester, test_method_name)()
    except Exception as e:
        print(f"❌ Test suite crashed: {e}")
        traceback.print_exc()
        results = [TestResult(tool_name=tool_name, test_name="suite_crash",
                              passed=False, duration_seconds=0, error=str(e))]

    duration = time.time() - start
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    warnings = sum(len(r.warnings) for r in results)

    for r in results:
        r.tool_name = tool_name

    report = ToolReport(
        tool_name=tool_name, total_tests=len(results),
        passed=passed, failed=failed, warnings=warnings,
        duration_seconds=round(duration, 2), results=results,
        sandbox_dir=str(sandbox_dir)
    )

    # Print results
    print(f"\n📋 Results:")
    for r in results:
        icon = "✅" if r.passed else "❌"
        timing = f"{r.duration_seconds:.2f}s"
        msg = "" if r.passed else f" — {r.error[:60]}"
        print(f"   {icon} {r.test_name:<40} [{timing}]{msg}")

    print(f"\n{'─'*60}")
    print(f"{report.status}  {passed}/{len(results)} tests passed in {duration:.2f}s")

    if failed > 0:
        print(f"\n🔍 Failure Details:")
        for r in results:
            if not r.passed:
                print(f"\n   ❌ {r.test_name}")
                print(f"   Error: {r.error}")
                if verbose and r.traceback:
                    print(f"   Traceback:\n{r.traceback}")

    # Save report
    report_path = sandbox_dir / "test_report.json"
    report_dict = asdict(report)
    report_path.write_text(json.dumps(report_dict, indent=2))
    print(f"\n📄 Full report: {report_path}")

    if not keep_data:
        shutil.rmtree(sandbox_dir)
        print(f"🧹 Sandbox cleaned up")
    else:
        print(f"📁 Sandbox data kept: {sandbox_dir}")

    return report


def run_all_tests(verbose: bool = False, keep_data: bool = False) -> Dict:
    """Run tests for all tools and produce a master report."""
    print(f"\n{'='*60}")
    print(f"🧪 FULL SANDBOX TEST SUITE")
    print(f"   Tools: {len(TOOL_REGISTRY)}")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    all_reports = {}
    total_start = time.time()

    for tool_name in TOOL_REGISTRY:
        report = run_tool_tests(tool_name, verbose=verbose, keep_data=keep_data)
        all_reports[tool_name] = report

    total_duration = time.time() - total_start

    # Master summary
    total_passed = sum(r.passed for r in all_reports.values())
    total_failed = sum(r.failed for r in all_reports.values())
    tools_passing = sum(1 for r in all_reports.values() if r.failed == 0)

    print(f"\n{'='*60}")
    print(f"📊 MASTER SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Tool':<35} {'Status':<15} {'Tests':<10} {'Time'}")
    print(f"{'─'*65}")
    for name, report in all_reports.items():
        print(f"{name:<35} {report.status:<15} {report.passed}/{report.total_tests:<8} {report.duration_seconds:.1f}s")

    print(f"\n{'─'*65}")
    print(f"{'TOTAL':<35} {'':15} {total_passed}/{total_passed+total_failed:<8} {total_duration:.1f}s")
    print(f"\nTools fully passing: {tools_passing}/{len(TOOL_REGISTRY)}")

    summary_path = SANDBOX_DIR / f"master_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "total_duration_seconds": round(total_duration, 2),
        "tools_tested": len(TOOL_REGISTRY),
        "tools_passing": tools_passing,
        "total_tests_passed": total_passed,
        "total_tests_failed": total_failed,
        "by_tool": {k: {"status": v.status, "passed": v.passed, "failed": v.failed}
                    for k, v in all_reports.items()}
    }, indent=2))
    print(f"\n📄 Master report: {summary_path}")
    return all_reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sandbox tool tester")
    parser.add_argument("--tool", choices=list(TOOL_REGISTRY.keys()), help="Specific tool to test")
    parser.add_argument("--all", action="store_true", help="Test all tools")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--keep-data", action="store_true", help="Keep sandbox data after test")
    parser.add_argument("--list", action="store_true", help="List available tools")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable tools:")
        for name in TOOL_REGISTRY:
            print(f"  {name}")
    elif args.all:
        run_all_tests(verbose=args.verbose, keep_data=args.keep_data)
    elif args.tool:
        run_tool_tests(args.tool, verbose=args.verbose, keep_data=args.keep_data)
    else:
        parser.print_help()
