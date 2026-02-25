# Satellite Bootstrap — Tools Directory
# CLAUDE.md for Visual Studio Code + Claude Code

## What This Directory Is
12 production tools that extend the core bootstrapping pipeline.
Each tool is standalone, independently testable, and integrates with
the Supabase backend and YOLOv8 model from the main pipeline.

**Always test a tool in the sandbox before running it on real data.**

---

## Directory Structure

```
tools/
├── data_acquisition/
│   ├── sentinel2_fetcher.py       ← Fetch Sentinel-2 imagery from ESA Copernicus
│   ├── scene_quality_filter.py    ← Reject cloudy/corrupt imagery before pipeline
│   └── chip_generator.py          ← Tile large scenes into 640x640 YOLO chips
│
├── model_performance/
│   ├── geo_aware_evaluator.py     ← Map model errors geographically
│   ├── rotation_invariance_tester.py ← Test detection across all orientations
│   └── confidence_calibrator.py   ← Temperature scaling for calibrated scores
│
├── active_learning/
│   ├── spatial_diversity_sampler.py  ← Geography-aware query batch selection
│   ├── pseudo_label_scorer.py        ← Score pseudo-label quality before human review
│   └── bootstrapping_dashboard.py   ← Real-time efficiency visualization
│
├── commercial/
│   ├── change_detection_engine.py    ← Compare same location across two timestamps
│   ├── geojson_exporter.py           ← Convert detections to ArcGIS/QGIS-ready GeoJSON
│   ├── batch_processor.py            ← Async batch processing for large portfolios
│   └── coverage_report_generator.py ← HTML reports for insurance customers
│
└── sandbox/
    └── tool_tester.py               ← TEST ALL TOOLS BEFORE PRODUCTION USE
```

---

## CRITICAL: Always Test Before Deploying

Before running any tool on real imagery, labels, or the Supabase database:

```bash
# Test a single tool
python tools/sandbox/tool_tester.py --tool chip_generator

# Test everything at once
python tools/sandbox/tool_tester.py --all

# Keep sandbox files to inspect inputs/outputs
python tools/sandbox/tool_tester.py --tool geojson_exporter --keep-data

# See verbose output + full tracebacks
python tools/sandbox/tool_tester.py --tool change_detection --verbose
```

The tester creates a temporary sandbox at `/tmp/satellite_sandbox_{timestamp}/`,
generates purely synthetic data, imports each tool directly (no subprocess),
and cleans up automatically unless `--keep-data` is passed.

---

## Tool Quick Reference

### data_acquisition/chip_generator.py
Tiles a large GeoTIFF or JPEG scene into overlapping 640×640 chips.
Writes a `{scene_id}_chip_index.json` alongside the chips.

```bash
python tools/data_acquisition/chip_generator.py \
    --input data/raw/scene.tif \
    --output data/unlabeled \
    --chip-size 640 \
    --overlap 64
```

**Public API:**
```python
from tools.data_acquisition.chip_generator import generate_chips
chip_index = generate_chips(scene_path, output_dir, chip_size=640, overlap=64)
```

---

### data_acquisition/scene_quality_filter.py
Rejects chips that are too cloudy, noisy, low-contrast, or no-data.
Writes a `quality_report.json` to the input directory.

```bash
python tools/data_acquisition/scene_quality_filter.py \
    --input-dir data/unlabeled \
    --max-cloud 0.20 \
    --move-rejected
```

**Public API:**
```python
from tools.data_acquisition.scene_quality_filter import run_filter
report = run_filter(input_dir, max_cloud=0.20, move_rejected=False)
```

---

### data_acquisition/sentinel2_fetcher.py
Downloads Sentinel-2 L2A scenes from ESA Copernicus Open Access Hub.
Requires `COPERNICUS_USER` and `COPERNICUS_PASSWORD` in `.env`.

```bash
python tools/data_acquisition/sentinel2_fetcher.py \
    --bbox -73.99 40.74 -73.97 40.76 \
    --date-range 2024-01-01 2024-01-31 \
    --output data/raw/sentinel2
```

**Dry-run (sandbox-safe):**
```python
from tools.data_acquisition.sentinel2_fetcher import fetch_scene
result = fetch_scene(bbox=(...), date_range=(...), output_dir=..., dry_run=True)
```

---

### model_performance/geo_aware_evaluator.py
Runs YOLOv8 inference on a validation set, matches predictions to ground
truth, and generates a geographic error heatmap showing where the model
makes the most false negatives.

```bash
python tools/model_performance/geo_aware_evaluator.py \
    --model models/baseline_v1/weights/best.pt \
    --val-dir data/yolo_format/val/images \
    --label-dir data/yolo_format/val/labels
```

Outputs: `geo_evaluation_report.json`, `geo_error_map.png` (or
`error_distribution.png` if no geo data is available).

---

### model_performance/rotation_invariance_tester.py
Rotates each chip at N angles (default: 0°, 45°, 90°, … 315°) and
measures how consistently the model detects the same buildings.

```bash
python tools/model_performance/rotation_invariance_tester.py \
    --images-dir data/yolo_format/val/images \
    --model models/baseline_v1/weights/best.pt \
    --angles 0 45 90 135 180 225 270 315
```

---

### model_performance/confidence_calibrator.py
Applies temperature scaling to raw YOLOv8 confidence scores so that a
score of 0.8 actually means the model is right ~80% of the time.

```bash
python tools/model_performance/confidence_calibrator.py \
    --scores-path outputs/raw_scores.json \
    --output calibrated_scores.json
```

---

### active_learning/spatial_diversity_sampler.py
Selects the next annotation batch from the unlabeled pool using both
uncertainty (model score) and spatial diversity (geographic spread) so
that the annotator sees the widest possible variety of scenes.

```bash
python tools/active_learning/spatial_diversity_sampler.py \
    --scores-path outputs/uncertainty_scores.json \
    --output outputs/next_batch.json \
    --batch-size 20
```

---

### active_learning/bootstrapping_dashboard.py
Renders an HTML (or PNG) dashboard showing per-round mAP50, precision,
recall, labeling efficiency, and projected convergence.

```bash
python tools/active_learning/bootstrapping_dashboard.py \
    --metrics-path outputs/metrics.json \
    --output-dir outputs/dashboard
```

---

### commercial/change_detection_engine.py
Compares two chips of the same geographic location taken at different
times and reports newly appeared / disappeared building detections.

```bash
python tools/commercial/change_detection_engine.py \
    --before data/scenes/site_2023.jpg \
    --after  data/scenes/site_2024.jpg \
    --output outputs/change_report.json
```

---

### commercial/geojson_exporter.py
Converts a detections JSON (produced by the inference pipeline) into a
FeatureCollection GeoJSON that can be loaded directly into QGIS, ArcGIS,
or Mapbox.

```bash
python tools/commercial/geojson_exporter.py \
    --detections outputs/detections.json \
    --chip-index data/chips/scene_chip_index.json \
    --output outputs/detections.geojson
```

---

### commercial/batch_processor.py
Async worker that iterates over a portfolio JSON and runs inference on
every property image, writing per-property result JSONs.

```bash
python tools/commercial/batch_processor.py \
    --portfolio data/portfolio.json \
    --images-dir data/portfolio_images \
    --output-dir outputs/batch_results
```

---

### commercial/coverage_report_generator.py
Generates a self-contained HTML report for an insurance portfolio
summarising detection counts, risk scores, and a map for each property.

```bash
python tools/commercial/coverage_report_generator.py \
    --portfolio outputs/batch_results/portfolio_summary.json \
    --output reports/coverage_report.html
```

---

## Environment Variables

All tools read from `/Users/kahlil/satellite-project/.env`.
Required variables:

| Variable              | Used by                              |
|-----------------------|--------------------------------------|
| `SUPABASE_URL`        | all tools (non-fatal if absent)      |
| `SUPABASE_KEY`        | all tools (non-fatal if absent)      |
| `MODEL_PATH`          | geo_aware_evaluator (CLI default)    |
| `COPERNICUS_USER`     | sentinel2_fetcher                    |
| `COPERNICUS_PASSWORD` | sentinel2_fetcher                    |

All Supabase writes are wrapped in try/except and will print a warning
then fall back to local-only mode if credentials are missing or invalid.

---

## Adding a New Tool

1. Create the file in the appropriate subdirectory.
2. Expose a single public entry-point function (e.g. `run_filter`, `generate_chips`).
3. Keep Supabase writes non-fatal (wrap in try/except).
4. Add a `test_<toolname>` function to `tools/sandbox/tool_tester.py` and
   register it in `TOOL_TESTS`.
5. Run `python tools/sandbox/tool_tester.py --tool <toolname>` and confirm PASS.
6. Document the CLI and public API in this file.

---

## Common Pitfalls

- **Never pass a real model path to the sandbox tester.** The tester is
  designed to work without loading any `.pt` weights. Tools that require a
  model should accept `model_path=None` and short-circuit gracefully.

- **Never hardcode absolute paths in tool source.** Use `Path(__file__)` as
  an anchor or accept paths as parameters. The one exception is the `.env`
  loader at the top of each file (which uses the project-absolute path).

- **Supabase writes must be non-fatal.** Every `supabase.table(...)` call
  must be inside a try/except that prints a warning and continues.

- **Do not call `sys.exit()` inside a library function.** Only call
  `sys.exit()` inside a `main()` function that is guarded by
  `if __name__ == "__main__"` or a dedicated CLI entry point.

- **chip_index.json format.** The chip_generator writes a *nested* format
  (`{scene_id, chips: [{filename, row, col, geo_bounds, ...}]}`).
  The geo_aware_evaluator expects a *flat* format (`{stem: {lon, lat}}`).
  Use the flat format when writing chip_index.json alongside val images.

---

## Never Do

- Run any tool directly on `data/labeled/` without backing it up first.
- Run `batch_processor` against the production Supabase without a dry-run flag.
- Commit `.env` or any file containing credentials to git.
- Skip the sandbox tester when a tool has been modified.
- Use `subprocess` to call another tool from within a tool — import and call
  the public function directly instead.
