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

The sandbox tester generates synthetic data and validates each tool
without touching real data, models, or the database.

```bash
# Test one tool
python tools/sandbox/tool_tester.py --tool chip_generator

# Test all tools
python tools/sandbox/tool_tester.py --all

# Verbose output to see exactly what failed
python tools/sandbox/tool_tester.py --tool geojson_exporter --verbose

# Keep sandbox files for manual inspection
python tools/sandbox/tool_tester.py --tool change_detection --keep-data

# List all testable tools
python tools/sandbox/tool_tester.py --list
```

**Available tool names for --tool flag:**
- chip_generator
- scene_quality_filter
- spatial_diversity_sampler
- geojson_exporter
- change_detection
- bootstrapping_dashboard
- batch_processor
- coverage_report

---

## Tool Quick Reference

### Data Acquisition

**sentinel2_fetcher.py**
- Fetches from ESA Copernicus Data Space (free account required)
- Filters by cloud cover before downloading
- Auto-chips scenes via chip_generator
```bash
python tools/data_acquisition/sentinel2_fetcher.py \
  --bbox -87.7 41.8 -87.5 42.0 \
  --start 2024-01-01 --end 2024-06-01 \
  --max-cloud 20 --limit 50
```
- Requires: COPERNICUS_USER, COPERNICUS_PASSWORD in .env
- Output: data/unlabeled/ (chipped imagery ready for AL loop)

**scene_quality_filter.py**
- Run BEFORE the active learning loop on any new imagery
- Rejects cloudy, NoData, low-contrast, and low-information chips
```bash
python tools/data_acquisition/scene_quality_filter.py \
  --input-dir data/unlabeled --max-cloud 0.20 --move-rejected
```
- Output: quality_report.json in input directory, optionally moves rejected to data/rejected/

**chip_generator.py**
- Tiles GeoTIFF or JPEG scenes into overlapping 640x640 chips
- Preserves geo metadata for coordinate reconstruction
- Handles scenes of any size (auto-pads edges)
```bash
python tools/data_acquisition/chip_generator.py \
  --input path/to/scene.tif --output data/unlabeled \
  --chip-size 640 --overlap 64
```
- Output: chips + {scene_id}_chip_index.json per scene

---

### Model Performance

**geo_aware_evaluator.py**
- Runs mAP evaluation with geographic error mapping
- Generates geo_error_map.png showing where FP/FN cluster
- Requires trained model at MODEL_PATH
```bash
python tools/model_performance/geo_aware_evaluator.py \
  --model models/baseline_v1/weights/best.pt \
  --val-dir data/yolo_format/val/images \
  --label-dir data/yolo_format/val/labels
```

**rotation_invariance_tester.py**
- Tests detection performance at 8 orientations (0, 45, 90...315°)
- Outputs polar plot showing orientation blind spots
- Critical for vehicle/aircraft class performance
```bash
python tools/model_performance/rotation_invariance_tester.py \
  --model models/baseline_v1/weights/best.pt \
  --test-dir data/yolo_format/val/images
```
- Output: rotation_invariance_report.json + polar plot PNG

**confidence_calibrator.py**
- Finds optimal temperature T for calibrated confidence scores
- Saves T to models/calibration_temperature.json
- Apply in inference_server.py: conf_calibrated = conf / T
```bash
python tools/model_performance/confidence_calibrator.py \
  --model models/baseline_v1/weights/best.pt \
  --val-dir data/yolo_format/val/images \
  --label-dir data/yolo_format/val/labels
```

---

### Active Learning

**spatial_diversity_sampler.py**
- Replaces the default uncertainty sampler for geographically diverse queries
- alpha=0.6 = 60% uncertainty weight, 40% spatial diversity weight
- Reads uncertainty scores from active learning loop output
```bash
python tools/active_learning/spatial_diversity_sampler.py \
  --unlabeled-dir data/unlabeled \
  --uncertainty-scores data/bootstrapped/uncertainty_scores.json \
  --budget 50 --alpha 0.6
```
- Output: diverse_query_batch.json

**pseudo_label_scorer.py**
- Scores pseudo-labels using ensemble disagreement BEFORE human review
- Sorts into: quick_confirm / normal_review / careful_review
- Requires 1+ model checkpoints (use best.pt if only one available)
```bash
python tools/active_learning/pseudo_label_scorer.py \
  --query-dir data/bootstrapped/iteration_01_query \
  --checkpoints models/baseline_v1/weights/best.pt \
  --output data/bootstrapped/scored_queue.json
```

**bootstrapping_dashboard.py**
- Renders efficiency dashboard from Supabase or local JSON files
- Pulls automatically from bootstrap_iterations table
- Use --export-png for demo/sales presentations
```bash
python tools/active_learning/bootstrapping_dashboard.py
python tools/active_learning/bootstrapping_dashboard.py --export-png reports/demo_dashboard.png
```

---

### Commercial

**change_detection_engine.py**
- Compares same location across two timestamps
- Uses ORB feature matching to align images before comparison
- Classifies: appeared / disappeared / moved / unchanged
- HIGH VALUE: primary insurance use case
```bash
python tools/commercial/change_detection_engine.py \
  --before data/scene_2023_01.jpg \
  --after data/scene_2024_01.jpg \
  --model models/baseline_v1/weights/best.pt \
  --output data/change_reports
```
- Output: JSON report + side-by-side visualization PNG

**geojson_exporter.py**
- Converts YOLO pixel detections to geographic GeoJSON
- Output loads directly in ArcGIS, QGIS, Google Earth
- Requires chip index for geo-referenced output
- Also exports change detection reports as GeoJSON
```bash
python tools/commercial/geojson_exporter.py \
  --detections data/detections.json \
  --chip-index data/unlabeled/scene_chip_index.json \
  --output data/exports/detections.geojson
```

**batch_processor.py**
- Processes large property portfolios asynchronously
- Three modes: submit / worker / status
- Worker polls Supabase queue; falls back to local JSON if offline
```bash
# Submit job
python tools/commercial/batch_processor.py submit \
  --images-dir data/customer_portfolio --client-id abc123

# Run worker (keep running to process queue)
python tools/commercial/batch_processor.py worker

# Check status
python tools/commercial/batch_processor.py status --job-id job_abc123
```
- Output: data/batch_results/{job_id}/detections.geojson

**coverage_report_generator.py**
- Generates HTML property coverage reports for insurance customers
- Portfolio JSON format: list of {id, address, image_path, policy_number, insured_value}
- Revenue-generating deliverable: charge per report
```bash
python tools/commercial/coverage_report_generator.py \
  --portfolio data/customer_portfolio.json \
  --client-name "Acme Insurance Co" \
  --output reports/acme_q4_2024.html
```
- Output: styled HTML report with annotated imagery

---

## Integration With Main Pipeline

Tools connect to the main pipeline at these points:

```
Sentinel2Fetcher → [chip_generator] → data/unlabeled/
                                            ↓
SceneQualityFilter → filters unlabeled/ before active learning
                                            ↓
SpatialDiversitySampler → replaces script 05's default sampler
PseudoLabelScorer → runs on query batch before Roboflow upload
BootstrappingDashboard → reads from bootstrap_iterations table
                                            ↓
[after training converges]
                                            ↓
GeoAwareEvaluator → evaluate on geo-tagged val set
RotationInvarianceTester → find orientation blind spots
ConfidenceCalibrator → calibrate before commercial deployment
                                            ↓
[commercial deployment]
                                            ↓
ChangeDetectionEngine → process before/after pairs
GeoJSONExporter → export results for customer GIS tools
BatchProcessor → handle large portfolio jobs
CoverageReportGenerator → generate customer deliverables
```

---

## Shared Inputs/Outputs

| File | Created By | Used By |
|---|---|---|
| data/unlabeled/*.jpg | chip_generator, sentinel2_fetcher | scene_quality_filter, script 05 |
| data/unlabeled/*_chip_index.json | chip_generator | geojson_exporter, spatial_diversity_sampler |
| data/bootstrapped/uncertainty_scores.json | script 05 | spatial_diversity_sampler |
| data/bootstrapped/iteration_*/metrics.json | script 05 | bootstrapping_dashboard |
| models/calibration_temperature.json | confidence_calibrator | inference_server.py |
| data/exports/*.geojson | geojson_exporter | Customer GIS tools |
| data/reports/*.html | coverage_report_generator | Customer delivery |

---

## Dependencies (additions to main requirements.txt)

```
sentinelsat>=0.14.0    # sentinel2_fetcher (optional — only if fetching live data)
rasterio>=1.3.0        # chip_generator (for GeoTIFF support)
matplotlib>=3.7.0      # all visualization tools
scipy>=1.11.0          # confidence_calibrator
```

Install all tool dependencies:
```bash
pip install sentinelsat rasterio matplotlib scipy --break-system-packages
```

---

## Coding Conventions (same as main pipeline)

- All tools are standalone: `python tools/category/tool_name.py`
- Use pathlib.Path, not os.path
- Use tqdm for all loops over files
- Print ✅ success, ❌ error, ⚠️ warning, 📊 stats
- Supabase writes are always non-fatal (fall back to local)
- Never commit data/ or models/ directories

---

## Environment Variables Needed

All tools inherit from the main .env file:
```
SUPABASE_URL                 # Required for dashboard, batch processor
SUPABASE_KEY                 # Required for dashboard, batch processor
SUPABASE_SERVICE_ROLE_KEY    # Required for batch processor worker
COPERNICUS_USER              # Required for sentinel2_fetcher
COPERNICUS_PASSWORD          # Required for sentinel2_fetcher
MODEL_PATH                   # Default model path for all tools
DATA_DIR                     # Default: ./data
MODELS_DIR                   # Default: ./models
```

---

## Never Do (applies to all tools)

- ❌ Run any tool on real data before running sandbox test
- ❌ Skip scene_quality_filter before feeding imagery to active learning loop
- ❌ Use spatial_diversity_sampler with alpha=0 (pure diversity = ignores uncertainty)
- ❌ Deploy inference_server.py to customers without running confidence_calibrator first
- ❌ Send coverage reports to real customers using demo/model-not-found mode output
- ❌ Run batch_processor worker with service role key exposed in logs
