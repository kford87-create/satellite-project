"""
coverage_report_generator.py
-----------------------------
Generate a self-contained, styled HTML coverage report for a customer
property portfolio using YOLOv8 satellite imagery inference.

CLI:
    python tools/commercial/coverage_report_generator.py \\
      --portfolio data/customer_portfolio.json \\
      --client-name "Acme Insurance Co" \\
      --output reports/acme_q4_2024.html
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw
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

CONF_THRESHOLD  = float(os.environ.get("CONF_THRESHOLD", "0.25"))
THUMBNAIL_WIDTH = 400  # px

# Risk thresholds
RISK_ALERT_MIN_DETECTIONS  = 10
RISK_REVIEW_MIN_DETECTIONS = 4

# Dark theme palette
BG_COLOR      = "#0a0e1a"
SURFACE_COLOR = "#111827"
ACCENT_COLOR  = "#3b82f6"

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
# Supabase write (non-fatal)
# ---------------------------------------------------------------------------

def _supabase_log_report(
    client_name: str,
    output_path: Path,
    property_count: int,
    alert_count: int,
) -> None:
    try:
        if supabase is None:
            return
        supabase.table("coverage_reports").insert(
            {
                "client_name":    client_name,
                "output_path":    str(output_path),
                "property_count": property_count,
                "alert_count":    alert_count,
                "generated_at":   datetime.now(timezone.utc).isoformat(),
            }
        ).execute()
        print("✅ Report metadata written to Supabase")
    except Exception as exc:
        print(f"⚠️  Supabase write skipped (falling back to local only): {exc}")


# ---------------------------------------------------------------------------
# YOLOv8 inference
# ---------------------------------------------------------------------------

def _run_inference(model: Any, image_path: Path) -> list[dict]:
    """
    Run YOLOv8 on *image_path*.
    Returns list of {class_name, confidence, bbox_xyxy}.
    """
    try:
        results = model(str(image_path), verbose=False)
    except Exception as exc:
        print(f"⚠️  Inference failed for {image_path.name}: {exc}")
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
            class_name = (
                names[cls_id] if names and cls_id in names else str(cls_id)
            )
            detections.append(
                {
                    "class_name":  class_name,
                    "confidence":  round(conf, 4),
                    "bbox_xyxy":   [x1, y1, x2, y2],
                }
            )
    return detections


# ---------------------------------------------------------------------------
# Image annotation + base64 thumbnail
# ---------------------------------------------------------------------------

BOX_COLORS = [
    (59, 130, 246),   # blue
    (34, 197, 94),    # green
    (239, 68, 68),    # red
    (234, 179, 8),    # yellow
    (168, 85, 247),   # purple
    (249, 115, 22),   # orange
    (20, 184, 166),   # teal
    (236, 72, 153),   # pink
]


def _class_color(class_name: str) -> tuple[int, int, int]:
    return BOX_COLORS[hash(class_name) % len(BOX_COLORS)]


def _annotate_image(img: Image.Image, detections: list[dict]) -> Image.Image:
    """Draw detection boxes on a PIL image copy."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        color = _class_color(det["class_name"])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        draw.text((x1 + 2, max(0, y1 - 12)), label, fill=color)
    return out


def _image_to_base64(img: Image.Image, max_width: int = THUMBNAIL_WIDTH) -> str:
    """Resize image to *max_width* and return base64-encoded JPEG data URI."""
    if img.width > max_width:
        ratio   = max_width / img.width
        new_h   = int(img.height * ratio)
        img     = img.resize((max_width, new_h), Image.LANCZOS)

    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------

def _classify_risk(detections: list[dict], demo: bool = False) -> str:
    """
    Return risk level string based on detection count.
    demo=True triggers random-ish risk for placeholder data.
    """
    n = len(detections)
    if demo:
        # Spread across all three for demo variety
        if n > 6:
            return "alert"
        if n > 2:
            return "review"
        return "normal"
    if n >= RISK_ALERT_MIN_DETECTIONS:
        return "alert"
    if n >= RISK_REVIEW_MIN_DETECTIONS:
        return "review"
    return "normal"


RISK_INDICATOR = {
    "normal": "🟢 Normal",
    "review": "🟡 Review",
    "alert":  "🔴 Alert",
}

RISK_BADGE_COLOR = {
    "normal": "#16a34a",
    "review": "#ca8a04",
    "alert":  "#dc2626",
}


# ---------------------------------------------------------------------------
# Demo placeholder data
# ---------------------------------------------------------------------------

def _make_demo_detections(prop_id: str) -> list[dict]:
    """Generate deterministic fake detections for demo mode."""
    rng = np.random.default_rng(seed=abs(hash(prop_id)) % (2**31))
    n = int(rng.integers(0, 12))
    classes = ["building", "vehicle", "tree", "road", "solar_panel"]
    dets: list[dict] = []
    for _ in range(n):
        cls = classes[rng.integers(0, len(classes))]
        x1  = float(rng.integers(10, 300))
        y1  = float(rng.integers(10, 300))
        x2  = x1 + float(rng.integers(20, 100))
        y2  = y1 + float(rng.integers(20, 100))
        conf = float(rng.uniform(0.4, 0.99))
        dets.append(
            {
                "class_name": cls,
                "confidence": round(conf, 4),
                "bbox_xyxy":  [x1, y1, x2, y2],
            }
        )
    return dets


def _make_demo_thumbnail() -> str:
    """Return a tiny solid-color base64 thumbnail for demo mode."""
    img = Image.new("RGB", (400, 300), color=(30, 40, 60))
    draw = ImageDraw.Draw(img)
    draw.text((140, 130), "Demo Image\n(model not found)", fill=(120, 160, 200))
    return _image_to_base64(img)


# ---------------------------------------------------------------------------
# Per-property analysis
# ---------------------------------------------------------------------------

def _analyze_property(
    prop: dict,
    model: Any | None,
    demo_mode: bool,
) -> dict:
    """
    Run inference (or generate demo data) for a single property.
    Returns an enriched dict with 'detections', 'thumbnail_b64', 'risk'.
    """
    image_path = Path(prop.get("image_path", ""))

    if demo_mode:
        detections  = _make_demo_detections(str(prop.get("id", "")))
        thumbnail   = _make_demo_thumbnail()
        risk        = _classify_risk(detections, demo=True)
        return {**prop, "detections": detections, "thumbnail_b64": thumbnail, "risk": risk}

    # Real model path
    if not image_path.exists():
        print(f"⚠️  Image not found for property {prop.get('id')}: {image_path}")
        detections = []
        thumbnail  = _make_demo_thumbnail()
    else:
        detections = _run_inference(model, image_path)
        try:
            img       = Image.open(image_path).convert("RGB")
            annotated = _annotate_image(img, detections)
            thumbnail = _image_to_base64(annotated)
        except Exception as exc:
            print(f"⚠️  Could not create thumbnail for {image_path.name}: {exc}")
            thumbnail = _make_demo_thumbnail()

    risk = _classify_risk(detections, demo=False)
    return {**prop, "detections": detections, "thumbnail_b64": thumbnail, "risk": risk}


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def _detection_table_rows(detections: list[dict]) -> str:
    """Aggregate detections by class and return HTML <tr> rows."""
    class_data: dict[str, list[float]] = defaultdict(list)
    for det in detections:
        class_data[det["class_name"]].append(det["confidence"])

    if not class_data:
        return '<tr><td colspan="3" style="color:#6b7280;text-align:center;">No detections</td></tr>'

    rows = []
    for cls, confs in sorted(class_data.items()):
        avg_conf = sum(confs) / len(confs)
        rows.append(
            f'<tr>'
            f'<td>{cls}</td>'
            f'<td style="text-align:center;">{len(confs)}</td>'
            f'<td style="text-align:center;">{avg_conf:.2f}</td>'
            f'</tr>'
        )
    return "\n".join(rows)


def _property_card(prop_result: dict, demo_mode: bool) -> str:
    prop_id      = prop_result.get("id", "N/A")
    address      = prop_result.get("address", "N/A")
    policy       = prop_result.get("policy_number", "N/A")
    insured_val  = prop_result.get("insured_value", "N/A")
    detections   = prop_result.get("detections", [])
    thumbnail    = prop_result.get("thumbnail_b64", "")
    risk         = prop_result.get("risk", "normal")
    risk_label   = RISK_INDICATOR.get(risk, "🟢 Normal")
    risk_color   = RISK_BADGE_COLOR.get(risk, "#16a34a")
    table_rows   = _detection_table_rows(detections)

    if isinstance(insured_val, (int, float)):
        insured_display = f"${insured_val:,.0f}"
    else:
        insured_display = str(insured_val)

    return f"""
    <div class="property-card" id="prop-{prop_id}">
      <div class="card-header">
        <div class="card-title">
          <span class="prop-id">#{prop_id}</span>
          <span class="prop-address">{address}</span>
        </div>
        <div class="risk-badge" style="background:{risk_color}20;color:{risk_color};border:1px solid {risk_color};">
          {risk_label}
        </div>
      </div>
      <div class="card-body">
        <div class="card-image-col">
          <img class="prop-image" src="{thumbnail}" alt="Property {prop_id}" />
        </div>
        <div class="card-info-col">
          <table class="meta-table">
            <tr><td class="meta-label">Policy</td><td>{policy}</td></tr>
            <tr><td class="meta-label">Insured Value</td><td>{insured_display}</td></tr>
            <tr><td class="meta-label">Detections</td><td>{len(detections)}</td></tr>
          </table>
          <table class="det-table">
            <thead>
              <tr>
                <th>Class</th>
                <th>Count</th>
                <th>Avg Conf</th>
              </tr>
            </thead>
            <tbody>
              {table_rows}
            </tbody>
          </table>
        </div>
      </div>
    </div>
    """


def _build_html_report(
    client_name: str,
    report_date: str,
    model_version: str,
    conf_threshold: float,
    property_results: list[dict],
    demo_mode: bool,
) -> str:
    total_properties  = len(property_results)
    total_detections  = sum(len(p.get("detections", [])) for p in property_results)
    alert_count       = sum(1 for p in property_results if p.get("risk") == "alert")
    review_count      = sum(1 for p in property_results if p.get("risk") == "review")
    normal_count      = sum(1 for p in property_results if p.get("risk") == "normal")

    all_confs: list[float] = []
    for p in property_results:
        for det in p.get("detections", []):
            all_confs.append(det["confidence"])
    avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0

    demo_banner = ""
    if demo_mode:
        demo_banner = """
        <div class="demo-banner">
          ⚠️  DEMO MODE — Model not found. Displaying placeholder data.
          All detections and risk indicators are synthetic.
        </div>
        """

    property_cards_html = "\n".join(
        _property_card(p, demo_mode) for p in property_results
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SatelliteVision — {client_name} Coverage Report</title>
  <style>
    /* ---- Reset & base ---- */
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                   "Helvetica Neue", Arial, sans-serif;
      background: {BG_COLOR};
      color: #e5e7eb;
      font-size: 14px;
      line-height: 1.6;
    }}
    a {{ color: {ACCENT_COLOR}; text-decoration: none; }}

    /* ---- Layout ---- */
    .page-wrapper {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px 16px 48px;
    }}

    /* ---- Demo banner ---- */
    .demo-banner {{
      background: #7c2d12;
      border: 1px solid #ea580c;
      color: #fed7aa;
      border-radius: 8px;
      padding: 12px 20px;
      margin-bottom: 24px;
      font-weight: 600;
      text-align: center;
    }}

    /* ---- Header ---- */
    .report-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 24px 28px;
      background: {SURFACE_COLOR};
      border-radius: 12px;
      border: 1px solid #1f2937;
      margin-bottom: 28px;
    }}
    .logo-block {{
      display: flex;
      align-items: center;
      gap: 14px;
    }}
    .logo-icon {{
      width: 44px;
      height: 44px;
      background: {ACCENT_COLOR};
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 22px;
    }}
    .logo-text {{
      font-size: 20px;
      font-weight: 700;
      color: #f9fafb;
      letter-spacing: -0.3px;
    }}
    .logo-sub {{
      font-size: 11px;
      color: #9ca3af;
      letter-spacing: 1px;
      text-transform: uppercase;
    }}
    .header-meta {{
      text-align: right;
      color: #9ca3af;
      font-size: 13px;
    }}
    .header-meta .client-name {{
      font-size: 17px;
      font-weight: 600;
      color: #f3f4f6;
      margin-bottom: 4px;
    }}

    /* ---- Executive Summary ---- */
    .section-title {{
      font-size: 16px;
      font-weight: 600;
      color: #f3f4f6;
      margin-bottom: 14px;
      padding-bottom: 6px;
      border-bottom: 1px solid #1f2937;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-bottom: 32px;
    }}
    .summary-card {{
      background: {SURFACE_COLOR};
      border: 1px solid #1f2937;
      border-radius: 10px;
      padding: 18px 20px;
      text-align: center;
    }}
    .summary-card .stat-value {{
      font-size: 32px;
      font-weight: 700;
      color: {ACCENT_COLOR};
      line-height: 1.1;
    }}
    .summary-card .stat-label {{
      font-size: 12px;
      color: #9ca3af;
      margin-top: 4px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .summary-card.alert-card .stat-value {{ color: #ef4444; }}
    .summary-card.review-card .stat-value {{ color: #eab308; }}
    .summary-card.normal-card .stat-value {{ color: #22c55e; }}

    /* ---- Risk flags ---- */
    .risk-flags {{
      background: {SURFACE_COLOR};
      border: 1px solid #1f2937;
      border-radius: 10px;
      padding: 16px 20px;
      margin-bottom: 32px;
    }}
    .risk-flags p {{
      color: #9ca3af;
      font-size: 13px;
    }}
    .risk-flag-item {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 8px 0;
      border-bottom: 1px solid #1f2937;
      font-size: 13px;
    }}
    .risk-flag-item:last-child {{ border-bottom: none; }}

    /* ---- Property cards ---- */
    .properties-section {{ margin-bottom: 40px; }}
    .property-card {{
      background: {SURFACE_COLOR};
      border: 1px solid #1f2937;
      border-radius: 12px;
      margin-bottom: 20px;
      overflow: hidden;
    }}
    .card-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 20px;
      background: #0f172a;
      border-bottom: 1px solid #1f2937;
    }}
    .card-title {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .prop-id {{
      font-size: 12px;
      color: #6b7280;
      font-family: monospace;
    }}
    .prop-address {{
      font-size: 14px;
      font-weight: 600;
      color: #f3f4f6;
    }}
    .risk-badge {{
      font-size: 12px;
      font-weight: 600;
      padding: 3px 10px;
      border-radius: 20px;
      white-space: nowrap;
    }}
    .card-body {{
      display: flex;
      gap: 20px;
      padding: 16px 20px;
    }}
    .card-image-col {{
      flex: 0 0 auto;
    }}
    .prop-image {{
      width: 280px;
      height: auto;
      border-radius: 8px;
      border: 1px solid #1f2937;
      display: block;
    }}
    .card-info-col {{
      flex: 1 1 auto;
      min-width: 0;
    }}
    .meta-table {{
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 14px;
      font-size: 13px;
    }}
    .meta-table td {{
      padding: 4px 8px 4px 0;
      vertical-align: top;
    }}
    .meta-label {{
      color: #9ca3af;
      white-space: nowrap;
      width: 110px;
    }}
    .det-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .det-table th {{
      text-align: left;
      color: #9ca3af;
      font-weight: 600;
      border-bottom: 1px solid #1f2937;
      padding: 6px 8px 6px 0;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .det-table td {{
      padding: 5px 8px 5px 0;
      border-bottom: 1px solid #0f172a;
      color: #d1d5db;
    }}
    .det-table tr:last-child td {{ border-bottom: none; }}

    /* ---- Footer ---- */
    .report-footer {{
      background: {SURFACE_COLOR};
      border: 1px solid #1f2937;
      border-radius: 10px;
      padding: 18px 24px;
      font-size: 12px;
      color: #6b7280;
      line-height: 1.8;
    }}
    .report-footer strong {{ color: #9ca3af; }}
    .disclaimer {{
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid #1f2937;
      color: #4b5563;
    }}

    /* ---- Responsive ---- */
    @media (max-width: 640px) {{
      .card-body {{ flex-direction: column; }}
      .prop-image {{ width: 100%; }}
      .report-header {{ flex-direction: column; gap: 16px; text-align: left; }}
      .header-meta {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <div class="page-wrapper">
    {demo_banner}

    <!-- Header -->
    <div class="report-header">
      <div class="logo-block">
        <div class="logo-icon">🛰</div>
        <div>
          <div class="logo-text">SatelliteVision</div>
          <div class="logo-sub">Aerial Intelligence Platform</div>
        </div>
      </div>
      <div class="header-meta">
        <div class="client-name">{client_name}</div>
        <div>Coverage Report</div>
        <div>{report_date}</div>
        <div>Properties analyzed: <strong style="color:#f3f4f6;">{total_properties}</strong></div>
      </div>
    </div>

    <!-- Executive Summary -->
    <div class="section-title">Executive Summary</div>
    <div class="summary-grid">
      <div class="summary-card">
        <div class="stat-value">{total_properties}</div>
        <div class="stat-label">Properties</div>
      </div>
      <div class="summary-card">
        <div class="stat-value">{total_detections}</div>
        <div class="stat-label">Total Detections</div>
      </div>
      <div class="summary-card">
        <div class="stat-value">{avg_conf:.2f}</div>
        <div class="stat-label">Avg Confidence</div>
      </div>
      <div class="summary-card alert-card">
        <div class="stat-value">{alert_count}</div>
        <div class="stat-label">🔴 Alert</div>
      </div>
      <div class="summary-card review-card">
        <div class="stat-value">{review_count}</div>
        <div class="stat-label">🟡 Review</div>
      </div>
      <div class="summary-card normal-card">
        <div class="stat-value">{normal_count}</div>
        <div class="stat-label">🟢 Normal</div>
      </div>
    </div>

    <!-- Risk Flags -->
    <div class="risk-flags">
      <div class="section-title" style="margin-bottom:12px;">Risk Flags</div>
      {''.join(
          f'<div class="risk-flag-item">'
          f'<span style="color:{RISK_BADGE_COLOR[p["risk"]]};">{RISK_INDICATOR[p["risk"]]}</span>'
          f'<span style="color:#9ca3af;">{p.get("address","N/A")}</span>'
          f'<span style="color:#6b7280;margin-left:auto;">{len(p.get("detections",[]))} detections</span>'
          f'</div>'
          for p in property_results if p.get("risk") in ("alert", "review")
      ) or '<p>No properties flagged for review or alert.</p>'}
    </div>

    <!-- Per-property cards -->
    <div class="properties-section">
      <div class="section-title">Property Analysis</div>
      {property_cards_html}
    </div>

    <!-- Footer -->
    <div class="report-footer">
      <strong>Model version:</strong> {model_version} &nbsp;|&nbsp;
      <strong>Confidence threshold:</strong> {conf_threshold} &nbsp;|&nbsp;
      <strong>Generated:</strong> {report_date}
      <div class="disclaimer">
        Disclaimer: This report is generated by automated AI analysis of satellite imagery
        and is intended for informational purposes only. Results may not reflect current
        ground conditions. SatelliteVision makes no warranty, express or implied, regarding
        the accuracy, completeness, or suitability of this analysis for any specific use.
        All risk indicators are algorithmic estimates and do not constitute professional
        insurance, legal, or engineering advice. Users should independently verify any
        findings before making decisions based on this report.
      </div>
    </div>
  </div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_coverage_report(
    portfolio_path: Path,
    client_name: str,
    output_path: Path,
    model_path: Path,
    conf_threshold: float = CONF_THRESHOLD,
) -> None:
    """Load portfolio, run inference, generate HTML report."""
    # Load portfolio
    try:
        portfolio: list[dict] = json.loads(portfolio_path.read_text())
    except Exception as exc:
        print(f"❌ Cannot read portfolio file: {exc}")
        sys.exit(1)

    if not isinstance(portfolio, list):
        print("❌ Portfolio JSON must be a list of property objects")
        sys.exit(1)

    print(f"📊 Portfolio loaded: {len(portfolio)} properties")

    # Load model (or enter demo mode)
    demo_mode   = False
    model: Any  = None
    model_version = "unknown"

    if not model_path.exists():
        print(f"⚠️  Model not found at {model_path} — entering demo mode")
        demo_mode = True
        model_version = "DEMO (model not found)"
    else:
        try:
            from ultralytics import YOLO

            print(f"📊 Loading model: {model_path}")
            model = YOLO(str(model_path))
            model_version = model_path.name
            print(f"✅ Model loaded: {model_version}")
        except ImportError:
            print("⚠️  ultralytics not installed — entering demo mode")
            demo_mode = True
            model_version = "DEMO (ultralytics not installed)"
        except Exception as exc:
            print(f"⚠️  Failed to load model ({exc}) — entering demo mode")
            demo_mode = True
            model_version = f"DEMO (load error)"

    # Process each property
    property_results: list[dict] = []
    alerts_flagged = 0

    for prop in tqdm(portfolio, desc="Analyzing properties", unit="property"):
        result = _analyze_property(prop, model, demo_mode)
        property_results.append(result)
        if result.get("risk") in ("alert", "review"):
            alerts_flagged += 1

    # Build HTML
    report_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html = _build_html_report(
        client_name=client_name,
        report_date=report_date,
        model_version=model_version,
        conf_threshold=conf_threshold,
        property_results=property_results,
        demo_mode=demo_mode,
    )

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"✅ HTML report saved → {output_path}")

    # Summary
    total_dets = sum(len(p.get("detections", [])) for p in property_results)
    alert_count = sum(1 for p in property_results if p.get("risk") == "alert")

    print("\n" + "=" * 50)
    print("📊 Report Summary")
    print("=" * 50)
    print(f"  Properties processed: {len(property_results)}")
    print(f"  Total detections:     {total_dets}")
    print(f"  Alerts flagged:       {alert_count}")
    print(f"  Report path:          {output_path}")
    if demo_mode:
        print("  ⚠️  DEMO MODE — synthetic data used")
    print("=" * 50)

    _supabase_log_report(
        client_name=client_name,
        output_path=output_path,
        property_count=len(property_results),
        alert_count=alert_count,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML coverage report for a property portfolio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--portfolio",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to portfolio JSON (list of property objects)",
    )
    parser.add_argument(
        "--client-name",
        required=True,
        metavar="NAME",
        help="Client name displayed in the report header",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="PATH",
        help="Output path for the HTML report",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(MODEL_PATH_DEFAULT),
        metavar="PATH",
        help="Path to YOLOv8 .pt weights file",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONF_THRESHOLD,
        metavar="FLOAT",
        help="YOLOv8 confidence threshold",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    portfolio_path: Path = args.portfolio.resolve()
    model_path: Path     = args.model.resolve()
    output_path: Path    = args.output

    if not portfolio_path.exists():
        print(f"❌ Portfolio file not found: {portfolio_path}")
        sys.exit(1)

    run_coverage_report(
        portfolio_path=portfolio_path,
        client_name=args.client_name,
        output_path=output_path,
        model_path=model_path,
        conf_threshold=args.conf,
    )


if __name__ == "__main__":
    main()
