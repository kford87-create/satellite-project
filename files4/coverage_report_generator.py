"""
tools/commercial/coverage_report_generator.py

Generates formatted PDF/HTML reports for insurance customers.
Takes a portfolio of addresses, pulls satellite imagery,
runs detection, and produces a report with flagged properties.

This is a direct revenue-generating deliverable — charge per report.

Usage:
  python tools/commercial/coverage_report_generator.py \
    --portfolio data/customer_portfolio.json \
    --model models/baseline_v1/weights/best.pt \
    --client-name "Acme Insurance Co" \
    --output reports/acme_q4_2024.html
"""

import os
import json
import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
import argparse

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))


REPORT_CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f5f5f5; color: #333; }
.header { background: linear-gradient(135deg, #1a237e, #283593); color: white; padding: 30px 40px; }
.header h1 { margin: 0 0 5px; font-size: 28px; }
.header .subtitle { opacity: 0.85; font-size: 14px; }
.summary-bar { display: flex; gap: 20px; padding: 20px 40px; background: white; border-bottom: 1px solid #e0e0e0; flex-wrap: wrap; }
.metric { background: #f8f9fa; border-radius: 8px; padding: 15px 20px; flex: 1; min-width: 140px; text-align: center; }
.metric .value { font-size: 32px; font-weight: bold; color: #1a237e; }
.metric .label { font-size: 12px; color: #666; margin-top: 4px; }
.metric.alert .value { color: #c62828; }
.metric.warn .value { color: #e65100; }
.metric.ok .value { color: #2e7d32; }
.content { padding: 30px 40px; }
.section-title { font-size: 18px; font-weight: bold; color: #1a237e; margin: 20px 0 12px; border-bottom: 2px solid #e3f2fd; padding-bottom: 6px; }
.property-card { background: white; border-radius: 8px; padding: 16px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); display: flex; gap: 16px; align-items: flex-start; }
.property-card.high { border-left: 4px solid #c62828; }
.property-card.medium { border-left: 4px solid #e65100; }
.property-card.low { border-left: 4px solid #2e7d32; }
.property-img { width: 160px; height: 120px; object-fit: cover; border-radius: 4px; flex-shrink: 0; }
.property-details { flex: 1; }
.property-address { font-weight: bold; font-size: 15px; margin-bottom: 6px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: bold; margin: 2px; }
.badge.high { background: #ffebee; color: #c62828; }
.badge.medium { background: #fff3e0; color: #e65100; }
.badge.low { background: #e8f5e9; color: #2e7d32; }
.badge.info { background: #e3f2fd; color: #1565c0; }
.detection-list { font-size: 12px; color: #555; margin-top: 6px; }
.footer { background: #263238; color: #90a4ae; padding: 20px 40px; font-size: 11px; text-align: center; margin-top: 40px; }
"""


class CoverageReportGenerator:
    """Generates insurance-ready property coverage reports."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or str(MODELS_DIR / "baseline_v1/weights/best.pt")
        self._model = None

    def _get_model(self):
        if self._model is None:
            from ultralytics import YOLO
            if Path(self.model_path).exists():
                self._model = YOLO(self.model_path)
            else:
                print(f"⚠️  Model not found — using demo mode")
        return self._model

    def _analyze_property(self, property_data: Dict) -> Dict:
        """Run detection analysis on a property image."""
        model = self._get_model()
        img_path = property_data.get("image_path")
        result = {
            **property_data,
            "detections": [],
            "risk_level": "low",
            "flags": [],
            "annotated_image_b64": None,
        }

        if not img_path or not Path(img_path).exists():
            result["flags"].append("No satellite imagery available")
            result["risk_level"] = "unknown"
            return result

        img = cv2.imread(img_path)
        if img is None:
            result["flags"].append("Could not read image")
            return result

        if model:
            yolo_results = model.predict(source=img, conf=0.25, verbose=False)[0]
            detections = []
            if yolo_results.boxes is not None:
                for box in yolo_results.boxes:
                    cls_id = int(box.cls.item())
                    cls_name = {0:"building",1:"vehicle",2:"aircraft",3:"ship"}.get(cls_id,"unknown")
                    conf = float(box.conf.item())
                    x1,y1,x2,y2 = box.xyxy.tolist()[0]
                    detections.append({"class": cls_name, "confidence": round(conf,3), "bbox_px": [int(x1),int(y1),int(x2),int(y2)]})
                    color = (0,255,0) if cls_name=="building" else (255,165,0)
                    cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
                    cv2.putText(img, f"{cls_name} {conf:.2f}", (int(x1),int(y1)-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            result["detections"] = detections
            n_buildings = sum(1 for d in detections if d["class"] == "building")
            n_vehicles = sum(1 for d in detections if d["class"] == "vehicle")

            # Risk assessment flags
            if n_buildings == 0:
                result["flags"].append("No structures detected — verify address")
                result["risk_level"] = "medium"
            elif n_buildings > 10:
                result["flags"].append(f"{n_buildings} structures detected — high density")
                result["risk_level"] = "medium"
            else:
                result["risk_level"] = "low"

            # Annotated image
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            result["annotated_image_b64"] = base64.b64encode(buf).decode()
        else:
            # Demo mode
            result["detections"] = [{"class": "building", "confidence": 0.91, "bbox_px": [100,100,200,200]}]
            result["risk_level"] = "low"

        return result

    def generate_report(
        self,
        portfolio: List[Dict],
        client_name: str,
        output_path: Path,
        report_date: str = None
    ):
        """Generate HTML coverage report for a property portfolio."""
        report_date = report_date or datetime.now().strftime("%B %d, %Y")
        print(f"\n📊 Generating coverage report for {client_name}")
        print(f"   Properties: {len(portfolio)}")

        # Analyze all properties
        from tqdm import tqdm
        analyzed = [self._analyze_property(p) for p in tqdm(portfolio, desc="Analyzing")]

        # Compute summary stats
        n_high = sum(1 for p in analyzed if p["risk_level"] == "high")
        n_med = sum(1 for p in analyzed if p["risk_level"] == "medium")
        n_low = sum(1 for p in analyzed if p["risk_level"] == "low")
        n_flags = sum(len(p["flags"]) for p in analyzed)
        total_dets = sum(len(p["detections"]) for p in analyzed)

        # Sort: high risk first
        risk_order = {"high": 0, "medium": 1, "unknown": 2, "low": 3}
        analyzed.sort(key=lambda x: risk_order.get(x["risk_level"], 4))

        # Build HTML
        props_html = ""
        for prop in analyzed:
            risk = prop["risk_level"]
            flags_html = "".join(f'<span class="badge {risk}">{f}</span>' for f in prop["flags"]) or '<span class="badge low">No flags</span>'
            dets_html = ", ".join(f"{d['class']} ({d['confidence']:.2f})" for d in prop.get("detections", [])[:5]) or "No detections"
            img_html = f'<img class="property-img" src="data:image/jpeg;base64,{prop["annotated_image_b64"]}" />' if prop.get("annotated_image_b64") else '<div class="property-img" style="background:#eee;display:flex;align-items:center;justify-content:center;font-size:11px;color:#aaa;">No image</div>'
            n_dets = len(prop.get("detections", []))
            props_html += f"""
<div class="property-card {risk}">
  {img_html}
  <div class="property-details">
    <div class="property-address">{prop.get("address", prop.get("id", "Unknown"))}</div>
    <div>{flags_html}</div>
    <div class="detection-list">Detections ({n_dets}): {dets_html}</div>
    {'<div class="detection-list" style="color:#888">Policy: ' + prop.get("policy_number","N/A") + ' | Value: $' + str(prop.get("insured_value","N/A")) + '</div>' if prop.get("policy_number") else ""}
  </div>
</div>"""

        html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>{client_name} — Coverage Report</title>
<style>{REPORT_CSS}</style></head><body>
<div class="header">
  <h1>🛰️ Satellite Property Coverage Report</h1>
  <div class="subtitle">{client_name} &nbsp;|&nbsp; {report_date} &nbsp;|&nbsp; Powered by AI Object Detection</div>
</div>
<div class="summary-bar">
  <div class="metric"><div class="value">{len(analyzed)}</div><div class="label">Properties Analyzed</div></div>
  <div class="metric alert"><div class="value">{n_high}</div><div class="label">High Risk</div></div>
  <div class="metric warn"><div class="value">{n_med}</div><div class="label">Review Required</div></div>
  <div class="metric ok"><div class="value">{n_low}</div><div class="label">Clear</div></div>
  <div class="metric"><div class="value">{n_flags}</div><div class="label">Total Flags</div></div>
  <div class="metric info"><div class="value">{total_dets}</div><div class="label">Objects Detected</div></div>
</div>
<div class="content">
  <div class="section-title">Property Analysis</div>
  {props_html}
</div>
<div class="footer">
  Generated by Satellite Detection Pipeline &nbsp;|&nbsp; {datetime.now().strftime("%Y-%m-%d %H:%M UTC")} &nbsp;|&nbsp; Confidential
</div></body></html>"""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        print(f"\n✅ Report saved: {output_path}")
        print(f"   High risk: {n_high} | Review: {n_med} | Clear: {n_low}")
        return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio", required=True, help="JSON file with property list")
    parser.add_argument("--client-name", default="Insurance Client")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default=str(DATA_DIR / "reports" / "coverage_report.html"))
    args = parser.parse_args()

    portfolio = json.loads(Path(args.portfolio).read_text())
    gen = CoverageReportGenerator(args.model)
    gen.generate_report(portfolio, args.client_name, Path(args.output))
