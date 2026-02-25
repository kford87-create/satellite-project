"""
tools/active_learning/bootstrapping_dashboard.py

Real-time visualization of the active learning bootstrapping loop.
Pulls metrics from Supabase and renders the efficiency curve —
mAP vs labels used, cost per label, FN rate trajectory.

Doubles as a sales tool when demoing to insurance/defense customers.

Usage:
  python tools/active_learning/bootstrapping_dashboard.py
  python tools/active_learning/bootstrapping_dashboard.py --export-png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
import argparse

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


def load_iterations_from_supabase() -> List[Dict]:
    """Load bootstrapping iteration metrics from Supabase."""
    try:
        from supabase import create_client
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        result = client.table("bootstrap_iterations").select("*").order("iteration").execute()
        return result.data or []
    except Exception as e:
        print(f"⚠️  Supabase unavailable ({e}) — loading from local files")
        return []


def load_iterations_from_local() -> List[Dict]:
    """Fallback: load from local JSON files in bootstrapped directory."""
    bootstrapped_dir = DATA_DIR / "bootstrapped"
    iterations = []
    for f in sorted(bootstrapped_dir.glob("iteration_*/metrics.json")):
        try:
            iterations.append(json.loads(f.read_text()))
        except Exception:
            pass
    return iterations


def load_fn_report() -> Optional[Dict]:
    """Load most recent false negative quantification report."""
    fn_dir = DATA_DIR / "bootstrapped" / "fn_analysis"
    reports = list(fn_dir.glob("fn_quantification_report.json"))
    if reports:
        return json.loads(reports[0].read_text())
    return None


class BootstrappingDashboard:
    """Renders the bootstrapping efficiency dashboard."""

    def __init__(self):
        self.iterations = load_iterations_from_supabase() or load_iterations_from_local()
        self.fn_report = load_fn_report()

        if not self.iterations:
            print("⚠️  No iteration data found — generating demo data")
            self.iterations = self._generate_demo_data()

    def _generate_demo_data(self) -> List[Dict]:
        """Generate realistic demo data for dashboard preview."""
        np.random.seed(42)
        iterations = []
        map_val = 0.25
        fn_rate = 0.45
        labels = 0
        for i in range(1, 11):
            gain = np.random.uniform(0.04, 0.09) * (1 - i/20)
            map_val = min(map_val + gain, 0.92)
            fn_rate = max(fn_rate - np.random.uniform(0.02, 0.06), 0.05)
            n_new = np.random.randint(40, 60)
            labels += n_new
            iterations.append({
                "iteration": i,
                "n_images_labeled": n_new,
                "cumulative_labels": labels,
                "map50": round(map_val, 4),
                "map50_95": round(map_val * 0.65, 4),
                "precision_score": round(map_val + np.random.uniform(-0.05, 0.05), 4),
                "recall_score": round(map_val - np.random.uniform(0, 0.08), 4),
                "fn_rate": round(fn_rate, 4),
                "map_gain_per_label": round(gain / n_new, 6),
            })
        return iterations

    def render(self, export_path: Optional[Path] = None):
        """Render the full dashboard."""
        iters = self.iterations
        if not iters:
            print("❌ No data to render")
            return

        x_labels = [d["cumulative_labels"] for d in iters]
        x_iter = [d["iteration"] for d in iters]
        map50 = [d.get("map50", 0) for d in iters]
        map50_95 = [d.get("map50_95", 0) for d in iters]
        fn_rate = [d.get("fn_rate", 0) for d in iters]
        efficiency = [d.get("map_gain_per_label", 0) * 1000 for d in iters]
        precision = [d.get("precision_score", 0) for d in iters]
        recall = [d.get("recall_score", 0) for d in iters]

        fig = plt.figure(figsize=(18, 10))
        fig.patch.set_facecolor("#0f1117")
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        # Color scheme
        C = {"map": "#00d4ff", "fn": "#ff4757", "efficiency": "#2ed573",
             "precision": "#ffa502", "recall": "#eccc68", "grid": "#2f3542",
             "text": "#a4b0be", "title": "#ffffff"}

        def styled_ax(ax, title):
            ax.set_facecolor("#1a1d27")
            ax.tick_params(colors=C["text"])
            ax.set_title(title, color=C["title"], fontsize=11, pad=8)
            ax.spines[:].set_color(C["grid"])
            ax.xaxis.label.set_color(C["text"])
            ax.yaxis.label.set_color(C["text"])
            ax.grid(True, color=C["grid"], alpha=0.5, linewidth=0.5)

        # Plot 1: mAP vs Labels Used (main efficiency curve)
        ax1 = fig.add_subplot(gs[0, 0])
        styled_ax(ax1, "mAP vs Labels Used")
        ax1.plot(x_labels, map50, color=C["map"], linewidth=2.5, marker="o", markersize=5, label="mAP@50")
        ax1.plot(x_labels, map50_95, color=C["map"], linewidth=1.5, linestyle="--", alpha=0.6, label="mAP@50-95")
        ax1.fill_between(x_labels, map50_95, map50, alpha=0.15, color=C["map"])
        ax1.set_xlabel("Cumulative Labels")
        ax1.set_ylabel("mAP")
        ax1.legend(facecolor="#1a1d27", labelcolor=C["text"], fontsize=8)
        ax1.set_ylim(0, 1)

        # Plot 2: False Negative Rate Trajectory
        ax2 = fig.add_subplot(gs[0, 1])
        styled_ax(ax2, "False Negative Rate")
        ax2.plot(x_iter, fn_rate, color=C["fn"], linewidth=2.5, marker="s", markersize=5)
        ax2.fill_between(x_iter, fn_rate, alpha=0.2, color=C["fn"])
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("FN Rate")
        ax2.set_ylim(0, 1)

        # Plot 3: Efficiency (mAP gain per 1000 labels)
        ax3 = fig.add_subplot(gs[0, 2])
        styled_ax(ax3, "Learning Efficiency (mAP gain / 1k labels)")
        bars = ax3.bar(x_iter, efficiency, color=C["efficiency"], alpha=0.8, edgecolor="none")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("mAP gain per 1k labels")

        # Plot 4: Precision / Recall
        ax4 = fig.add_subplot(gs[1, 0])
        styled_ax(ax4, "Precision & Recall")
        ax4.plot(x_iter, precision, color=C["precision"], linewidth=2, marker="^", label="Precision")
        ax4.plot(x_iter, recall, color=C["recall"], linewidth=2, marker="v", label="Recall")
        ax4.set_xlabel("Iteration")
        ax4.legend(facecolor="#1a1d27", labelcolor=C["text"], fontsize=8)
        ax4.set_ylim(0, 1)

        # Plot 5: Labels per iteration
        ax5 = fig.add_subplot(gs[1, 1])
        styled_ax(ax5, "Labels Added per Iteration")
        n_labels = [d.get("n_images_labeled", 0) for d in iters]
        ax5.bar(x_iter, n_labels, color="#747d8c", alpha=0.9, edgecolor="none")
        ax5.set_xlabel("Iteration")
        ax5.set_ylabel("Images labeled")

        # Plot 6: Summary stats panel
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_facecolor("#1a1d27")
        ax6.axis("off")
        latest = iters[-1]
        fn_reduction = (iters[0].get("fn_rate", 1) - latest.get("fn_rate", 0)) / max(iters[0].get("fn_rate", 1), 1e-6)

        summary_lines = [
            ("CURRENT STATUS", "", "#00d4ff"),
            ("", "", ""),
            ("Iterations completed", str(len(iters)), C["text"]),
            ("Total labels used", str(latest.get("cumulative_labels", 0)), C["text"]),
            ("Current mAP@50", f"{latest.get('map50', 0):.4f}", C["map"]),
            ("Current FN rate", f"{latest.get('fn_rate', 0):.4f}", C["fn"]),
            ("FN reduction", f"{fn_reduction:.1%}", C["efficiency"]),
            ("", "", ""),
            ("BENCHMARKS", "", "#00d4ff"),
            ("Start mAP@50", f"{iters[0].get('map50', 0):.4f}", C["text"]),
            ("Start FN rate", f"{iters[0].get('fn_rate', 0):.4f}", C["text"]),
        ]

        for i, (label, value, color) in enumerate(summary_lines):
            if label and value:
                ax6.text(0.05, 0.95 - i*0.082, label, transform=ax6.transAxes,
                         color=C["text"], fontsize=9, va="top")
                ax6.text(0.95, 0.95 - i*0.082, value, transform=ax6.transAxes,
                         color=color, fontsize=9, va="top", ha="right", fontweight="bold")
            elif label:
                ax6.text(0.5, 0.95 - i*0.082, label, transform=ax6.transAxes,
                         color=color, fontsize=10, va="top", ha="center", fontweight="bold")

        fig.suptitle("Satellite Detection Bootstrapping Dashboard",
                     color=C["title"], fontsize=16, fontweight="bold", y=0.98)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        fig.text(0.99, 0.01, f"Generated: {ts}", color=C["text"], fontsize=7, ha="right")

        if export_path:
            plt.savefig(export_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"📊 Dashboard exported: {export_path}")
        else:
            plt.show()

        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-png", default=None, help="Export to PNG instead of showing")
    args = parser.parse_args()

    dash = BootstrappingDashboard()
    export = Path(args.export_png) if args.export_png else \
             DATA_DIR / "bootstrapped" / "dashboard.png"
    dash.render(export_path=export)
    print(f"✅ Dashboard rendered")
