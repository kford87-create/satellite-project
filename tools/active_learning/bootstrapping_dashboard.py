"""
bootstrapping_dashboard.py
--------------------------
Generates a 2x2 matplotlib dashboard visualising the progress of the
YOLOv8 active-learning bootstrapping pipeline.

Data is loaded from Supabase first; if unavailable it falls back to local
iteration metrics files and the fn_quantification_report.json.

Usage:
    python tools/active_learning/bootstrapping_dashboard.py
    python tools/active_learning/bootstrapping_dashboard.py \
      --export-png reports/demo_dashboard.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV_PATH = Path("/Users/kahlil/satellite-project/.env")
load_dotenv(ENV_PATH)

import os  # noqa: E402 – after dotenv

# ---------------------------------------------------------------------------
# Supabase (non-fatal)
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://obdsgqjkjjmmtbcfjhnn.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

try:
    from supabase import create_client, Client as SupabaseClient

    _supabase: SupabaseClient | None = (
        create_client(SUPABASE_URL, SUPABASE_KEY)
        if SUPABASE_URL and SUPABASE_KEY
        else None
    )
except Exception:
    _supabase = None

# ---------------------------------------------------------------------------
# Project root (two levels up from this file)
# ---------------------------------------------------------------------------
_TOOL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TOOL_DIR.parent.parent

# ---------------------------------------------------------------------------
# Theme colours
# ---------------------------------------------------------------------------
BG_COLOR = "#0a0e1a"
ACCENT_BLUE = "#3b82f6"
ACCENT_GREEN = "#22c55e"
ACCENT_YELLOW = "#eab308"
ACCENT_RED = "#ef4444"
AXES_BG = "#111827"
GRID_COLOR = "#1f2937"
TEXT_COLOR = "#e5e7eb"
MUTED_COLOR = "#6b7280"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_iterations_from_supabase() -> list[dict[str, Any]] | None:
    """
    Query the `bootstrap_iterations` table.

    Expected columns:
        iteration, n_images_labeled, cumulative_labels,
        map50, precision_score, recall_score, map_gain_per_label, fn_rate

    Returns None on failure.
    """
    try:
        if _supabase is None:
            return None
        result = (
            _supabase.table("bootstrap_iterations")
            .select(
                "iteration, n_images_labeled, cumulative_labels, "
                "map50, precision_score, recall_score, "
                "map_gain_per_label, fn_rate"
            )
            .order("iteration")
            .execute()
        )
        if result.data:
            print(f"✅ Loaded {len(result.data)} iterations from Supabase bootstrap_iterations")
            return result.data
        return None
    except Exception as exc:
        print(f"⚠️  Supabase bootstrap_iterations unavailable: {exc}")
        return None


def _load_fn_report_from_supabase() -> dict[str, Any] | None:
    """
    Query the latest entry in the `fn_reports` table and return its report_json.
    Returns None on failure.
    """
    try:
        if _supabase is None:
            return None
        result = (
            _supabase.table("fn_reports")
            .select("report_json")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if result.data:
            raw = result.data[0]["report_json"]
            # report_json may be a dict (JSONB) or a string
            if isinstance(raw, str):
                return json.loads(raw)
            return raw
        return None
    except Exception as exc:
        print(f"⚠️  Supabase fn_reports unavailable: {exc}")
        return None


def _load_iterations_from_local() -> list[dict[str, Any]]:
    """
    Scan data/bootstrapped/iteration_*/metrics.json files.

    Expected metrics.json schema:
        {
          "iteration": 1,
          "n_images_labeled": 50,
          "cumulative_labels": 100,
          "mAP50": 0.42,
          "precision": 0.61,
          "recall": 0.55,
          "map_gain_per_label": 0.003,
          "fn_rate": 0.38
        }

    Both snake_case and camelCase/shorthand keys are accepted.
    """
    bootstrapped_dir = _PROJECT_ROOT / "data" / "bootstrapped"
    if not bootstrapped_dir.is_dir():
        return []

    iteration_dirs = sorted(
        d for d in bootstrapped_dir.iterdir()
        if d.is_dir() and d.name.startswith("iteration_")
    )

    rows: list[dict[str, Any]] = []
    for it_dir in tqdm(iteration_dirs, desc="Scanning local iteration dirs", unit="dir"):
        metrics_file = it_dir / "metrics.json"
        if not metrics_file.exists():
            print(f"⚠️  No metrics.json in {it_dir.name} — skipping")
            continue
        try:
            raw = json.loads(metrics_file.read_text())

            def _get(*keys: str, default: float = 0.0) -> float:
                for k in keys:
                    v = raw.get(k)
                    if v is not None:
                        return float(v)
                return default

            row: dict[str, Any] = {
                "iteration": int(raw.get("iteration", 0)),
                "n_images_labeled": int(raw.get("n_images_labeled", 0)),
                "cumulative_labels": int(raw.get("cumulative_labels", 0)),
                "map50": _get("map50", "mAP50", "map_50"),
                "precision_score": _get("precision_score", "precision"),
                "recall_score": _get("recall_score", "recall"),
                "map_gain_per_label": _get("map_gain_per_label"),
                "fn_rate": _get("fn_rate"),
            }
            rows.append(row)
        except Exception as exc:
            print(f"⚠️  Could not parse {metrics_file}: {exc}")

    if rows:
        print(f"✅ Loaded {len(rows)} iterations from local metrics.json files")
    return rows


def _load_fn_report_from_local() -> dict[str, Any] | None:
    """Load fn_quantification_report.json from canonical local path."""
    local_path = _PROJECT_ROOT / "data" / "bootstrapped" / "fn_analysis" / "fn_quantification_report.json"
    if not local_path.exists():
        return None
    try:
        data = json.loads(local_path.read_text())
        print(f"✅ Loaded FN report from {local_path}")
        return data
    except Exception as exc:
        print(f"⚠️  Could not load local FN report: {exc}")
        return None


def _extract_fn_rates_from_report(
    report: dict[str, Any],
) -> list[tuple[int, float]]:
    """
    Extract (iteration, fn_rate) pairs from the first profile found in the FN report.
    Falls back to an empty list if the format is unexpected.
    """
    try:
        # The report is keyed by profile name; take the first one
        profile_key = next(iter(report))
        profile_data = report[profile_key]
        pairs = []
        for entry in profile_data.get("iteration_results", []):
            iteration = int(entry.get("iteration", 0))
            fnr = float(entry.get("false_negative_rate", 0.0))
            pairs.append((iteration, fnr))
        return pairs
    except Exception as exc:
        print(f"⚠️  Could not extract FN rates from report: {exc}")
        return []


# ---------------------------------------------------------------------------
# Dashboard drawing
# ---------------------------------------------------------------------------

def _apply_theme(ax: Any) -> None:
    """Apply dark theme to a single Axes."""
    ax.set_facecolor(AXES_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.7)


def _build_dashboard(
    iterations: list[dict[str, Any]],
    fn_report: dict[str, Any] | None,
    export_png: Path | None,
) -> None:
    """Render the 2x2 dashboard and either save it or show it interactively."""
    try:
        import matplotlib
        if export_png:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("❌ matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    if not iterations:
        print("❌ No iteration data available to plot. Check Supabase or local metrics files.")
        sys.exit(1)

    # Sort by cumulative_labels ascending for sensible x-axis
    iterations = sorted(iterations, key=lambda r: (r["cumulative_labels"], r["iteration"]))

    cumulative_labels = [r["cumulative_labels"] for r in iterations]
    map50_vals = [r["map50"] for r in iterations]
    precision_vals = [r["precision_score"] for r in iterations]
    recall_vals = [r["recall_score"] for r in iterations]
    gain_vals = [r["map_gain_per_label"] for r in iterations]
    iteration_labels = [str(r["iteration"]) for r in iterations]

    # FN rate: prefer iterations table, then FN report
    fn_rates_from_iters = [r.get("fn_rate") for r in iterations]
    use_iter_fn = any(v is not None and v > 0 for v in fn_rates_from_iters)

    fn_x: list[int] = []
    fn_y: list[float] = []

    if use_iter_fn:
        fn_x = list(range(len(iterations)))
        fn_y = [v if v is not None else 0.0 for v in fn_rates_from_iters]
        fn_x_label = "Iteration"
    elif fn_report:
        pairs = _extract_fn_rates_from_report(fn_report)
        fn_x = [p[0] for p in pairs]
        fn_y = [p[1] for p in pairs]
        fn_x_label = "Iteration"
    else:
        fn_x = list(range(len(iterations)))
        fn_y = []
        fn_x_label = "Iteration"

    # ---- Figure setup -------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        "Satellite Active Learning — Bootstrapping Dashboard",
        color=TEXT_COLOR,
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.subplots_adjust(hspace=0.38, wspace=0.32, left=0.08, right=0.97, top=0.92, bottom=0.08)

    ax_map, ax_gain, ax_pr, ax_fn = (
        axes[0][0], axes[0][1], axes[1][0], axes[1][1]
    )

    # ── Top-left: mAP50 vs cumulative labels ─────────────────────────────────
    ax_map.plot(
        cumulative_labels, map50_vals,
        color=ACCENT_BLUE, linewidth=2, marker="o", markersize=6,
        label="mAP50",
    )
    for x, y in zip(cumulative_labels, map50_vals):
        ax_map.annotate(
            f"{y:.3f}",
            (x, y),
            textcoords="offset points", xytext=(0, 8),
            fontsize=7, color=ACCENT_BLUE, ha="center",
        )
    ax_map.set_xlabel("Cumulative Labels")
    ax_map.set_ylabel("mAP50")
    ax_map.set_title("mAP50 vs Cumulative Labels")
    ax_map.set_ylim(0, min(1.05, max(map50_vals) * 1.25 + 0.05))
    ax_map.legend(fontsize=9, facecolor=AXES_BG, labelcolor=TEXT_COLOR, edgecolor=GRID_COLOR)
    _apply_theme(ax_map)

    # ── Top-right: mAP gain per label (bar chart) ─────────────────────────────
    x_pos = range(len(iterations))
    bars = ax_gain.bar(
        x_pos, gain_vals,
        color=[ACCENT_GREEN if g > 0 else ACCENT_RED for g in gain_vals],
        width=0.6, alpha=0.85,
    )
    ax_gain.set_xticks(list(x_pos))
    ax_gain.set_xticklabels(iteration_labels, fontsize=8)
    ax_gain.set_xlabel("Iteration")
    ax_gain.set_ylabel("mAP Gain / Label")
    ax_gain.set_title("mAP Gain per Label (Diminishing Returns)")
    ax_gain.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.4f}"))
    # Add value labels on bars
    for bar, val in zip(bars, gain_vals):
        h = bar.get_height()
        ax_gain.text(
            bar.get_x() + bar.get_width() / 2,
            h + abs(h) * 0.04 if h >= 0 else h - abs(h) * 0.04,
            f"{val:.4f}",
            ha="center", va="bottom" if h >= 0 else "top",
            fontsize=7, color=TEXT_COLOR,
        )
    _apply_theme(ax_gain)

    # ── Bottom-left: Precision vs Recall curve across iterations ──────────────
    scatter = ax_pr.scatter(
        recall_vals, precision_vals,
        c=range(len(iterations)),
        cmap="cool",
        s=70, zorder=3,
    )
    ax_pr.plot(recall_vals, precision_vals, color=MUTED_COLOR, linewidth=1, linestyle="--", zorder=2)
    for i, (r, p) in enumerate(zip(recall_vals, precision_vals)):
        ax_pr.annotate(
            f"it{iteration_labels[i]}",
            (r, p),
            textcoords="offset points", xytext=(5, 4),
            fontsize=7, color=TEXT_COLOR,
        )
    cbar = fig.colorbar(scatter, ax=ax_pr, pad=0.02)
    cbar.set_label("Iteration index", color=TEXT_COLOR, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision vs Recall Across Iterations")
    ax_pr.set_xlim(-0.02, 1.05)
    ax_pr.set_ylim(-0.02, 1.05)
    _apply_theme(ax_pr)

    # ── Bottom-right: False Negative rate over iterations ────────────────────
    if fn_y:
        ax_fn.plot(
            fn_x, fn_y,
            color=ACCENT_YELLOW, linewidth=2, marker="s", markersize=6,
            label="FN Rate",
        )
        for x, y in zip(fn_x, fn_y):
            ax_fn.annotate(
                f"{y:.2f}",
                (x, y),
                textcoords="offset points", xytext=(0, 8),
                fontsize=7, color=ACCENT_YELLOW, ha="center",
            )
        ax_fn.set_ylim(0, min(1.0, max(fn_y) * 1.3 + 0.05))
        ax_fn.legend(fontsize=9, facecolor=AXES_BG, labelcolor=TEXT_COLOR, edgecolor=GRID_COLOR)
    else:
        ax_fn.text(
            0.5, 0.5, "No FN rate data available",
            ha="center", va="center", color=MUTED_COLOR, fontsize=10,
            transform=ax_fn.transAxes,
        )
    ax_fn.set_xlabel(fn_x_label)
    ax_fn.set_ylabel("False Negative Rate")
    ax_fn.set_title("False Negative Rate Over Iterations")
    _apply_theme(ax_fn)

    # ---- Save or show -------------------------------------------------------
    if export_png:
        export_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(export_png), dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        print(f"✅ Dashboard saved → {export_png}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Stats printing
# ---------------------------------------------------------------------------

def _print_summary_stats(iterations: list[dict[str, Any]]) -> None:
    """Print headline stats to stdout."""
    if not iterations:
        print("⚠️  No iteration data to summarise")
        return

    sorted_iters = sorted(iterations, key=lambda r: r["cumulative_labels"])
    latest = sorted_iters[-1]
    first = sorted_iters[0]

    current_map = latest.get("map50", 0.0) or 0.0
    total_labels = latest.get("cumulative_labels", 0) or 0
    first_map = first.get("map50", 0.0) or 0.0

    print(f"\n📊 Dashboard Summary:")
    print(f"   Current mAP50            : {current_map:.4f}")
    print(f"   Total labels used        : {total_labels:,}")

    if first_map > 0:
        efficiency = (current_map - first_map) / first_map * 100
        print(f"   mAP improvement vs it 1  : +{efficiency:.1f}%")
    else:
        print(f"   mAP improvement vs it 1  : n/a (baseline mAP50 is 0)")

    # Project labels needed for 95% of current max mAP (simple linear extrapolation)
    if len(sorted_iters) >= 2:
        target_map = 0.95
        map_vals = [r["map50"] or 0.0 for r in sorted_iters]
        label_vals = [r["cumulative_labels"] or 0 for r in sorted_iters]

        if current_map >= target_map:
            print(f"   Labels to mAP50 >= {target_map:.0%}   : already achieved ✅")
        else:
            # Fit a linear trend to the last two points
            if len(sorted_iters) >= 2 and (label_vals[-1] - label_vals[-2]) > 0:
                map_slope = (map_vals[-1] - map_vals[-2]) / (label_vals[-1] - label_vals[-2])
                if map_slope > 0:
                    labels_needed = total_labels + (target_map - current_map) / map_slope
                    print(
                        f"   Projected labels to {target_map:.0%} mAP : "
                        f"~{labels_needed:,.0f} (linear extrapolation)"
                    )
                else:
                    print(f"   Projected labels to {target_map:.0%} mAP : n/a (no recent improvement)")
            else:
                print(f"   Projected labels to {target_map:.0%} mAP : insufficient data")
    else:
        print(f"   Projected labels to 95% mAP : insufficient data (need >= 2 iterations)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render a 2x2 bootstrapping progress dashboard. "
            "Loads data from Supabase or falls back to local JSON files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--export-png",
        type=Path,
        default=None,
        metavar="PATH",
        help="Save dashboard to PNG instead of opening an interactive window",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    export_png: Path | None = args.export_png.resolve() if args.export_png else None

    # ---- Load iteration data ------------------------------------------------
    print("🔍 Attempting to load iteration data from Supabase...")
    iterations = _load_iterations_from_supabase()

    if not iterations:
        print("⚠️  Falling back to local metrics.json files...")
        iterations = _load_iterations_from_local()

    if not iterations:
        print("❌ No iteration data found in Supabase or local files.")
        print("   Create data/bootstrapped/iteration_N/metrics.json files or populate")
        print("   the bootstrap_iterations Supabase table.")
        sys.exit(1)

    # ---- Load FN report -----------------------------------------------------
    print("🔍 Attempting to load FN report from Supabase...")
    fn_report = _load_fn_report_from_supabase()

    if fn_report is None:
        print("⚠️  Falling back to local fn_quantification_report.json...")
        fn_report = _load_fn_report_from_local()

    # ---- Print stats --------------------------------------------------------
    _print_summary_stats(iterations)

    # ---- Render dashboard ---------------------------------------------------
    print("\n🔍 Rendering dashboard...")
    _build_dashboard(iterations, fn_report, export_png)


if __name__ == "__main__":
    main()
