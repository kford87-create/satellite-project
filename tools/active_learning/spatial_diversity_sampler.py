"""
spatial_diversity_sampler.py
-----------------------------
Selects a spatially diverse, uncertainty-weighted batch of images for
active-learning annotation.

Usage:
    python tools/active_learning/spatial_diversity_sampler.py \
      --unlabeled-dir data/unlabeled \
      --uncertainty-scores data/bootstrapped/uncertainty_scores.json \
      --budget 50 --alpha 0.6
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
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


def _supabase_write_batch(batch: list[dict[str, Any]]) -> None:
    """Non-fatal write of selected batch metadata to Supabase."""
    try:
        if _supabase is None:
            return
        rows = [
            {
                "filename": item["filename"],
                "uncertainty_score": item["uncertainty_score"],
                "diversity_score": item["diversity_score"],
                "combined_score": item["combined_score"],
                "lat": (item["geo_coords"] or {}).get("lat"),
                "lng": (item["geo_coords"] or {}).get("lng"),
            }
            for item in batch
        ]
        _supabase.table("diversity_query_batches").insert(rows).execute()
    except Exception as exc:
        print(f"⚠️  Supabase write skipped (falling back to local only): {exc}")


# ---------------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------------

def _parse_filename_pseudo_spatial(filename: str, total: int, idx: int) -> tuple[float, float]:
    """
    Extract pseudo-spatial coordinates from a filename that encodes row/col.

    Chip filenames produced by chip_generator.py follow the pattern:
        <scene_id>_<RRRR>_<CCCC>.jpg
    e.g. scene_0023_0005.jpg  →  row=23, col=5

    If the pattern is not matched we fall back to (idx/total, idx/total).
    """
    stem = Path(filename).stem
    parts = stem.rsplit("_", 2)
    if len(parts) == 3:
        try:
            row_idx = int(parts[1])
            col_idx = int(parts[2])
            # Estimate a rough "total grid size" from the observed max
            denom = max(total, 1)
            return float(row_idx) / denom, float(col_idx) / denom
        except ValueError:
            pass
    # Ultimate fallback
    safe_total = max(total, 1)
    return float(idx) / safe_total, float(idx) / safe_total


def _load_geo_coords(
    filename: str,
    chip_index_lookup: dict[str, dict[str, Any]],
) -> dict[str, float] | None:
    """
    Return {"lat": ..., "lng": ...} for *filename* if geo metadata is available,
    else return None.

    chip_index_lookup maps filename → chip metadata dict (may include geo_bounds).
    """
    meta = chip_index_lookup.get(filename)
    if meta is None:
        return None
    bounds = meta.get("geo_bounds")
    if not bounds:
        return None
    try:
        lat = (bounds["north"] + bounds["south"]) / 2.0
        lng = (bounds["west"] + bounds["east"]) / 2.0
        return {"lat": lat, "lng": lng}
    except (KeyError, TypeError):
        return None


def _build_chip_index_lookup(unlabeled_dir: Path) -> dict[str, dict[str, Any]]:
    """
    Scan *unlabeled_dir* (and one level up) for *_chip_index.json files and
    return a flat mapping from filename → chip-meta dict.
    """
    lookup: dict[str, dict[str, Any]] = {}
    search_dirs = [unlabeled_dir, unlabeled_dir.parent]
    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for index_file in search_dir.glob("*_chip_index.json"):
            try:
                data = json.loads(index_file.read_text())
                for chip in data.get("chips", []):
                    fname = chip.get("filename")
                    if fname:
                        lookup[fname] = chip
            except Exception as exc:
                print(f"⚠️  Could not read chip index {index_file.name}: {exc}")
    return lookup


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def _euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _min_distance_to_selected(
    candidate_coord: tuple[float, float],
    selected_coords: list[tuple[float, float]],
) -> float:
    """Return min Euclidean distance from *candidate_coord* to any already-selected point."""
    if not selected_coords:
        return 1.0  # Treat as maximally diverse when set is empty
    return min(_euclidean_distance(candidate_coord, s) for s in selected_coords)


# ---------------------------------------------------------------------------
# Core sampler
# ---------------------------------------------------------------------------

def run_sampler(
    unlabeled_dir: Path,
    uncertainty_scores_path: Path,
    budget: int,
    alpha: float,
    output_dir: Path,
) -> list[dict[str, Any]]:
    # ---- Load uncertainty scores -----------------------------------------
    if not uncertainty_scores_path.exists():
        print(f"❌ Uncertainty scores file not found: {uncertainty_scores_path}")
        sys.exit(1)

    raw_scores: dict[str, float] = json.loads(uncertainty_scores_path.read_text())
    print(f"📊 Loaded uncertainty scores for {len(raw_scores):,} images")

    # ---- Discover images in unlabeled dir -----------------------------------
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    all_images = [
        p for p in sorted(unlabeled_dir.iterdir())
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    if not all_images:
        print(f"❌ No images found in: {unlabeled_dir}")
        sys.exit(1)

    print(f"📊 Found {len(all_images):,} images in {unlabeled_dir}")

    # ---- Build chip index lookup for geo coords -----------------------------
    chip_index_lookup = _build_chip_index_lookup(unlabeled_dir)
    geo_available = sum(
        1 for img in all_images if _load_geo_coords(img.name, chip_index_lookup) is not None
    )
    if geo_available > 0:
        print(f"📊 Geo coords available for {geo_available:,}/{len(all_images):,} images")
    else:
        print("⚠️  No chip index geo data found — using pseudo-spatial proxy from filenames")

    # ---- Build candidate list -----------------------------------------------
    n_total = len(all_images)
    candidates: list[dict[str, Any]] = []

    for idx, img_path in enumerate(tqdm(all_images, desc="Building candidate list", unit="img")):
        fname = img_path.name
        uncertainty = float(raw_scores.get(fname, 0.0))

        geo_coords = _load_geo_coords(fname, chip_index_lookup)
        if geo_coords is not None:
            coord: tuple[float, float] = (geo_coords["lat"], geo_coords["lng"])
        else:
            row_proxy, col_proxy = _parse_filename_pseudo_spatial(fname, n_total, idx)
            coord = (row_proxy, col_proxy)
            geo_coords = None  # Explicitly None in output

        candidates.append(
            {
                "filename": fname,
                "uncertainty_score": uncertainty,
                "coord": coord,
                "geo_coords": geo_coords,
            }
        )

    # ---- Normalise spatial coords to [0,1]² ---------------------------------
    coords_arr = np.array([c["coord"] for c in candidates], dtype=np.float64)
    coord_min = coords_arr.min(axis=0)
    coord_max = coords_arr.max(axis=0)
    coord_range = coord_max - coord_min
    coord_range[coord_range == 0] = 1.0  # Avoid div-by-zero for degenerate data

    for i, c in enumerate(candidates):
        norm_coord = (coords_arr[i] - coord_min) / coord_range
        c["norm_coord"] = (float(norm_coord[0]), float(norm_coord[1]))

    # ---- Greedy diversity-weighted selection --------------------------------
    effective_budget = min(budget, len(candidates))
    if effective_budget < budget:
        print(
            f"⚠️  Budget ({budget}) exceeds pool size ({len(candidates)}); "
            f"selecting all {effective_budget} images"
        )

    selected_indices: list[int] = []
    selected_norm_coords: list[tuple[float, float]] = []

    # Tracking sets for fast exclusion
    remaining_indices = list(range(len(candidates)))

    for step in tqdm(range(effective_budget), desc="Greedy selection", unit="img"):
        best_idx_in_remaining: int | None = None
        best_score = -1.0

        # For the first selection, spatial diversity = 1 for all; just pick max uncertainty
        for pool_pos, cand_idx in enumerate(remaining_indices):
            cand = candidates[cand_idx]
            uncertainty = cand["uncertainty_score"]

            raw_div = _min_distance_to_selected(cand["norm_coord"], selected_norm_coords)
            # raw_div is already in [0, sqrt(2)] for normalised coords; scale to ~[0,1]
            diversity_score = min(raw_div / math.sqrt(2.0), 1.0)

            combined = alpha * uncertainty + (1.0 - alpha) * diversity_score
            if combined > best_score:
                best_score = combined
                best_idx_in_remaining = pool_pos

        chosen_pool_pos = best_idx_in_remaining  # type: ignore[assignment]
        chosen_cand_idx = remaining_indices[chosen_pool_pos]
        chosen = candidates[chosen_cand_idx]

        # Recompute final diversity score for output record
        raw_div_final = _min_distance_to_selected(chosen["norm_coord"], selected_norm_coords)
        final_diversity = min(raw_div_final / math.sqrt(2.0), 1.0)
        final_combined = alpha * chosen["uncertainty_score"] + (1.0 - alpha) * final_diversity

        chosen["diversity_score"] = round(final_diversity, 6)
        chosen["combined_score"] = round(final_combined, 6)

        selected_indices.append(chosen_cand_idx)
        selected_norm_coords.append(chosen["norm_coord"])
        remaining_indices.pop(chosen_pool_pos)

    # ---- Build output records -----------------------------------------------
    selected_batch: list[dict[str, Any]] = []
    for idx in selected_indices:
        c = candidates[idx]
        selected_batch.append(
            {
                "filename": c["filename"],
                "uncertainty_score": round(float(c["uncertainty_score"]), 6),
                "diversity_score": c.get("diversity_score", 0.0),
                "combined_score": c.get("combined_score", 0.0),
                "geo_coords": c["geo_coords"],
            }
        )

    # ---- Statistics ---------------------------------------------------------
    unc_vals = np.array([r["uncertainty_score"] for r in selected_batch])
    div_vals = np.array([r["diversity_score"] for r in selected_batch])
    comb_vals = np.array([r["combined_score"] for r in selected_batch])

    print(f"\n📊 Selection statistics (n={len(selected_batch)}):")
    print(f"   Uncertainty  — mean={unc_vals.mean():.4f}  std={unc_vals.std():.4f}  "
          f"min={unc_vals.min():.4f}  max={unc_vals.max():.4f}")
    print(f"   Diversity    — mean={div_vals.mean():.4f}  std={div_vals.std():.4f}  "
          f"min={div_vals.min():.4f}  max={div_vals.max():.4f}")
    print(f"   Combined     — mean={comb_vals.mean():.4f}  std={comb_vals.std():.4f}  "
          f"min={comb_vals.min():.4f}  max={comb_vals.max():.4f}")
    print(f"   Alpha (uncertainty weight): {alpha}  |  1-alpha (diversity weight): {1 - alpha:.1f}")

    # Geographic spread
    geo_selected = [r for r in selected_batch if r["geo_coords"] is not None]
    if geo_selected:
        lats = [r["geo_coords"]["lat"] for r in geo_selected]
        lngs = [r["geo_coords"]["lng"] for r in geo_selected]
        lat_span = max(lats) - min(lats)
        lng_span = max(lngs) - min(lngs)
        print(f"\n📊 Geographic spread ({len(geo_selected)} images with real coords):")
        print(f"   Lat range : {min(lats):.5f} → {max(lats):.5f}  (span {lat_span:.5f}°)")
        print(f"   Lng range : {min(lngs):.5f} → {max(lngs):.5f}  (span {lng_span:.5f}°)")
    else:
        norm_coords_sel = np.array([candidates[i]["norm_coord"] for i in selected_indices])
        row_span = norm_coords_sel[:, 0].max() - norm_coords_sel[:, 0].min()
        col_span = norm_coords_sel[:, 1].max() - norm_coords_sel[:, 1].min()
        print(f"\n📊 Pseudo-spatial spread (normalised [0,1]):")
        print(f"   Row proxy span: {row_span:.4f}  Col proxy span: {col_span:.4f}")

    # ---- Save output --------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "diverse_query_batch.json"
    out_path.write_text(json.dumps(selected_batch, indent=2))
    print(f"\n✅ Saved {len(selected_batch)} selected images → {out_path}")

    # ---- Supabase (non-fatal) -----------------------------------------------
    _supabase_write_batch(selected_batch)

    return selected_batch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Greedy spatial-diversity + uncertainty sampler for active learning. "
            "Selects a budget of images that balances model uncertainty with "
            "geographic/spatial diversity."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--unlabeled-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory of unlabeled image chips",
    )
    parser.add_argument(
        "--uncertainty-scores",
        required=True,
        type=Path,
        metavar="JSON",
        help="JSON file mapping {filename: uncertainty_score} (scores in [0,1])",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=50,
        metavar="N",
        help="Number of images to select",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        metavar="FLOAT",
        help="Weight for uncertainty score (1-alpha goes to spatial diversity)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory to write diverse_query_batch.json "
            "(defaults to parent directory of --uncertainty-scores)"
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    unlabeled_dir: Path = args.unlabeled_dir.resolve()
    uncertainty_scores_path: Path = args.uncertainty_scores.resolve()
    output_dir: Path = (
        args.output_dir.resolve()
        if args.output_dir
        else uncertainty_scores_path.parent
    )

    if not unlabeled_dir.is_dir():
        print(f"❌ --unlabeled-dir does not exist: {unlabeled_dir}")
        sys.exit(1)

    if not 0.0 <= args.alpha <= 1.0:
        print(f"❌ --alpha must be in [0, 1], got {args.alpha}")
        sys.exit(1)

    if args.budget <= 0:
        print(f"❌ --budget must be a positive integer, got {args.budget}")
        sys.exit(1)

    print(f"📊 Unlabeled dir : {unlabeled_dir}")
    print(f"📊 Uncertainty   : {uncertainty_scores_path}")
    print(f"📊 Budget        : {args.budget}")
    print(f"📊 Alpha         : {args.alpha}")
    print(f"📊 Output dir    : {output_dir}")
    print()

    run_sampler(
        unlabeled_dir=unlabeled_dir,
        uncertainty_scores_path=uncertainty_scores_path,
        budget=args.budget,
        alpha=args.alpha,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
