"""
tools/active_learning/spatial_diversity_sampler.py

Augments the uncertainty sampler with geographic diversity constraints.
Prevents the active learning loop from wasting its labeling budget
on 50 images from the same city block.

Combines uncertainty score + spatial spread to select a maximally
informative AND geographically diverse query batch.

Usage:
  python tools/active_learning/spatial_diversity_sampler.py \
    --unlabeled-dir data/unlabeled \
    --uncertainty-scores data/bootstrapped/uncertainty_scores.json \
    --budget 50
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))


class SpatialDiversitySampler:
    """
    Selects query batch using combined uncertainty + spatial diversity.
    Uses a greedy algorithm: pick the next image that maximizes
    (alpha * uncertainty) + (1-alpha * spatial_distance_from_selected).
    Think of it as: pick the most uncertain image that's also
    as far as possible from images you've already selected.
    """

    def __init__(self, alpha: float = 0.6):
        """
        Args:
            alpha: Weight for uncertainty vs diversity (0=pure diversity, 1=pure uncertainty)
                   Default 0.6 = 60% uncertainty, 40% spatial diversity
        """
        self.alpha = alpha

    def load_uncertainty_scores(self, scores_path: Path) -> Dict[str, float]:
        """Load uncertainty scores from active learning loop output."""
        data = json.loads(scores_path.read_text())
        if isinstance(data, list):
            return {item["image"]: item["uncertainty_score"] for item in data}
        return data

    def load_chip_coordinates(self, chip_index_dir: Path) -> Dict[str, Tuple[float, float]]:
        """
        Load geographic coordinates for each chip from chip index files.
        Returns {chip_name: (lat, lon)} dict.
        """
        coords = {}
        for index_file in chip_index_dir.glob("*_chip_index.json"):
            for entry in json.loads(index_file.read_text()):
                if "lat_min" in entry and "lon_min" in entry:
                    coords[entry["chip_name"]] = (entry["lat_min"], entry["lon_min"])
        return coords

    def _pixel_coords_from_name(self, chip_name: str) -> Tuple[float, float]:
        """
        Extract approximate pixel coordinates from chip filename.
        Format: {scene_id}_r{row}_c{col}.jpg
        Used when geo coordinates are unavailable.
        """
        try:
            parts = Path(chip_name).stem.split("_")
            row = float(next(p[1:] for p in parts if p.startswith("r") and p[1:].isdigit()))
            col = float(next(p[1:] for p in parts if p.startswith("c") and p[1:].isdigit()))
            return row, col
        except Exception:
            return np.random.uniform(0, 10000), np.random.uniform(0, 10000)

    def select_diverse_batch(
        self,
        uncertainty_scores: Dict[str, float],
        coordinates: Optional[Dict[str, Tuple[float, float]]],
        budget: int,
        top_k_pool: int = 500
    ) -> List[Dict]:
        """
        Greedy selection of budget images balancing uncertainty and spatial diversity.

        Args:
            uncertainty_scores: {image_name: uncertainty_score}
            coordinates: {image_name: (lat, lon)} — optional
            budget: Number of images to select
            top_k_pool: Pre-filter to top-K most uncertain before diversity selection

        Returns:
            List of selected image dicts with scores
        """
        # Pre-filter to top-K most uncertain (efficiency)
        sorted_by_uncertainty = sorted(uncertainty_scores.items(), key=lambda x: x[1], reverse=True)
        pool = sorted_by_uncertainty[:top_k_pool]

        if len(pool) <= budget:
            return [{"image": name, "uncertainty": score, "selection_reason": "pool_smaller_than_budget"}
                    for name, score in pool]

        # Normalize uncertainty scores to 0-1
        scores = np.array([s for _, s in pool])
        names = [n for n, _ in pool]
        norm_uncertainty = (scores - scores.min()) / max(scores.max() - scores.min(), 1e-6)

        # Get coordinates for the pool
        def get_coord(name):
            if coordinates and name in coordinates:
                return coordinates[name]
            return self._pixel_coords_from_name(name)

        coords_array = np.array([get_coord(n) for n in names])

        # Normalize coordinates to 0-1
        coord_range = coords_array.max(axis=0) - coords_array.min(axis=0)
        coord_range = np.where(coord_range == 0, 1, coord_range)
        norm_coords = (coords_array - coords_array.min(axis=0)) / coord_range

        # Greedy selection
        selected_indices = []
        remaining = list(range(len(names)))

        # Start with highest uncertainty
        first = int(np.argmax(norm_uncertainty))
        selected_indices.append(first)
        remaining.remove(first)

        while len(selected_indices) < budget and remaining:
            selected_coords = norm_coords[selected_indices]

            # For each remaining candidate, compute min distance to any selected
            best_score, best_idx = -1, -1
            for idx in remaining:
                distances = np.linalg.norm(selected_coords - norm_coords[idx], axis=1)
                min_dist = float(distances.min())
                combined = self.alpha * norm_uncertainty[idx] + (1 - self.alpha) * min_dist
                if combined > best_score:
                    best_score, best_idx = combined, idx

            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        # Build result
        selected = []
        for idx in selected_indices:
            lat, lon = get_coord(names[idx])
            selected.append({
                "image": names[idx],
                "uncertainty_score": round(float(scores[idx]), 4),
                "normalized_uncertainty": round(float(norm_uncertainty[idx]), 4),
                "lat": lat, "lon": lon,
                "selection_rank": len(selected) + 1
            })

        return selected

    def run(self, unlabeled_dir: Path, uncertainty_path: Path, budget: int, output_path: Optional[Path] = None) -> List[Dict]:
        """Full pipeline: load scores + coords, select batch, save result."""
        print(f"\n🌍 Spatial Diversity Sampler")
        print(f"   Budget: {budget} images | Alpha: {self.alpha} (uncertainty/diversity balance)")

        uncertainty_scores = self.load_uncertainty_scores(uncertainty_path)
        print(f"   Loaded {len(uncertainty_scores)} uncertainty scores")

        coordinates = self.load_chip_coordinates(unlabeled_dir)
        print(f"   Loaded {len(coordinates)} geo-coordinates")

        if not coordinates:
            print("   ⚠️  No geo coordinates found — using pixel coordinates from filenames")

        selected = self.select_diverse_batch(uncertainty_scores, coordinates or None, budget)

        print(f"\n✅ Selected {len(selected)} images")

        # Compute coverage stats
        if all(s.get("lat") for s in selected):
            lats = [s["lat"] for s in selected]
            lons = [s["lon"] for s in selected]
            lat_range = max(lats) - min(lats)
            lon_range = max(lons) - min(lons)
            print(f"   Geographic coverage: {lat_range:.4f}° lat × {lon_range:.4f}° lon")

        avg_uncertainty = np.mean([s["uncertainty_score"] for s in selected])
        print(f"   Average uncertainty: {avg_uncertainty:.4f}")

        output_path = output_path or unlabeled_dir / "diverse_query_batch.json"
        output_path.write_text(json.dumps(selected, indent=2))
        print(f"📋 Query batch saved: {output_path}")
        return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlabeled-dir", default=str(DATA_DIR / "unlabeled"))
    parser.add_argument("--uncertainty-scores", required=True)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    sampler = SpatialDiversitySampler(alpha=args.alpha)
    sampler.run(
        unlabeled_dir=Path(args.unlabeled_dir),
        uncertainty_path=Path(args.uncertainty_scores),
        budget=args.budget,
        output_path=Path(args.output) if args.output else None
    )
