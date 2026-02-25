"""
scripts/06_false_negative_quantification.py

Theoretically quantifies the operational impact of false negative detections.

This directly addresses the SBIR Phase I requirement:
"theoretically quantify the impact of unintentional false negative detections
on detector performance"

A false negative = a real object the model MISSED detecting.
In satellite imagery applications this is often more dangerous than a false positive:
  - Insurance: Missing a damaged roof = incorrect claim payout
  - Defense: Missing a vehicle = operational gap in situational awareness
  - Construction monitoring: Missing new structure = missed permit violation

This module builds a mathematical model that answers:
"If our detector has X% false negative rate, what is the downstream
operational impact on decision-making?"
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from scipy import stats


@dataclass
class ApplicationProfile:
    """
    Defines the operational cost structure for a specific use case.
    Different applications have different costs for missing detections.
    """
    name: str
    false_negative_cost: float    # Cost of missing one real object
    false_positive_cost: float    # Cost of flagging one non-object
    objects_per_km2: float        # Expected object density in scene
    decision_threshold: float     # Min confidence to act on a detection
    description: str


# Pre-defined application profiles
APPLICATION_PROFILES = {
    "insurance_building_damage": ApplicationProfile(
        name="Insurance: Building Damage Assessment",
        false_negative_cost=15000.0,   # Avg cost of missed damage claim ($)
        false_positive_cost=500.0,     # Cost of unnecessary inspection ($)
        objects_per_km2=50.0,          # Buildings per km² in suburban area
        decision_threshold=0.5,
        description="Detect damaged buildings post-disaster for insurance claims"
    ),
    "defense_vehicle_tracking": ApplicationProfile(
        name="Defense: Vehicle Activity Monitoring",
        false_negative_cost=1.0,       # Normalized operational impact score
        false_positive_cost=0.1,       # Normalized cost of false alert
        objects_per_km2=5.0,           # Vehicles per km² in monitored area
        decision_threshold=0.4,
        description="Track vehicle movements at sensitive facilities"
    ),
    "construction_monitoring": ApplicationProfile(
        name="Commercial: Construction Site Monitoring",
        false_negative_cost=2500.0,    # Cost of missed unauthorized structure ($)
        false_positive_cost=200.0,     # Cost of unnecessary site visit ($)
        objects_per_km2=10.0,          # Structures per km² at construction sites
        decision_threshold=0.5,
        description="Detect new structures and changes at construction sites"
    ),
}


class FalseNegativeModel:
    """
    Mathematical model for quantifying false negative impact.

    Uses precision-recall analysis weighted by operational costs.
    """

    def __init__(self, application: ApplicationProfile):
        self.app = application

    def compute_expected_cost(
        self,
        false_negative_rate: float,
        false_positive_rate: float,
        scene_area_km2: float = 1.0
    ) -> Dict:
        """
        Compute expected operational cost for a given FN/FP rate.

        Args:
            false_negative_rate: Fraction of real objects missed (0-1)
            false_positive_rate: Fraction of detections that are wrong (0-1)
            scene_area_km2: Area being analyzed

        Returns:
            Cost breakdown dictionary
        """
        # Expected objects in scene
        n_real_objects = self.app.objects_per_km2 * scene_area_km2

        # Expected errors
        n_false_negatives = n_real_objects * false_negative_rate
        n_true_positives = n_real_objects * (1 - false_negative_rate)
        n_false_positives = n_true_positives * (false_positive_rate / max(1 - false_positive_rate, 1e-10))

        # Costs
        fn_cost = n_false_negatives * self.app.false_negative_cost
        fp_cost = n_false_positives * self.app.false_positive_cost
        total_cost = fn_cost + fp_cost

        return {
            "false_negative_rate": false_negative_rate,
            "false_positive_rate": false_positive_rate,
            "n_real_objects": n_real_objects,
            "n_false_negatives": n_false_negatives,
            "n_false_positives": n_false_positives,
            "fn_cost": fn_cost,
            "fp_cost": fp_cost,
            "total_cost": total_cost,
            "cost_dominated_by": "false_negatives" if fn_cost > fp_cost else "false_positives",
            "fn_to_fp_cost_ratio": fn_cost / max(fp_cost, 1e-10),
        }

    def compute_bootstrapping_impact(
        self,
        fn_rates_by_iteration: List[float],
        scene_area_km2: float = 100.0
    ) -> List[Dict]:
        """
        Model how bootstrapping iterations progressively reduce false negative cost.

        This is the key deliverable: showing that each labeling iteration
        reduces operational risk by a quantifiable amount.
        """
        results = []

        for i, fn_rate in enumerate(fn_rates_by_iteration):
            # As model improves, assume FP rate also improves (correlated)
            # Model: FP rate reduces proportionally with FN rate improvements
            fp_rate = fn_rate * 0.5  # Simplified correlation model

            cost = self.compute_expected_cost(fn_rate, fp_rate, scene_area_km2)
            cost["iteration"] = i
            cost["labels_used"] = i * 50  # 50 labels per iteration

            # Compute marginal value of this iteration's labels
            if i > 0:
                prev_cost = results[-1]["total_cost"]
                cost["cost_reduction"] = prev_cost - cost["total_cost"]
                cost["value_per_label"] = cost["cost_reduction"] / 50  # 50 labels per iter
            else:
                cost["cost_reduction"] = 0.0
                cost["value_per_label"] = 0.0

            results.append(cost)

        return results

    def theoretical_fn_floor(
        self,
        object_density: float,
        image_resolution_m: float = 0.5
    ) -> float:
        """
        Compute theoretical minimum false negative rate given physical constraints.

        Objects smaller than ~2x image resolution cannot be reliably detected.
        This sets a hard lower bound on achievable FN rate.

        Args:
            object_density: Objects per km²
            image_resolution_m: Ground sample distance in meters

        Returns:
            Theoretical minimum FN rate (floor)
        """
        # Objects smaller than ~3x3 pixels are unreliably detectable
        min_detectable_size_m = image_resolution_m * 3

        # Estimate fraction of objects below detection threshold
        # Assumes log-normal size distribution (common in satellite imagery)
        # Parameters estimated from SpaceNet building size statistics
        mean_size_m = 15.0   # Average building width ~15m
        std_size_m = 8.0     # Standard deviation

        # Log-normal parameters
        sigma = np.log(1 + (std_size_m / mean_size_m) ** 2) ** 0.5
        mu = np.log(mean_size_m) - sigma ** 2 / 2

        # Fraction of objects below minimum detectable size
        fn_floor = stats.lognorm.cdf(min_detectable_size_m, s=sigma, scale=np.exp(mu))

        return fn_floor


def generate_fn_report(
    output_dir: Path,
    fn_rates: List[float] = None,
    application_keys: List[str] = None
):
    """
    Generate comprehensive false negative quantification report.

    Args:
        output_dir: Where to save the report
        fn_rates: FN rates at each bootstrapping iteration (simulated if None)
        application_keys: Which application profiles to analyze
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if fn_rates is None:
        # Simulate typical bootstrapping improvement curve
        # Starts high, improves rapidly, then plateaus (diminishing returns)
        fn_rates = [0.45, 0.35, 0.27, 0.21, 0.17, 0.14, 0.12, 0.11, 0.10, 0.10]

    if application_keys is None:
        application_keys = list(APPLICATION_PROFILES.keys())

    all_results = {}

    print("\n🔬 False Negative Quantification Report")
    print("=" * 60)

    for app_key in application_keys:
        app = APPLICATION_PROFILES[app_key]
        model = FalseNegativeModel(app)

        print(f"\n📊 Application: {app.name}")

        # Compute impact across bootstrapping iterations
        iteration_results = model.compute_bootstrapping_impact(
            fn_rates_by_iteration=fn_rates,
            scene_area_km2=100.0
        )

        # Compute theoretical FN floor
        fn_floor = model.theoretical_fn_floor(
            object_density=app.objects_per_km2,
            image_resolution_m=0.5  # 50cm resolution (commercial satellite standard)
        )

        print(f"   Theoretical FN floor (physical limit): {fn_floor:.3f} ({fn_floor*100:.1f}%)")
        print(f"   Baseline FN rate (iteration 0):        {fn_rates[0]:.3f} ({fn_rates[0]*100:.1f}%)")
        print(f"   Final FN rate (iteration {len(fn_rates)-1}):         {fn_rates[-1]:.3f} ({fn_rates[-1]*100:.1f}%)")

        initial_cost = iteration_results[0]["total_cost"]
        final_cost = iteration_results[-1]["total_cost"]
        total_savings = initial_cost - final_cost
        total_labels = len(fn_rates) * 50

        print(f"\n   Cost Analysis (per 100 km² analysis area):")
        print(f"   Initial annual cost:   ${initial_cost:>12,.0f}")
        print(f"   Final annual cost:     ${final_cost:>12,.0f}")
        print(f"   Total cost reduction:  ${total_savings:>12,.0f}")
        print(f"   Labels used:           {total_labels}")
        print(f"   ROI per label:         ${total_savings/total_labels:>8,.0f}")

        all_results[app_key] = {
            "profile": asdict(app),
            "fn_floor": fn_floor,
            "iteration_results": iteration_results,
            "summary": {
                "initial_cost": initial_cost,
                "final_cost": final_cost,
                "total_savings": total_savings,
                "total_labels": total_labels,
                "roi_per_label": total_savings / total_labels
            }
        }

    # Save JSON report
    report_path = output_dir / "fn_quantification_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Full report saved to: {report_path}")

    # Generate visualization
    plot_fn_impact(fn_rates, all_results, output_dir)

    return all_results


def plot_fn_impact(fn_rates: List[float], results: Dict, output_dir: Path):
    """Generate visualization of FN impact across bootstrapping iterations."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("False Negative Impact: Bootstrapping Iterations", fontsize=14)

    iterations = list(range(len(fn_rates)))

    # Plot 1: FN rate reduction
    axes[0].plot(iterations, [r * 100 for r in fn_rates], "b-o", linewidth=2)
    axes[0].set_xlabel("Bootstrapping Iteration")
    axes[0].set_ylabel("False Negative Rate (%)")
    axes[0].set_title("FN Rate Reduction Over Iterations")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Cost reduction by application
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for i, (app_key, app_results) in enumerate(results.items()):
        costs = [r["total_cost"] for r in app_results["iteration_results"]]
        # Normalize for comparison
        normalized = [c / costs[0] for c in costs]
        app_name = APPLICATION_PROFILES[app_key].name.split(":")[0]
        axes[1].plot(iterations, normalized, f"-o", color=colors[i], label=app_name, linewidth=2)

    axes[1].set_xlabel("Bootstrapping Iteration")
    axes[1].set_ylabel("Normalized Operational Cost")
    axes[1].set_title("Cost Reduction by Application")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Value per label
    for i, (app_key, app_results) in enumerate(results.items()):
        vpls = [r.get("value_per_label", 0) for r in app_results["iteration_results"][1:]]
        app_name = APPLICATION_PROFILES[app_key].name.split(":")[0]
        axes[2].plot(iterations[1:], vpls, f"-o", color=colors[i], label=app_name, linewidth=2)

    axes[2].set_xlabel("Bootstrapping Iteration")
    axes[2].set_ylabel("$ Value Per Label Added")
    axes[2].set_title("Marginal Value of Each Labeling Iteration")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "fn_impact_visualization.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"📈 Visualization saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    output_dir = Path("./data/bootstrapped/fn_analysis")
    generate_fn_report(output_dir=output_dir)
