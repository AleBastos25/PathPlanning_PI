"""Main entry point for PathPlanning_PI."""
import sys
import os
from pathlib import Path

# Add project root to path if executed directly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.experiments import run_all_experiments

def main():
    print("PathPlanning_PI - Path Planning Algorithms")
    print("=" * 40)
    
    # Configuration
    base_dir = Path(__file__).parent.parent
    scenarios_dir = base_dir / "scenarios"
    output_dir = base_dir / "output/results"
    
    # MANUAL CONTROL OF WAYPOINTS (m) PER EXPERIMENT
    # Edit this dictionary to override default m=10
    # Format: "filename": number_of_waypoints
    manual_waypoints = {
        # Example overrides:
        "scenario1.txt": 4, 
        "scenario2.txt": 4,
    }
    
    print(f"Running experiments from {scenarios_dir}...")
    if manual_waypoints:
        print("Manual waypoint overrides active:")
        for k, v in manual_waypoints.items():
            print(f"  - {k}: m={v}")
            
    run_all_experiments(
        scenarios_dir=scenarios_dir,
        output_dir=output_dir,
        manual_waypoints=manual_waypoints
    )
    
    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()
