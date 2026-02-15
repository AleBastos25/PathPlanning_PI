"""Run a single scenario with one PSOConfig (edit config at top)."""
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.algorithms.pso import PSOConfig
from src.utils.experiments import run_experiment

# Edit: which scenario (0-4) and config
SCENARIO_INDEX = 0
OUTPUT_DIR = _project_root / "output" / "results"
SAVE_PLOTS = True
USE_RADIUS = False

def main():
    scenarios_dir = _project_root / "scenarios"
    scenario_files = sorted(scenarios_dir.glob("scenario*.txt"))
    if not scenario_files or SCENARIO_INDEX < 0 or SCENARIO_INDEX >= len(scenario_files):
        print(f"SCENARIO_INDEX must be 0..{len(scenario_files) - 1}")
        return
    scenario_path = scenario_files[SCENARIO_INDEX]
    config = PSOConfig(
        n_particles=10,
        n_waypoints=4,
        max_iters=100,
        seed=42,
        use_restart=False,
        use_sa=False,
        use_dimlearn=False,
    )
    print("=" * 50)
    print(f"Cenário: {scenario_path.name}")
    print("=" * 50)
    result = run_experiment(
        scenario_path,
        config,
        output_dir=OUTPUT_DIR if SAVE_PLOTS else None,
        save_plots=SAVE_PLOTS,
        radius=USE_RADIUS,
        verbose=True,
    )
    print("\nConcluído.")
    return result

if __name__ == "__main__":
    main()
