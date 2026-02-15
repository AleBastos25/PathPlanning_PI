"""
Q8: For every scenario in test files, run basic PSO (no improvements).
Output: path length, iterations, CPU time. Saves images and a CSV in output/Q8.
Basic = defaults only: no restart, SA, DL, seed_position, init_avoid_obstacles;
no custom init_line_prob, vmax, etc.
"""
import sys
import csv
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.algorithms.pso import PSOConfig
from src.utils.experiments import run_experiment

SCENARIOS_DIR = _project_root / "scenarios"
OUTPUT_DIR = _project_root / "output" / "Q8"
CONFIG_NAME = "basic"


def get_scenario_files():
    """Return sorted list of scenario*.txt paths."""
    if not SCENARIOS_DIR.exists():
        return []
    files = sorted(SCENARIOS_DIR.glob("scenario*.txt"))
    return files


def main():
    # Strictly basic config: no improvements, no custom init/vmax/avoid_obstacles
    basic_config = PSOConfig(
        n_particles=30,
        n_waypoints=7,
        max_iters=500,
        w=.75,
        c1=1.25,
        c2=1.25,
        vmax=None,
        penalty_factor=10000000000000.0,
        seed=42,
        init_line_prob=0,
        use_restart=False,
        use_sa=False,
        use_dimlearn=False,
        init_avoid_obstacles=False,
    )

    scenario_paths = get_scenario_files()
    if not scenario_paths:
        print("No scenario*.txt found in", SCENARIOS_DIR)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    print("Q8: Basic PSO on all scenarios")
    print("=" * 60)
    for scenario_path in scenario_paths:
        name = scenario_path.stem
        if name == "scenario4" or name == "scenario3":
            basic_config.n_particles = 300
            basic_config.max_iters = 1000

        print(f"  Running {name}...", end=" ", flush=True)
        r = run_experiment(
            scenario_path,
            basic_config,
            output_dir=OUTPUT_DIR,
            save_plots=True,
            config_name=CONFIG_NAME,
            radius=False,
            verbose=False,
            plot_feasible_colors=True,
        )
        print(f"L={r.path_length:.2f} iters={r.iterations} time={r.cpu_time:.3f}s")
        rows.append({
            "scenario": name,
            "n_particles": r.n_particles,
            "path_length": r.path_length,
            "L_coll": r.L_coll,
            "iterations": r.iterations,
            "cpu_time_sec": round(r.cpu_time, 4),
            "converged": r.converged,
        })

    csv_path = OUTPUT_DIR / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scenario", "n_particles", "path_length", "L_coll", "iterations", "cpu_time_sec", "converged"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 60)
    print(f"Images saved to: {OUTPUT_DIR}")
    print(f"CSV saved to:   {csv_path}")


if __name__ == "__main__":
    main()
