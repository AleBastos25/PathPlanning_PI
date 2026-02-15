"""Run scenario1 with basic, restart, SA, and DL configs; print comparison."""
import csv
import sys
from pathlib import Path
from dataclasses import replace

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.algorithms.pso import PSOConfig
from src.utils.experiments import run_experiment

SCENARIO_PATH = _project_root / "scenarios" / "scenario1.txt"
OUTPUT_DIR = _project_root / "output" / SCENARIO_PATH.stem
SAVE_PLOTS = True
USE_RADIUS = False

def main():
    base = PSOConfig(
    n_particles=30,
    n_waypoints=7,
    max_iters=500,
    seed=42,
    w=.75,
    c1=1.25,
    c2=1.25,
    vmax=None,
    penalty_factor=10000000000000.0,
    init_line_prob=0,


    use_restart=False,
    use_sa=False,
    use_dimlearn=False,
    init_avoid_obstacles=False,
    restart_period_T=10,
    dl_patience_Q=10,
    sa_beta=0.92,
    )

    configs = [
        ("basic", base),
        ("restart", replace(base, use_restart=True)),
        ("sa", replace(base, use_sa=True)),
        ("dl", replace(base, use_dimlearn=True)),
        ("all", replace(base, use_restart=True, use_sa=True, use_dimlearn=True)),
    ]
    print("=" * 55)
    print(f"Scenario: {SCENARIO_PATH.name}")
    print("=" * 55)
    results = []
    for name, config in configs:
        r = run_experiment(
            SCENARIO_PATH,
            config,
            output_dir=OUTPUT_DIR if SAVE_PLOTS else None,
            save_plots=SAVE_PLOTS,
            config_name=name,
            radius=USE_RADIUS,
            verbose=True,
            plot_feasible_colors=True,
        )
        results.append((name, r))
    print()
    print(f"{'Config':<10} {'path_length':>12} {'score':>12} {'converged':>10}")
    print("-" * 48)
    for name, r in results:
        print(f"{name:<10} {r.path_length:>12.2f} {r.fitness:>12.2f} {str(r.converged):>10}")
    print("=" * 55)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "results.csv"
    fieldnames = ["config", "scenario", "n_particles", "path_length", "L_coll", "iterations", "cpu_time_sec", "converged"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, r in results:
            writer.writerow({
                "config": name,
                "scenario": SCENARIO_PATH.stem,
                "n_particles": r.n_particles,
                "path_length": round(r.path_length, 4),
                "L_coll": round(r.L_coll, 4),
                "iterations": r.iterations,
                "cpu_time_sec": round(r.cpu_time, 4),
                "converged": r.converged,
            })
    print(f"CSV saved to: {csv_path}")

if __name__ == "__main__":
    main()
