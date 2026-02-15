import csv
import sys
from pathlib import Path
from dataclasses import replace
import numpy as np

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.algorithms.pso import PSOConfig
from src.utils.experiments import run_experiment

# Seed waypoints for scenario 4: one particle starts here to test convergence from a "close" path

SCENARIO_PATH = _project_root / "scenarios" / "scenario4.txt"
OUTPUT_DIR = _project_root / "output" / SCENARIO_PATH.stem
SAVE_PLOTS = True
USE_RADIUS = False

def main():
    # Scenario 4 is hard (10 obstacles, 1000x1000): need more particles, iters, and init along line
    base = PSOConfig(
        n_particles=1000,
        n_waypoints=8,
        max_iters=1000,
        seed=100,
        init_line_prob=.5,
        init_line_noise=1000,
        w=0.8,
        c1=1.3,
        c2=1.3,
        vmax=100,
        penalty_factor=1000000000000,
        restart_period_T=50,
        restart_keep_gbest=True,
        dl_patience_Q=300,
        sa_T0=10000,
        sa_beta=0.92,
        sa_Tmin=1e-12,
        init_avoid_obstacles=True,
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
