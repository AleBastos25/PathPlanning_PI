"""Experiment utilities for running and collecting PSO results."""

import csv
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import List, Optional

from ..loader import load_scenario
from ..visualization import plot_scenario
from ..algorithms.pso import PSOParams, PSOResult, pso_plan
from .plotting import plot_convergence


@dataclass
class ExperimentResult:
    scenario_name: str
    n_obstacles: int
    n_waypoints: int
    n_particles: int
    iterations: int
    path_length: float
    fitness: float
    cpu_time: float
    converged: bool
    L_coll: float
    L_out: float
    restarts_used: int = 1


def run_experiment(
    scenario_path: Path,
    params: PSOParams,
    output_dir: Optional[Path] = None,
    save_plots: bool = True,
    verbose: bool = True,
) -> ExperimentResult:
    scenario_id = scenario_path.stem.replace("scenario", "")
    
    if verbose:
        print(f"\nProcessing {scenario_path.name}...")
    
    # Load scenario
    scenario = load_scenario(scenario_path)
    
    if verbose:
        print(f"  Environment: {scenario.xmax} x {scenario.ymax}")
        print(f"  Obstacles: {len(scenario.obstacles)}")
        print(f"  Robot radius: {scenario.R}")
        if params and params.n_particles and params.n_waypoints:
            print(f"  Running PSO with {params.n_particles} particles, {params.n_waypoints} waypoints...")
        else:
            print(f"  Running PSO with auto-configuration...")
    
    # Run PSO
    result = pso_plan(scenario, params)
    
    if verbose:
        print(f"  Converged: {result.converged}")
        print(f"  Waypoints used: {result.n_waypoints}")
        print(f"  Particles used: {result.n_particles}")
        print(f"  Restarts: {result.restarts_used}")
        print(f"  Iterations: {result.iters_used}")
        print(f"  CPU Time: {result.cpu_time:.3f}s")
        print(f"  Path Length: {result.L:.2f}")
        print(f"  Fitness: {result.best_f:.2f}")
        if result.L_coll > 0:
            print(f"  L_coll: {result.L_coll:.4f} (WARNING: collision!)")
    
    # Save plots
    if save_plots and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Path plot
        path_plot_file = output_dir / f"scenario_{scenario_id}_pso_path.png"
        plot_scenario(
            scenario,
            path=result.best_path,
            save_to=path_plot_file,
            show=False,
        )
        if verbose:
            print(f"  Saved: {path_plot_file.name}")
        
        # Convergence plot
        conv_plot_file = output_dir / f"scenario_{scenario_id}_pso_convergence.png"
        plot_convergence(result.history, scenario_id, conv_plot_file)
        if verbose:
            print(f"  Saved: {conv_plot_file.name}")
    
    return ExperimentResult(
        scenario_name=scenario_path.name,
        n_obstacles=len(scenario.obstacles),
        n_waypoints=result.n_waypoints,
        n_particles=result.n_particles,
        iterations=result.iters_used,
        path_length=result.L,
        fitness=result.best_f,
        cpu_time=result.cpu_time,
        converged=result.converged,
        L_coll=result.L_coll,
        L_out=result.L_out,
        restarts_used=result.restarts_used,
    )


def run_all_experiments(
    scenarios_dir: Path,
    params: Optional[PSOParams] = None,
    output_dir: Optional[Path] = None,
    save_plots: bool = True,
    verbose: bool = True,
    manual_waypoints: Optional[dict[str, int]] = None,
) -> List[ExperimentResult]:
    """Run PSO experiments on all scenarios in a directory.
    
    Args:
        scenarios_dir: Directory containing scenario files.
        params: PSO parameters. If None, uses auto-configuration.
        output_dir: Directory to save output files.
        save_plots: Whether to save plots.
        verbose: Whether to print progress.
        manual_waypoints: Dict mapping scenario filename (e.g. 'scenario1.txt') to waypoints count.
    
    Returns:
        List of ExperimentResult for all scenarios.
    """
    scenario_files = sorted(scenarios_dir.glob("scenario*.txt"))
    
    if verbose:
        print(f"Found {len(scenario_files)} scenarios: {[f.stem for f in scenario_files]}")
        if params is None:
            print("Using auto-configuration for PSO parameters.")
    
    results = []
    best_overall_feasible: Optional[ExperimentResult] = None
    
    for scenario_file in scenario_files:
        current_params = params
        
        # Apply manual overrides if specified
        if manual_waypoints and scenario_file.name in manual_waypoints:
            m_override = manual_waypoints[scenario_file.name]
            if verbose:
                 print(f"  [Override] Using manual waypoints m={m_override} for {scenario_file.name}")
            
            if current_params is None:
                current_params = PSOParams(n_waypoints=m_override)
            else:
                current_params = replace(current_params, n_waypoints=m_override)

        result = run_experiment(
            scenario_file, current_params, output_dir, save_plots, verbose
        )
        results.append(result)
        
        # Track global best feasible across all scenarios
        if result.converged:
             if best_overall_feasible is None or result.path_length < best_overall_feasible.path_length:
                 best_overall_feasible = result
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Completed {len(results)} experiments.")
        
        if best_overall_feasible:
            print(f"BEST OVERALL FEASIBLE SOLUTION:")
            print(f"  Scenario: {best_overall_feasible.scenario_name}")
            print(f"  Length:   {best_overall_feasible.path_length:.4f}")
            print(f"  Fitness:  {best_overall_feasible.fitness:.4f}")
        else:
            print("NO FEASIBLE SOLUTION found in any experiment.")
    
    return results


def save_results_csv(
    results: List[ExperimentResult],
    output_path: Path,
) -> None:
    """Save experiment results to CSV file."""
    if not results:
        return
    
    fieldnames = [
        "scenario_name", "n_obstacles", "n_waypoints", "n_particles",
        "iterations", "path_length", "fitness", "cpu_time",
        "converged", "L_coll", "L_out", "restarts_used"
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def print_results_summary(
    results: List[ExperimentResult],
    params: Optional[PSOParams] = None,
) -> None:
    """Print formatted summary table of results."""
    print("\n" + "=" * 110)
    print("PSO EXPERIMENT RESULTS SUMMARY")
    print("=" * 110)
    
    if params:
        print(f"\nBase parameters: max_iters={params.max_iters}, "
              f"w={params.w_start}->{params.w_end}, c1={params.c1}, c2={params.c2}")
        print(f"Restarts: max={params.max_restarts}, Diversity threshold={params.diversity_threshold}")
    else:
        print("\nUsing auto-configuration for all parameters.")
    print()
    
    # Table header
    header = (f"{'Scenario':<18} {'Obs':>4} {'WP':>4} {'Part':>5} {'Iters':>6} "
              f"{'PathLen':>9} {'Fitness':>10} {'CPU(s)':>7} {'Rest':>4} {'Conv':>5}")
    print(header)
    print("-" * len(header))
    
    for r in results:
        print(f"{r.scenario_name:<18} {r.n_obstacles:>4} {r.n_waypoints:>4} {r.n_particles:>5} "
              f"{r.iterations:>6} {r.path_length:>9.2f} {r.fitness:>10.2f} "
              f"{r.cpu_time:>7.3f} {r.restarts_used:>4} {'Yes' if r.converged else 'No':>5}")
    
    print("-" * len(header))
    print(f"\nTotal scenarios: {len(results)}")
    print(f"Converged: {sum(1 for r in results if r.converged)}/{len(results)}")
    if results:
        print(f"Avg CPU time: {sum(r.cpu_time for r in results)/len(results):.3f}s")
        print(f"Avg restarts: {sum(r.restarts_used for r in results)/len(results):.1f}")
