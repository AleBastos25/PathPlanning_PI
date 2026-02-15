"""Experiment utilities for running and collecting PSO results."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..loader import load_scenario
from ..visualization import plot_scenario
from ..algorithms.pso import PSOConfig, pso_plan
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


def run_experiment(
    scenario_path: Path,
    config: PSOConfig,
    output_dir: Optional[Path] = None,
    save_plots: bool = True,
    config_name: Optional[str] = None,
    radius: bool = False,
    verbose: bool = True,
    plot_feasible_colors: bool = False,
) -> ExperimentResult:
    """Run PSO on one scenario. config_name used only for plot file names.
    plot_feasible_colors: if True, convergence plot uses green/coral for feasible vs non-feasible."""
    scenario_id = scenario_path.stem.replace("scenario", "")
    scenario = load_scenario(scenario_path)
    if not radius:
        scenario.R = 0.0
    result = pso_plan(scenario, config)

    
    if save_plots and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{config_name}" if config_name else ""
        safe_suffix = suffix.replace("+", "_").replace(" ", "_")[:20]
        path_plot_file = output_dir / f"scenario_{scenario_id}_pso_path{safe_suffix}.png"
        plot_scenario(scenario, path=result.best_path, save_to=path_plot_file, show=False)
        conv_plot_file = output_dir / f"scenario_{scenario_id}_pso_convergence{safe_suffix}.png"
        plot_convergence(
            result.history,
            scenario_id,
            conv_plot_file,
            history_feasible=result.history_feasible if plot_feasible_colors else None,
        )

    return ExperimentResult(
        scenario_name=scenario_path.name,
        n_obstacles=len(scenario.obstacles),
        n_waypoints=result.n_waypoints,
        n_particles=config.n_particles,
        iterations=result.iters_used,
        path_length=result.L,
        fitness=result.best_f,
        cpu_time=result.cpu_time,
        converged=result.converged,
        L_coll=result.L_coll,
        L_out=result.L_out,
    )


