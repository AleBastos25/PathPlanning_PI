"""PathPlanning_PI - Path Planning with PSO."""

from .loader import load_scenario, ScenarioValidationError, ScenarioParseError
from .models import ObstacleRect, Point, Scenario
from .visualization import plot_scenario
from .geometry import (
    dist,
    segment_length,
    point_in_bounds,
    point_in_rect,
    segment_intersects_rect,
    segment_rect_intersection_length,
    segment_outside_bounds_length,
    path_length,
    path_collides,
    path_collision_length,
    path_out_of_bounds,
    path_outside_bounds_length,
)

# Algorithms
from .algorithms import (
    PSOParams,
    ConfiguredParams,
    Particle,
    FitnessComponents,
    PSOResult,
    SpatialGrid,
    pso_plan,
    build_path,
    unpack_waypoints,
    pack_waypoints,
    build_spatial_grid,
    evaluate,
    is_feasible,
    estimate_waypoints,
    estimate_particles,
    configure_params,
    compute_adaptive_penalties,
    inflate_obstacle,
    inflate_obstacles,
)

# Utils
from .utils import (
    ExperimentResult,
    run_experiment,
    run_all_experiments,
    save_results_csv,
    print_results_summary,
    plot_convergence,
)
