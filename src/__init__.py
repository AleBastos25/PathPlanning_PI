"""PathPlanning_PI - Path Planning with PSO."""

from .loader import load_scenario, ScenarioValidationError, ScenarioParseError
from .models import ObstacleRect, Point, Scenario
from .visualization import plot_scenario
from .geometry import (
    dist,
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
    PSOConfig,
    Particle,
    FitnessComponents,
    PSOResult,
    pso_plan,
    build_path,
    unpack_waypoints,
    pack_waypoints,
    evaluate,
    compute_adaptive_penalties,
)

# Utils
from .utils import ExperimentResult, run_experiment, plot_convergence
