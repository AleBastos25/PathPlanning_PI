from .loader import load_scenario, ScenarioValidationError, ScenarioParseError
from .models import ObstacleRect, Point, Scenario
from .visualization import plot_scenario
from .geometry import (
    dist,
    point_in_bounds,
    point_in_rect,
    segment_intersects_rect,
    path_length,
    path_collides,
    path_out_of_bounds,
)

