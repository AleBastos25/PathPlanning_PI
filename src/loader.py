from pathlib import Path
from typing import List

from .models import ObstacleRect, Point, Scenario


class ScenarioValidationError(Exception):
    """Raised when scenario validation fails."""
    pass


class ScenarioParseError(Exception):
    """Raised when scenario file cannot be parsed."""
    pass


def _point_in_bounds(p: Point, xmax: float, ymax: float) -> bool:
    """Check if point is within [0, xmax] x [0, ymax]."""
    return 0 <= p.x <= xmax and 0 <= p.y <= ymax


def _point_in_rect(p: Point, rect: ObstacleRect) -> bool:
    """Check if a point is inside a rectangle."""
    return rect.x_min <= p.x <= rect.x_max and rect.y_min <= p.y <= rect.y_max


def _parse_values(filepath: Path) -> List[float]:
    """Parse file content into list of floats."""
    try:
        content = filepath.read_text()
    except FileNotFoundError:
        raise ScenarioParseError(f"File not found: {filepath}")
    except PermissionError:
        raise ScenarioParseError(f"Permission denied: {filepath}")
    
    if not content.strip():
        raise ScenarioParseError(f"File is empty: {filepath}")
    
    values: List[float] = []
    for i, token in enumerate(content.split()):
        try:
            values.append(float(token))
        except ValueError:
            raise ScenarioParseError(
                f"Invalid number at position {i}: '{token}' in {filepath}"
            )
    
    return values


def _validate_field_count(values: List[float], filepath: Path) -> None:
    """Validate that file has correct number of fields."""

    min_fields = 11
    if len(values) < min_fields:
        raise ScenarioParseError(
            f"Not enough values in {filepath}: expected at least {min_fields}, got {len(values)}"
        )
    
    # Remaining values after header must be multiple of 4 (obstacles)
    remaining = len(values) - min_fields
    if remaining % 4 != 0:
        raise ScenarioParseError(
            f"Invalid obstacle data in {filepath}: {remaining} values is not a multiple of 4"
        )


def _validate_environment(xmax: float, ymax: float) -> None:
    """Validate environment dimensions."""
    if xmax <= 0:
        raise ScenarioValidationError(f"xmax must be positive, got {xmax}")
    if ymax <= 0:
        raise ScenarioValidationError(f"ymax must be positive, got {ymax}")


def _validate_robot_radius(R: float) -> None:
    """Validate robot radius."""
    if R < 0:
        raise ScenarioValidationError(f"Robot radius R must be non-negative, got {R}")


def _validate_point_in_bounds(
    p: Point, name: str, xmax: float, ymax: float
) -> None:
    """Validate that a point is within environment bounds."""
    if not _point_in_bounds(p, xmax, ymax):
        raise ScenarioValidationError(
            f"{name} ({p.x}, {p.y}) is outside environment bounds [0, {xmax}] x [0, {ymax}]"
        )


def _validate_point_not_in_obstacle(
    p: Point, name: str, obstacles: List[ObstacleRect]
) -> None:
    """Validate that a point is not inside any obstacle."""
    for i, obs in enumerate(obstacles):
        if _point_in_rect(p, obs):
            raise ScenarioValidationError(
                f"{name} ({p.x}, {p.y}) is inside obstacle {i} "
                f"[({obs.xo}, {obs.yo}) to ({obs.x_max}, {obs.y_max})]"
            )


def _validate_obstacle(
    obs: ObstacleRect, index: int, xmax: float, ymax: float
) -> None:
    """Validate a single obstacle."""
    # Check positive dimensions
    if obs.lx <= 0:
        raise ScenarioValidationError(
            f"Obstacle {index}: width (lx) must be positive, got {obs.lx}"
        )
    if obs.ly <= 0:
        raise ScenarioValidationError(
            f"Obstacle {index}: height (ly) must be positive, got {obs.ly}"
        )
    
    # Check origin is non-negative
    if obs.xo < 0 or obs.yo < 0:
        raise ScenarioValidationError(
            f"Obstacle {index}: origin ({obs.xo}, {obs.yo}) has negative coordinates"
        )
    
    # Check obstacle is within environment bounds
    if obs.x_max > xmax or obs.y_max > ymax:
        raise ScenarioValidationError(
            f"Obstacle {index}: extends beyond environment bounds. "
            f"Obstacle ends at ({obs.x_max}, {obs.y_max}), environment is ({xmax}, {ymax})"
        )


def load_scenario(filepath: str | Path) -> Scenario:
    """Load a scenario from file.
    
    Args:
        filepath: Path to the scenario file.
        validate: If True, validates all constraints. If False, only parses.
    
    Returns:
        Scenario object with all data.
    
    Raises:
        ScenarioParseError: If file cannot be parsed.
        ScenarioValidationError: If validation fails (when validate=True).
    
    File format (whitespace-separated values):
        xmax ymax
        start1_x start1_y
        goal1_x goal1_y
        start2_x start2_y
        goal2_x goal2_y
        R
        [xo yo lx ly] ...  (0 or more obstacles)
    """
    filepath = Path(filepath)
    
    # Parse file
    values = _parse_values(filepath)
    _validate_field_count(values, filepath)
    
    # Extract values
    i = 0
    xmax, ymax = values[i], values[i + 1]
    i += 2

    start1 = Point(values[i], values[i + 1])
    i += 2
    goal1 = Point(values[i], values[i + 1])
    i += 2

    start2 = Point(values[i], values[i + 1])
    i += 2
    goal2 = Point(values[i], values[i + 1])
    i += 2

    R = values[i]
    i += 1

    obstacles: List[ObstacleRect] = []
    while i < len(values):
        obstacles.append(
            ObstacleRect(values[i], values[i + 1], values[i + 2], values[i + 3])
        )
        i += 4

    _validate_environment(xmax, ymax)
    _validate_robot_radius(R)
    
    for idx, obs in enumerate(obstacles):
        _validate_obstacle(obs, idx, xmax, ymax)
    
    # Validate points after obstacles are parsed (to check collision)
    _validate_point_in_bounds(start1, "start1", xmax, ymax)
    _validate_point_in_bounds(goal1, "goal1", xmax, ymax)
    _validate_point_in_bounds(start2, "start2", xmax, ymax)
    _validate_point_in_bounds(goal2, "goal2", xmax, ymax)
    
    _validate_point_not_in_obstacle(start1, "start1", obstacles)
    _validate_point_not_in_obstacle(goal1, "goal1", obstacles)
    _validate_point_not_in_obstacle(start2, "start2", obstacles)
    _validate_point_not_in_obstacle(goal2, "goal2", obstacles)

    return Scenario(
        xmax=xmax,
        ymax=ymax,
        start1=start1,
        goal1=goal1,
        start2=start2,
        goal2=goal2,
        R=R,
        obstacles=obstacles,
    )
