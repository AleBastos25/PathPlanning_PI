from pathlib import Path
from typing import List

from .models import ObstacleRect, Point, Scenario


def point_in_rect(p: Point, rect: ObstacleRect) -> bool:
    """Check if a point is inside a rectangle (inclusive boundaries)."""
    return rect.x_min <= p.x <= rect.x_max and rect.y_min <= p.y <= rect.y_max


def load_scenario(filepath: str | Path) -> Scenario:
    """Load a scenario from file."""
    content = Path(filepath).read_text()
    values = [float(v) for v in content.split()]

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
        obstacles.append(ObstacleRect(values[i], values[i + 1], values[i + 2], values[i + 3]))
        i += 4

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
