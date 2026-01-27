from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Point:
    """Represents a 2D point."""

    x: float
    y: float

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))


@dataclass(frozen=True)
class ObstacleRect:
    """Represents a rectangular obstacle.

    Attributes:
        xo: X coordinate of the bottom-left corner.
        yo: Y coordinate of the bottom-left corner.
        lx: Width of the rectangle (extent in X direction).
        ly: Height of the rectangle (extent in Y direction).
    """

    xo: float
    yo: float
    lx: float
    ly: float

    @property
    def x_min(self) -> float:
        """Left edge X coordinate."""
        return self.xo

    @property
    def x_max(self) -> float:
        """Right edge X coordinate."""
        return self.xo + self.lx

    @property
    def y_min(self) -> float:
        """Bottom edge Y coordinate."""
        return self.yo

    @property
    def y_max(self) -> float:
        """Top edge Y coordinate."""
        return self.yo + self.ly


@dataclass
class Scenario:
    """Represents a complete path planning scenario.

    Attributes:
        xmax: Maximum X coordinate of the environment.
        ymax: Maximum Y coordinate of the environment.
        start1: Starting point for robot 1.
        goal1: Goal point for robot 1.
        start2: Starting point for robot 2.
        goal2: Goal point for robot 2.
        R: Robot radius.
        obstacles: List of rectangular obstacles.
    """

    xmax: float
    ymax: float
    start1: Point
    goal1: Point
    start2: Point
    goal2: Point
    R: float
    obstacles: List[ObstacleRect]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return environment bounds as (x_min, y_min, x_max, y_max)."""
        return (0, 0, self.xmax, self.ymax)
