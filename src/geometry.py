import math
from .models import Point, ObstacleRect


def dist(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)


def point_in_bounds(p: Point, xmax: float, ymax: float) -> bool:
    """Check if point is within [0, xmax] x [0, ymax]."""
    return 0 <= p.x <= xmax and 0 <= p.y <= ymax


def point_in_rect(p: Point, rect: ObstacleRect) -> bool:
    """Check if point is inside rectangle (inclusive)."""
    return rect.x_min <= p.x <= rect.x_max and rect.y_min <= p.y <= rect.y_max


def _ccw(a: Point, b: Point, c: Point) -> float:
    """Returns positive if CCW, negative if CW, 0 if collinear."""
    return (c.y - a.y) * (b.x - a.x) - (b.y - a.y) * (c.x - a.x)


def _on_segment(p: Point, q: Point, r: Point) -> bool:
    """Check if point r lies on segment pq (assuming collinear)."""
    return (min(p.x, q.x) <= r.x <= max(p.x, q.x) and
            min(p.y, q.y) <= r.y <= max(p.y, q.y))


def segment_intersects_segment(a: Point, b: Point, c: Point, d: Point) -> bool:
    """Check if segment AB intersects segment CD."""
    d1 = _ccw(c, d, a)
    d2 = _ccw(c, d, b)
    d3 = _ccw(a, b, c)
    d4 = _ccw(a, b, d)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Check collinear cases
    if d1 == 0 and _on_segment(c, d, a):
        return True
    if d2 == 0 and _on_segment(c, d, b):
        return True
    if d3 == 0 and _on_segment(a, b, c):
        return True
    if d4 == 0 and _on_segment(a, b, d):
        return True

    return False


def segment_intersects_rect(p: Point, q: Point, rect: ObstacleRect) -> bool:
    """Check if segment PQ intersects rectangle."""
    # If either endpoint is inside the rectangle
    if point_in_rect(p, rect) or point_in_rect(q, rect):
        return True

    # Rectangle corners
    bl = Point(rect.x_min, rect.y_min)  # bottom-left
    br = Point(rect.x_max, rect.y_min)  # bottom-right
    tr = Point(rect.x_max, rect.y_max)  # top-right
    tl = Point(rect.x_min, rect.y_max)  # top-left

    # Check intersection with each edge
    if segment_intersects_segment(p, q, bl, br):  # bottom
        return True
    if segment_intersects_segment(p, q, br, tr):  # right
        return True
    if segment_intersects_segment(p, q, tr, tl):  # top
        return True
    if segment_intersects_segment(p, q, tl, bl):  # left
        return True

    return False


def path_length(path: list[Point]) -> float:
    """Calculate total length of a path."""
    total = 0.0
    for i in range(len(path) - 1):
        total += dist(path[i], path[i + 1])
    return total


def path_collides(path: list[Point], obstacles: list[ObstacleRect]) -> bool:
    """Check if any segment of the path collides with any obstacle."""
    for i in range(len(path) - 1):
        p, q = path[i], path[i + 1]
        for rect in obstacles:
            if segment_intersects_rect(p, q, rect):
                return True
    return False


def path_out_of_bounds(path: list[Point], xmax: float, ymax: float) -> bool:
    """Check if any point in the path is out of bounds."""
    for p in path:
        if not point_in_bounds(p, xmax, ymax):
            return True
    return False
