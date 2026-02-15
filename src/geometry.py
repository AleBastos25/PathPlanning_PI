import math
from typing import Tuple, Optional
from .models import Point, ObstacleRect


# Numerical tolerance for floating point comparisons
EPS = 1e-9


def dist(a: Point, b: Point) -> float:
    return math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)


def segment_length(a: Point, b: Point) -> float:
    return dist(a, b)


def point_in_bounds(p: Point, xmax: float, ymax: float) -> bool:
    return -EPS <= p.x <= xmax + EPS and -EPS <= p.y <= ymax + EPS


def point_in_rect(p: Point, rect: ObstacleRect) -> bool:
    return (rect.x_min - EPS <= p.x <= rect.x_max + EPS and 
            rect.y_min - EPS <= p.y <= rect.y_max + EPS)


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
    """Check if segment PQ intersects rectangle.
    
    Treats the obstacle as effectively slightly smaller (by EPS) to allow 
    tangency/grazing of the boundary without being considered a collision.
    """
    # Shrink bounds by EPS to allow boundary touching
    x_min = rect.x_min + EPS
    y_min = rect.y_min + EPS
    x_max = rect.x_max - EPS
    y_max = rect.y_max - EPS
    
    # If shrunk rectangle is invalid (obstacle smaller than 2*EPS), use centroid check or original
    if x_min >= x_max or y_min >= y_max:
        # Fallback for tiny obstacles: just check if midpoint is inside original
        mid = Point((p.x+q.x)/2, (p.y+q.y)/2)
        return point_in_rect(mid, rect)

    result = _liang_barsky_clip(p, q, x_min, y_min, x_max, y_max)
    return result is not None


def _liang_barsky_clip(
    a: Point, b: Point,
    x_min: float, y_min: float, x_max: float, y_max: float
) -> Optional[Tuple[float, float]]:
    """Liang-Barsky algorithm for clipping segment AB against axis-aligned rectangle.
    
    Returns:
        Tuple (t_enter, t_exit) in [0,1] if segment intersects rectangle interior,
        None if no intersection.
    """
    dx = b.x - a.x
    dy = b.y - a.y
    
    # Handle degenerate segment (point)
    if abs(dx) < EPS and abs(dy) < EPS:
        # Segment is a point - check if inside
        if x_min <= a.x <= x_max and y_min <= a.y <= y_max:
            return (0.0, 0.0)  # Zero-length intersection
        return None
    
    t_enter = 0.0
    t_exit = 1.0
    
    # Check each boundary: left, right, bottom, top
    # For each boundary, compute t value where line crosses it
    
    p = [-dx, dx, -dy, dy]
    q = [a.x - x_min, x_max - a.x, a.y - y_min, y_max - a.y]
    
    for i in range(4):
        if abs(p[i]) < EPS:
            # Line is parallel to this boundary
            if q[i] < 0:
                # Line is outside this boundary
                return None
        else:
            t = q[i] / p[i]
            if p[i] < 0:
                # Line enters through this boundary
                t_enter = max(t_enter, t)
            else:
                # Line exits through this boundary
                t_exit = min(t_exit, t)
    
    if t_enter > t_exit + EPS:
        return None
    
    # Clamp to [0, 1]
    t_enter = max(0.0, t_enter)
    t_exit = min(1.0, t_exit)
    
    if t_enter > t_exit + EPS:
        return None
    
    return (t_enter, t_exit)


def segment_rect_intersection_length(a: Point, b: Point, rect: ObstacleRect) -> float:
    """Calculate length of segment AB that lies inside rectangle.
    
    Uses Liang-Barsky clipping algorithm.
    
    Args:
        a, b: Segment endpoints.
        rect: Rectangle to check intersection with.
    
    Returns:
        Length of the portion of segment inside rectangle. Returns 0 if no intersection.
    """
    result = _liang_barsky_clip(a, b, rect.x_min, rect.y_min, rect.x_max, rect.y_max)
    
    if result is None:
        return 0.0
    
    t_enter, t_exit = result
    seg_len = dist(a, b)
    
    return (t_exit - t_enter) * seg_len


def segment_outside_bounds_length(
    a: Point, b: Point, xmax: float, ymax: float
) -> float:
    """Calculate length of segment AB that lies outside environment bounds [0,xmax]x[0,ymax].
    
    Args:
        a, b: Segment endpoints.
        xmax, ymax: Environment bounds.
    
    Returns:
        Length of the portion of segment outside bounds.
    """
    total_len = dist(a, b)
    
    if total_len < EPS:
        return 0.0
    
    # Calculate length inside bounds
    result = _liang_barsky_clip(a, b, 0.0, 0.0, xmax, ymax)
    
    if result is None:
        # Entire segment is outside
        return total_len
    
    t_enter, t_exit = result
    inside_len = (t_exit - t_enter) * total_len
    
    return max(0.0, total_len - inside_len)


# =============================================================================
# Path utilities
# =============================================================================


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


def path_collision_length(path: list[Point], obstacles: list[ObstacleRect]) -> float:
    """Calculate total length of path segments inside obstacles."""
    total = 0.0
    for i in range(len(path) - 1):
        p, q = path[i], path[i + 1]
        for rect in obstacles:
            total += segment_rect_intersection_length(p, q, rect)
    return total


def path_out_of_bounds(path: list[Point], xmax: float, ymax: float) -> bool:
    """Check if any point in the path is out of bounds."""
    for p in path:
        if not point_in_bounds(p, xmax, ymax):
            return True
    return False


def path_outside_bounds_length(path: list[Point], xmax: float, ymax: float) -> float:
    """Calculate total length of path segments outside bounds."""
    total = 0.0
    for i in range(len(path) - 1):
        total += segment_outside_bounds_length(path[i], path[i + 1], xmax, ymax)
    return total

# =============================================================================
# Obstacle utilities
# =============================================================================

def obstacles_inflate(obstacles: list[ObstacleRect], R: float) -> list[ObstacleRect]:
    return [ObstacleRect(obs.x_min - R, obs.y_min - R, (obs.x_max - obs.x_min) + 2*R, (obs.y_max - obs.y_min) + 2*R) for obs in obstacles]

def candidate_obstacles(a: Point, b: Point, obstacles: list[ObstacleRect]) -> list[ObstacleRect]:
    return [obs for obs in obstacles if segment_intersects_rect(a, b, obs)]