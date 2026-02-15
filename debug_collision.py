import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models import Point, ObstacleRect
from src.geometry import segment_intersects_rect

def test_collision():
    # Define obstacle: x=[100, 340], y=[75, 275]
    obs = ObstacleRect(100.0, 75.0, 240.0, 200.0) 
    # Note: ObstacleRect(x,y,w,h) -> x_min=x, x_max=x+w, ...
    
    print(f"Obstacle: x[{obs.x_min}, {obs.x_max}], y[{obs.y_min}, {obs.y_max}]")

    # Case 1: Tangent Vertical (Right Edge)
    p1 = Point(340.0, 50.0)
    p2 = Point(340.0, 300.0)
    collision = segment_intersects_rect(p1, p2, obs)
    print(f"Case 1 (Tangent Vertical x=340): {collision} (Expected: ?)")
    
    # Case 2: Corner Touch
    p3 = Point(340.0, 275.0)
    p4 = Point(400.0, 350.0)
    collision2 = segment_intersects_rect(p3, p4, obs)
    print(f"Case 2 (Corner Touch 340,275): {collision2} (Expected: ?)")

    # Case 3: Epsilon Clear
    p5 = Point(340.0001, 50.0)
    p6 = Point(340.0001, 300.0)
    collision3 = segment_intersects_rect(p5, p6, obs)
    print(f"Case 3 (Epsilon Clear x=340.0001): {collision3}")

if __name__ == "__main__":
    test_collision()
