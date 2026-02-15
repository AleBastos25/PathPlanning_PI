import sys
import os
import math
import random
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from src.models import Point, ObstacleRect
from src.geometry import dist, point_in_rect

@dataclass
class State4:
    x1: float
    y1: float
    x2: float
    y2: float

@dataclass
class Node4:
    state: State4
    parent: Optional['Node4'] = None
    cost: float = 0.0

@dataclass
class Scenario2Robots:
    width: float
    height: float
    start1: Point
    goal1: Point
    start2: Point
    goal2: Point
    radius: float
    obstacles: List[ObstacleRect]
    interesting_points: List[Point] = None

def robots_separated(p1: Point, p2: Point, radius: float) -> bool:
    return dist(p1, p2) >= 2 * radius

def is_valid_state(state: State4, scenario: Scenario2Robots) -> bool:
    p1 = Point(state.x1, state.y1)
    p2 = Point(state.x2, state.y2)
    if not (0 <= state.x1 <= scenario.width and 0 <= state.y1 <= scenario.height): 
        # print("Bounds fail 1")
        return False


    if not (0 <= state.x2 <= scenario.width and 0 <= state.y2 <= scenario.height):
        # print("Bounds fail 2")
        return False

    if not robots_separated(p1, p2, scenario.radius):
       # print(f"Sep fail logic: {dist(p1, p2)} < {2*scenario.radius}")
       return False


    for obs in scenario.obstacles:
        if point_in_rect(p1, obs) or point_in_rect(p2, obs): 
             # print("Obstacle collision")
             return False

    return True


def edge_free_4d(s_from: State4, s_to: State4, scenario: Scenario2Robots, n_checks: int = 10) -> bool:
    for i in range(1, n_checks + 1):
        t = i / float(n_checks)
        ix1 = s_from.x1 + (s_to.x1 - s_from.x1) * t
        iy1 = s_from.y1 + (s_to.y1 - s_from.y1) * t
        ix2 = s_from.x2 + (s_to.x2 - s_from.x2) * t
        iy2 = s_from.y2 + (s_to.y2 - s_from.y2) * t
        state_t = State4(ix1, iy1, ix2, iy2)
        if not is_valid_state(state_t, scenario): return False
    return True

def dist4(s1: State4, s2: State4) -> float:
    return math.sqrt((s1.x1 - s2.x1)**2 + (s1.y1 - s2.y1)**2 + 
                     (s1.x2 - s2.x2)**2 + (s1.y2 - s2.y2)**2)



@dataclass
class RRTParams2Robots:
    step_size: float = 5.0
    max_iters: int = 5000
    p_intelligent: float = 0.2
    use_intelligent_sampling: bool = True
    use_path_opt: bool = False

def load_scenario_2robots(file_path: str) -> Scenario2Robots:
    with open(file_path, 'r') as f:
        content = f.read()
    tokens = content.replace(',', ' ').split()
    vals = [float(x) for x in tokens]
    
    width = vals[0]
    height = vals[1]
    start1 = Point(vals[2], vals[3])
    goal1 = Point(vals[4], vals[5])
    start2 = Point(vals[6], vals[7])
    goal2 = Point(vals[8], vals[9])
    radius = vals[10]
    obs_start_idx = 11
    
    obstacles = []
    for i in range(obs_start_idx, len(vals), 4):
        obstacles.append(ObstacleRect(vals[i], vals[i+1], vals[i+2], vals[i+3]))
        
    interesting = []
    for obs in obstacles:
        interesting.append(Point(obs.x_min, obs.y_min))
        interesting.append(Point(obs.x_max, obs.y_min))
        interesting.append(Point(obs.x_max, obs.y_max))
        interesting.append(Point(obs.x_min, obs.y_max))
        
    return Scenario2Robots(
        width=width, height=height,
        start1=start1, goal1=goal1,
        start2=start2, goal2=goal2,
        radius=radius,
        obstacles=obstacles,
        interesting_points=interesting
    )

    
class RRT2Robots:
    def __init__(self, scenario: Scenario2Robots, params: RRTParams2Robots):
        self.scenario = scenario
        self.params = params
        self.nodes: List[Node4] = []
        start_state = State4(scenario.start1.x, scenario.start1.y, scenario.start2.x, scenario.start2.y)
        self.nodes.append(Node4(state=start_state, parent=None, cost=0.0))
        
    def _nearest(self, state: State4) -> Node4:
        min_d = float('inf')
        nearest_node = None
        for n in self.nodes:
            d = dist4(n.state, state)
            if d < min_d:
                min_d = d
                nearest_node = n
        return nearest_node

    def _steer(self, from_state: State4, to_state: State4) -> State4:
        d = dist4(from_state, to_state)
        if d <= self.params.step_size:
            return to_state
        
        ratio = self.params.step_size / d
        new_x1 = from_state.x1 + (to_state.x1 - from_state.x1) * ratio
        new_y1 = from_state.y1 + (to_state.y1 - from_state.y1) * ratio
        new_x2 = from_state.x2 + (to_state.x2 - from_state.x2) * ratio
        new_y2 = from_state.y2 + (to_state.y2 - from_state.y2) * ratio
        return State4(new_x1, new_y1, new_x2, new_y2)

    def _sample_intelligent(self) -> State4:
        # Simple version of intelligent sampling
        if random.random() < self.params.p_intelligent:
             return State4(self.scenario.goal1.x, self.scenario.goal1.y,
                          self.scenario.goal2.x, self.scenario.goal2.y)
        return self._sample_uniform()

    def _sample_uniform(self) -> State4:



        return State4(
            random.uniform(0, self.scenario.width),
            random.uniform(0, self.scenario.height),
            random.uniform(0, self.scenario.width),
            random.uniform(0, self.scenario.height)
        )

    def _optimize_path_4d(self, path: List[State4]) -> List[State4]:
        if len(path) <= 2: return path
        simplified = [path[0]]
        curr_idx = 0
        while curr_idx < len(path) - 1:
            next_idx = curr_idx + 1
            for i in range(len(path) - 1, curr_idx, -1):
                if edge_free_4d(path[curr_idx], path[i], self.scenario):
                    next_idx = i
                    break
            simplified.append(path[next_idx])
            curr_idx = next_idx
        return simplified

    def _get_path(self, node: Node4) -> List[State4]:
        path = []
        curr = node
        while curr:
            path.append(curr.state)
            curr = curr.parent
        return path[::-1]

    def solve(self):

        print("Starting solve loop")

        for k in range(self.params.max_iters):
            if k % 100 == 0:
                print(f"Iter {k}, Nodes: {len(self.nodes)}")
            
            # 1. Sample
            if self.params.use_intelligent_sampling:
                s_rand = self._sample_intelligent()
            else:
                 s_rand = self._sample_uniform()
            
            # 2. Nearest
            nearest_node = self._nearest(s_rand)
            if not nearest_node: continue
            
            # 3. Steer
            s_new = self._steer(nearest_node.state, s_rand)
            
            # 4. Validity
            if is_valid_state(s_new, self.scenario):
                if edge_free_4d(nearest_node.state, s_new, self.scenario):
                    # Add node
                    new_node = Node4(state=s_new, parent=nearest_node, cost=nearest_node.cost + dist4(nearest_node.state, s_new))
                    self.nodes.append(new_node)
                    
                    # Check Goal (Distance to goal)
                    # Simplified goal check
                    d1 = dist(Point(s_new.x1, s_new.y1), self.scenario.goal1)
                    d2 = dist(Point(s_new.x2, s_new.y2), self.scenario.goal2)
                    if d1 < 10.0 and d2 < 10.0:
                        print(f"Goal Reached in {k} iters!")
                        path = self._get_path(new_node)
                        print(f"Path length: {len(path)}")
                        optimized = self._optimize_path_4d(path)
                        print(f"Optimized path length: {len(optimized)}")
                        return optimized, None, {}
        print("Done")
        return [], [], {}


def plot_results(scenario, path1, path2, filename):
    import matplotlib.pyplot as plt
    print("Plotting results...")
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(0, scenario.width)
    ax.set_ylim(0, scenario.height)

    # Draw Obstacles
    for obs in scenario.obstacles:
        rect = plt.Rectangle((obs.x_min, obs.y_min), obs.x_max - obs.x_min, obs.y_max - obs.y_min, color='gray')
        ax.add_patch(rect)

    # Draw Start/Goal
    ax.plot(scenario.start1.x, scenario.start1.y, 'ro', label='Start 1')
    ax.plot(scenario.goal1.x, scenario.goal1.y, 'rx', label='Goal 1')
    ax.plot(scenario.start2.x, scenario.start2.y, 'bo', label='Start 2')
    ax.plot(scenario.goal2.x, scenario.goal2.y, 'bx', label='Goal 2')

    # Draw Paths
    if path1:
        xs1 = [p.x for p in path1]
        ys1 = [p.y for p in path1]
        ax.plot(xs1, ys1, 'r-', linewidth=2, label='Robot 1')
    
    if path2:
        xs2 = [p.x for p in path2]
        ys2 = [p.y for p in path2]
        ax.plot(xs2, ys2, 'b-', linewidth=2, label='Robot 2')

    plt.legend()
    plt.title("RRT 2 Robots (4D)")
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    # plt.show() # Don't show in headless env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", type=str)
    args = parser.parse_args()
    
    print(f"Loading {args.scenario}")
    scenario = load_scenario_2robots(args.scenario)
    print("Scenario loaded")
    
    params = RRTParams2Robots(max_iters=5000)
    solver = RRT2Robots(scenario, params)

    path, _, _ = solver.solve()
    
    if path:
        print("Path found! Plotting...")
        path1 = [Point(s.x1, s.y1) for s in path]
        path2 = [Point(s.x2, s.y2) for s in path]
        plot_results(scenario, path1, path2, "result_scenario0.png")
    else:
        print("No path found.")



def test():
    print("Starting minimal test with LOADER")
    # scenario = Scenario2Robots(...)
    scenario = load_scenario_2robots("scenarios_2robots/scenario0_2robots.txt")
    print("Scenario loaded")
    params = RRTParams2Robots(max_iters=5000)
    solver = RRT2Robots(scenario, params)

    try:
        path, _, _ = solver.solve()
        if path:
            path1 = [Point(s.x1, s.y1) for s in path]
            path2 = [Point(s.x2, s.y2) for s in path]
            plot_results(scenario, path1, path2, "test_result.png")
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        test()
