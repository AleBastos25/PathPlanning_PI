import math
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

from src.models import Point, ObstacleRect, Scenario
from src.geometry import dist, point_in_rect

# -----------------------------------------------------------------------------
# 1. Robust Stream-Based Loader & Adaptation Logic
# -----------------------------------------------------------------------------

def place_near(
    ref_point: Point, 
    R: float, 
    xmax: float, 
    ymax: float, 
    obstacles: List[ObstacleRect], 
    other_robot_pos: Point
) -> Optional[Point]:
    """
    Tries to place a point near 'ref_point' such that it's valid and 
    separated from 'other_robot_pos' by at least 2*R.
    Strategy: Try fixed offsets with increasing d = k * R.
    Prioritize closer points (k ~ 2.0).
    """
    # Try to be as close as possible to the limit 2R
    k_factors = [2.01, 2.5, 3.0, 4.0, 6.0] 
    
    for k in k_factors:
        d = k * R
        offsets = [
            (d, 0), (0, d), (-d, 0), (0, -d),       # Cardinals
            (d, d), (d, -d), (-d, d), (-d, -d)      # Diagonals
        ]
        
        for dx, dy in offsets:
            p = Point(ref_point.x + dx, ref_point.y + dy)
            
            # Check bounds
            if not (0 <= p.x <= xmax and 0 <= p.y <= ymax):
                continue
                
            # Check obstacles
            in_obs = False
            for obs in obstacles:
                if point_in_rect(p, obs):
                    in_obs = True
                    break
            if in_obs:
                continue
            
            # Check separation
            if dist(p, other_robot_pos) >= 2 * R:
                return p

    return None

def load_scenario_robust(filepath: str, R_override: Optional[float] = None) -> Scenario:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"{filepath} not found")
        
    # 1. Stream-based parsing
    text = path.read_text()
    tokens = text.split()
    clean_tokens = [t for t in tokens if not t.startswith("#")]
    
    try:
        vals = [float(x) for x in clean_tokens]
    except ValueError as e:
        raise ValueError(f"Scenario parsing error: {e}")

    # 2. Detect Format
    is_2_robots = (len(vals) >= 11 and (len(vals) - 11) % 4 == 0)
    is_1_robot = (len(vals) >= 6 and (len(vals) - 6) % 4 == 0)
    
    if not is_2_robots and not is_1_robot:
        raise ValueError(f"Invalid file format. Token count: {len(vals)}")
        
    # 3. Parse and Adapt
    if is_2_robots:
        # Format: xmax, ymax, x1, y1, gx1, gy1, x2, y2, gx2, gy2, R
        xmax, ymax = vals[0], vals[1]
        s1 = Point(vals[2], vals[3])
        g1 = Point(vals[4], vals[5])
        s2 = Point(vals[6], vals[7])
        g2 = Point(vals[8], vals[9])
        R_file = vals[10]
        obs_start = 11
        
        R = R_override if R_override is not None else R_file
        
    else: # is_1_robot
        # Format: xmax, ymax, x1, y1, gx1, gy1
        xmax, ymax = vals[0], vals[1]
        s1 = Point(vals[2], vals[3])
        g1 = Point(vals[4], vals[5])
        obs_start = 6
        
        # Default R: 10.0 (Small but visible)
        R = R_override if R_override is not None else 10.0 
        
        # Parse obstacles first needed for adaptation
        obstacles_temp = []
        i = obs_start
        while i < len(vals):
             obstacles_temp.append(ObstacleRect(vals[i], vals[i+1], vals[i+2], vals[i+3]))
             i += 4
             
        # Adapt: Generate Robot 2
        # s2 near s1
        s2 = place_near(s1, R, xmax, ymax, obstacles_temp, s1)
        if s2 is None:
            raise ValueError(f"Could not place Start 2 near Start 1 with R={R}")
            
        # g2 near g1
        g2 = place_near(g1, R, xmax, ymax, obstacles_temp, g1)
        if g2 is None:
            raise ValueError(f"Could not place Goal 2 near Goal 1 with R={R}")

    # Parse obstacles (if not already done/parsed again is cheap)
    obstacles = []
    i = obs_start
    while i < len(vals):
        obstacles.append(ObstacleRect(vals[i], vals[i+1], vals[i+2], vals[i+3]))
        i += 4
        
    # 4. Final Validation/Fix check (for 2-robot files that might violate R)
    # Check starts
    if dist(s1, s2) < 2 * R:
        print(f"Warning: Starts too close ({dist(s1, s2):.2f} < {2*R}). Attempting fix...")
        # Fix s2 relative to s1
        fixed = place_near(s1, R, xmax, ymax, obstacles, s1)
        if fixed: s2 = fixed
        else: print("Start fix failed.")

    # Check goals
    if dist(g1, g2) < 2 * R:
        print(f"Warning: Goals too close ({dist(g1, g2):.2f} < {2*R}). Attempting fix...")
        fixed = place_near(g1, R, xmax, ymax, obstacles, g1)
        if fixed: g2 = fixed
        else: print("Goal fix failed.")

    return Scenario(xmax, ymax, s1, g1, s2, g2, R, obstacles)


# -----------------------------------------------------------------------------
# 2. Basic 4D RRT Implementation (Anytime)
# -----------------------------------------------------------------------------

@dataclass
class State4:
    x1: float
    y1: float
    x2: float
    y2: float

@dataclass
class Node4:
    state: State4
    parent: Optional[int]
    cost: float

class RRT2RobotsBasic:
    def __init__(self, scenario: Scenario, max_iters: int = 5000, step_size: float = 20.0):
        self.scenario = scenario
        self.max_iters = max_iters
        self.step_size = step_size
        self.nodes: List[Node4] = []
        
        # Anytime metrics
        self.best_path: List[State4] = []
        self.best_makespan: float = float('inf')
        self.first_success_iter: Optional[int] = None
        
    def _dist4(self, s1: State4, s2: State4) -> float:
        return math.sqrt(
            (s1.x1-s2.x1)**2 + (s1.y1-s2.y1)**2 + 
            (s1.x2-s2.x2)**2 + (s1.y2-s2.y2)**2
        )

    def _state_valid(self, s: State4) -> bool:
        # 1. Bounds
        if not (0 <= s.x1 <= self.scenario.xmax and 0 <= s.y1 <= self.scenario.ymax): return False
        if not (0 <= s.x2 <= self.scenario.xmax and 0 <= s.y2 <= self.scenario.ymax): return False
        
        p1 = Point(s.x1, s.y1)
        p2 = Point(s.x2, s.y2)
        
        # 2. Separation
        if dist(p1, p2) < 2 * self.scenario.R:
            return False
            
        # 3. Obstacles
        for obs in self.scenario.obstacles:
            if point_in_rect(p1, obs) or point_in_rect(p2, obs):
                return False
                
        return True

    def _edge_free_4d(self, s_from: State4, s_to: State4, steps: int = 20) -> bool:
        # Increased steps to 20 for better safety
        for i in range(1, steps + 1):
            t = i / steps
            # Linear Interp
            s_interp = State4(
                s_from.x1 + t * (s_to.x1 - s_from.x1),
                s_from.y1 + t * (s_to.y1 - s_from.y1),
                s_from.x2 + t * (s_to.x2 - s_from.x2),
                s_from.y2 + t * (s_to.y2 - s_from.y2)
            )
            if not self._state_valid(s_interp):
                return False
        return True

    def _nearest_idx(self, query: State4) -> int:
        best_idx = -1
        best_dist = float('inf')
        for i, node in enumerate(self.nodes):
            d = self._dist4(node.state, query)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _steer(self, s_from: State4, s_to: State4) -> State4:
        d = self._dist4(s_from, s_to)
        if d <= self.step_size:
            return s_to
        ratio = self.step_size / d
        return State4(
            s_from.x1 + ratio * (s_to.x1 - s_from.x1),
            s_from.y1 + ratio * (s_to.y1 - s_from.y1),
            s_from.x2 + ratio * (s_to.x2 - s_from.x2),
            s_from.y2 + ratio * (s_to.y2 - s_from.y2)
        )
    
    def _calculate_makespan(self, path: List[State4]) -> float:
        if not path: return float('inf')
        d1 = 0.0
        d2 = 0.0
        for i in range(len(path) - 1):
            p_curr = path[i]
            p_next = path[i+1]
            d1 += math.sqrt((p_next.x1 - p_curr.x1)**2 + (p_next.y1 - p_curr.y1)**2)
            d2 += math.sqrt((p_next.x2 - p_curr.x2)**2 + (p_next.y2 - p_curr.y2)**2)
        return max(d1, d2)
    
    def shortcut_path_4d(self, path: List[State4]) -> List[State4]:
        """Greedy shortcutting in 4D space."""
        if len(path) < 3: return path
        
        new_path = [path[0]]
        curr_idx = 0
        
        while curr_idx < len(path) - 1:
            # Look ahead from far to near
            found_shortcut = False
            for look_idx in range(len(path) - 1, curr_idx + 1, -1):
                if self._edge_free_4d(path[curr_idx], path[look_idx], steps=20):
                    new_path.append(path[look_idx])
                    curr_idx = look_idx
                    found_shortcut = True
                    break
            
            if not found_shortcut:
                # Should not happen if adjacent nodes are edge free
                curr_idx += 1
                new_path.append(path[curr_idx])
                
        return new_path

    def plan(self) -> Tuple[List[State4], float, Optional[int]]:
        start_state = State4(
            self.scenario.start1.x, self.scenario.start1.y,
            self.scenario.start2.x, self.scenario.start2.y
        )
        
        if not self._state_valid(start_state):
            print("CRITICAL: Start state is invalid (collision or bounds).")
            return [], 0.0, None

        self.nodes = [Node4(start_state, None, 0.0)]
        # Tighter goal tolerance
        goal_eps = self.step_size * 1.0 
        
        print(f"Starting Anytime RRT (Max Iters: {self.max_iters})...")
        
        for k in range(self.max_iters):
            if k % 500 == 0:
                print(f"Iter {k}, Tree: {len(self.nodes)}, Best Makespan: {self.best_makespan:.2f}")
            
            # Sample (Goal Bias 10%)
            if random.random() < 0.10:
                s_rand = State4(
                    self.scenario.goal1.x, self.scenario.goal1.y,
                    self.scenario.goal2.x, self.scenario.goal2.y
                )
            else:
                s_rand = State4(
                    random.uniform(0, self.scenario.xmax),
                    random.uniform(0, self.scenario.ymax),
                    random.uniform(0, self.scenario.xmax),
                    random.uniform(0, self.scenario.ymax)
                )
                if not self._state_valid(s_rand): continue
                
            # Extend
            near_idx = self._nearest_idx(s_rand)
            near_node = self.nodes[near_idx]
            s_new = self._steer(near_node.state, s_rand)
            
            if self._edge_free_4d(near_node.state, s_new):
                new_cost = near_node.cost + self._dist4(near_node.state, s_new)
                new_node = Node4(s_new, near_idx, new_cost)
                self.nodes.append(new_node)
                
                # Check Goal
                d1 = dist(Point(s_new.x1, s_new.y1), self.scenario.goal1)
                d2 = dist(Point(s_new.x2, s_new.y2), self.scenario.goal2)
                
                if d1 <= goal_eps and d2 <= goal_eps:
                    # Found a solution!
                    raw_path = self._reconstruct_path(len(self.nodes) - 1)
                    curr_makespan = self._calculate_makespan(raw_path)
                    
                    if curr_makespan < self.best_makespan:
                        print(f"New Checkpoint! Iter {k}, Makespan: {curr_makespan:.2f}")
                        self.best_makespan = curr_makespan
                        self.best_path = raw_path
                        if self.first_success_iter is None:
                            self.first_success_iter = k
                        # Don't return, keep searching (Anytime)
                    
        print("Max iters reached.")
        
        if self.best_path:
            # Post-Process: 4D Shortcut
            print("Optimizing best path with 4D Shortcut...")
            optimized_path = self.shortcut_path_4d(self.best_path)
            optimized_makespan = self._calculate_makespan(optimized_path)
            print(f"Optimization: {self.best_makespan:.2f} -> {optimized_makespan:.2f}")
            print(f"Optimization: {self.best_makespan:.2f} -> {optimized_makespan:.2f}")
            return optimized_path, optimized_makespan, self.first_success_iter
            
        return [], 0.0, None

    def _reconstruct_path(self, idx: int) -> List[State4]:
        path = []
        curr = idx
        while curr is not None:
            path.append(self.nodes[curr].state)
            curr = self.nodes[curr].parent
        return path[::-1]

# -----------------------------------------------------------------------------
# 3. Visualization
# -----------------------------------------------------------------------------

def compute_distances(path: List[State4]) -> Tuple[List[float], float]:
    dists = []
    min_dist = float('inf')
    for s in path:
        d = math.sqrt((s.x1 - s.x2)**2 + (s.y1 - s.y2)**2)
        dists.append(d)
        if d < min_dist:
            min_dist = d
    return dists, min_dist

def plot_results(scenario: Scenario, nodes: List[Node4], path: List[State4] = None, save_to: str = "out.png"):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # 1. Plot Map
        fig, ax = plt.subplots(figsize=(10, 10))
        
        for obs in scenario.obstacles:
            ax.add_patch(patches.Rectangle(
                (obs.xo, obs.yo), obs.lx, obs.ly, 
                edgecolor='black', facecolor='gray', alpha=0.5
            ))
            
        # Tree
        limit_nodes = nodes[::3] if len(nodes) > 3000 else nodes
        for node in limit_nodes:
            if node.parent is not None:
                parent = nodes[node.parent]
                plt.plot([parent.state.x1, node.state.x1], [parent.state.y1, node.state.y1], 'r-', alpha=0.1)
                plt.plot([parent.state.x2, node.state.x2], [parent.state.y2, node.state.y2], 'b-', alpha=0.1)

        # Path
        if path:
            x1 = [s.x1 for s in path]; y1 = [s.y1 for s in path]
            x2 = [s.x2 for s in path]; y2 = [s.y2 for s in path]
            plt.plot(x1, y1, 'r-', linewidth=2, label='Robot 1')
            plt.plot(x2, y2, 'b-', linewidth=2, label='Robot 2')
            
        plt.plot(scenario.start1.x, scenario.start1.y, 'ro', label='S1')
        plt.plot(scenario.start2.x, scenario.start2.y, 'bo', label='S2')
        plt.plot(scenario.goal1.x, scenario.goal1.y, 'rx', label='G1')
        plt.plot(scenario.goal2.x, scenario.goal2.y, 'bx', label='G2')
        
        plt.xlim(0, scenario.xmax)
        plt.ylim(0, scenario.ymax)
        plt.legend()
        plt.title(f"RRT 4D Basic (Anytime) - {len(nodes)} Nodes")
        plt.savefig(save_to)
        plt.close()
        print(f"Map plot saved to {save_to}")
        
        # 2. Plot Distances (if path exists)
        if path:
            dist_file = save_to.replace(".png", "_dist.png")
            dists, min_d = compute_distances(path)
            
            plt.figure(figsize=(10, 4))
            plt.plot(dists, 'k-', label='Inter-Robot Distance')
            plt.axhline(y=2*scenario.R, color='r', linestyle='--', label=f'Limit 2R ({2*scenario.R:.1f})')
            plt.ylim(0, max(max(dists), 3*scenario.R))
            plt.xlabel("Path Steps")
            plt.ylabel("Distance")
            plt.title(f"Distance vs. Time (Min Dist: {min_d:.2f})")
            plt.legend()
            plt.grid(True)
            plt.savefig(dist_file)
            plt.close()
            print(f"Distance plot saved to {dist_file}")
            print(f"Minimum Distance encountered: {min_d:.2f} (Required: {2*scenario.R})")

        
    except ImportError:
        print("Matplotlib not missing? Skipping plot.")
    except Exception as e:
        print(f"Plot failed: {e}")

# -----------------------------------------------------------------------------
# 4. Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", help="Path to scenario file")
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--step", type=float, default=20.0)
    parser.add_argument("--r-override", type=float, default=None)
    parser.add_argument("--out", type=str, default="rrt_basic_result.png")
    
    args = parser.parse_args()
    
    try:
        scen = load_scenario_robust(args.scenario, args.r_override)
        print(f"Loaded Scenario: {scen.xmax}x{scen.ymax}, R={scen.R}")
        print(f"S1: ({scen.start1.x:.2f}, {scen.start1.y:.2f})  S2: ({scen.start2.x:.2f}, {scen.start2.y:.2f})")
        print(f"G1: ({scen.goal1.x:.2f}, {scen.goal1.y:.2f})  G2: ({scen.goal2.x:.2f}, {scen.goal2.y:.2f})")
        
        start_time = time.time()
        solver = RRT2RobotsBasic(scen, max_iters=args.iters, step_size=args.step)
        path, cost, first_iter = solver.plan()
        duration = time.time() - start_time
        
        print(f"Planning complete in {duration:.4f}s")
        print(f"Total Iterations: {args.iters}")
        
        if path:
            print(f"Final Path Found! Steps: {len(path)}, Makespan: {cost:.2f}")
            if first_iter is not None:
                print(f"First Success Iter: {first_iter}")
        else:
            print("No path found.")
            
        plot_results(scen, solver.nodes, path, save_to=args.out)
        
    except Exception as e:
        print(f"Error: {e}")
