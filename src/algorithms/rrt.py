import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..models import Point, ObstacleRect, Scenario
from ..geometry import (
    dist,
    segment_intersects_rect,
    point_in_bounds,
    path_length,
)

@dataclass
class Node:
    """Represents a node in the RRT tree."""
    pos: Point
    parent: Optional[int] = None  # Index of parent node in the tree list
    cost: float = 0.0  # Cost from start to this node
    children: List[int] = field(default_factory=list) # Indices of children

@dataclass
class RRTParams:
    step_size: float = 10.0
    neighbor_radius: float = 30.0
    max_iters: int = 5000
    goal_sample_rate: float = 0.05
    use_path_optimization: bool = False
    
    # Boundary-Aware Parameters
    use_boundary_aware: bool = False
    p_vertex: float = 0.05     # Bias for Vertex-First sampling (Normal mode)

    p_edge: float = 0.0        # Unused in Normal mode
    r_corner: float = 10.0     # Jitter radius around vertex (proportional to step)
    epsilon_out: float = 2.0   # Small offset from edge
    jitter_edge: float = 2.0   # Tangential jitter along edge
    
    T_stuck: int = 500         # Iters without improvement
    B_edge: int = 200          # Budget for edge-follow
    
    # Vertex Attachment
    use_vertex_attach: bool = False  # Default OFF for control
    attach_every: int = 50     # How often to retry attachment
    r_attach: float = 30.0     # Radius to force-attach
    r_dup: float = 5.0         # Deduplication radius (don't add if node exists)
    K_attach: int = 3          # Max vertices to attach per call
    M_attach_period: int = 50  # (Legacy/Alias to attach_every if needed, but we use attach_every now)
    goal_connect_threshold: float = 50.0  # Only try goal connect if close

@dataclass
class RRTResult:
    path: Optional[List[Point]]
    cost: float
    iters: int
    cpu_time: float
    tree_nodes: List[Node]
    first_success_iter: Optional[int] = None

class RRTStar:
    def __init__(self, scenario: Scenario, params: RRTParams):
        self.scenario = scenario
        self.params = params
        self.nodes: List[Node] = []
        self.goal_idx: Optional[int] = None
        
        # Pre-compute geometry by obstacle
        self.VerticesByObs: List[List[Point]] = []
        self.EdgesByObs: List[List[Tuple[Point, Point]]] = []
        
        # Also keep flat lists for global sampling if needed
        self.V: List[Point] = []
        
        for obs in self.scenario.obstacles:
            # Vertices
            p1 = Point(obs.x_min, obs.y_min)
            p2 = Point(obs.x_max, obs.y_min)
            p3 = Point(obs.x_max, obs.y_max)
            p4 = Point(obs.x_min, obs.y_max)
            
            verts = [p1, p2, p3, p4]
            self.VerticesByObs.append(verts)
            self.V.extend(verts)
            
            # Edges
            edges = [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]
            self.EdgesByObs.append(edges)
        
        # Initialize tree
        start_node = Node(pos=scenario.start1, parent=None, cost=0.0)
        self.nodes.append(start_node)

    def _sample_boundary_aware(self, mode: str, blocking_obs_idx: Optional[int]) -> Point:
        # 1. Goal Bias
        if random.random() < self.params.goal_sample_rate:
            return self.scenario.goal1
            
        if mode == 'EDGE_FOLLOW':
            # Target specific obstacle if known, else random
            target_idx = blocking_obs_idx if blocking_obs_idx is not None else -1
            if target_idx == -1 or target_idx >= len(self.scenario.obstacles):
                if not self.scenario.obstacles:
                     return self._sample_uniform()
                target_idx = random.randint(0, len(self.scenario.obstacles) - 1)
            
            # Select random edge from this obstacle
            edges = self.EdgesByObs[target_idx]
            p1, p2 = random.choice(edges)
            
            # Sample along edge
            t = random.random()
            
            # Vector along edge
            ux = p2.x - p1.x
            uy = p2.y - p1.y
            length = math.sqrt(ux*ux + uy*uy)
            if length < 1e-6:
                return self._sample_uniform()
                
            # Normalized tangent
            tx = ux / length
            ty = uy / length
            
            # Base point
            px = p1.x + t * ux
            py = p1.y + t * uy
            
            # Calculate Outward Normal
            # Edge vector is (ux, uy). Normals are (-uy, ux) or (uy, -ux).
            # We want the one pointing NOT towards the center.
            obs = self.scenario.obstacles[target_idx]
            center_x = (obs.x_min + obs.x_max) / 2.0
            center_y = (obs.y_min + obs.y_max) / 2.0
            
            # Vector from center to point
            cx = px - center_x
            cy = py - center_y
            
            # Normal candidate 1
            nx1, ny1 = -ty, tx
            
            # Check dot product
            if (nx1 * cx + ny1 * cy) > 0:
                nx, ny = nx1, ny1
            else:
                nx, ny = -nx1, -ny1
                
            # Apply offset and tangential jitter
            # Jitter
            jit = random.uniform(-self.params.jitter_edge, self.params.jitter_edge)
            
            final_x = px + self.params.epsilon_out * nx + jit * tx
            final_y = py + self.params.epsilon_out * ny + jit * ty
            
            return Point(final_x, final_y)
            
        else: # NORMAL mode
            r = random.random()
            if r < self.params.p_vertex and self.V:
                # If blocked, prefer vertices of blocking obs
                if blocking_obs_idx is not None and blocking_obs_idx < len(self.VerticesByObs):
                    v_list = self.VerticesByObs[blocking_obs_idx]
                    v = random.choice(v_list)
                else:
                    v = random.choice(self.V)
                
                angle = random.uniform(0, 2 * math.pi)
                dist_r = random.uniform(0, self.params.r_corner)
                return Point(v.x + dist_r * math.cos(angle), v.y + dist_r * math.sin(angle))
            else:
                return self._sample_uniform()

    def _sample_uniform(self) -> Point:
        x = random.uniform(0, self.scenario.xmax)
        y = random.uniform(0, self.scenario.ymax)
        return Point(x, y)
        
    def _sample(self) -> Point:
        if random.random() < self.params.goal_sample_rate:
            return self.scenario.goal1
        return self._sample_uniform()

    def _try_attach_vertices(self, blocking_obs_idx: Optional[int]):
        """Inject vertices into the tree if reachable."""
        potential_candidates = []
        # Priority to blocking obs
        if blocking_obs_idx is not None and blocking_obs_idx < len(self.VerticesByObs):
            potential_candidates.extend(self.VerticesByObs[blocking_obs_idx])
        
        # Add all other V if needed
        # Actually, simpler: just use V but try to prioritize blocking obs in selection
        if not potential_candidates:
             potential_candidates = list(self.V)
        else:
             # Extend with some global ones to not starve others?
             # For now, let's stick to blocking + random fill
             needed = self.params.K_attach - len(potential_candidates)
             if needed > 0:
                 rest = [v for v in self.V if v not in potential_candidates] # simple logic
                 if rest:
                     potential_candidates.extend(random.choices(rest, k=needed))
        
        # Select strictly K_attach
        if not potential_candidates:
            return
            
        if len(potential_candidates) > self.params.K_attach:
            candidates = random.sample(potential_candidates, self.params.K_attach)
        else:
            candidates = potential_candidates

        for v in candidates:
            # 1. Check deduplication (is there a node very close?)
            nearest_idx = self._nearest_node_idx(v)
            if nearest_idx == -1: continue
            nearest_node = self.nodes[nearest_idx]
            
            if dist(nearest_node.pos, v) <= self.params.r_dup:
                continue # Already in tree
                
            # 2. Check reachability and utility
            if dist(nearest_node.pos, v) <= self.params.r_attach:
                # Utility: Is vertex closer to goal than parent?
                # Heuristic to avoid going backwards too much
                d_v_goal = dist(v, self.scenario.goal1)
                d_u_goal = dist(nearest_node.pos, self.scenario.goal1)
                
                # Check collision free
                if d_v_goal < d_u_goal:
                     is_free, blocking = self._check_collision(nearest_node.pos, v)
                     if is_free:
                        cost = nearest_node.cost + dist(nearest_node.pos, v)
                        self._add_node(v, nearest_idx, cost)

    def _check_collision(self, p1: Point, p2: Point) -> Tuple[bool, Optional[int]]:
        """Returns (is_free, blocking_obs_index)."""
        if not point_in_bounds(p2, self.scenario.xmax, self.scenario.ymax):
            return False, None
        
        for i, obs in enumerate(self.scenario.obstacles):
            if segment_intersects_rect(p1, p2, obs):
                return False, i
        return True, None

    def _nearest_node_idx(self, p: Point) -> int:
        min_dist = float('inf')
        nearest_idx = -1
        for i, node in enumerate(self.nodes):
            d = dist(node.pos, p)
            if d < min_dist:
                min_dist = d
                nearest_idx = i
        return nearest_idx

    def _steer(self, from_pos: Point, to_point: Point) -> Point:
        d = dist(from_pos, to_point)
        if d <= self.params.step_size:
            return to_point
        theta = math.atan2(to_point.y - from_pos.y, to_point.x - from_pos.x)
        return Point(
            from_pos.x + self.params.step_size * math.cos(theta),
            from_pos.y + self.params.step_size * math.sin(theta)
        )
        
    def _add_node(self, pos: Point, parent_idx: Optional[int], cost: float) -> int:
        new_node = Node(pos=pos, parent=parent_idx, cost=cost)
        self.nodes.append(new_node)
        new_idx = len(self.nodes) - 1
        if parent_idx is not None:
            self.nodes[parent_idx].children.append(new_idx)
        return new_idx

    def _set_parent(self, child_idx: int, new_parent_idx: int, new_cost: float):
        child = self.nodes[child_idx]
        old_parent_idx = child.parent
        if old_parent_idx is not None:
             if child_idx in self.nodes[old_parent_idx].children:
                 self.nodes[old_parent_idx].children.remove(child_idx)
        child.parent = new_parent_idx
        child.cost = new_cost
        if new_parent_idx is not None:
            self.nodes[new_parent_idx].children.append(child_idx)
        self._propagate_cost_to_children(child_idx)

    def _propagate_cost_to_children(self, node_idx: int):
        node = self.nodes[node_idx]
        for child_idx in node.children:
            child = self.nodes[child_idx]
            d = dist(node.pos, child.pos)
            child.cost = node.cost + d
            self._propagate_cost_to_children(child_idx)

    def _get_neighbors_indices(self, p: Point) -> List[int]:
        indices = []
        r = self.params.neighbor_radius
        for i, node in enumerate(self.nodes):
            if dist(node.pos, p) <= r:
                indices.append(i)
        return indices
    
    def _greedy_simplify(self, path: List[Point]) -> List[Point]:
        """Runs a single forward greedy shortcut pass."""
        if len(path) <= 2: return list(path)
        simplified = [path[0]]
        current_idx = 0
        while current_idx < len(path) - 1:
            next_idx = current_idx + 1
            # Try to connect to the furthest reachable node
            for i in range(len(path) - 1, current_idx, -1):
                # Using [0] to get is_free boolean from _check_collision
                if self._check_collision(path[current_idx], path[i])[0]:
                    next_idx = i
                    break
            simplified.append(path[next_idx])
            current_idx = next_idx
        return simplified

    def _optimize_path(self, path: List[Point]) -> List[Point]:
        """Bidirectional greedy optimization (Forward + Backward)."""
        if len(path) <= 2: return path
        
        # 1. Forward Pass
        fwd_path = self._greedy_simplify(path)
        
        # 2. Backward Pass (Simplify reversed path, then reverse back)
        rev_path_input = path[::-1]
        rev_simplified = self._greedy_simplify(rev_path_input)
        bwd_path = rev_simplified[::-1]
        
        # Compare lengths
        len_fwd = path_length(fwd_path)
        len_bwd = path_length(bwd_path)
        
        # Return the shorter one
        if len_bwd < len_fwd:
            return bwd_path
        return fwd_path


    def plan(self) -> RRTResult:
        start_time = time.perf_counter()
        
        # State machine variables
        stuck_counter = 0
        edge_follow_budget = 0
        best_dist_to_goal_seen = float('inf')
        last_blocking_obs_idx: Optional[int] = None
        first_success_iter: Optional[int] = None
        
        # Initial dist
        best_dist_to_goal_seen = dist(self.scenario.start1, self.scenario.goal1)
        
        for i_iter in range(self.params.max_iters):
            # 1. Select Mode
            mode = 'NORMAL'
            if self.params.use_boundary_aware:
                if edge_follow_budget > 0:
                    mode = 'EDGE_FOLLOW'
                    edge_follow_budget -= 1
                else:
                    mode = 'NORMAL'
                
                # Refined Attachment Trigger
                if self.params.use_vertex_attach:
                     # Check schedule or blocked state?
                     # Plan says: if enabled AND (iter % every == 0 OR improved)
                     # Actually, attaching when improved is good to "lock in" progress.
                     # Attaching periodically helps exploration.
                     is_scheduled = (i_iter % self.params.attach_every == 0)
                     # We can also attach if we just switched to stuck, etc.
                     # Staying simple per plan:
                     if is_scheduled: # Simplify to period for stability
                          self._try_attach_vertices(last_blocking_obs_idx)
                
                rnd_point = self._sample_boundary_aware(mode, last_blocking_obs_idx)
            else:
                rnd_point = self._sample()

            # 2. Extend Tree
            nearest_idx = self._nearest_node_idx(rnd_point)
            nearest_node = self.nodes[nearest_idx]
            new_pos = self._steer(nearest_node.pos, rnd_point)
            
            # Check Collision & Capture Blocking Obs
            is_free, blocking_idx = self._check_collision(nearest_node.pos, new_pos)
            
            if not is_free:
                # Update blocking info
                if blocking_idx is not None:
                    last_blocking_obs_idx = blocking_idx
                continue
            
            # 3. Add Node (Standard RRT*)
            parent_idx = nearest_idx
            min_cost = nearest_node.cost + dist(nearest_node.pos, new_pos)
            neighbor_indices = self._get_neighbors_indices(new_pos)
            
            for i in neighbor_indices:
                neighbor = self.nodes[i]
                is_free_neighbor, _ = self._check_collision(neighbor.pos, new_pos)
                if is_free_neighbor:
                    cost = neighbor.cost + dist(neighbor.pos, new_pos)
                    if cost < min_cost:
                        min_cost = cost
                        parent_idx = i
            
            new_node_idx = self._add_node(new_pos, parent_idx, min_cost)
            
            # Rewire
            for i in neighbor_indices:
                if i == parent_idx: continue
                neighbor = self.nodes[i]
                is_free_neighbor, _ = self._check_collision(new_pos, neighbor.pos)
                if is_free_neighbor:
                    new_neighbor_cost = min_cost + dist(new_pos, neighbor.pos)
                    if new_neighbor_cost < neighbor.cost:
                        self._set_parent(i, new_node_idx, new_neighbor_cost)
            
            # 4. Progress Update
            d_goal = dist(new_pos, self.scenario.goal1)
            improved = False
            
            if d_goal < best_dist_to_goal_seen - 1e-3: # Small epsilon
                best_dist_to_goal_seen = d_goal
                improved = True
                
            # 5. State Machine Update
            if self.params.use_boundary_aware:
                if improved:
                    stuck_counter = 0
                    edge_follow_budget = 0
                    # Optional: attach on improvement?
                    if self.params.use_vertex_attach:
                        self._try_attach_vertices(last_blocking_obs_idx)
                else:
                    stuck_counter += 1
                    
                if stuck_counter >= self.params.T_stuck:
                    edge_follow_budget = self.params.B_edge
                    stuck_counter = 0

            # 6. Goal Connection
            # Only try if reasonably close to avoid expensive checks or if mode is simple
            if d_goal < self.params.goal_connect_threshold:
                is_free_goal, _ = self._check_collision(new_pos, self.scenario.goal1)
                if is_free_goal:
                    goal_cost = min_cost + d_goal
                    if self.goal_idx is None:
                        self.goal_idx = self._add_node(self.scenario.goal1, new_node_idx, goal_cost)
                        if first_success_iter is None:
                            first_success_iter = i_iter
                        # Reset stuck because we found goal!
                        stuck_counter = 0
                        best_dist_to_goal_seen = 0
                    else:
                        current_goal_node = self.nodes[self.goal_idx]
                        if goal_cost < current_goal_node.cost:
                            self._set_parent(self.goal_idx, new_node_idx, goal_cost)

        cpu_time = time.perf_counter() - start_time
        
        # Path reconstruction
        path = None
        final_cost = float('inf')
        if self.goal_idx is not None:
            path = []
            curr_idx = self.goal_idx
            final_cost = self.nodes[self.goal_idx].cost
            while curr_idx is not None:
                path.append(self.nodes[curr_idx].pos)
                curr_idx = self.nodes[curr_idx].parent
            path.reverse()
            
            if self.params.use_path_optimization:
                path = self._optimize_path(path)
                final_cost = path_length(path)

        return RRTResult(
            path=path,
            cost=final_cost,
            iters=self.params.max_iters,
            cpu_time=cpu_time,
            tree_nodes=self.nodes,
            first_success_iter=first_success_iter
        )
