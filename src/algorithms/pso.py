from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set, Tuple
import time
import numpy as np

from ..models import Point, Scenario, ObstacleRect
from ..geometry import (
    path_collision_length,
    path_outside_bounds_length,
    path_length,
    clamp_points,
    obstacles_inflate,
)


# Numerical tolerance for feasibility checks
EPS_FEASIBLE = 1e-9


@dataclass
class PSOParams:
    # Core
    n_waypoints: Optional[int] = 10
    n_particles: Optional[int] = 30
    max_iters: Optional[int] = 100
    
    # Inertia
    w: float = 0.9
    
    # Acceleration
    c1: float = 2.0
    c2: float = 2.0
    
    # Penalties
    penalty_factor: Optional[float] = 100
    penalty_lambda: Optional[float] = None  # Computed from penalty_factor if None
    
    # Convergence tolerance
    tol: float = 1e-6
    
    # Initialization
    init_line_prob: float = 0.5
    init_line_noise: float = 100.0
    
    # Random seed
    seed: Optional[int] = 42


def compute_adaptive_penalties(scenario: Scenario, penalty_factor: float) -> float:
    L_max = 2 * (scenario.xmax + scenario.ymax)
    penalty_lambda = penalty_factor * L_max
    return penalty_lambda


@dataclass
class Particle:
    x: np.ndarray
    v: np.ndarray
    f: float = float("inf")
    
    # General pbest
    pbest_x: np.ndarray = field(default_factory=lambda: np.array([]))
    pbest_f: float = float("inf")
    
    # Feasible pbest
    pbest_feas_x: Optional[np.ndarray] = None
    pbest_feas_f: float = float("inf")
    pbest_feas_L: float = float("inf")


@dataclass
class FitnessComponents:
    f: float
    L: float
    L_coll: float
    L_out: float
    feasible: bool


@dataclass
class PSOResult:
    best_x: np.ndarray
    best_f: float
    best_path: List[Point]
    iters_used: int
    cpu_time: float
    converged: bool
    L: float
    L_coll: float
    L_out: float
    n_waypoints: int
    history: List[float] = field(default_factory=list)


def unpack_waypoints(x: np.ndarray, m: int) -> List[Point]:
    """Convert position vector to waypoints."""
    return [Point(x[2*i], x[2*i+1]) for i in range(m)]


def pack_waypoints(waypoints: List[Point]) -> np.ndarray:
    """Convert waypoints to position vector."""
    x = np.empty(2 * len(waypoints))
    for i, wp in enumerate(waypoints):
        x[2*i], x[2*i+1] = wp.x, wp.y
    return x


def build_path(scenario: Scenario, x: np.ndarray, m: int) -> List[Point]:
    """Build complete path from position vector."""
    return [scenario.start1] + unpack_waypoints(x, m) + [scenario.goal1]


# =============================================================================
# 8. Fitness Evaluation
# =============================================================================


def evaluate(
    x: np.ndarray,
    scenario: Scenario,
    m: int,
    penalty_lambda: float,
    inflated_obstacles: List[ObstacleRect],
) -> FitnessComponents:
    path = build_path(scenario, x, m)
    
    L = path_length(path)
    L_coll = path_collision_length(path, inflated_obstacles)
    L_out = path_outside_bounds_length(path, scenario.xmax, scenario.ymax)
    
    # Compute fitness with linear penalization
    f = L + penalty_lambda * (L_coll + L_out)
    
    # STRICT feasibility check:
    feasible = (L_coll <= EPS_FEASIBLE) and (L_out <= EPS_FEASIBLE)
    
    
    return FitnessComponents(f=f, L=L, L_coll=L_coll, L_out=L_out, feasible=feasible)


# =============================================================================
# 9. Particle Initialization
# =============================================================================


def init_particle(
    scenario: Scenario,
    cfg: PSOParams,
    rng: np.random.Generator,
    inflated_obstacles: List[ObstacleRect],
    v0: float = 0.1,
) -> Particle:
    """Initialize a particle."""
    n_waypoints = cfg.n_waypoints
    x = np.zeros(2 * n_waypoints)
    
    if rng.random() < cfg.init_line_prob:
        # Line initialization with noise
        start, goal = scenario.start1, scenario.goal1
        for i in range(n_waypoints):
            t = (i + 1) / (n_waypoints + 1)
            x[2*i] = start.x + t * (goal.x - start.x) + rng.normal(0, cfg.init_line_noise)
            x[2*i+1] = start.y + t * (goal.y - start.y) + rng.normal(0, cfg.init_line_noise)
    else:
        # Random initialization
        x[0::2] = rng.uniform(0, scenario.xmax, size=n_waypoints)
        x[1::2] = rng.uniform(0, scenario.ymax, size=n_waypoints)
    
    # Clamp positions to be within the scenario bounds
    x = clamp_points(x, scenario.xmax, scenario.ymax)
    
    # Initialize velocity
    v = rng.uniform(-v0, v0, size=2 * n_waypoints)
    
    # Evaluate
    comp = evaluate(x, scenario, n_waypoints, cfg.penalty_lambda, inflated_obstacles)
    
    particle = Particle(x=x, v=v, f=comp.f, pbest_x=x.copy(), pbest_f=comp.f)
    
    if comp.feasible:
        particle.pbest_feas_x = x.copy()
        particle.pbest_feas_f = comp.f
        particle.pbest_feas_L = comp.L
    
    return particle


# =============================================================================
# 10. PSO Update with Adaptive Inertia and Mutation
# =============================================================================


def update_particle(
    particle: Particle,
    gbest_x: np.ndarray,
    scenario: Scenario,
    n_waypoints: int,
    cfg: PSOParams,
    rng: np.random.Generator,
    inflated_obstacles: List[ObstacleRect],
) -> FitnessComponents:
    D = len(particle.x)
    
    r1, r2 = rng.random(D), rng.random(D) 
    pbest = particle.pbest_feas_x if particle.pbest_feas_x is not None else particle.pbest_x
    
    # Velocity update with adaptive inertia
    cognitive = cfg.c1 * r1 * (pbest - particle.x)
    social = cfg.c2 * r2 * (gbest_x - particle.x)
    particle.v = cfg.w * particle.v + cognitive + social
    
    # Position update
    particle.x = particle.x + particle.v
    
    
    particle.x = clamp_points(particle.x, scenario.xmax, scenario.ymax)
    
    # Evaluate
    comp = evaluate(particle.x, scenario, n_waypoints, cfg.penalty_lambda, inflated_obstacles)
    particle.f = comp.f
    
    # Update general pbest
    if comp.f < particle.pbest_f:
        particle.pbest_x = particle.x.copy()
        particle.pbest_f = comp.f
    
    # Update feasible pbest
    if comp.feasible and comp.f < particle.pbest_feas_f:
        particle.pbest_feas_x = particle.x.copy()
        particle.pbest_feas_f = comp.f
        particle.pbest_feas_L = comp.L
    
    return comp


# =============================================================================
# 11. Main PSO Loop with Restarts
# =============================================================================


def pso_single_run(
    scenario: Scenario,
    cfg: PSOParams,
    rng: np.random.Generator,
    inflated_obstacles: List[ObstacleRect],
) -> Tuple[Optional[np.ndarray], float, float, List[float], int]:
    """Single PSO run. Returns (best_feas_x, best_feas_f, best_feas_L, history, iters)."""
    # Initialize swarm
    swarm = [init_particle(scenario, cfg, rng, inflated_obstacles)
             for _ in range(cfg.n_particles)]
    
    # Find initial bests
    gbest_gen_x = min(swarm, key=lambda p: p.pbest_f).pbest_x.copy()
    gbest_gen_f = min(p.pbest_f for p in swarm)
    
    gbest_feas_x: Optional[np.ndarray] = None
    gbest_feas_f = float("inf")
    gbest_feas_L = float("inf")
    
    for p in swarm:
        if p.pbest_feas_x is not None and p.pbest_feas_f < gbest_feas_f:
            gbest_feas_x = p.pbest_feas_x.copy()
            gbest_feas_f = p.pbest_feas_f
            gbest_feas_L = p.pbest_feas_L
    
    history = [gbest_feas_f]
    iters_used = 0
    
    for it in range(1, cfg.max_iters + 1):
        iters_used = it
        # Choose gbest for social component
        gbest_for_social = gbest_feas_x if gbest_feas_x is not None else gbest_gen_x
        
        for particle in swarm:
            comp = update_particle(
                particle, gbest_for_social, scenario, cfg.n_waypoints, cfg, rng, inflated_obstacles
            )
            
            # Update general gbest
            if particle.pbest_f < gbest_gen_f:
                gbest_gen_x = particle.pbest_x.copy()
                gbest_gen_f = particle.pbest_f
            
            # Update feasible gbest
            if particle.pbest_feas_x is not None and particle.pbest_feas_f < gbest_feas_f - cfg.tol:
                gbest_feas_x = particle.pbest_feas_x.copy()
                gbest_feas_f = particle.pbest_feas_f
                gbest_feas_L = particle.pbest_feas_L
        
        history.append(gbest_feas_f)
    
    return gbest_feas_x, gbest_feas_f, gbest_feas_L, history, iters_used


def pso_plan(
    scenario: Scenario,
    cfg: PSOParams,
) -> PSOResult:
    """Main PSO path planning function."""
    start_time = time.perf_counter()
    
    # Compute penalty_lambda if not provided
    if cfg.penalty_lambda is None:
        cfg.penalty_lambda = compute_adaptive_penalties(scenario, cfg.penalty_factor)
    
    # Create RNG and compute inflated obstacles
    rng = np.random.default_rng(cfg.seed)
    inflated_obstacles = obstacles_inflate(scenario.obstacles, scenario.R)
    
    # Run PSO
    feas_x, feas_f, feas_L, history, iters = pso_single_run(
        scenario, cfg, rng, inflated_obstacles
    )
    cpu_time = time.perf_counter() - start_time
    
    n_waypoints = cfg.n_waypoints
    
    # Build result
    if feas_x is not None:
        # Feasible solution found
        best_path = build_path(scenario, feas_x, n_waypoints)
        final_comp = evaluate(
            feas_x, 
            scenario, 
            n_waypoints, 
            cfg.penalty_lambda, 
            inflated_obstacles
        )
        
        return PSOResult(
            best_x=feas_x,
            best_f=final_comp.f,
            best_path=best_path,
            iters_used=iters,
            cpu_time=cpu_time,
            converged=True,
            L=final_comp.L,
            L_coll=final_comp.L_coll,
            L_out=final_comp.L_out,
            n_waypoints=n_waypoints,
            history=history,
        )
    else:
        # No feasible found - run a fresh swarm to get best general solution
        swarm = [init_particle(scenario, cfg, rng, inflated_obstacles)
                 for _ in range(cfg.n_particles)]
        best_gen = min(swarm, key=lambda p: p.pbest_f)
        best_path = build_path(scenario, best_gen.pbest_x, n_waypoints)
        final_comp = evaluate(
            best_gen.pbest_x, 
            scenario, 
            n_waypoints, 
            cfg.penalty_lambda, 
            inflated_obstacles
        )
        
        return PSOResult(
            best_x=best_gen.pbest_x,
            best_f=final_comp.f,
            best_path=best_path,
            iters_used=iters,
            cpu_time=cpu_time,
            converged=False,
            L=final_comp.L,
            L_coll=final_comp.L_coll,
            L_out=final_comp.L_out,
            n_waypoints=n_waypoints,
            history=history,
        )
