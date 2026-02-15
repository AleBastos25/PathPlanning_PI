from dataclasses import dataclass, field, replace as dc_replace
from typing import Optional, List, Tuple
import math
import time
import numpy as np

from ..models import Point, Scenario, ObstacleRect
from ..geometry import (
    path_collision_length,
    path_outside_bounds_length,
    path_length,
    obstacles_inflate,
    point_in_rect,
)


# Numerical tolerance for feasibility checks
EPS_FEASIBLE = 1e-9


@dataclass
class PSOConfig:
    # Base PSO
    n_particles: int = 30
    n_waypoints: int = 5
    max_iters: int = 100
    w: float = 0.9
    c1: float = 2.0
    c2: float = 2.0
    vmax: Optional[float] = None  # None = no clamp
    penalty_factor: float = 100.0
    penalty_lambda: Optional[float] = None
    eps_feasible: float = 1e-9
    patience_global: Optional[int] = None  # stop if no improvement for this many iters
    seed: Optional[int] = 42
    init_line_prob: float = 0.5
    init_line_noise: float = 100.0
    tol: float = 1e-6

    # Flags
    use_restart: bool = False
    use_sa: bool = False
    use_dimlearn: bool = False

    # Restart (only if use_restart=True): periodic full-swarm reinit every T iters
    restart_period_T: Optional[int] = None
    restart_keep_gbest: bool = False  # if True, one particle keeps current gbest position after restart
    
    # SA (only if use_sa=True): T_k = T0 * beta^k; accept worse gbest with prob exp(-delta/T)
    sa_T0: Optional[float] = None
    sa_beta: float = 0.92  # faster cooling so SA only helps early, then greedy
    sa_Tmin: float = 1e-12
    sa_max_relative_delta: Optional[float] = 0.12  # reject if (f_cand - f_curr) / f_curr > this (None = no cap)

    # Dimensional learning (only if use_dimlearn=True)
    dl_patience_Q: Optional[int] = None

    # If True, initial waypoints are resampled until outside all (inflated) obstacles
    init_avoid_obstacles: bool = False


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
    
    # Dimensional learning: stall counter (incremented when pbest does not improve)
    pbest_stall_counter: int = 0


@dataclass
class FitnessComponents:
    f: float
    L: float
    L_coll: float
    L_out: float
    feasible: bool


def compare_solutions(
    metrics_a: FitnessComponents,
    metrics_b: FitnessComponents,
    f_a: float,
    f_b: float,
) -> bool:
    """Lexicographic: True if A is strictly better than B (feasible > infeasible; then lower L or lower violation)."""
    v_a = metrics_a.L_coll + metrics_a.L_out
    v_b = metrics_b.L_coll + metrics_b.L_out
    if metrics_a.feasible and not metrics_b.feasible:
        return True
    if not metrics_a.feasible and metrics_b.feasible:
        return False
    if metrics_a.feasible and metrics_b.feasible:
        return metrics_a.L < metrics_b.L
    return v_a < v_b


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
    history_feasible: List[bool] = field(default_factory=list)  # per-iteration: was global best feasible?


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
    eps_feasible: float = EPS_FEASIBLE,
) -> FitnessComponents:
    path = build_path(scenario, x, m)
    L = path_length(path)
    L_coll = path_collision_length(path, inflated_obstacles)
    L_out = path_outside_bounds_length(path, scenario.xmax, scenario.ymax)
    f = L + penalty_lambda * (L_coll**2 + L_out**2)
    feasible = (L_coll <= eps_feasible) and (L_out <= eps_feasible)
    return FitnessComponents(f=f, L=L, L_coll=L_coll, L_out=L_out, feasible=feasible)


def evaluate_position(
    x: np.ndarray,
    scenario: Scenario,
    config: PSOConfig,
    m: int,
    penalty_lambda: float,
    inflated_obstacles: List[ObstacleRect],
) -> FitnessComponents:
    """Evaluate position and return metrics (uses config.eps_feasible)."""
    return evaluate(
        x, scenario, m, penalty_lambda, inflated_obstacles, config.eps_feasible
    )


# =============================================================================
# 9. Particle Initialization
# =============================================================================


def _sample_point_outside_obstacles(
    scenario: Scenario,
    obstacles: List[ObstacleRect],
    rng: np.random.Generator,
    max_tries: int = 200,
) -> Tuple[float, float]:
    """Sample (x, y) in [0, xmax] x [0, ymax] that is not inside any obstacle."""
    for _ in range(max_tries):
        wx = rng.uniform(0, scenario.xmax)
        wy = rng.uniform(0, scenario.ymax)
        if not any(point_in_rect(Point(wx, wy), r) for r in obstacles):
            return (wx, wy)
    return (rng.uniform(0, scenario.xmax), rng.uniform(0, scenario.ymax))


def _sample_position(
    scenario: Scenario,
    n_waypoints: int,
    init_line_prob: float,
    init_line_noise: float,
    rng: np.random.Generator,
    xmax_ymax: float,
) -> np.ndarray:
    x = np.zeros(2 * n_waypoints)
    if rng.random() < init_line_prob:
        start, goal = scenario.start1, scenario.goal1
        for i in range(n_waypoints):
            t = (i + 1) / (n_waypoints + 1)
            x[2*i] = start.x + t * (goal.x - start.x) + rng.normal(0, init_line_noise)
            x[2*i+1] = start.y + t * (goal.y - start.y) + rng.normal(0, init_line_noise)
    else:
        x[0::2] = rng.uniform(0, scenario.xmax, size=n_waypoints)
        x[1::2] = rng.uniform(0, scenario.ymax, size=n_waypoints)
    return np.clip(x, 0, xmax_ymax)


def init_particle(
    scenario: Scenario,
    config: PSOConfig,
    rng: np.random.Generator,
    inflated_obstacles: List[ObstacleRect],
    penalty_lambda: float,
    v0: float = 0.1,
    x_override: Optional[np.ndarray] = None,
) -> Particle:
    """Initialize a single particle. If x_override is set, use it as position (clipped to bounds)."""
    m = config.n_waypoints
    xmax_ymax = max(scenario.xmax, scenario.ymax)
    if x_override is not None and len(x_override) == 2 * m:
        x = np.clip(np.asarray(x_override, dtype=float), 0, xmax_ymax)
    else:
        x = _sample_position(
            scenario, m, config.init_line_prob, config.init_line_noise, rng, xmax_ymax
        )
        if config.init_avoid_obstacles and inflated_obstacles:
            for i in range(m):
                if any(point_in_rect(Point(x[2 * i], x[2 * i + 1]), r) for r in inflated_obstacles):
                    x[2 * i], x[2 * i + 1] = _sample_point_outside_obstacles(
                        scenario, inflated_obstacles, rng
                    )
            x = np.clip(x, 0, xmax_ymax)
    v = rng.uniform(-v0, v0, size=2 * m)
    comp = evaluate_position(
        x, scenario, config, m, penalty_lambda, inflated_obstacles
    )
    particle = Particle(
        x=x, v=v, f=comp.f,
        pbest_x=x.copy(), pbest_f=comp.f,
        pbest_stall_counter=0,
    )
    if comp.feasible:
        particle.pbest_feas_x = x.copy()
        particle.pbest_feas_f = comp.f
        particle.pbest_feas_L = comp.L
    return particle


def initialize_swarm(
    scenario: Scenario,
    config: PSOConfig,
    rng: np.random.Generator,
    inflated_obstacles: List[ObstacleRect],
    penalty_lambda: float,
) -> List[Particle]:
    """Create initial swarm."""
    particles = []
    for _ in range(len(particles), config.n_particles):
        particles.append(
            init_particle(scenario, config, rng, inflated_obstacles, penalty_lambda)
        )
    return particles


# =============================================================================
# 10. PSO Update: velocity/position step and pbest update (with optional SA)
# =============================================================================


def sa_accept_as_gbest(
    f_candidate: float,
    f_current: float,
    T: float,
    rng: np.random.Generator,
    max_relative_delta: Optional[float] = None,
) -> bool:
    """
    Accept candidate as new global best with probability min(1, exp(-(f(s)-f(g_k))/T)).
    If f_candidate <= f_current, always accept. Else accept with prob exp(-(f_candidate - f_current)/T).
    If max_relative_delta is set and f_current > 0, reject when (f_candidate - f_current) / f_current > max_relative_delta.
    """
    if f_candidate <= f_current:
        return True
    if T <= 0:
        return False
    delta = f_candidate - f_current
    if max_relative_delta is not None and f_current > 0:
        if delta / f_current > max_relative_delta:
            return False
    return rng.random() < math.exp(-delta / T)


def step_velocity_position(
    particle: Particle,
    gbest_x: np.ndarray,
    scenario: Scenario,
    config: PSOConfig,
    rng: np.random.Generator,
) -> None:
    """Update velocity and position only (no evaluation). Uses config.w, c1, c2, vmax."""
    D = len(particle.x)
    r1, r2 = rng.random(D), rng.random(D)
    pbest = particle.pbest_feas_x if particle.pbest_feas_x is not None else particle.pbest_x
    cognitive = config.c1 * r1 * (pbest - particle.x)
    social = config.c2 * r2 * (gbest_x - particle.x)
    particle.v = config.w * particle.v + cognitive + social
    if config.vmax is not None:
        particle.v = np.clip(particle.v, -config.vmax, config.vmax)
    particle.x = particle.x + particle.v
    xmax_ymax = max(scenario.xmax, scenario.ymax)
    particle.x = np.clip(particle.x, 0, xmax_ymax)


def update_pbest(
    particle: Particle,
    comp: FitnessComponents,
    config: PSOConfig,
) -> Tuple[bool, bool]:
    """
    Update pbest (and feasible pbest) from current position metrics.
    Returns (general_pbest_updated, feasible_pbest_updated).
    """
    updated_gen = False
    updated_feas = False
    if comp.f < particle.pbest_f:
        particle.pbest_x = particle.x.copy()
        particle.pbest_f = comp.f
        updated_gen = True
    if comp.feasible and (particle.pbest_feas_x is None or comp.f < particle.pbest_feas_f):
        particle.pbest_feas_x = particle.x.copy()
        particle.pbest_feas_f = comp.f
        particle.pbest_feas_L = comp.L
        updated_feas = True
    return updated_gen, updated_feas


# =============================================================================
# 11. Restart (periodic full-swarm reinit) and Dimensional Learning
# =============================================================================
def reinit_all_particles(
    swarm: List[Particle],
    scenario: Scenario,
    config: PSOConfig,
    rng: np.random.Generator,
    inflated_obstacles: List[ObstacleRect],
    penalty_lambda: float,
    xmax_ymax: float,
    keep_position: Optional[np.ndarray] = None,
) -> None:
    """Re-initialize all particles (positions, velocity, pbest). Does not change gbest.
    If keep_position is set, the first particle keeps that position (and pbest from it)."""
    m = config.n_waypoints
    for i, p in enumerate(swarm):
        if keep_position is not None and len(keep_position) == 2 * m and i == 0:
            x = np.clip(np.asarray(keep_position, dtype=float), 0, xmax_ymax)
            p.x = x
            p.v = rng.uniform(-0.05, 0.05, size=len(p.x))
        else:
            p.x = _sample_position(
                scenario, m, config.init_line_prob, config.init_line_noise, rng, xmax_ymax
            )
            if config.init_avoid_obstacles and inflated_obstacles:
                for j in range(m):
                    if any(point_in_rect(Point(p.x[2 * j], p.x[2 * j + 1]), r) for r in inflated_obstacles):
                        p.x[2 * j], p.x[2 * j + 1] = _sample_point_outside_obstacles(
                            scenario, inflated_obstacles, rng
                        )
                p.x = np.clip(p.x, 0, xmax_ymax)
            p.v = rng.uniform(-0.1, 0.1, size=len(p.x))
        comp = evaluate_position(p.x, scenario, config, m, penalty_lambda, inflated_obstacles)
        p.f = comp.f
        p.pbest_x = p.x.copy()
        p.pbest_f = comp.f
        p.pbest_feas_x = None
        p.pbest_feas_f = float("inf")
        p.pbest_feas_L = float("inf")
        if comp.feasible:
            p.pbest_feas_x = p.x.copy()
            p.pbest_feas_f = comp.f
            p.pbest_feas_L = comp.L
        p.pbest_stall_counter = 0


def dimensional_learning_update(
    particle: Particle,
    gbest_x: np.ndarray,
    scenario: Scenario,
    config: PSOConfig,
    rng: np.random.Generator,
    inflated_obstacles: List[ObstacleRect],
    penalty_lambda: float,
) -> Tuple[bool, int, int]:
    """
    Copy dimensions from gbest into pbest one by one (sequential, lexicographic acceptance).
    Returns (improved, dims_tested, dims_accepted).
    """
    D = len(particle.pbest_x)
    dims = list(range(D))
    p = particle.pbest_x.copy()
    p_metrics = evaluate_position(p, scenario, config, config.n_waypoints, penalty_lambda, inflated_obstacles)
    dims_tested = 0
    dims_accepted = 0
    for j in dims:
        p_new = p.copy()
        p_new[j] = gbest_x[j]
        p_new = np.clip(p_new, 0, max(scenario.xmax, scenario.ymax))
        dims_tested += 1
        comp_new = evaluate_position(p_new, scenario, config, config.n_waypoints, penalty_lambda, inflated_obstacles)
        accept = compare_solutions(comp_new, p_metrics, comp_new.f, p_metrics.f)
        if accept:
            p = p_new
            p_metrics = comp_new
            dims_accepted += 1
    improved = dims_accepted > 0
    if improved:
        particle.pbest_x = p.copy()
        particle.pbest_f = p_metrics.f
        if p_metrics.feasible:
            particle.pbest_feas_x = p.copy()
            particle.pbest_feas_f = p_metrics.f
            particle.pbest_feas_L = p_metrics.L
        else:
            particle.pbest_feas_x = None
            particle.pbest_feas_f = float("inf")
            particle.pbest_feas_L = float("inf")
        particle.pbest_stall_counter = 0
    return improved, dims_tested, dims_accepted


# =============================================================================
# 12. Main PSO Loop (legacy and config-based)
# =============================================================================


@dataclass
class _ConfigRunResult:
    feas_x: Optional[np.ndarray]
    feas_f: float
    feas_L: float
    best_gen_x: np.ndarray
    best_gen_f: float
    history: List[float]
    history_feasible: List[bool]
    iters_used: int


def _resolve_config(config: PSOConfig, scenario: Scenario) -> PSOConfig:
    """Fill in None fields that depend on scenario/m."""
    m = config.n_waypoints
    cfg = config
    if cfg.use_restart and cfg.restart_period_T is None:
        cfg = dc_replace(cfg, restart_period_T=max(25, config.max_iters // 4))
    if cfg.dl_patience_Q is None:
        cfg = dc_replace(cfg, dl_patience_Q=10 * m)
    if cfg.sa_T0 is None and cfg.use_sa:
        L_max = 2 * (scenario.xmax + scenario.ymax)
        cfg = dc_replace(cfg, sa_T0=L_max * 0.015)  # lower T0 so we don't accept very bad moves
    return cfg


def pso_single_run_config(
    scenario: Scenario,
    config: PSOConfig,
    rng: np.random.Generator,
    inflated_obstacles: List[ObstacleRect],
    penalty_lambda: float,
) -> _ConfigRunResult:
    """Single PSO run with optional restart, SA, dimensional learning."""
    cfg = _resolve_config(config, scenario)
    m = cfg.n_waypoints
    xmax_ymax = max(scenario.xmax, scenario.ymax)
    swarm = initialize_swarm(scenario, cfg, rng, inflated_obstacles, penalty_lambda)

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

    # Plot value: feasible best if available, else general best
    history = [gbest_feas_f if gbest_feas_x is not None else gbest_gen_f]
    history_feasible = [gbest_feas_x is not None]
    T = cfg.sa_T0 if cfg.use_sa else 0.0
    T_period = cfg.restart_period_T if cfg.use_restart else (cfg.max_iters + 1)
    Q = cfg.dl_patience_Q or (10 * m)

    for k in range(1, cfg.max_iters + 1):
        gbest_for_social = gbest_feas_x if gbest_feas_x is not None else gbest_gen_x

        for particle in swarm:
            step_velocity_position(particle, gbest_for_social, scenario, cfg, rng)
            comp = evaluate_position(
                particle.x, scenario, cfg, m, penalty_lambda, inflated_obstacles
            )
            particle.f = comp.f
            upd_gen, upd_feas = update_pbest(particle, comp, cfg)
            if upd_gen or upd_feas:
                particle.pbest_stall_counter = 0
            else:
                particle.pbest_stall_counter += 1
            # gbest update: with SA (enunciado) adopt s as gbest with prob min(1, exp(-(f(s)-f(g_k))/T))
            if cfg.use_sa and T > 0:
                max_rel = cfg.sa_max_relative_delta
                if sa_accept_as_gbest(particle.pbest_f, gbest_gen_f, T, rng, max_rel):
                    gbest_gen_x = particle.pbest_x.copy()
                    gbest_gen_f = particle.pbest_f
                if particle.pbest_feas_x is not None and sa_accept_as_gbest(particle.pbest_feas_f, gbest_feas_f, T, rng, max_rel):
                    gbest_feas_x = particle.pbest_feas_x.copy()
                    gbest_feas_f = particle.pbest_feas_f
                    gbest_feas_L = particle.pbest_feas_L
            else:
                if particle.pbest_f < gbest_gen_f:
                    gbest_gen_x = particle.pbest_x.copy()
                    gbest_gen_f = particle.pbest_f
                if particle.pbest_feas_x is not None and particle.pbest_feas_f < gbest_feas_f - cfg.tol:
                    gbest_feas_x = particle.pbest_feas_x.copy()
                    gbest_feas_f = particle.pbest_feas_f
                    gbest_feas_L = particle.pbest_feas_L

        if cfg.use_dimlearn:
            for particle in swarm:
                if particle.pbest_stall_counter >= Q:
                    gx = gbest_feas_x if gbest_feas_x is not None else gbest_gen_x
                    improved, _tested, _accepted = dimensional_learning_update(
                        particle, gx, scenario, cfg, rng, inflated_obstacles, penalty_lambda
                    )
                    if improved:
                        if cfg.use_sa and T > 0:
                            max_rel = cfg.sa_max_relative_delta
                            if particle.pbest_feas_x is not None and sa_accept_as_gbest(particle.pbest_feas_f, gbest_feas_f, T, rng, max_rel):
                                gbest_feas_x = particle.pbest_feas_x.copy()
                                gbest_feas_f = particle.pbest_feas_f
                                gbest_feas_L = particle.pbest_feas_L
                            if sa_accept_as_gbest(particle.pbest_f, gbest_gen_f, T, rng, max_rel):
                                gbest_gen_x = particle.pbest_x.copy()
                                gbest_gen_f = particle.pbest_f
                        else:
                            if particle.pbest_feas_x is not None and particle.pbest_feas_f < gbest_feas_f - cfg.tol:
                                gbest_feas_x = particle.pbest_feas_x.copy()
                                gbest_feas_f = particle.pbest_feas_f
                                gbest_feas_L = particle.pbest_feas_L
                            if particle.pbest_f < gbest_gen_f:
                                gbest_gen_x = particle.pbest_x.copy()
                                gbest_gen_f = particle.pbest_f

        if cfg.use_restart and T_period and k % T_period == 0:
            reinit_all_particles(
                swarm, scenario, cfg, rng, inflated_obstacles, penalty_lambda, xmax_ymax,
                keep_position=gbest_gen_x.copy() if cfg.restart_keep_gbest else None,
            )
            if cfg.use_sa:
                T = cfg.sa_T0 or 0.0

        # Only cool when we already have a feasible solution; keep T high to keep exploring until then
        if cfg.use_sa and gbest_feas_x is not None:
            T = max(cfg.sa_Tmin, cfg.sa_beta * T)

        history.append(gbest_feas_f if gbest_feas_x is not None else gbest_gen_f)
        history_feasible.append(gbest_feas_x is not None)

    return _ConfigRunResult(
        feas_x=gbest_feas_x,
        feas_f=gbest_feas_f,
        feas_L=gbest_feas_L,
        best_gen_x=gbest_gen_x,
        best_gen_f=gbest_gen_f,
        history=history,
        history_feasible=history_feasible,
        iters_used=k,
    )


def pso_plan(scenario: Scenario, config: PSOConfig) -> PSOResult:
    """Main PSO path planning. Single entry: PSOConfig only."""
    start_time = time.perf_counter()
    penalty_lambda = config.penalty_lambda
    if penalty_lambda is None:
        penalty_lambda = compute_adaptive_penalties(scenario, config.penalty_factor)
    rng = np.random.default_rng(config.seed)
    inflated_obstacles = obstacles_inflate(scenario.obstacles, scenario.R)
    run = pso_single_run_config(
        scenario, config, rng, inflated_obstacles, penalty_lambda
    )
    cpu_time = time.perf_counter() - start_time
    m = config.n_waypoints
    if run.feas_x is not None:
        best_path = build_path(scenario, run.feas_x, m)
        final_comp = evaluate_position(
            run.feas_x, scenario, config, m, penalty_lambda, inflated_obstacles
        )
        return PSOResult(
            best_x=run.feas_x, best_f=final_comp.f, best_path=best_path,
            iters_used=run.iters_used, cpu_time=cpu_time, converged=True,
            L=final_comp.L, L_coll=final_comp.L_coll, L_out=final_comp.L_out,
            n_waypoints=m, history=run.history, history_feasible=run.history_feasible,
        )
    best_gen_x = run.best_gen_x
    best_path = build_path(scenario, best_gen_x, m)
    final_comp = evaluate_position(
        best_gen_x, scenario, config, m, penalty_lambda, inflated_obstacles
    )
    return PSOResult(
        best_x=best_gen_x, best_f=final_comp.f, best_path=best_path,
        iters_used=run.iters_used, cpu_time=cpu_time, converged=False,
        L=final_comp.L, L_coll=final_comp.L_coll, L_out=final_comp.L_out,
        n_waypoints=m, history=run.history, history_feasible=run.history_feasible,
    )
