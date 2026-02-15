import argparse
from pathlib import Path

from src.loader import load_scenario
from src.algorithms.rrt import RRTStar, RRTParams
from src.visualization import plot_scenario

def main():
    parser = argparse.ArgumentParser(description="Run RRT* Path Planning")
    parser.add_argument("scenario", type=str, help="Path to scenario file")
    parser.add_argument("--iters", type=int, default=1000, help="Max iterations")
    parser.add_argument("--step", type=float, default=0.5, help="Step size (delta_s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--opt", action="store_true", help="Enable path optimization (shortcut)")
    
    # New features
    parser.add_argument("--intelligent", action="store_true", help="Enable Boundary-Aware RRT (Vertex-First + Edge-Follow)")
    
    parser.add_argument("--out", type=str, default="rrt_result.png", help="Output filename for plot")
    
    # Vertex Attachment Control
    parser.add_argument("--attach", action="store_true", help="Enable vertex attachment mechanism")
    parser.add_argument("--p-vertex", type=float, default=0.05, help="Probability of Vertex-First sampling in Normal mode")
    
    args = parser.parse_args()
    
    # Load scenario
    scenario_path = Path(args.scenario)
    if not scenario_path.exists():
        print(f"Error: Scenario file {scenario_path} not found.")
        return

    print(f"Loading scenario from {scenario_path}...")
    scenario = load_scenario(scenario_path)
    
    # Configure RRT
    params = RRTParams(
        step_size=args.step,
        neighbor_radius=args.step * 3.0,
        max_iters=args.iters,
        use_path_optimization=args.opt,
        use_boundary_aware=args.intelligent,
        use_vertex_attach=args.attach,
        p_vertex=args.p_vertex,
    )
    
    print(f"Running RRT* (Output: {args.out})...")
    rrt = RRTStar(scenario, params)
    result = rrt.plan()
    
    print(f"Planning complete in {result.cpu_time:.4f}s")
    print(f"Iterations: {result.iters}")
    
    if result.path:
        print(f"Path found! Cost: {result.cost:.4f}")
        print(f"Path length (nodes): {len(result.path)}")
        if result.first_success_iter is not None:
             print(f"First Success Iter: {result.first_success_iter}")
    else:
        print("No path found.")

    # Visualize
    # Create traces from tree
    traces = []
    for node in result.tree_nodes:
        if node.parent is not None:
            parent = result.tree_nodes[node.parent]
            traces.append((parent.pos, node.pos))
            
    print("Plotting results...")
    plot_scenario(
        scenario, 
        path=result.path, 
        traces=traces, 
        save_to=args.out,
        show=False
    )
    print(f"Result saved to {args.out}")

if __name__ == "__main__":
    main()
