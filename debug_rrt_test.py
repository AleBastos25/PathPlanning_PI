import sys
from src.algorithms.rrt_2robots_basic import RRT2RobotsBasic, load_scenario_unified

def run_debug():
    try:
        scenario = load_scenario_unified("scenarios/test_2robots.txt", R_override=20.0)
        solver = RRT2RobotsBasic(scenario, max_iters=2000, step_size=30.0)
        
        print("Starting Debug Run...")
        
        print("Running full plan...")
        path, cost = solver.plan()
        
        if path:
             print(f"SUCCESS! Path len: {len(path)}, Cost: {cost}")
        else:
             print("FAILURE: No path found.")
             
        # with open("debug_log.txt", "w") as f: ...
            
        print("Debug run done. Wrote debug_log.txt")

    except Exception as e:
        print(f"Exception: {e}")
        with open("debug_log.txt", "a") as f:
            f.write(f"EXCEPTION: {e}\n")

if __name__ == "__main__":
    run_debug()
