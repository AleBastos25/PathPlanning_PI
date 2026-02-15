import re
from pathlib import Path

def parse_log(filepath):
    content = Path(filepath).read_text(encoding='utf-16')
    data = {}
    
    # Extract CPU Time
    m_time = re.search(r"Planning complete in ([\d\.]+)s", content)
    if m_time: data['time'] = float(m_time.group(1))
    
    # Extract Total Iters
    m_iters = re.search(r"Iterations: (\d+)", content)
    if m_iters: data['iters'] = int(m_iters.group(1))
    
    # Extract Path Cost
    m_cost = re.search(r"Path found! Cost: ([\d\.]+)", content)
    if m_cost: data['cost'] = float(m_cost.group(1))
    
    # Extract Path Length
    m_len = re.search(r"Path length \(nodes\): (\d+)", content)
    if m_len: data['len'] = int(m_len.group(1))
    
    # Extract First Success Iter
    m_first = re.search(r"First Success Iter: (\d+)", content)
    if m_first: data['first'] = int(m_first.group(1))
    
    return data

scenarios = ["scenario0", "scenario1", "scenario2", "scenario3", "scenario4", "test_2robots", "test_empty"]

with open("output/rrt_2robots/summary.txt", "w") as f:
    for s in scenarios:
        try:
            data = parse_log(f"output/rrt_2robots/log_{s}.txt")
            f.write(f"{s}: {data}\n")
        except Exception as e:
            f.write(f"{s}: Error: {e}\n")
