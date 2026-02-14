"""Plotting utilities for experiments."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def plot_convergence(
    history: List[float],
    scenario_id: str,
    save_to: Path,
    fontsize: int = 16,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(range(len(history)), history, linewidth=2, color="blue")
    ax.set_xlabel("Iteration", fontsize=fontsize)
    ax.set_ylabel("Global Best Fitness", fontsize=fontsize)
    ax.set_title(f"PSO Convergence - Scenario {scenario_id}", fontsize=fontsize + 2)
    ax.tick_params(axis="both", labelsize=fontsize - 2)
    ax.grid(True, alpha=0.3)
    
    # Use log scale if range is large
    if len(history) > 1:
        min_val = min(history)
        max_val = max(history)
        if min_val > 0 and max_val / min_val > 100:
            ax.set_yscale("log")
    
    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
