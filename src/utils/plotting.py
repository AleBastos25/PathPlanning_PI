"""Plotting utilities for experiments."""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_convergence(
    history: List[float],
    scenario_id: str,
    save_to: Path,
    fontsize: int = 16,
    history_feasible: Optional[List[bool]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    n = len(history)
    x = list(range(n))

    if history_feasible is not None and len(history_feasible) == n:
        # Two y-axes when both types present: left for feasible, right for non-feasible (different scales)
        color_feasible = "green"
        color_infeasible = "coral"
        feas_vals = [history[i] for i in range(n) if history_feasible[i]]
        infeas_vals = [history[i] for i in range(n) if not history_feasible[i]]
        has_both = feas_vals and infeas_vals
        ax2 = ax.twinx() if has_both else None
        i = 0
        while i < n:
            feas = history_feasible[i]
            j = i + 1
            while j < n and history_feasible[j] == feas:
                j += 1
            if feas:
                ax.plot(x[i:j], history[i:j], linewidth=2, color=color_feasible)
            else:
                (ax2 or ax).plot(x[i:j], history[i:j], linewidth=2, color=color_infeasible)
            i = j
        if feas_vals:
            lo, hi = min(feas_vals), max(feas_vals)
            margin = (hi - lo) * 0.05 or 1.0
            ax.set_ylim(lo - margin, hi + margin)
            ax.set_ylabel("Feasible (Global Best Fitness)" if has_both else "Global Best Fitness", fontsize=fontsize, color=color_feasible if has_both else "black")
            ax.tick_params(axis="y", labelcolor=color_feasible if has_both else "black", labelsize=fontsize - 2)
        if infeas_vals:
            lo, hi = min(infeas_vals), max(infeas_vals)
            margin = (hi - lo) * 0.05 or 1.0
            if ax2 is not None:
                ax2.set_ylim(lo - margin, hi + margin)
                ax2.set_ylabel("Non-feasible (Global Best Fitness)", fontsize=fontsize, color=color_infeasible)
                ax2.tick_params(axis="y", labelcolor=color_infeasible, labelsize=fontsize - 2)
            else:
                ax.set_ylim(lo - margin, hi + margin)
                ax.set_ylabel("Non-feasible (Global Best Fitness)", fontsize=fontsize, color=color_infeasible)
                ax.tick_params(axis="y", labelcolor=color_infeasible, labelsize=fontsize - 2)
        handles = [
            Line2D([0], [0], color=color_feasible, linewidth=2),
            Line2D([0], [0], color=color_infeasible, linewidth=2),
        ]
        ax.legend(handles, ["Feasible", "Non-feasible"], loc="upper right", fontsize=fontsize - 2)
    else:
        ax.plot(x, history, linewidth=2, color="blue")

    use_two_axes = history_feasible is not None and len(history_feasible) == n
    ax.set_xlabel("Iteration", fontsize=fontsize)
    if not use_two_axes:
        ax.set_ylabel("Global Best Fitness", fontsize=fontsize)
    ax.set_title(f"PSO Convergence - Scenario {scenario_id}", fontsize=fontsize + 2)
    ax.tick_params(axis="x", labelsize=fontsize - 2)
    if not use_two_axes:
        ax.tick_params(axis="y", labelsize=fontsize - 2)
    ax.grid(True, alpha=0.3)

    if n > 1 and not use_two_axes:
        valid = [v for v in history if v != float("inf") and v != float("-inf")]
        if valid:
            min_val = min(valid)
            max_val = max(valid)
            if min_val > 0 and max_val / min_val > 100:
                ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
