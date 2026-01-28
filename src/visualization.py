from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .models import Point, Scenario


def plot_scenario(
    scenario: Scenario,
    path: Optional[List[Point]] = None,
    traces: Optional[List[Tuple[Point, Point]]] = None,
    save_to: Optional[str | Path] = None,
    show: bool = True,
) -> Tuple[Figure, Axes]:
    """Plot the environment, obstacles, start/goal, and optionally a path."""
    # Extract scenario data
    xmax, ymax = scenario.xmax, scenario.ymax
    start1, goal1 = scenario.start1, scenario.goal1
    obstacles = scenario.obstacles

    fontsize = 16
    
    # Initialize figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Add padding so points at boundaries are visible
    padding = max(xmax, ymax) * 0.02
    ax.set_xlim(-padding, xmax + padding)
    ax.set_ylim(-padding, ymax + padding)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=fontsize)

    # Draw environment border
    border = patches.Rectangle(
        (0, 0), xmax, ymax,
        linewidth=2,
        edgecolor="black",
        facecolor="none",
    )
    ax.add_patch(border)

    # Draw obstacles
    for obs in obstacles:
        rect = patches.Rectangle(
            (obs.xo, obs.yo), obs.lx, obs.ly,
            linewidth=1,
            edgecolor="darkgray",
            facecolor="gray",
            alpha=0.7,
        )
        ax.add_patch(rect)

    # Draw traces (if provided)
    if traces:
        for p1, p2 in traces:
            ax.plot(
                [p1.x, p2.x], [p1.y, p2.y],
                color="lightblue",
                linewidth=0.5,
                alpha=0.5,
            )

    # Draw path (if provided)
    if path and len(path) >= 2:
        xs = [p.x for p in path]
        ys = [p.y for p in path]
        ax.plot(xs, ys, color="blue", linewidth=2, label="Path", zorder=5)

    # Draw start and goal
    ax.plot(
        start1.x, start1.y,
        marker="o",
        markersize=12,
        color="green",
        label="Start",
        zorder=10,
    )
    ax.plot(
        goal1.x, goal1.y,
        marker="*",
        markersize=15,
        color="red",
        label="Goal",
        zorder=10,
    )

    # Finalize
    ax.set_xlabel("X", fontsize=fontsize)
    ax.set_ylabel("Y", fontsize=fontsize)
    ax.set_title(f"Path Planning ({xmax:.0f} x {ymax:.0f})", fontsize=fontsize + 4)
    ax.legend(loc="best", fontsize=fontsize)


    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax
