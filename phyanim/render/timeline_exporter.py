from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

from phyanim.core.animation import PhysicsAnimation


class TimelineExporter:
    """Renderer-neutral exporter for inspecting solved physics timelines."""

    def to_dict(self, animation: PhysicsAnimation) -> dict[str, object]:
        return {
            "objects": {
                object_id: {
                    "parameters": {
                        name: asdict(parameter) for name, parameter in obj.parameters.items()
                    },
                    "state_variables": {
                        name: asdict(variable) for name, variable in obj.state_variables.items()
                    },
                    "cartesian_position": obj.cartesian_position_variables(),
                    "has_mobject": obj.mobject is not None,
                }
                for object_id, obj in animation.objects.items()
            },
            "keyframes": [asdict(keyframe) for keyframe in animation.keyframes],
            "trajectories": [asdict(trajectory) for trajectory in animation.trajectories],
        }

    def write_json(self, animation: PhysicsAnimation, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(animation), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def plot_variables(
        self,
        animation: PhysicsAnimation,
        *,
        variables: list[str] | None = None,
        objects: list[str] | None = None,
        title: str | None = None,
        output_path: str | Path | None = None,
        show: bool = True,
    ) -> plt.Figure:
        """Plot all state variables over time, merging segments.

        Parameters
        ----------
        animation : PhysicsAnimation
            The solved physics animation.
        variables : list[str] | None
            Specific variable names to plot (e.g., ['x', 'v']). None means all.
        objects : list[str] | None
            Specific object IDs to plot. None means all.
        title : str | None
            Plot title.
        output_path : str | Path | None
            If provided, save the figure to this path.
        show : bool
            Whether to call plt.show().

        Returns
        -------
        plt.Figure
            The matplotlib figure object.
        """
        # Collect all (time, value) pairs for each variable, merged across segments
        merged_data: dict[str, tuple[list[float], list[float]]] = defaultdict(
            lambda: ([], [])
        )

        for trajectory_id, trajectory in enumerate(animation.trajectories):
            segment_start_time = trajectory.times[0] if trajectory.times else 0.0
            obj_states = trajectory.object_states or {}

            for var_name, values in trajectory.states.items():
                # Filter by objects if specified
                if objects:
                    # Check if this variable belongs to any of the specified objects
                    found = False
                    for obj_id in objects:
                        if obj_id in obj_states and var_name in obj_states[obj_id]:
                            found = True
                            break
                    if not found:
                        continue

                # Filter by variables if specified
                if variables and var_name not in variables:
                    continue

                times = trajectory.times
                next_idx = 0

                if trajectory_id > 0:
                    for idx, t in enumerate(times):
                        if t > merged_data[var_name][0][-1]:
                            next_idx = idx
                            break

                merged_data[var_name][0].extend(times[next_idx:])
                merged_data[var_name][1].extend(values[next_idx:])

        # Sort by time for each variable
        for var_name in merged_data:
            times, values = merged_data[var_name]
            sorted_pairs = sorted(zip(times, values))
            merged_data[var_name] = ([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs])

        # Create subplot grid
        n_vars = len(merged_data)
        if n_vars == 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "No variables to plot", ha="center", va="center")
            ax.set_title("Empty")
            return fig

        fig, axes = plt.subplots(n_vars, 1, figsize=(12, 4 * n_vars), squeeze=False)
        if title:
            fig.suptitle(title, fontsize=14, y=1.02)

        for ax, (var_name, (times, values)) in zip(axes.flat, merged_data.items()):
            ax.plot(times, values, linewidth=1.5, marker="o", markersize=2)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(var_name)
            ax.set_title(f"Variable: {var_name}")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig
