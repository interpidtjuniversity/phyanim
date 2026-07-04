"""LLM-friendly trajectory data wrapper around PhysicsContext.

This module exposes the solved physics state in a clean, well-documented API
that generated code (code/hybrid render modes) can consume directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from phyanim.core.animation import PhysicsAnimation
from phyanim.core.context import PhysicsContext
from phyanim.utils.util import is_numeric


@dataclass
class SegmentInfo:
    """Metadata for a single solved physics segment."""

    segment_id: str
    start_time: float
    end_time: float
    triggered_event: str | None = None
    object_ids: list[str] = field(default_factory=list)


@dataclass
class ObjectInfo:
    """Metadata for a physical object in the solved animation."""

    object_id: str
    state_variables: list[str]
    parameters: dict[str, float]
    cartesian_position: list[tuple[str, str]] = field(default_factory=list)


class TrajectoryData:
    """Solved physics trajectory data for code/hybrid render modes.

    Created via :func:`solve_animation`.  All time parameters are in
    *physics* time (the real simulation clock), not render time.

    Attributes
    ----------
    times: list[float]
        All sampled time points across the entire animation.
    total_time: float
        Total physics duration in seconds.
    segments: list[SegmentInfo]
        Per-segment metadata (id, start/end time, triggered event, objects).
    objects: dict[str, ObjectInfo]
        Per-object metadata (state variables, parameters, cartesian position).

    Examples
    --------
    ::

        trajectory = solve_animation(animation)
        x = trajectory.value_at("ball_x", 2.5)
        pos = trajectory.eval_position("ball", 2.5)
        times, values = trajectory.sample("ball_x", dt=0.01)
    """

    def __init__(
        self,
        ctx: PhysicsContext,
        animation: PhysicsAnimation,
    ) -> None:
        self._ctx = ctx
        self._animation = animation

        # Time axis.
        self.times: list[float] = list(ctx.times)
        self.total_time: float = ctx.total_time

        # Segment metadata.
        self.segments: list[SegmentInfo] = []
        for tra in ctx.trajectories:
            self.segments.append(
                SegmentInfo(
                    segment_id=tra.segment_id,
                    start_time=tra.times[0],
                    end_time=tra.times[-1],
                    triggered_event=tra.triggered_event,
                    object_ids=list(tra.object_ids),
                )
            )

        # Object metadata.
        self.objects: dict[str, ObjectInfo] = {}
        for obj_id, obj in ctx.objects.items():
            self.objects[obj_id] = ObjectInfo(
                object_id=obj_id,
                state_variables=list(obj.state_variables.keys()),
                parameters=obj.parameter_values(),
                cartesian_position=obj.cartesian_position_variables(),
            )

        # Segment-id → index mapping for fast lookups.
        self._seg_index_map: dict[str, int] = {
            seg.segment_id: i for i, seg in enumerate(self.segments)
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_context(
        cls, ctx: PhysicsContext, animation: PhysicsAnimation
    ) -> "TrajectoryData":
        """Build from a solved PhysicsContext."""
        return cls(ctx, animation)

    # ------------------------------------------------------------------
    # Core value queries
    # ------------------------------------------------------------------

    def value_at(self, name: str, t: float) -> float:
        """Return the value of *name* (state or derived variable) at physics time *t*.

        ``name="t"`` returns *t* itself.
        """
        return self._ctx.value_at_time(name, t)

    def eval_expr(self, expr: str, t: float) -> float:
        """Evaluate a SymPy expression at physics time *t*.

        The expression may reference any state variable, derived variable,
        parameter, ``t`` (absolute time), ``t_start``/``t_end`` (segment
        bounds).
        """
        if is_numeric(expr):
            return float(expr)
        return self._ctx.eval_expr(expr, "float", t)

    def eval_expr_vector(self, exprs: list[str] | tuple[str, ...], t: float) -> np.ndarray:
        """Evaluate a vector SymPy expression at physics time *t*."""
        return np.array([self.eval_expr(expr, t) for expr in exprs])

    def eval_expr_bool(self, expr: str, t: float) -> bool:
        """Evaluate a boolean SymPy expression at physics time *t*."""
        return self._ctx.eval_expr(expr, "bool", t)

    def eval_position(self, obj_id: str, t: float) -> list[tuple[float, float]]:
        """Return the cartesian position(s) of *obj_id* at physics time *t*.

        For single-point objects returns ``[(x, y)]``; for two-endpoint
        objects (e.g. springs) returns ``[(x1, y1), (x2, y2)]``.
        """
        return self._ctx.eval_position(obj_id, t)

    # ------------------------------------------------------------------
    # Segment queries
    # ------------------------------------------------------------------

    def find_segment(self, t: float) -> str:
        """Return the segment_id that owns physics time *t*."""
        idx = self._ctx.find_trajectory_index(t)
        return self._ctx.tra_map[idx]

    def segment_times(self, segment_id: str) -> tuple[float, float]:
        """Return ``(start_time, end_time)`` for *segment_id*."""
        idx = self._seg_index_map.get(segment_id)
        if idx is None:
            raise ValueError(f"Unknown segment_id: {segment_id}")
        seg = self.segments[idx]
        return seg.start_time, seg.end_time

    def segment_index(self, t: float) -> int:
        """Return the trajectory index that owns physics time *t*."""
        return self._ctx.find_trajectory_index(t)

    # ------------------------------------------------------------------
    # Registered event queries
    # ------------------------------------------------------------------

    def all_events(self) -> dict[str, list[float]]:
        """Return all registered events and their trigger times.

        Returns
        -------
        dict[str, list[float]]
            Maps ``event_id`` to a list of physics-time seconds at which
            the event's expression crossed zero.

        Example::

            trajectory = solve_animation(animation)
            for event_id, times in trajectory.all_events().items():
                print(f"{event_id}: triggered at {times}")
        """
        return dict(self._ctx.event_trigger_map)

    def event_trigger_times(self, event_id: str) -> list[float]:
        """Return the trigger times for a specific registered event.

        Parameters
        ----------
        event_id:
            The event ID passed to ``animation.register_event()``.

        Returns
        -------
        list[float]
            Sorted list of physics-time seconds at which the event
            was triggered.  Empty if the event never triggered or
            was not registered.
        """
        return list(self._ctx.event_trigger_map.get(event_id, []))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self, name: str, dt: float | None = None
    ) -> tuple[list[float], list[float]]:
        """Sample *name* across the entire animation timeline.

        Parameters
        ----------
        name:
            State or derived variable name (or ``"t"``).
        dt:
            Sampling interval in seconds.  ``None`` uses the animation's
            ``sample_dt``.

        Returns
        -------
        tuple[list[float], list[float]]
            ``(times, values)`` aligned to the sampling grid.
        """
        step = dt if dt is not None else self._animation.sample_dt
        times = np.arange(0.0, self.total_time, step).tolist()
        if not times or times[-1] < self.total_time:
            times.append(self.total_time)
        values = [self.value_at(name, float(t)) for t in times]
        return times, values

    def sample_segment(
        self,
        name: str,
        segment_id: str,
        dt: float | None = None,
    ) -> tuple[list[float], list[float]]:
        """Sample *name* within a single segment."""
        t0, t1 = self.segment_times(segment_id)
        step = dt if dt is not None else self._animation.sample_dt
        times = np.arange(t0, t1, step).tolist()
        if not times or times[-1] < t1:
            times.append(t1)
        values = [self.value_at(name, float(t)) for t in times]
        return times, values


def solve_animation(animation: PhysicsAnimation) -> TrajectoryData:
    """Solve *animation* and return a :class:`TrajectoryData`.

    The animation is solved in-place (``animation.solved`` becomes ``True``).
    If already solved, the existing ``physics_ctx`` is reused.
    """
    if not animation.solved:
        animation.solve()
    assert animation.physics_ctx is not None
    return TrajectoryData.from_context(animation.physics_ctx, animation)
