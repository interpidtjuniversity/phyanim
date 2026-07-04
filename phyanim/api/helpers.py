"""Rendering helper functions for code/hybrid render modes.

These utilities wrap common manim patterns so that LLM-generated code
can be concise and readable.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from manim import Line, Mobject, ValueTracker

from phyanim.api.trajectory import TrajectoryData

def _to_3d(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    if v.size == 2:
        return np.array([v[0], v[1], 0.0])
    return v[:3]

def create_tracker(initial: float = 0.0) -> ValueTracker:
    """Create a manim ValueTracker at *initial* value."""
    return ValueTracker(initial)


def attach_position_updater(
    mob: Mobject,
    trajectory: TrajectoryData,
    obj_id: str,
    tracker: ValueTracker,
) -> None:
    """Attach a per-frame updater that moves *mob* to follow *obj_id*.

    For multi-point objects (e.g. springs) the mobject must implement
    ``point_change_callbacks()`` returning one callback per cartesian
    position pair — the framework calls each callback with ``(x, y)``.

    Parameters
    ----------
    mob:
        The manim Mobject to move.
    trajectory:
        Solved trajectory data from :func:`solve_animation`.
    obj_id:
        Object ID whose position drives *mob*.
    tracker:
        ValueTracker whose value represents the current render time.
    """
    obj = trajectory._ctx.objects.get(obj_id)
    if obj is None:
        raise ValueError(f"Unknown object id: {obj_id}")

    cartesian_positions = obj.cartesian_position_variables()
    callbacks = None

    if len(cartesian_positions) > 1:
        if hasattr(mob, "point_change_callbacks"):
            callbacks = mob.point_change_callbacks()
            if len(callbacks) != len(cartesian_positions):
                raise ValueError(
                    f"Object '{obj_id}' has {len(cartesian_positions)} "
                    f"cartesian positions, but its mobject provides "
                    f"{len(callbacks)} callbacks."
                )
    elif len(cartesian_positions) == 1:
        if hasattr(mob, "point_change_callbacks"):
            callbacks = mob.point_change_callbacks()

    def updater(m: Mobject) -> None:
        positions = trajectory.eval_position(obj_id, tracker.get_value())

        if callbacks is not None:
            for callback, pos in zip(callbacks, positions):
                x, y = pos
                callback(x, y)
        elif len(positions) == 1:
            x, y = positions[0]
            m.move_to(np.array([x, y, 0.0]))

    mob.add_updater(updater)

    return updater

def attach_expr_updater(
    mob: Mobject,
    trajectory: TrajectoryData,
    tracker: ValueTracker,
    expr: str,
    apply_fn: Callable[[Mobject, float], None],
) -> None:
    """Attach a generic updater that evaluates *expr* each frame.

    Each frame, the expression is evaluated at the physics time corresponding
    to the tracker's value, and *apply_fn* is called with ``(mob, value)``.

    Parameters
    ----------
    mob:
        The manim Mobject to update.
    trajectory:
        Solved trajectory data from :func:`solve_animation`.
    expr:
        SymPy expression string (e.g. ``"0.5 + 0.5*Abs(vy)/(Abs(vy)+5)"``).
        May reference any state variable, derived variable, parameter, or ``t``.
    tracker:
        ValueTracker whose value represents the current render time.
    apply_fn:
        Callback ``(mob, value) → None`` that applies the evaluated value
        to the mobject (e.g. ``lambda m, v: m.set_opacity(v)``).

    Example::

        attach_expr_updater(
            ball, trajectory, "0.5 + 0.5*Abs(vy)/(Abs(vy)+5)",
            tracker, lambda m, v: m.set_opacity(v),
        )
    """

    def updater(m: Mobject) -> None:
        val = trajectory.eval_expr(expr, tracker.get_value())
        apply_fn(m, val)

    mob.add_updater(updater)

    return updater


def interpolate_trajectory(
    times: list[float] | np.ndarray,
    values: list[float] | np.ndarray,
    t: float,
) -> float:
    """Linearly interpolate a sampled trajectory at time *t*.

    Equivalent to numpy's ``np.interp`` but returns a plain float.
    Values outside the range are clamped to the first/last sample.

    Parameters
    ----------
    times:
        Sample time points (must be sorted ascending).
    values:
        Sample values at each time point (same length as *times*).
    t:
        Query time.
    """
    times_arr = np.asarray(times, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    return float(np.interp(t, times_arr, values_arr))

# Attach a line to animation with a trajectory.
def attach_line(line: Line, trajectory: TrajectoryData, tracker: ValueTracker, start_point_names: list[str] | tuple[str, ...] | None = None, end_point_names: list[str] | tuple[str, ...] | None = None, dir_vector_names: list[str] | tuple[str, ...] | None = None):

    def update_line(m: Mobject) -> None:
        dir_vector = None

        if dir_vector_names is not None:
            if start_point_names is None:
                raise ValueError("start_point_names must be provided if dir_vector_names is provided")
            start_point = _to_3d(trajectory.eval_expr_vector(start_point_names, tracker.get_value()))
            dir_vector = _to_3d(trajectory.eval_expr_vector(dir_vector_names, tracker.get_value()))
        else:
            if start_point_names is None or end_point_names is None:
                raise ValueError("start_point_names and end_point_names must be provided if dir_vector_names is not provided")
            start_point = _to_3d(trajectory.eval_expr_vector(start_point_names, tracker.get_value()))
            end_point = _to_3d(trajectory.eval_expr_vector(end_point_names, tracker.get_value()))
            dir_vector = end_point - start_point

        if np.linalg.norm(dir_vector) < 1e-9:
            m.set_opacity(0.0)
        else:
            m.set_opacity(1.0)
            end_point = start_point + dir_vector
            m.put_start_and_end_on(start_point, end_point)

    line.add_updater(update_line)
    return update_line

def attach_mobject(mob: Mobject, trajectory: TrajectoryData, tracker: ValueTracker, position_names : list[str] | tuple[str, ...]):
    def updater(m: Mobject) -> None:
        positions = _to_3d(trajectory.eval_expr_vector(position_names, tracker.get_value()))
        m.move_to(positions)
    
    mob.add_updater(updater)
    return updater

# 当事件发生后开始执行 updater
def attach_mobject_with_event(
    mob: Mobject,
    trajectory: TrajectoryData,
    tracker: ValueTracker,
    event_id: str,
    update_fn: Callable[[Mobject, TrajectoryData, float], None],
    event_index: int = 0,
    fade_in: float = 0.0,
) -> Callable:
    """Attach an updater that only starts running after a registered event triggers.

    The mobject remains in its initial state until the specified event
    fires. After the event, the updater runs every frame just like a
    normal manim updater.

    Parameters
    ----------
    mob:
        The manim Mobject to update.
    trajectory:
        Solved trajectory data containing registered events.
    tracker:
        ValueTracker whose value is the current render time.
    event_id:
        The event ID registered via ``animation.register_event(...)``.
    update_fn:
        Callback ``(mob, trajectory, physics_t) → None`` called each frame
        after the event triggers. Use this to update position, color, etc.
        建议用 def 定义而非 lambda，因为 def 支持多行逻辑。

        Example::

            def update_arrow(mob, traj, t):
                x = traj.value_at("x", t)
                vx = traj.value_at("vx", t)
                start = np.array([x, 0, 0])
                end = start + np.array([vx * 0.2, 0, 0])
                mob.put_start_and_end_on(start, end)

    event_index:
        Which occurrence of the event to use (0 = first trigger, 1 = second, ...).
        Defaults to 0 (first trigger).
    fade_in:
        If > 0, fade in the mobject over this many seconds starting at the
        event trigger time. Defaults to 0 (instant appear).

    Returns
    -------
    Callable
        The updater function. Pass this to ``detach_mobject_with_event``
        to remove it later.

    Example::

        # Show a force arrow after collision
        def update_arrow(mob, traj, t):
            x = traj.value_at("x1", t)
            mob.put_start_and_end_on(
                np.array([x, 0, 0]),
                np.array([x + 2, 0, 0]),
            )

        arrow = Arrow(color=RED)
        updater = attach_mobject_with_event(
            arrow, trajectory, tracker, "collision", update_arrow
        )
        self.add(arrow)

        # Later, stop updating after "separation" event
        detach_mobject_with_event(arrow, trajectory, tracker, "separation", updater)
    """
    trigger_times = trajectory.event_trigger_times(event_id)
    if event_index < len(trigger_times):
        trigger_t = trigger_times[event_index]
    else:
        # Event never triggered — updater will never run.
        trigger_t = float("inf")

    def updater(m: Mobject) -> None:
        t = tracker.get_value()
        if t < trigger_t:
            m.set_opacity(0.0)
            return

        # After event: run the update function
        if fade_in > 0:
            elapsed = t - trigger_t
            opacity = min(1.0, elapsed / fade_in)
            m.set_opacity(opacity)
        else:
            m.set_opacity(1.0)

        update_fn(m, trajectory, t)

    mob.add_updater(updater)
    return updater


# 当事件发生后移除 updater（冻结 mobject 在事件发生时的状态）
def detach_mobject_with_event(
    mob: Mobject,
    trajectory: TrajectoryData,
    tracker: ValueTracker,
    event_id: str,
    updater: Callable,
    event_index: int = 0,
    fade_out: float = 0.0,
) -> None:
    """Remove an updater after a registered event triggers.

    The mobject freezes at the state it was in when the event fired.
    Optionally fade out over a few seconds.

    Parameters
    ----------
    mob:
        The manim Mobject whose updater to remove.
    trajectory:
        Solved trajectory data containing registered events.
    tracker:
        ValueTracker whose value is the current render time.
    event_id:
        The event ID that triggers the detach.
    updater:
        The updater function returned by ``attach_mobject_with_event``
        or ``attach_position_updater`` etc.
    event_index:
        Which occurrence of the event to use (0 = first trigger).
    fade_out:
        If > 0, fade out the mobject over this many seconds starting at
        the event trigger time, then remove the updater. Defaults to 0
        (instant freeze).
    """
    trigger_times = trajectory.event_trigger_times(event_id)
    if event_index < len(trigger_times):
        trigger_t = trigger_times[event_index]
    else:
        # Event never triggered — don't modify updater.
        return

    if fade_out <= 0:
        # 即时冻结：事件触发时直接移除 updater
        def stopper(m: Mobject) -> None:
            if tracker.get_value() >= trigger_t:
                mob.remove_updater(updater)
                mob.remove_updater(stopper)
        mob.add_updater(stopper)
    else:
        # 渐进淡出：事件后逐渐降低透明度，完成后移除 updater
        def fader(m: Mobject) -> None:
            t = tracker.get_value()
            if t >= trigger_t:
                elapsed = t - trigger_t
                if elapsed >= fade_out:
                    m.set_opacity(0.0)
                    mob.remove_updater(updater)
                    mob.remove_updater(fader)
                else:
                    m.set_opacity(1.0 - elapsed / fade_out)
        mob.add_updater(fader)