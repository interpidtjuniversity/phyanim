"""Rendering helper functions for code/hybrid render modes.

These utilities wrap common manim patterns so that LLM-generated code
can be concise and readable.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from manim import Mobject, ValueTracker

from phyanim.api.trajectory import TrajectoryData


def create_tracker(initial: float = 0.0) -> ValueTracker:
    """Create a manim ValueTracker at *initial* value."""
    return ValueTracker(initial)


def attach_position_updater(
    mob: Mobject,
    trajectory: TrajectoryData,
    obj_id: str,
    tracker: ValueTracker,
    render_to_physics: Callable[[float], float] | None = None,
) -> None:
    """Attach a per-frame updater that moves *mob* to follow *obj_id*.

    The updater reads ``tracker.get_value()`` as render time, converts to
    physics time (via *render_to_physics* if provided, else identity), and
    moves the mobject to the object's cartesian position.

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
    render_to_physics:
        Optional callable mapping render time → physics time.  Use this
        when you have time-warping (slow/freeze).  If ``None``, render
        time equals physics time (1:1 mapping).
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
        render_t = tracker.get_value()
        physics_t = render_to_physics(render_t) if render_to_physics else render_t
        positions = trajectory.eval_position(obj_id, physics_t)

        if callbacks is not None:
            for callback, pos in zip(callbacks, positions):
                x, y = pos
                callback(x, y)
        elif len(positions) == 1:
            x, y = positions[0]
            m.move_to(np.array([x, y, 0.0]))

    mob.add_updater(updater)


def attach_expr_updater(
    mob: Mobject,
    trajectory: TrajectoryData,
    expr: str,
    tracker: ValueTracker,
    apply_fn: Callable[[Mobject, float], None],
    render_to_physics: Callable[[float], float] | None = None,
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
    render_to_physics:
        Optional render→physics time mapping.  ``None`` means 1:1.

    Example::

        attach_expr_updater(
            ball, trajectory, "0.5 + 0.5*Abs(vy)/(Abs(vy)+5)",
            tracker, lambda m, v: m.set_opacity(v),
        )
    """

    def updater(m: Mobject) -> None:
        render_t = tracker.get_value()
        physics_t = render_to_physics(render_t) if render_to_physics else render_t
        val = trajectory.eval_expr(expr, physics_t)
        apply_fn(m, val)

    mob.add_updater(updater)


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


def create_bound_mobject(
    mob: Mobject,
    trajectory: TrajectoryData,
    obj_id: str,
    tracker: ValueTracker,
    *,
    follow_position: bool = True,
    render_to_physics: Callable[[float], float] | None = None,
) -> Mobject:
    """Create a custom mobject and bind it to a physical object's state.

    This is the primary entry point for hybrid-mode custom rendering.
    The LLM can create *any* manim mobject (Circle, VGroup, custom VMobject,
    imported geometry, etc.) and bind it to physics state with one call.

    By default the mobject follows the object's cartesian position.
    Additional visual property bindings can be chained via
    :func:`attach_expr_updater`.

    Parameters
    ----------
    mob:
        Any manim Mobject to bind (created by the caller).
    trajectory:
        Solved trajectory data from :func:`solve_animation`.
    obj_id:
        Object ID whose state drives *mob*.
    tracker:
        ValueTracker representing the current render time.
    follow_position:
        If True (default), attach a position updater so *mob* follows
        the object's cartesian_position.  Set to False if you want to
        control position manually (e.g. a static label that only changes
        color based on state).
    render_to_physics:
        Optional render→physics time mapping.

    Returns
    -------
    Mobject
        The same *mob* (for chaining), with updaters attached.

    Example::

        # Create a custom glowing circle and bind it to "ball"
        glow = Circle(radius=0.3, color=YELLOW).set_opacity(0.3)
        create_bound_mobject(glow, trajectory, "ball", tracker)
        self.add(glow)

        # Chain with expression binding for dynamic opacity
        attach_expr_updater(glow, trajectory, "Abs(vy)/10", tracker,
            lambda m, v: m.set_opacity(min(0.8, v)))
    """
    if follow_position:
        attach_position_updater(mob, trajectory, obj_id, tracker, render_to_physics)
    return mob
