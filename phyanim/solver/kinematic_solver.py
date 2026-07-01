"""Kinematic solver: evaluates closed-form expressions or sample arrays.

Unlike :class:`ScipySegmentSolver`, this solver does **not** integrate an
ODE.  Instead it directly evaluates SymPy expressions (or interpolates
sample arrays) at each sample time, producing a :class:`Trajectory` that
is structurally identical to the ODE solver output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from phyanim.core.events import PhysicsEvent
from phyanim.core.expressions import SympyExpressionCompiler
from phyanim.core.keyframe import PhysicsKeyFrame
from phyanim.core.segment import PhysicsSegment
from phyanim.core.solution import SegmentResult
from phyanim.core.trajectory import Trajectory

from phyanim.solver.solver import Solver


@dataclass
class KinematicSegmentSolver(Solver):
    """Solver for ``mode=kinematic`` / ``mode=sampled`` segments.

    No ODE integration is performed.  State values are obtained either by
    evaluating closed-form SymPy expressions or by linearly interpolating
    pre-computed sample arrays.
    """

    sample_dt: float | None = None

    def solve(
        self,
        segment: PhysicsSegment,
        start_keyframe: PhysicsKeyFrame,
        parameters: dict[str, float],
        end_event: PhysicsEvent,
    ) -> SegmentResult:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("KinematicSegmentSolver requires numpy.") from exc

        t0 = start_keyframe.time
        planned_t1 = t0 + segment.duration

        # Generate sample times.
        if self.sample_dt is not None:
            n_samples = int(round((planned_t1 - t0) / self.sample_dt)) + 1
            times = np.linspace(t0, planned_t1, n_samples)
        else:
            times = np.array([t0, planned_t1])

        # --- Evaluate states ---
        if segment.kinematic_expressions is not None:
            state_history = self._eval_expressions(
                segment, parameters, times, t0
            )
        elif segment.kinematic_samples is not None:
            state_history = self._interp_samples(
                segment, times
            )
        else:
            raise ValueError(
                f"Segment '{segment.segment_id}' has no kinematic data "
                "(neither kinematic_expressions nor kinematic_samples)."
            )

        # --- Evaluate derived quantities ---
        derived_history = self._eval_derived(segment, parameters, times, state_history)

        # --- Event detection (zero-crossing on sampled values) ---
        actual_t1, triggered_event_name = self._detect_event(
            end_event, segment, parameters, times, state_history, t0, planned_t1
        )

        # Trim to actual end time if event triggered early.
        if actual_t1 < planned_t1:
            mask = times <= actual_t1
            if not mask[-1]:
                # Ensure the event time is included.
                times = np.append(times[mask], actual_t1)
                # Re-evaluate at the appended point.
                if segment.kinematic_expressions is not None:
                    extra = self._eval_expressions(
                        segment, parameters, np.array([actual_t1]), t0
                    )
                else:
                    extra = self._interp_samples(segment, np.array([actual_t1]))
                for name in state_history:
                    state_history[name] = state_history[name][mask.tolist()].tolist() + extra[name]
                    if name in derived_history:
                        derived_history[name] = derived_history[name][mask.tolist()].tolist() + self._eval_derived_single(segment, parameters, actual_t1, state_history, name)
            else:
                times = times[mask]
                for name in state_history:
                    state_history[name] = state_history[name][mask].tolist()
                for name in derived_history:
                    derived_history[name] = derived_history[name][mask].tolist()

        # Build object-state history.
        object_state_history = self._split_state_history_by_object(segment, state_history)

        # End state.
        end_state = {name: state_history[name][-1] for name in segment.state_vector}
        end_object_states = segment.split_state_by_object(end_state)

        trajectory = Trajectory(
            segment_id=segment.segment_id,
            object_ids=list(segment.object_ids),
            times=times.astype(float).tolist(),
            states=state_history,
            object_states=object_state_history,
            derived=derived_history,
            triggered_event=triggered_event_name,
        )

        end_keyframe = PhysicsKeyFrame(
            time=actual_t1,
            object_states=end_object_states,
            parameters={
                name: value
                for name, value in parameters.items()
                if name not in ["t_start", "t_end"]
            },
            segment_id=segment.segment_id,
            event_name=triggered_event_name,
        )

        return SegmentResult(
            trajectory=trajectory,
            solution=None,
            start_keyframe=start_keyframe,
            end_keyframe=end_keyframe,
            triggered_event=triggered_event_name,
        )

    # ------------------------------------------------------------------
    # Expression evaluation
    # ------------------------------------------------------------------

    def _eval_expressions(
        self,
        segment: PhysicsSegment,
        parameters: dict[str, float],
        times: Any,
        t0: float,
    ) -> dict[str, list[float]]:
        """Evaluate closed-form SymPy expressions at each sample time."""
        assert segment.kinematic_expressions is not None
        compiler = SympyExpressionCompiler()
        symbol_names = set(segment.state_vector) | set(parameters) | {"t"}
        compiled = compiler.compile_mapping(
            segment.kinematic_expressions, symbol_names=symbol_names
        )

        state_history: dict[str, list[float]] = {}
        for name in segment.state_vector:
            expr = compiled[name]
            values = []
            for t in times:
                # State dict is empty for kinematic mode — expressions depend
                # only on parameters and t.
                val = expr.evaluate({}, parameters, float(t))
                values.append(float(val))
            state_history[name] = values
        return state_history

    # ------------------------------------------------------------------
    # Sample interpolation
    # ------------------------------------------------------------------

    def _interp_samples(
        self,
        segment: PhysicsSegment,
        times: Any,
    ) -> dict[str, list[float]]:
        """Linearly interpolate pre-computed sample arrays."""
        import numpy as np

        assert segment.kinematic_samples is not None
        state_history: dict[str, list[float]] = {}
        for name in segment.state_vector:
            sample = segment.kinematic_samples[name]
            sample_times = np.asarray(sample["times"], dtype=float)
            sample_values = np.asarray(sample["values"], dtype=float)
            interp = np.interp(times, sample_times, sample_values)
            state_history[name] = interp.tolist()
        return state_history

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def _eval_derived(
        self,
        segment: PhysicsSegment,
        parameters: dict[str, float],
        times: Any,
        state_history: dict[str, list[float]],
    ) -> dict[str, list[float]]:
        if not segment.derived_equations:
            return {}

        compiler = SympyExpressionCompiler()
        symbol_names = set(segment.state_vector) | set(parameters) | {"t"}
        compiled = compiler.compile_mapping(
            segment.derived_equations, symbol_names=symbol_names
        )

        derived_history: dict[str, list[float]] = {}
        for name, expr in compiled.items():
            values = []
            for idx, t in enumerate(times):
                state_at_t = {
                    s: state_history[s][idx] for s in segment.state_vector
                }
                val = expr.evaluate(state_at_t, parameters, float(t))
                values.append(float(val))
            derived_history[name] = values
        return derived_history

    def _eval_derived_single(
        self,
        segment: PhysicsSegment,
        parameters: dict[str, float],
        t: float,
        state_history: dict[str, list[float]],
        name: str,
    ) -> list[float]:
        """Evaluate a single derived variable at time *t*."""
        if not segment.derived_equations or name not in segment.derived_equations:
            return [0.0]
        compiler = SympyExpressionCompiler()
        symbol_names = set(segment.state_vector) | set(parameters) | {"t"}
        compiled = compiler.compile_mapping(
            {name: segment.derived_equations[name]}, symbol_names=symbol_names
        )
        expr = compiled[name]
        state_at_t = {s: state_history[s][-1] for s in segment.state_vector}
        val = expr.evaluate(state_at_t, parameters, float(t))
        return [float(val)]

    # ------------------------------------------------------------------
    # Event detection
    # ------------------------------------------------------------------

    def _detect_event(
        self,
        end_event: PhysicsEvent,
        segment: PhysicsSegment,
        parameters: dict[str, float],
        times: Any,
        state_history: dict[str, list[float]],
        t0: float,
        planned_t1: float,
    ) -> tuple[float, str]:
        """Detect zero-crossing of the end event on sampled values.

        For kinematic mode, event detection is done by scanning the
        pre-computed sample array for sign changes.  If no event triggers,
        the planned end time is used (countdown-style events always work).
        """
        import numpy as np

        # Countdown events: just use planned_t1.
        condition = end_event.condition
        if condition.expression is None:
            return planned_t1, end_event.name

        # Build event function from the expression.
        # Event expressions can reference state variables, derived variables, parameters, and t.
        compiler = SympyExpressionCompiler()
        symbol_names = set(segment.state_vector) | set(parameters) | set(segment.derived_equations) | {"t"}
        compiled_expr = compiler.compile(
            condition.expression, symbol_names=symbol_names
        )

        # Pre-compute derived quantities at each sample time.
        derived_history = self._eval_derived(segment, parameters, times, state_history)

        # Evaluate event function at each sample time.
        event_values = []
        for idx, t in enumerate(times):
            state_at_t = {
                s: state_history[s][idx] for s in segment.state_vector
            }
            # Merge derived values into state for event expression resolution.
            for dname, dvals in derived_history.items():
                state_at_t[dname] = dvals[idx]
            val = compiled_expr.evaluate(state_at_t, parameters, float(t))
            event_values.append(float(val))

        event_values = np.array(event_values)
        direction = condition.direction

        # Scan for zero-crossings.
        for i in range(len(event_values) - 1):
            f0 = event_values[i]
            f1 = event_values[i + 1]

            if direction == 1:  # negative to positive
                crossed = f0 <= 0 < f1 or (f0 < 0 and f1 >= 0)
            elif direction == -1:  # positive to negative
                crossed = f0 >= 0 > f1 or (f0 > 0 and f1 <= 0)
            else:  # any direction
                crossed = (f0 < 0 <= f1) or (f0 > 0 >= f1) or (f0 == 0 and f1 != 0)

            if crossed and times[i] > t0 + 1e-12:
                # Linear interpolation for precise crossing time.
                if f1 != f0:
                    alpha = -f0 / (f1 - f0)
                    event_t = times[i] + alpha * (times[i + 1] - times[i])
                else:
                    event_t = times[i]
                return float(event_t), end_event.name

        # No event triggered — use planned end time.
        return planned_t1, end_event.name
