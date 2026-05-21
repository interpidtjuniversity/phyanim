from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from phyanim.core.keyframe import PhysicsKeyFrame
from phyanim.core.trajectory import InterpolatedStateFunction, Trajectory


@dataclass(frozen=True)
class StateFunction:
    """Named pure function that maps an absolute segment time to a scalar value."""

    name: str
    domain: tuple[float, float]
    function: Callable[[float], float]
    clamp: bool = False

    def __call__(self, time: float) -> float:
        t0, t1 = self.domain
        query_time = time
        if query_time < t0:
            if not self.clamp:
                raise ValueError(f"time {time} is before domain start {t0}.")
            query_time = t0
        elif query_time > t1:
            if not self.clamp:
                raise ValueError(f"time {time} is after domain end {t1}.")
            query_time = t1
        return float(self.function(query_time))


@dataclass(frozen=True)
class SegmentSolution:
    """Continuous query interface for one solved physics segment."""

    segment_id: str
    object_ids: tuple[str, ...]
    state_vector: tuple[str, ...]
    state_owners: dict[str, str]
    domain: tuple[float, float]
    state_functions: dict[str, StateFunction]
    derived_functions: dict[str, StateFunction] = field(default_factory=dict)

    @property
    def t0(self) -> float:
        return self.domain[0]

    @property
    def t1(self) -> float:
        return self.domain[1]

    def state_at(self, time: float) -> dict[str, float]:
        return {name: function(time) for name, function in self.state_functions.items()}

    def object_state_at(self, object_id: str, time: float) -> dict[str, float]:
        return {
            name: function(time)
            for name, function in self.state_functions.items()
            if self.state_owners[name] == object_id
        }

    def object_states_at(self, time: float) -> dict[str, dict[str, float]]:
        return {
            object_id: self.object_state_at(object_id, time)
            for object_id in self.object_ids
        }

    def derived_at(self, time: float) -> dict[str, float]:
        return {name: function(time) for name, function in self.derived_functions.items()}

    @classmethod
    def from_trajectory(cls, trajectory: Trajectory, *, clamp: bool = False) -> "SegmentSolution":
        if not trajectory.times:
            raise ValueError("Cannot build SegmentSolution from an empty trajectory.")
        domain = (trajectory.times[0], trajectory.times[-1])
        state_functions = {
            name: StateFunction(
                name=name,
                domain=domain,
                function=InterpolatedStateFunction(
                    tuple(trajectory.times),
                    tuple(values),
                    clamp=clamp,
                ),
                clamp=clamp,
            )
            for name, values in trajectory.states.items()
        }
        derived_functions = {
            name: StateFunction(
                name=name,
                domain=domain,
                function=InterpolatedStateFunction(
                    tuple(trajectory.times),
                    tuple(values),
                    clamp=clamp,
                ),
                clamp=clamp,
            )
            for name, values in trajectory.derived.items()
        }
        return cls(
            segment_id=trajectory.segment_id,
            object_ids=tuple(trajectory.object_ids),
            state_vector=tuple(trajectory.states.keys()),
            state_owners={
                name: object_id
                for object_id, states in trajectory.object_states.items()
                for name in states
            },
            domain=domain,
            state_functions=state_functions,
            derived_functions=derived_functions,
        )


@dataclass(frozen=True)
class SegmentResult:
    """Uniform solver result consumed by PhysicsAnimation."""

    trajectory: Trajectory
    solution: SegmentSolution
    start_keyframe: PhysicsKeyFrame
    end_keyframe: PhysicsKeyFrame
    triggered_event: str | None = None
