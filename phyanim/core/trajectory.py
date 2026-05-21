from __future__ import annotations

from dataclasses import dataclass, field
from bisect import bisect_right

from phyanim.core.keyframe import PhysicsKeyFrame


# 当只记载了某些帧的状态变量时，需要进行插值来得到其他时间的状态变量
@dataclass(frozen=True)
class InterpolatedStateFunction:
    """Pure time-to-value lookup backed by sampled trajectory values."""

    times: tuple[float, ...]
    values: tuple[float, ...]
    clamp: bool = True

    def __post_init__(self) -> None:
        if len(self.times) != len(self.values):
            raise ValueError("times and values must have the same length.")
        if not self.times:
            raise ValueError("InterpolatedStateFunction requires at least one sample.")
        if any(right < left for left, right in zip(self.times, self.times[1:])):
            raise ValueError("times must be sorted in ascending order.")

    def __call__(self, time: float) -> float:
        if len(self.times) == 1:
            return self.values[0]

        start_time = self.times[0]
        end_time = self.times[-1]
        if time == start_time:
            return self.values[0]
        if time == end_time:
            return self.values[-1]
        if time < start_time:
            if self.clamp:
                return self.values[0]
            raise ValueError(f"time {time} is before trajectory start {start_time}.")
        if time > end_time:
            if self.clamp:
                return self.values[-1]
            raise ValueError(f"time {time} is after trajectory end {end_time}.")
        
        # 近似为直线插值
        right_index = bisect_right(self.times, time)
        left_index = right_index - 1
        left_time = self.times[left_index]
        right_time = self.times[right_index]
        left_value = self.values[left_index]
        right_value = self.values[right_index]
        if right_time == left_time:
            return right_value
        ratio = (time - left_time) / (right_time - left_time)
        return left_value + ratio * (right_value - left_value)


@dataclass
class Trajectory:
    """Sampled state history produced by solving a segment."""

    segment_id: str
    object_ids: list[str]
    times: list[float]
    states: dict[str, list[float]]
    object_states: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    derived: dict[str, list[float]] = field(default_factory=dict)

    # 这个参数暂时没用，后续可能会用
    triggered_event: str | None = None

    def state_function(self, state_name: str, *, clamp: bool = True) -> InterpolatedStateFunction:
        if state_name not in self.states:
            raise KeyError(f"Trajectory has no state named '{state_name}'.")
        return InterpolatedStateFunction(tuple(self.times), tuple(self.states[state_name]), clamp=clamp)

    def derived_function(self, quantity_name: str, *, clamp: bool = True) -> InterpolatedStateFunction:
        if quantity_name not in self.derived:
            raise KeyError(f"Trajectory has no derived quantity named '{quantity_name}'.")
        return InterpolatedStateFunction(tuple(self.times), tuple(self.derived[quantity_name]), clamp=clamp)

    def state_functions(self, *, clamp: bool = True) -> dict[str, InterpolatedStateFunction]:
        return {name: self.state_function(name, clamp=clamp) for name in self.states}

    def derived_functions(self, *, clamp: bool = True) -> dict[str, InterpolatedStateFunction]:
        return {name: self.derived_function(name, clamp=clamp) for name in self.derived}

    # 保存轨迹结束快照
    def end_keyframe(self, parameters: dict[str, float]) -> PhysicsKeyFrame:
        if not self.times:
            raise ValueError("Cannot create a keyframe from an empty trajectory.")
        # 返回最后一帧的状态变量值
        if self.object_states:
            object_states = {
                object_id: {name: values[-1] for name, values in states.items()}
                for object_id, states in self.object_states.items()
            }
        else:
            raise ValueError("Trajectory requires object_states to build a keyframe.")
        return PhysicsKeyFrame(
            time=self.times[-1],
            object_states=object_states,
            parameters=dict(parameters),
            segment_id=self.segment_id,
            event_name=self.triggered_event,
        )
