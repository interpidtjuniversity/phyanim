from __future__ import annotations

from dataclasses import dataclass, field

from phyanim.core.events import PhysicsEvent
from phyanim.core.keyframe import PhysicsKeyFrame
from phyanim.core.objects import PhysicObject
from phyanim.core.segment import PhysicsSegment
from phyanim.core.solution import SegmentResult, StateFunction
from phyanim.core.trajectory import InterpolatedStateFunction, Trajectory
from phyanim.solver.scipy_solver import ScipySegmentSolver


@dataclass
class PhysicsAnimation:
    """Coordinates objects, sequential segments, keyframes, and timelines."""

    objects: dict[str, PhysicObject] = field(default_factory=dict)
    segments: list[PhysicsSegment] = field(default_factory=list)
    global_parameters: dict[str, float] = field(default_factory=dict)
    initial_keyframe: PhysicsKeyFrame | None = None
    keyframes: list[PhysicsKeyFrame] = field(default_factory=list)
    trajectories: list[Trajectory] = field(default_factory=list)
    segment_results: list[SegmentResult] = field(default_factory=list)
    segment_end_events: list[PhysicsEvent] = field(default_factory=list)

    # 将object添加到动画中，initial_state必须在object.state_variables中定义
    def add_object(self, obj: PhysicObject, initial_state: dict[str, float]) -> None:
        invalid_global_parameters = [name for name in self.global_parameters if not name.isidentifier()]
        if invalid_global_parameters:
            raise ValueError(f"Global parameter names must be valid identifiers: {invalid_global_parameters}")
        if obj.object_id in self.objects:
            raise ValueError(f"Duplicate object id: {obj.object_id}")
        existing_parameters = set(self.global_parameters)
        for existing in self.objects.values():
            existing_parameters.update(existing.parameter_values())
        duplicate_parameters = set(obj.parameter_values()) & existing_parameters
        if duplicate_parameters:
            raise ValueError(
                f"Object '{obj.object_id}' parameters duplicate existing parameters: "
                f"{sorted(duplicate_parameters)}"
            )
        duplicate_states = {
            name
            for existing in self.objects.values()
            for name in existing.state_variables
        } & set(obj.state_variables)
        if duplicate_states:
            raise ValueError(
                f"Object '{obj.object_id}' state variables duplicate existing object states: "
                f"{sorted(duplicate_states)}"
            )
        duplicate_parameter_states = set(obj.state_variables) & existing_parameters
        if duplicate_parameter_states:
            raise ValueError(
                f"Object '{obj.object_id}' state variables duplicate existing parameters: "
                f"{sorted(duplicate_parameter_states)}"
            )
        for name in initial_state:
            if not name.isidentifier():
                raise ValueError(f"Initial state name '{name}' must be a valid identifier.")
        self.objects[obj.object_id] = obj
        if self.initial_keyframe is None:
            self.initial_keyframe = PhysicsKeyFrame(
                time=0.0,
                object_states={obj.object_id: dict(initial_state)},
                parameters=self.global_parameters,
            )
        else:
            self.initial_keyframe.object_states[obj.object_id] = dict(initial_state)

    # 每个段后面都需要一个 terminal 事件，用于结束当前段。
    def add_segment(self, segment: PhysicsSegment, end_event: PhysicsEvent) -> None:
        duplicate_segment_parameters = set(segment.parameters) & set(self.global_parameters)
        if duplicate_segment_parameters:
            raise ValueError(
                f"Segment '{segment.segment_id}' parameters duplicate global parameters: "
                f"{sorted(duplicate_segment_parameters)}"
            )
        if not end_event.condition.terminal:
            raise ValueError(
                f"Segment end event '{end_event.name}' must be terminal."
            )
        self.segments.append(segment)
        self.segment_end_events.append(end_event)

    def solve(self, solver: ScipySegmentSolver | None = None) -> list[Trajectory]:
        if self.initial_keyframe is None:
            raise ValueError("PhysicsAnimation requires an initial keyframe before solving.")
        solver = solver or ScipySegmentSolver()
        self.keyframes = [self.initial_keyframe]
        self.trajectories = []
        self.segment_results = []
        current_keyframe = self.initial_keyframe

        for segment, end_event in zip(self.segments, self.segment_end_events):
            # 段调度、事件后的状态衔接都在框架层完成；求解器只处理当前连续段。
            parameters = self._segment_parameters(segment)
            # 一般认为状态转移瞬间完成
            parameters.setdefault("t_start", current_keyframe.time)
            parameters.setdefault("t_end", current_keyframe.time + segment.duration)
            result = solver.solve(segment, current_keyframe, parameters, end_event=end_event)
            self.segment_results.append(result)
            self.trajectories.append(result.trajectory)
            # 段结束帧
            end_keyframe = self._merge_keyframe_states(current_keyframe, result.end_keyframe)
            self.keyframes.append(end_keyframe)
            # 段状态转移
            current_keyframe = self._apply_segment_event(segment, end_event, result, end_keyframe)
            self.keyframes.append(current_keyframe)

        return list(self.trajectories)

    def solve_segments(self, solver: ScipySegmentSolver | None = None) -> list[SegmentResult]:
        self.solve(solver=solver)
        return list(self.segment_results)

    def _segment_parameters(self, segment: PhysicsSegment) -> dict[str, float]:
        parameters = segment.merged_parameters(self.global_parameters)
        duplicate_state_parameters = set(segment.state_vector) & set(parameters)
        if duplicate_state_parameters:
            raise ValueError(
                f"Segment '{segment.segment_id}' state names duplicate parameter names: "
                f"{sorted(duplicate_state_parameters)}"
            )
        for object_id in segment.object_ids:
            obj = self.objects.get(object_id)
            if obj is None:
                raise ValueError(f"Segment '{segment.segment_id}' references unknown object '{object_id}'.")
            object_parameters = obj.parameter_values()
            duplicate = set(parameters) & set(object_parameters)
            if duplicate:
                raise ValueError(
                    f"Segment '{segment.segment_id}' object '{object_id}' parameter names "
                    f"duplicate existing parameters: {sorted(duplicate)}"
                )
            duplicate_state_object_parameters = set(segment.state_vector) & set(object_parameters)
            if duplicate_state_object_parameters:
                raise ValueError(
                    f"Segment '{segment.segment_id}' state names duplicate object '{object_id}' "
                    f"parameters: {sorted(duplicate_state_object_parameters)}"
                )
            parameters.update(object_parameters)
        return parameters

    def build_solution_state_functions(
        self,
    ) -> dict[str, dict[str, dict[str, StateFunction]]]:
        """Return segment -> object -> state -> dense f(time) functions when available."""

        functions: dict[str, dict[str, dict[str, StateFunction]]] = {}
        for result in self.segment_results:
            segment_functions = functions.setdefault(result.solution.segment_id, {})
            for object_id in result.solution.object_ids:
                segment_functions[object_id] = {
                    name: function
                    for name, function in result.solution.state_functions.items()
                    if result.solution.state_owners[name] == object_id
                }
        return functions

    def build_state_functions(
        self, *, clamp: bool = True
    ) -> dict[str, dict[str, dict[str, InterpolatedStateFunction]]]:
        """Return segment -> object -> state -> f(time) lookup functions."""

        functions: dict[str, dict[str, dict[str, InterpolatedStateFunction]]] = {}
        for trajectory in self.trajectories:
            segment_functions = functions.setdefault(trajectory.segment_id, {})
            if trajectory.object_states:
                for object_id, states in trajectory.object_states.items():
                    segment_functions[object_id] = {
                        name: trajectory.state_function(name, clamp=clamp)
                        for name in states
                    }
            else:
                raise ValueError("Trajectory is missing object_states.")
        return functions

    def build_derived_functions(
        self, *, clamp: bool = True
    ) -> dict[str, dict[str, dict[str, InterpolatedStateFunction]]]:
        """Return segment -> object -> derived quantity -> f(time) lookup functions."""

        functions: dict[str, dict[str, dict[str, InterpolatedStateFunction]]] = {}
        for trajectory in self.trajectories:
            segment_functions = functions.setdefault(trajectory.segment_id, {})
            for object_id in trajectory.object_ids:
                segment_functions[object_id] = trajectory.derived_functions(clamp=clamp)
        return functions

    def _merge_keyframe_states(
        self,
        previous_keyframe: PhysicsKeyFrame,
        segment_keyframe: PhysicsKeyFrame,
    ) -> PhysicsKeyFrame:
        object_states = {
            object_id: dict(state)
            for object_id, state in previous_keyframe.object_states.items()
        }
        # 有些state变了，有些没变
        for object_id, state in segment_keyframe.object_states.items():
            object_states[object_id] = dict(state)
        return PhysicsKeyFrame(
            time=segment_keyframe.time,
            object_states=object_states,
            parameters=segment_keyframe.parameters,
            segment_id=segment_keyframe.segment_id,
            event_name=segment_keyframe.event_name,
        )

    def _apply_segment_event(
        self,
        segment: PhysicsSegment,
        end_event: PhysicsEvent,
        result: SegmentResult,
        boundary_keyframe: PhysicsKeyFrame,
    ) -> PhysicsKeyFrame:
        # 没有触发事件，直接返回边界帧
        if result.triggered_event is None:
            raise ValueError(
                    f"Segment '{segment.segment_id}' end_event '{end_event.event_name}' is not triggered, can not use transition."
                )

        flat_state = segment.flatten_keyframe_state(boundary_keyframe.object_states)
        parameters = boundary_keyframe.parameters
        # 事件是段和段之间的边界：所有状态跃迁都发生在下一段开始前。
        # 这里必然不能为None，如果是最后一个segment则需要使用恒等映射transition进行兜底
        if end_event.transition is not None:
            flat_state = end_event.transition.apply(flat_state, parameters, time=boundary_keyframe.time)

        object_states = {
            object_id: dict(state)
            for object_id, state in boundary_keyframe.object_states.items()
        }
        object_states.update(segment.split_state_by_object(flat_state))
        return PhysicsKeyFrame(
            time=boundary_keyframe.time,
            object_states=object_states,
            parameters=boundary_keyframe.parameters,
            segment_id=boundary_keyframe.segment_id,
            event_name=boundary_keyframe.event_name,
        )
