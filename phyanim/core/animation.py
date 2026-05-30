from __future__ import annotations

from dataclasses import dataclass, field

from phyanim.core.context import PhysicsContext, AnnotationContext, TransitionContext, TimeWrapperContext
from phyanim.core.events import PhysicsEvent
from phyanim.core.keyframe import PhysicsKeyFrame
from phyanim.core.objects import PhysicObject2D
from phyanim.core.segment import PhysicsSegment
from phyanim.core.solution import SegmentResult
from phyanim.core.trajectory import InterpolatedStateFunction, Trajectory
from phyanim.solver.scipy_solver import ScipySegmentSolver
from phyanim.core.enhance.annotation import Annotation
from phyanim.core.enhance.transition import Transition
from phyanim.core.enhance.timewrapper import TimeWrapper
from phyanim.solver.annotation_solver import DefaultAnnotationSolver
from phyanim.solver.transition_solver import DefaultTransitionSolver
from phyanim.solver.timewrapper_solver import DefaultTimeWrapperSolver
from phyanim.core.layer import Layer

from typing import Tuple, Callable


@dataclass
class PhysicsAnimation:
    """Coordinates objects, sequential segments, keyframes, and timelines."""
    # 变量相关
    objects: dict[str, PhysicObject2D] = field(default_factory=dict)
    segments: list[PhysicsSegment] = field(default_factory=list)
    global_parameters: dict[str, float] = field(default_factory=dict)
    initial_keyframe: PhysicsKeyFrame | None = None
    keyframes: list[PhysicsKeyFrame] = field(default_factory=list)
    trajectories: list[Trajectory] = field(default_factory=list)
    segment_results: list[SegmentResult] = field(default_factory=list)
    segment_end_events: list[PhysicsEvent] = field(default_factory=list)
    segment_parameters: dict[str, dict[str, float]] = field(default_factory=dict)
    # 变量是否全部求解成功
    solved: bool = False

    # 注释相关
    annotations: dict[str, Annotation] = field(default_factory=dict)

    # 过渡相关
    transitions: dict[str, Transition] = field(default_factory=dict)

    # 时间缩放相关
    time_wrappers: dict[str, TimeWrapper] = field(default_factory=dict)

    # 将object添加到动画中，initial_state必须在object.state_variables中定义
    def add_object(self, obj: PhysicObject2D, initial_state: dict[str, float]) -> None:
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
                # 有状态转移的事件必须是终端事件，应用状态转移函数后开启新的段
        if end_event.transition is None or not end_event.condition.terminal:
            raise ValueError(
                f"Event '{end_event.name}' must be terminal and has a state transition."
            )
        self.segments.append(segment)
        self.segment_end_events.append(end_event)


    def add_annotation(self, annotation: Annotation) -> None:
        if annotation.id is None:
            raise ValueError(
                f"Annotation '{annotation}' id should not be None or empty"
            )
        self.annotations[annotation.id] = annotation
    
    def add_transition(self, transition: Transition) -> None:
        if transition.id is None:
            raise ValueError(
                f"Transition '{transition}' id should not be None or empty"
            )
        self.transitions[transition.id] = transition
    
    def add_time_wrapper(self, time_wrapper: TimeWrapper) -> None:
        if time_wrapper.id is None:
            raise ValueError(
                f"TimeWrapper '{time_wrapper}' id should not be None or empty"
            )
        self.time_wrappers[time_wrapper.id] = time_wrapper

    # 求解变量和注释
    def solve(self, solver: ScipySegmentSolver, anno_solver: DefaultAnnotationSolver, transition_solver: DefaultTransitionSolver, time_wrapper_solver: DefaultTimeWrapperSolver):
        physics_ctx = self.solve_simulation(solver)
        annotation_ctx = self.solve_annotation(anno_solver, physics_ctx)
        transition_ctx = self.solve_transition(transition_solver, physics_ctx)
        total_time, time_mapping_func = self.solve_time_wrapper(time_wrapper_solver, physics_ctx)

        main_layer = Layer(
            id="main_layer",
            contexts=[physics_ctx, annotation_ctx, transition_ctx],
            time_mapping_func=time_mapping_func,
            total_time=total_time,
        )

        return main_layer

    # initial_keyframe必须包含所有状态变量的初始值（必须强行保证，否则可能造成数据丢失）
    def solve_simulation(self, solver: ScipySegmentSolver | None = None) -> PhysicsContext:
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
            self.segment_parameters[segment.segment_id] = parameters
            # 一般认为状态转移瞬间完成
            # t_start和t_end当作是这个段内的参数处理
            parameters.setdefault("t_start", current_keyframe.time)
            parameters.setdefault("t_end", current_keyframe.time + segment.duration)
            result = solver.solve(segment, current_keyframe, parameters, end_event=end_event)
            self.segment_results.append(result)
            self.trajectories.append(result.trajectory)
            # 段结束帧来自从y的解码值，状态可能补全因此需要和前一帧进行合并
            end_keyframe = self._merge_keyframe_states(current_keyframe, result.end_keyframe)
            self.keyframes.append(end_keyframe)
            # 段状态转移
            current_keyframe = self._apply_segment_event(segment, end_event, result, end_keyframe)
            self.keyframes.append(current_keyframe)

        self.solved = True

        return PhysicsContext(
                self.build_state_functions(),
                self.build_derived_functions(),
                self.trajectories,
                self.segments,
                self.segment_parameters,
                self.objects
            )

    def solve_annotation(self, solver: DefaultAnnotationSolver | None = None, physics_ctx: PhysicsContext | None = None) -> AnnotationContext:
        return solver.solve(physics_ctx)

    def solve_transition(self, solver: DefaultTransitionSolver | None = None, physics_ctx: PhysicsContext | None = None) -> TransitionContext:
        return solver.solve(physics_ctx)

    def solve_time_wrapper(self, solver: DefaultTimeWrapperSolver | None = None, physics_ctx: PhysicsContext | None = None) -> Tuple[float, Callable[[float], float]] :
        return solver.solve(physics_ctx)

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

    def build_state_functions(
        self, *, clamp: bool = True
    ) -> dict[str, dict[str, dict[str, InterpolatedStateFunction]]]:
        """Return segment -> object -> state -> f(time) lookup functions."""
        functions: dict[str, dict[str, dict[str, InterpolatedStateFunction]]] = {}
        for i, trajectory in enumerate(self.trajectories):
            start_keyframe = self.keyframes[2 * i]
            t_start = trajectory.times[0]
            t_end = trajectory.times[-1]
            segment_functions = functions.setdefault(trajectory.segment_id, {})

            for object_id in self.objects.keys():
                obj_functions: dict[str, InterpolatedStateFunction] = {}
                # 全量的state
                obj_state = list(self.objects[object_id].state_variables.keys())
                for state in obj_state:
                    # 本段有解
                    if state in trajectory.states:
                        obj_functions[state] = trajectory.state_function(state, clamp=clamp)
                    # 本段没有解，从上一帧继承
                    elif state in start_keyframe.object_states[object_id].keys():
                        state_value = start_keyframe.object_states[object_id][state]
                        obj_functions[state] = InterpolatedStateFunction(
                            times=(t_start, t_end),
                            values=(state_value, state_value),
                            clamp=clamp,
                        )

                    else:
                        raise ValueError(
                            f"Segment '{trajectory.segment_id}' object '{object_id}' state '{state}' is not found in trajectory or keyframe."
                        )
                segment_functions[object_id] = obj_functions
            
        return functions

    def build_derived_functions(
        self, *, clamp: bool = True
    ) -> dict[str, dict[str, InterpolatedStateFunction]]:
        """Return segment -> derived_name -> f(time) lookup functions.
        
        Derived variables have no object ownership.
        Variables absent from a segment are carried over as constants
        from the last segment that computed them.
        """
        # 收集所有 segment 里出现过的派生量名称
        all_derived_names: set[str] = set()
        for trajectory in self.trajectories:
            all_derived_names.update(trajectory.derived.keys())

        functions: dict[str, dict[str, InterpolatedStateFunction]] = {}
        # 记录每个派生量最后一次被计算到的值，用于后续 segment 的常值继承
        last_known: dict[str, float] = {}

        for trajectory in self.trajectories:
            t_start = trajectory.times[0]
            t_end = trajectory.times[-1]
            seg_functions: dict[str, InterpolatedStateFunction] = {}

            for name in all_derived_names:
                if name in trajectory.derived:
                    seg_functions[name] = trajectory.derived_function(name, clamp=clamp)
                    # 更新最后已知值为本段末尾值
                    last_known[name] = trajectory.derived[name][-1]
                elif name in last_known:
                    seg_functions[name] = InterpolatedStateFunction(
                        times=(t_start, t_end),
                        values=(last_known[name], last_known[name]),
                        clamp=clamp,
                    )
                else:
                    raise ValueError(
                        f"Segment '{trajectory.segment_id}': derived variable '{name}' "
                        f"has no value in this trajectory or any prior segment."
                    )

            functions[trajectory.segment_id] = seg_functions

        return functions

    # previous_keyframe是全量的，segment_keyframe可能不全量，需要合并
    def _merge_keyframe_states(
        self,
        previous_keyframe: PhysicsKeyFrame,
        segment_keyframe: PhysicsKeyFrame,
    ) -> PhysicsKeyFrame:
        object_states = {
            object_id: dict(state)
            for object_id, state in previous_keyframe.object_states.items()
        }
        # 有些state变了，有些没变，保留该段内未更新的状态变量
        for object_id, state in segment_keyframe.object_states.items():
            object_states.setdefault(object_id, {}).update(state)
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
        # 没有触发事件，报错
        if result.triggered_event is None:
            raise ValueError(
                    f"Segment '{segment.segment_id}' end_event '{end_event.name}' is not triggered, can not use transition."
                )

        # 这里必然保证了是全量状态变量
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
        for object_id, partial_state in segment.split_state_by_object(flat_state).items():
            object_states.setdefault(object_id, {}).update(partial_state)
        return PhysicsKeyFrame(
            time=boundary_keyframe.time,
            object_states=object_states,
            parameters=boundary_keyframe.parameters,
            segment_id=boundary_keyframe.segment_id,
            event_name=boundary_keyframe.event_name,
        )
