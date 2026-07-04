from __future__ import annotations

from dataclasses import dataclass, field

from phyanim.core.context import PhysicsContext
from phyanim.core.events import PhysicsEvent
from phyanim.core.keyframe import PhysicsKeyFrame
from phyanim.core.objects import PhysicObject2D
from phyanim.core.segment import PhysicsSegment
from phyanim.core.solution import SegmentResult
from phyanim.core.trajectory import InterpolatedStateFunction, Trajectory
from phyanim.core.validation import SymbolRegistry, normalize_numeric_mapping, require_identifiers
from phyanim.solver.solver import Solver
from phyanim.solver.scipy_solver import ScipySegmentSolver
from phyanim.core.layer import Layer

@dataclass
class PhysicsAnimation:
    """Coordinates objects, sequential segments, keyframes, and timelines."""
    # 求解精确度
    sample_dt: float = 1 / 60
    engine: str = "scipy"

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
    # 物理上下文
    physics_ctx: PhysicsContext = None
    # physics_layer，它是physics_ctx的展示层
    physics_layer: Layer | None = None
    render_layer: Layer | None = None
    # 注册的观察事件（不影响求解，在求解后扫描轨迹检测触发时刻）
    registered_events: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.global_parameters = normalize_numeric_mapping(
            self.global_parameters,
            kind="Global parameter",
        )

    @property
    def symbol_registry(self) -> SymbolRegistry:
        registry = SymbolRegistry.from_global_parameters(self.global_parameters)
        for obj in self.objects.values():
            registry.reserve_mapping(
                obj.parameters,
                owner=f"object '{obj.object_id}' parameters",
                kind="parameter",
            )
            registry.reserve_mapping(
                obj.state_variables,
                owner=f"object '{obj.object_id}' states",
                kind="state",
            )
        return registry

    # 将object添加到动画中，initial_state必须在object.state_variables中定义
    def add_object(self, obj: PhysicObject2D, initial_state: dict[str, float | int | str]) -> "PhysicsAnimation":
        if obj.object_id in self.objects:
            raise ValueError(f"Duplicate object id: {obj.object_id}")
        normalized_state = normalize_numeric_mapping(
            initial_state,
            kind=f"Object '{obj.object_id}' initial state",
        )
        missing = set(obj.state_variables) - set(normalized_state)
        unknown = set(normalized_state) - set(obj.state_variables)
        if missing:
            raise ValueError(f"Object '{obj.object_id}' missing initial states: {sorted(missing)}")
        if unknown:
            raise ValueError(f"Object '{obj.object_id}' has unknown initial states: {sorted(unknown)}")
        registry = self.symbol_registry
        registry.reserve_mapping(
            obj.parameters,
            owner=f"object '{obj.object_id}' parameters",
            kind="parameter",
        )
        registry.reserve_mapping(
            obj.state_variables,
            owner=f"object '{obj.object_id}' states",
            kind="state",
        )
        self.objects[obj.object_id] = obj
        if self.initial_keyframe is None:
            self.initial_keyframe = PhysicsKeyFrame(
                time=0.0,
                object_states={obj.object_id: dict(normalized_state)},
                parameters=self.global_parameters,
            )
        else:
            self.initial_keyframe.object_states[obj.object_id] = dict(normalized_state)
        return self

    # 每个段后面都需要一个 terminal 事件，用于结束当前段。
    def add_segment(self, segment: PhysicsSegment, end_event: PhysicsEvent) -> "PhysicsAnimation":
        self._validate_segment(segment)
        # 有状态转移的事件必须是终端事件，应用状态转移函数后开启新的段
        if end_event.transition is None or not end_event.condition.terminal:
            raise ValueError(
                f"Event '{end_event.name}' must be terminal and has a state transition."
            )
        self.segments.append(segment)
        self.segment_end_events.append(end_event)
        return self

    def add_equation_segment(
        self,
        segment_id: str,
        *,
        objects: list[str],
        equations: dict[str, str],
        end_event: PhysicsEvent,
        duration: float,
        owners: dict[str, str] | None = None,
        parameters: dict[str, float] | None = None,
        derived: dict[str, str] | None = None,
    ) -> "PhysicsAnimation":
        return self.add_segment(
            PhysicsSegment.from_equations(
                segment_id,
                objects=objects,
                equations=equations,
                duration=duration,
                owners=owners,
                parameters=parameters,
                derived=derived,
            ),
            end_event=end_event,
        )

    def register_event(
        self,
        event_id: str,
        expression: str,
        direction: int = 0,
        event_type: str = "crossing",
    ) -> "PhysicsAnimation":
        """注册一个观察事件，在求解后扫描轨迹检测其触发时刻。

        观察事件不影响物理求解过程。

        Parameters
        ----------
        event_id:
            事件唯一标识符。
        expression:
            - event_type="crossing": 零点检测表达式（SymPy 语法）。可引用
              所有状态变量、derived 变量、参数、t。当表达式穿越零点时触发。
            - event_type="bool": 布尔表达式。当条件从 False 变为 True 时触发。
              可用运算符：> < >= <= == != & | ~。可引用所有状态变量、
              derived 变量、参数、t。支持内置极值函数：
              is_local_max(expr)  — expr 在该采样点是局部极大值
              is_local_min(expr)  — expr 在该采样点是局部极小值
              is_global_max(expr) — expr 在该采样点是全局最大值
              is_global_min(expr) — expr 在该采样点是全局最小值
              极值函数的 expr 参数是一个 SymPy 表达式字符串。
        direction:
            -1 = 正→负穿越，1 = 负→正穿越，0 = 任意方向。
            仅对 crossing 类型有效。
        event_type:
            "crossing"（零点穿越，默认）或 "bool"（布尔条件）。

        Returns
        -------
        PhysicsAnimation
            self（支持链式调用）。

        示例::

            # 零点穿越
            animation.register_event("at_peak", "vy", direction=-1)

            # 布尔条件
            animation.register_event("fast_and_high", "(x > 5) & (vx > 100)", event_type="bool")

            # 极值检测
            animation.register_event("peak", "is_local_max(y)", event_type="bool")
            animation.register_event("valley", "is_local_min(y)", event_type="bool")
            animation.register_event("highest", "is_global_max(y)", event_type="bool")
        """
        if direction not in (-1, 0, 1):
            raise ValueError("direction must be -1, 0, or 1")
        if event_type not in ("crossing", "bool"):
            raise ValueError("event_type must be 'crossing' or 'bool'")
        self.registered_events.append({
            "event_id": event_id,
            "expression": expression,
            "direction": direction,
            "event_type": event_type,
        })
        return self

    def solve(self, solver: Solver | None = None) -> PhysicsContext:
        """求解动画"""
        solver = solver or self._build_solver()
        self.physics_ctx = self.solve_simulation(solver)
        return self.physics_ctx

    def _build_solver(self) -> Solver:
        if self.engine == "scipy":
            return ScipySegmentSolver(sample_dt=self.sample_dt)
        if self.engine == "heyoka":
            try:
                from phyanim.solver.heyoka_solver import HeyokaSegmentSolver
            except ImportError as exc:
                raise ImportError(
                    "heyoka engine requires heyoka.py. Install it with: pip install heyoka"
                ) from exc
            return HeyokaSegmentSolver(sample_dt=self.sample_dt)
        raise ValueError(f"Unknown engine: {self.engine}. Supported engines are 'scipy' and 'heyoka'.")

    def get_physics_layer(self) -> Layer:     
        if self.physics_layer is not None:
            return self.physics_layer
        self.physics_layer = Layer(id="physics_layer", name_space="physics")
        return self.physics_layer

    def get_render_layer(self) -> Layer:
        if self.render_layer is not None:
            return self.render_layer
        self.render_layer = Layer(id="render_layer", name_space="render", sample_dt=self.sample_dt)
        return self.render_layer

    # initial_keyframe必须包含所有状态变量的初始值（必须强行保证，否则可能造成数据丢失）
    def solve_simulation(self, solver: Solver | None = None) -> PhysicsContext:
        if self.initial_keyframe is None:
            raise ValueError("PhysicsAnimation requires an initial keyframe before solving.")
        solver = solver or self._build_solver()
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

        ctx = PhysicsContext(
                self.build_state_functions(),
                self.build_derived_functions(),
                self.trajectories,
                self.segments,
                self.segment_parameters,
                self.objects
            )

        # 求解注册的观察事件：扫描所有采样时间点，检测零点穿越。
        ctx.event_trigger_map = self._solve_registered_events(ctx)

        return ctx

    def _solve_registered_events(self, ctx: PhysicsContext) -> dict[str, list[float]]:
        """求解所有注册事件的触发时刻。

        支持两种事件类型：
        - crossing: 零点穿越检测，用 brentq 精确定位。
        - bool: 布尔条件检测，当条件从 False 变为 True 时触发。
          支持内置极值函数 is_local_max/is_local_min/is_global_max/is_global_min。
        """

        result: dict[str, list[float]] = {}

        for reg in self.registered_events:
            event_id = reg["event_id"]
            expression = reg["expression"]
            direction = reg["direction"]
            event_type = reg.get("event_type", "crossing")

            if event_type == "bool":
                trigger_times = self._solve_bool_event(ctx, expression)
            else:
                trigger_times = self._solve_crossing_event(ctx, expression, direction)

            result[event_id] = trigger_times
        
        # 将段退出事件的触发结果也加入
        for seg_result in self.segment_results:
            result[seg_result.triggered_event] = [seg_result.solution.t1]

        return result

    def _solve_crossing_event(
        self, ctx: PhysicsContext, expression: str, direction: int
    ) -> list[float]:
        """求解零点穿越事件的触发时刻。"""
        from phyanim.core.enhance.trigger import CrossingTrigger

        trigger = CrossingTrigger(expression=expression, direction=direction)

        def eval_func(t: float) -> float:
            return ctx.eval_expr(expression, "float", t)

        trigger_times: list[float] = []
        old_t: float | None = None

        for t in ctx.times:
            if old_t is not None:
                event_t = trigger(old_t, t, eval_func)
                if event_t is not None:
                    trigger_times.append(event_t)
            old_t = t

        return trigger_times

    def _solve_bool_event(
        self, ctx: PhysicsContext, expression: str
    ) -> list[float]:
        """求解布尔条件事件的触发时刻。

        支持内置极值函数 is_local_max/is_local_min/is_global_max/is_global_min。
        当布尔表达式从 False 变为 True 时触发。
        """
        import re

        # 1. 预处理：提取极值函数调用，替换为占位符。
        # is_local_max(expr) → __extremum_0__, is_global_min(expr) → __extremum_1__, ...
        extremum_calls: list[dict] = []
        pattern = re.compile(
            r"(is_local_max|is_local_min|is_global_max|is_global_min)\s*\(\s*(.+?)\s*\)"
        )

        def replace_extremum(match: re.Match) -> str:
            func_name = match.group(1)
            inner_expr = match.group(2)
            idx = len(extremum_calls)
            placeholder = f"__extremum_{idx}__"
            extremum_calls.append({
                "func": func_name,
                "expr": inner_expr,
                "placeholder": placeholder,
            })
            return placeholder

        processed_expr = pattern.sub(replace_extremum, expression)

        # 2. 预计算极值函数在所有采样点的值。
        extremum_values: list[list[bool]] = []  # [call_idx][time_idx] → bool
        for call in extremum_calls:
            values = self._eval_extremum_func(ctx, call["func"], call["expr"])
            extremum_values.append(values)

        # 3. 在每个采样点评估整个布尔表达式。
        bool_values: list[bool] = []
        for ti, t in enumerate(ctx.times):
            # 构建极值占位符的值
            extremum_map = {}
            for ci, call in enumerate(extremum_calls):
                extremum_map[call["placeholder"]] = extremum_values[ci][ti]

            val = self._eval_bool_with_extremum(ctx, processed_expr, t, extremum_map)
            bool_values.append(val)

        # 4. 检测 False → True 跳变。
        trigger_times: list[float] = []
        for i in range(1, len(bool_values)):
            if not bool_values[i - 1] and bool_values[i]:
                trigger_times.append(ctx.times[i])

        return trigger_times

    def _eval_extremum_func(
        self, ctx: PhysicsContext, func_name: str, expr: str
    ) -> list[bool]:
        """评估极值函数在所有采样点的布尔值。

        Parameters
        ----------
        func_name: is_local_max / is_local_min / is_global_max / is_global_min
        expr: 被检测的表达式字符串
        """
        values = [ctx.eval_expr(expr, "float", t) for t in ctx.times]

        if not values:
            return []

        results: list[bool] = [False] * len(values)

        if func_name == "is_global_max":
            global_max = max(values)
            for i, v in enumerate(values):
                results[i] = abs(v - global_max) < 1e-12

        elif func_name == "is_global_min":
            global_min = min(values)
            for i, v in enumerate(values):
                results[i] = abs(v - global_min) < 1e-12

        elif func_name == "is_local_max":
            for i in range(len(values)):
                if i == 0 or i == len(values) - 1:
                    continue  # 边界点不算极值
                if values[i] > values[i - 1] and values[i] > values[i + 1]:
                    results[i] = True

        elif func_name == "is_local_min":
            for i in range(len(values)):
                if i == 0 or i == len(values) - 1:
                    continue
                if values[i] < values[i - 1] and values[i] < values[i + 1]:
                    results[i] = True

        return results

    def _eval_bool_with_extremum(
        self,
        ctx: PhysicsContext,
        expr: str,
        t: float,
        extremum_map: dict[str, bool],
    ) -> bool:
        """评估包含极值占位符的布尔表达式。

        策略：
        1. 将极值占位符替换为 Python True/False。
        2. 将 & 和 | 替换为 Python and/or。
        3. 对剩余的 sympy 子表达式（如 y > 3）用 eval_expr 评估。
        4. 用 Python 逻辑合并所有部分。
        """
        if not extremum_map:
            try:
                return ctx.eval_expr(expr, "bool", t)
            except Exception:
                return False

        # 替换占位符为 True/False
        processed = expr
        for placeholder, val in extremum_map.items():
            processed = processed.replace(placeholder, "True" if val else "False")

        # 将 & 和 | 替换为 Python and/or（注意保留优先级，加括号）
        # 先把 True/False 用括号包裹
        processed = processed.replace("True", "(True)").replace("False", "(False)")
        # 替换逻辑运算符
        processed = processed.replace("&", " and ").replace("|", " or ")

        # 现在需要评估所有非 True/False 的 sympy 子表达式
        # 用正则找到所有不是 True/False/and/or 的表达式片段
        # 更简单的方法：把 (y > 3) 这种子表达式替换为 eval_expr 的结果
        import re

        # 找到所有括号内的表达式（排除 True/False/and/or）
        def replace_sympy_expr(match):
            inner = match.group(1).strip()
            if inner in ("True", "False"):
                return match.group(0)
            try:
                val = ctx.eval_expr(inner, "bool", t)
                return "True" if val else "False"
            except Exception:
                try:
                    val = ctx.eval_expr(inner, "float", t)
                    return "True" if val > 0 else "False"
                except Exception:
                    return "False"

        # 匹配 (expr) 但不匹配 (True) (False)
        processed = re.sub(r"\(([^()]+)\)", replace_sympy_expr, processed)

        try:
            return bool(eval(processed))
        except Exception:
            return False

    def _validate_segment(self, segment: PhysicsSegment) -> None:
        if segment.segment_id in {existing.segment_id for existing in self.segments}:
            raise ValueError(f"Duplicate segment id: {segment.segment_id}")
        unknown_objects = sorted(set(segment.object_ids) - set(self.objects))
        if unknown_objects:
            raise ValueError(f"Segment '{segment.segment_id}' references unknown objects: {unknown_objects}")
        require_identifiers(segment.parameters, kind=f"Segment '{segment.segment_id}' parameter")
        duplicate_segment_parameters = set(segment.parameters) & set(self.global_parameters)
        if duplicate_segment_parameters:
            raise ValueError(
                f"Segment '{segment.segment_id}' parameters duplicate global parameters: "
                f"{sorted(duplicate_segment_parameters)}"
            )
        object_state_names = {
            name
            for object_id in segment.object_ids
            for name in self.objects[object_id].state_variables
        }
        unknown_states = sorted(set(segment.state_vector) - object_state_names)
        if unknown_states:
            raise ValueError(
                f"Segment '{segment.segment_id}' states are not declared by its objects: {unknown_states}"
            )

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
