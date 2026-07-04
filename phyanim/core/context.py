from typing import Tuple, Callable

from phyanim.core.trajectory import InterpolatedStateFunction, Trajectory
from phyanim.core.segment import PhysicsSegment
from phyanim.core.expressions import CompiledExpression
from phyanim.core.objects import PhysicObject2D
from phyanim.core.enhance.annotation import Annotation
from phyanim.core.enhance.trigger import CrossingTrigger, Trigger
from phyanim.core.enhance.transition import Transition

import sympy as sp
from manim import ValueTracker, Mobject, VGroup, PI, LEFT, RIGHT, DOWN, UP, RED, GREEN, BLUE, WHITE, YELLOW, YELLOW_C, RED_C, BLUE_C, GREEN_C, ORANGE, PURPLE, TEAL_C, GOLD_C, MAROON_C, PURE_RED, PURE_GREEN, PURE_BLUE
import numpy as np
from phyanim.utils.util import is_numeric
from phyanim.core.enhance.visual_binding import VisualBinding
from phyanim.core.expressions import SympyExpressionCompiler

class Context:

    def set_render_to_physics_mapping_func(self, func: Callable[[float], float]):
        self.render_to_physics_mapping_func = func

    def set_physics_to_render_mapping_func(self, func: Callable[[float], float]):
        self.physics_to_render_mapping_func = func

# animation 上下文(求解后使用)
class PhysicsContext(Context):
    """求解后的物理上下文，管理整个动画中的变量。"""
    """ 参数必须来自已经求解的animation """
    def __init__(
        self, 
        state_funcs: dict[str, dict[str, dict[str, InterpolatedStateFunction]]],
        derived_funcs: dict[str, dict[str, InterpolatedStateFunction]],
        trajectories: list[Trajectory],
        segments: list[PhysicsSegment],
        segment_parameters: dict[str, dict[str, float]],
        objects: dict[str, PhysicObject2D]
        ):

        super().__init__()
        self.state_funcs = state_funcs
        self.derived_funcs = derived_funcs
        self.trajectories = trajectories
        self.segments = segments
        self.segment_parameters = segment_parameters
        self.objects = objects
        # 注册事件的触发时刻（由 animation._solve_registered_events 填充）
        self.event_trigger_map: dict[str, list[float]] = {}

        # 这里times中可能有重复时刻
        self.times_set : set[float] = set()
        self.start_times = []
        self.end_times = []
        self.tra_map = {}

        for tra_idx, tra in enumerate(self.trajectories):
            self.times_set.update(tra.times)
            self.start_times.append(tra.times[0])
            self.end_times.append(tra.times[-1])
            self.tra_map[tra_idx] = tra.segment_id
        
        self.times = sorted(self.times_set)
        self.total_time = max(self.end_times)


        self.cached_seg_funcs = {}
        self.cached_seg_symbols_map = {}
        self.cached_expressions = {}

        self.initialize_segment_parameters()

    def get_context_time(self) -> float:
        """将渲染时间映射到物理时间。"""
        return self.total_time

    def initialize_segment_parameters(self):
        """初始化每个段的符号参数字典。"""
        for segment in self.segments:
            # 获取段内所有符号参数
            segment_parameters = self.segment_parameters[segment.segment_id]
            symbol_names = set(segment.state_vector) | set(segment_parameters) | set(self.derived_funcs[segment.segment_id].keys()) |{"t"}
            invalid_symbol_names = sorted(name for name in symbol_names if not name.isidentifier())
            if invalid_symbol_names:
                raise ValueError(
                    "SymPy expressions only support plain identifier symbols: "
                    f"{symbol_names}"
                )
            locals_map = {
                name: sp.Symbol(name)
                for name in symbol_names
            }
            self.cached_seg_symbols_map[segment.segment_id] = locals_map

    
    def find_trajectory_index(self, t: float) -> int:
        """根据全局时间 t 找到当前属于哪一段 trajectory。"""
        eps = 1e-9
        for i, (t0, t1) in enumerate(zip(self.start_times, self.end_times)):
            is_last = i == len(self.start_times) - 1

            if is_last:
                if t0 - eps <= t <= t1 + eps:
                    return i
            else:
                # 中间段建议使用左闭右开，避免边界 t 同时属于两段
                if t0 - eps <= t < t1 - eps:
                    return i

        raise ValueError(
            f"Time {t} is out of range for any trajectory. "
            f"Trajectory intervals: {list(zip(self.start_times, self.end_times))}"
        )
    
    # 支持所有state变量和derived变量
    def value_at_time(self, name: str, t: float) -> float:
        if name == "t":
            return t
        """评估 name 在时间 t 处的数值。"""
        tra_idx = self.find_trajectory_index(t)
        seg_id = self.tra_map[tra_idx]

        # 先查缓存
        seg_states_func = self.cached_seg_funcs.setdefault(seg_id, {})
        if name in seg_states_func:
            return seg_states_func[name](t)

        for obj_id in self.state_funcs[seg_id].keys():
            for state_name in self.state_funcs[seg_id][obj_id].keys():
                if state_name == name:
                    seg_states_func[name] = self.state_funcs[seg_id][obj_id][name]
                    return seg_states_func[name](t)

        for derived_name in self.derived_funcs[seg_id].keys():
            if derived_name == name:
                seg_states_func[name] = self.derived_funcs[seg_id][derived_name]
                return seg_states_func[name](t)
        
        raise ValueError(
            f"Variable {name} not found in any state or derived function at segment {seg_id}."
        )

    """
    1.支持Piecewise写法(值, 条件)
        expr = "Piecewise((2.0, EK > 5), (1.0, (EK <= 5) & (P < 10)), (0, True))", return_type="float"
        {"EK": 8, "P": 11}   # 2.0
        {"EK": 8, "P": 9}   # 2.0
        {"EK": 3, "P": 9}   # 1.0
        {"EK": 3, "P": 11}   # 0.0
    2.支持返回bool类型
        expr = "(EK > 5) & (P < 10)", return_type="bool"
        {"EK": 8, "P": 9}   # True
        {"EK": 8, "P": 11}   # False
        {"EK": 3, "P": 9}   # False
        {"EK": 3, "P": 11}   # False
    3.支持数字表达式直接求值
        expr = "EK + P", return_type="float"
        {"EK": 8, "P": 11}   # 19.0
    """
    def eval_expr(self, expr: str, return_type: str, t: float) -> float:
        """评估 expr 在时间 t 处的数值。"""
        tra_idx = self.find_trajectory_index(t)
        seg_id = self.tra_map[tra_idx]

        if expr not in self.cached_expressions:
            parsed = sp.sympify(expr, locals=self.cached_seg_symbols_map[seg_id])
            names = tuple(sorted(str(symbol) for symbol in parsed.free_symbols))
            symbols = [sp.Symbol(name) for name in names]
            function = sp.lambdify(symbols, parsed, modules="numpy")
            self.cached_expressions[expr] = {
                "names": names,
                "compiled_expression": CompiledExpression(expression=expr, names=names, function=function),
            }
        
        compiled_expression = self.cached_expressions[expr]["compiled_expression"]
        names = self.cached_expressions[expr]["names"]
        values = []
        for name in names:
            values.append(self.value_at_time(name, t))
        
        result = float(compiled_expression.function(*values))

        match return_type:
            case "float":
                return result
            case "bool":
                return bool(result)
            case _:
                raise ValueError(
                    f"Return type {return_type} not supported. "
                    f"Available types: float, bool."
                )

    # 给trigger使用
    def build_eval_exper_func(self, trigger: Trigger) -> callable:
        """根据 trigger 构建评估函数。"""
        if isinstance(trigger, CrossingTrigger):
            def function(t: float) -> float:
                return self.eval_expr(trigger.expression, "float", t)
            return function
        else:
            return None


    def eval_position(self, obj_id: str, t: float) -> list[Tuple[float, float]]:
        """评估 obj_id 在时间 t 处的位置，返回具体数值 x, y。

        每个坐标分量可以是：
        - 纯变量名（状态变量 / derived 变量 / 其它对象的状态变量）
        - 字面量数字字符串（如 "0.5"）
        - SymPy 表达式（如 "x_p + L/2"），可引用状态变量、derived 变量、参数、t
        """

        tra_idx = self.find_trajectory_index(t)
        seg_id = self.tra_map[tra_idx]

        obj = self.objects[obj_id]

        cartesian_positions = []

        cartesian_position_variables = obj.cartesian_position_variables()
        for cartesian_position_variable in cartesian_position_variables:
            x_expr, y_expr = cartesian_position_variable
            x_val = self._resolve_coordinate(x_expr, obj_id, seg_id, t)
            y_val = self._resolve_coordinate(y_expr, obj_id, seg_id, t)
            cartesian_positions.append((float(x_val), float(y_val)))

        return cartesian_positions

    def _resolve_coordinate(
        self, expr: str, obj_id: str, seg_id: str, t: float
    ) -> float:
        """解析一个笛卡尔坐标分量。

        支持纯变量名、字面量数字、以及 SymPy 表达式。
        表达式可引用所有对象的状态变量、derived 变量、参数、t。
        """
        from phyanim.utils.util import is_numeric

        # 1. 字面量数字
        if is_numeric(expr):
            return float(expr)

        # 2. 尝试直接用 value_at_time（覆盖自身状态、derived、t）
        try:
            return self.value_at_time(expr, t)
        except (ValueError, KeyError):
            pass
        
        # 4. 作为 SymPy 表达式求值（如 "x_p + L/2"）
        #    eval_expr 的符号表包含本段所有状态变量 + derived + 参数 + t，
        #    但不包含其它对象的状态变量。我们需要扩展符号表。
        return self._eval_expr_full(expr, seg_id, t)

    def _eval_expr_full(self, expr: str, seg_id: str, t: float) -> float:
        """求值表达式，符号表包含所有对象的状态变量 + derived + 参数 + t。"""
        import sympy as sp

        # 构建完整的符号表
        symbol_names = set(self.cached_seg_symbols_map[seg_id].keys())

        # 添加其它对象的状态变量
        for obj_id in self.state_funcs[seg_id]:
            symbol_names.update(self.state_funcs[seg_id][obj_id].keys())

        # 添加段参数
        segment_parameters = self.segment_parameters.get(seg_id, {})
        symbol_names.update(segment_parameters.keys())

        locals_map = {name: sp.Symbol(name) for name in symbol_names}
        parsed = sp.sympify(expr, locals=locals_map)
        names = tuple(sorted(str(symbol) for symbol in parsed.free_symbols))

        # 逐个求值
        values = []
        for name in names:
            if name == "t":
                values.append(t)
            elif name in segment_parameters:
                values.append(float(segment_parameters[name]))
            else:
                values.append(self.value_at_time(name, t))

        symbols = [sp.Symbol(name) for name in names]
        func = sp.lambdify(symbols, parsed, modules="numpy")
        return float(func(*values))

    def get_entities(self, tracker: ValueTracker) -> list[Mobject]:
        #================================物理实体开始================================    
        mobs = []
        # 给mobject添加更新器
        for obj_id, obj in self.objects.items():
            if obj.mobject is None:
                raise ValueError(f"Object '{obj_id}' has no mobject for rendering.")

            mob = obj.mobject
            mobs.append(mob)

            cartesian_position_variables = obj.cartesian_position_variables()

            callbacks = None

            if len(cartesian_position_variables) > 1:
                if hasattr(mob, "point_change_callbacks"):
                    callbacks = mob.point_change_callbacks()

                    if len(callbacks) != len(cartesian_position_variables):
                        raise ValueError(
                            f"Object '{obj_id}' has {len(cartesian_position_variables)} "
                            f"cartesian positions, but its mobject provides {len(callbacks)} callbacks."
                        )
                else:
                    raise TypeError(
                        f"Object '{obj_id}' uses multiple cartesian positions, "
                        "but its mobject implements neither set_control_points() "
                        "nor point_change_callbacks()."
                    )
            elif len(cartesian_position_variables) == 1:
                if hasattr(mob, "point_change_callbacks"):
                    callbacks = mob.point_change_callbacks()

                    if len(callbacks) != 1:
                        raise ValueError(
                            f"Object '{obj_id}' has 1 cartesian position, "
                            f"but its mobject provides {len(callbacks)} callbacks."
                        )

            # 创建轨迹 mobject（如果开启了轨迹追踪）
            trace_mobs = self._create_trace_mobs(obj)

            # 给物理实体添加updater
            mob.add_updater(self.make_obj_position_updater(obj_id, callbacks, tracker, trace_mobs))

            # 轨迹 mobject 需要在物理实体之后添加，确保渲染层级
            mobs.extend(trace_mobs)

        return mobs

    def _create_trace_mobs(self, obj: PhysicObject2D) -> list[Mobject]:
        """为开启了轨迹追踪的对象创建轨迹 VMobject。

        每个笛卡尔坐标点对应一条独立的轨迹线。
        """
        from manim import VMobject
        from phyanim.core.objects import TraceConfig

        if obj.trace_config is None:
            return []

        config = obj.trace_config
        n_points = len(obj.cartesian_position_variables())
        trace_mobs = []
        for _ in range(n_points):
            line = VMobject()
            line.set_stroke(color=config.color, width=config.stroke_width)
            line.set_opacity(config.opacity)
            # 初始为一个不可见的退化线段（单点）
            line.start_new_path(np.array([0.0, 0.0, 0.0]))
            line.add_line_to(np.array([0.0, 0.0, 0.0]))
            trace_mobs.append(line)
        return trace_mobs

    # 普通物理对象
    def make_obj_position_updater(
        self,
        current_obj_id: str,
        current_callbacks=None,
        tracker: ValueTracker = None,
        trace_mobs: list[Mobject] | None = None,
    ):
        obj = self.objects[current_obj_id]
        # 预编译 visual_bindings 的表达式
        binding_compiled = {}
        if obj.visual_bindings:
            for binding in obj.visual_bindings:
                symbol_names = set(binding.variables) | {"t"}
                compiler = SympyExpressionCompiler()
                binding_compiled[id(binding)] = compiler.compile(binding.expression, symbol_names=symbol_names)

        # 轨迹追踪状态：每个笛卡尔坐标点维护一个 (physics_t, x, y) 列表
        trace_points: list[list[tuple[float, float, float]]] | None = None
        if trace_mobs:
            trace_points = [[] for _ in trace_mobs]

        def updater(m):
            t = tracker.get_value()
            physics_t = self.render_to_physics_mapping_func(t)
            cartesian_positions = self.eval_position(current_obj_id, physics_t)

            if current_callbacks is not None:
                for callback, position in zip(current_callbacks, cartesian_positions):
                    x, y = position
                    callback(x, y)
            elif len(cartesian_positions) == 1:
                x, y = cartesian_positions[0]
                m.move_to(np.array([x, y, 0.0]))

            # Visual bindings: 更新颜色/透明度/缩放/旋转等视觉属性
            for binding in obj.visual_bindings:
                expr = binding_compiled[id(binding)]
                # 收集变量值
                state_dict = {}
                for var_name in binding.variables:
                    try:
                        state_dict[var_name] = self.value_at_time(var_name, physics_t)
                    except (KeyError, ValueError):
                        pass
                val = expr.evaluate(state_dict, {}, physics_t)

                if binding.attribute == "color":
                    color = _resolve_color(val)
                    if color is not None:
                        m.set_color(color)
                elif binding.attribute == "opacity":
                    m.set_opacity(float(val))
                elif binding.attribute == "stroke_width":
                    m.set_stroke(width=float(val))
                elif binding.attribute == "scale":
                    _apply_scale(m, float(val))
                elif binding.attribute == "rotation":
                    _apply_rotation(m, float(val), binding.center)

            for callback, position in zip(current_callbacks or [], cartesian_positions):
                x, y = position
                callback(x, y)

            # 轨迹追踪更新
            if trace_mobs and trace_points is not None:
                self._update_trace(
                    obj, trace_mobs, trace_points, cartesian_positions, physics_t
                )
        return updater

    def _update_trace(
        self,
        obj: PhysicObject2D,
        trace_mobs: list[Mobject],
        trace_points: list[list[tuple[float, float, float]]],
        cartesian_positions: list[tuple[float, float]],
        physics_t: float,
    ) -> None:
        """每帧更新轨迹线。

        追加当前物理时刻的位置采样点，tail 模式下裁剪超出 keeping_t 窗口的旧点。
        """
        from manim import VMobject

        config = obj.trace_config
        if config is None:
            return

        for i, (x, y) in enumerate(cartesian_positions):
            pts = trace_points[i]
            pts.append((physics_t, float(x), float(y)))

            # tail 模式：裁剪掉 keeping_t 之前的点
            if config.mode == "tail":
                cutoff = physics_t - config.keeping_t
                # 保留第一个 >= cutoff 的点之前的一个点，保证线段连续
                while len(pts) > 2 and pts[1][0] < cutoff:
                    pts.pop(0)
                # 如果第一个点也太旧，用 cutoff 处的插值替换
                if len(pts) > 1 and pts[0][0] < cutoff:
                    t0, x0, y0 = pts[0]
                    t1, x1, y1 = pts[1]
                    if t1 > t0:
                        alpha = (cutoff - t0) / (t1 - t0)
                        pts[0] = (cutoff, x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0))

            # 构建 manim 点序列
            if len(pts) < 2:
                continue

            mob = trace_mobs[i]
            points = [np.array([px, py, 0.0]) for _, px, py in pts]
            # VMobject 的 set_points_as_corners 会自动闭合路径（末点→首点），
            # 闭合后 fill 会渲染出一个面。通过 set_fill(opacity=0) 关闭填充，
            # 这样即使路径闭合也只显示描边（轨迹线），不显示填充面。
            mob.set_points_as_corners(points)
            mob.set_fill(opacity=0)
            mob.set_stroke(color=config.color, width=config.stroke_width, opacity=config.opacity)


# 注释上下文，需要使用AnimationContext和annotationsolver来进行初始化
from phyanim.core.enhance.annotation import Annotation, ArrowContent, TextContent, MathTexContent
from phyanim.core.enhance.transition import TransitionContent
class AnnotationContext(Context):
    """注释上下文，管理整个动画的注释。"""
    def __init__(self, 
        annotations: dict[str, Annotation], 
        timeline: dict[str, list[Tuple[float, float]]], 
        value_at_time: Callable[[str, float], float],
        eval_expr: Callable[[str, str, float], float],
        build_eval_exper_func: Callable[[Trigger], callable],
        name_space: str
    ):
        super().__init__()
        self.timeline = timeline
        self.annotations = annotations
        self.value_at_time = value_at_time
        self.eval_expr = eval_expr
        self.build_eval_exper_func = build_eval_exper_func
        self.name_space = name_space

        # 已经触发的注释字典，键为注释规格，值为触发时间列表（在渲染时使用，记录注释触发的次数等上下文）
        self.triggered_annotations = {}

    def make_anno_position_updater(
        self,
        anno_id: str,
        content,
        callback=None,
        tracker: ValueTracker = None,
        physics_ctx: PhysicsContext = None,
    ):
        def updater(m):
            t = tracker.get_value()
            physics_t = self.render_to_physics_mapping_func(t)

            # 这里待优化，变量r需要自适应，因为timeline是根据space_name而不同的
            if self.name_space == "physics":
                t = physics_t
            elif self.name_space == "render":
                t = t
                
            timeline = self.timeline.get(anno_id, [])
            visible = any(start <= t <= end for start, end in timeline)
            m.set_opacity(1.0 if visible else 0.0)

            if not visible:
                return

            x_pos_name, y_pos_name = content.pos_variables

            x_pos_value, y_pos_value = None, None
            if not is_numeric(x_pos_name):
                if self.name_space == "physics":
                    x_pos_value = self.value_at_time(x_pos_name, t)
                elif self.name_space == "render":
                    x_pos_value = self.value_at_time(x_pos_name, physics_t, t, physics_ctx)
            else:
                x_pos_value = float(x_pos_name)
            if not is_numeric(y_pos_name):
                if self.name_space == "physics":
                    y_pos_value = self.value_at_time(y_pos_name, t)
                elif self.name_space == "render":
                    y_pos_value = self.value_at_time(y_pos_name, physics_t, t, physics_ctx)
            else:
                y_pos_value = float(y_pos_name)

            if isinstance(content, ArrowContent):
                x_shift_name, y_shift_name = content.shift_variables
                x_shift_value, y_shift_value = None, None
                if not is_numeric(x_shift_name):
                    if self.name_space == "physics":
                        x_shift_value = self.value_at_time(x_shift_name, t)
                    elif self.name_space == "render":
                        x_shift_value = self.value_at_time(x_shift_name, physics_t, t, physics_ctx)
                else:
                    x_shift_value = float(x_shift_name)
                if not is_numeric(y_shift_name):
                    if self.name_space == "physics":
                        y_shift_value = self.value_at_time(y_shift_name, t)
                    elif self.name_space == "render":
                        y_shift_value = self.value_at_time(y_shift_name, physics_t, t, physics_ctx)
                else:
                    y_shift_value = float(y_shift_name)
                callback(x_pos_value, y_pos_value, x_shift_value, y_shift_value)
                
            elif isinstance(content, (MathTexContent, TextContent)):
                callback(x_pos_value, y_pos_value)

        return updater

    def get_entities(self, tracker: ValueTracker, physics_ctx: PhysicsContext) -> list[Mobject]:
        #================================注释实体开始================================
        entities = []
        for anno_id, anno in self.annotations.items():
            # 这里的TransitionContent 需要单独运镜处理
            if isinstance(anno.content, TransitionContent):
                continue
            entities.append(anno.content.obj)
            call_back = anno.content.change_callback()
            anno.content.obj.add_updater(self.make_anno_position_updater(anno_id, anno.content, call_back, tracker, physics_ctx))
        return entities
    
class TransitionContext(Context):
    """转换上下文，管理整个动画的转换。"""
    def __init__(self, 
        transitions: dict[str, Transition], 
        timeline: dict[str, list[float]], 
        value_at_time: Callable[[str, float], float],
        eval_expr: Callable[[str, str, float], float],
        build_eval_exper_func: Callable[[Trigger], callable],
        name_space: str
    ):
        super().__init__()
        self.transitions = transitions
        self.timeline = timeline
        
        self.triggered_transitions = {}
        self.value_at_time = value_at_time
        self.eval_expr = eval_expr
        self.build_eval_exper_func = build_eval_exper_func
        self.name_space = name_space

    def make_sequential(self, transition, groups, times, tracker, physics_ctx: PhysicsContext, style="scale", duration=0.5, smooth_func : Callable[[float], float] = lambda x: x * x * (3 - 2 * x)):
        """
        根据 tracker 在多个公式之间依次切换，支持多种动画风格。
        formulas: [mob_a, mob_b, mob_c, mob_d]
        times:    [2, 4, 6, 7, 8]  — 各切换时间点
        style:    动画风格:
            "fade"        — 纯淡入淡出
            "scale"       — 缩成一点再扩散（推荐）
            "slide_left"  — 左滑出 / 右滑入
            "slide_down"  — 下滑出 / 上滑入
            "spin"        — 旋转缩放出 / 旋转放大入
        duration: 过渡持续时间（秒）
        """
        container = VGroup(*groups)
        n = len(groups)
        refs = [f.copy() for f in groups]

        def apply_out(mob, ref, alpha):
            """alpha: 0=完全可见, 1=完全消失"""
            a = smooth_func(alpha)
            if style == "fade":
                mob.set_opacity(1 - a)
            elif style == "scale":
                s = 1 - a
                mob.scale(s)
                mob.set_opacity(max(0, 1 - a * 1.5))
            elif style == "slide_left":
                mob.shift(LEFT * a * 2)
                mob.set_opacity(1 - a)
            elif style == "slide_down":
                mob.shift(DOWN * a * 2)
                mob.set_opacity(1 - a)
            elif style == "spin":
                s = 1 - a
                mob.scale(s)
                mob.rotate(a * PI / 2)
                mob.set_opacity(max(0, 1 - a * 1.5))

        def apply_in(mob, ref, alpha):
            """alpha: 0=完全不可见, 1=完全可见"""
            a = smooth_func(alpha)
            if style == "fade":
                mob.set_opacity(a)
            elif style == "scale":
                mob.scale(max(0.01, a))
                mob.set_opacity(min(1, a * 1.5))
            elif style == "slide_left":
                mob.shift(RIGHT * (1 - a) * 2)
                mob.set_opacity(a)
            elif style == "slide_down":
                mob.shift(UP * (1 - a) * 2)
                mob.set_opacity(a)
            elif style == "spin":
                mob.scale(max(0.01, a))
                mob.rotate(-(1 - a) * PI / 2)
                mob.set_opacity(min(1, a * 1.5))

        # 这里m是最大的那个group(group1(str1,str2), group2(str3,str4,str5,str6))
        def updater(m):
            t = tracker.get_value()
            physics_t = self.render_to_physics_mapping_func(t)
            if self.name_space == "physics":
                t = physics_t
            elif self.name_space == "render":
                t = t
            
            x_pos_name, y_pos_name = transition.content.pos_variables

            x_pos_value, y_pos_value = None, None
            if not is_numeric(x_pos_name):
                if self.name_space == "physics":
                    x_pos_value = self.value_at_time(x_pos_name, t)
                elif self.name_space == "render":
                    x_pos_value = self.value_at_time(x_pos_name, physics_t, t, physics_ctx)
            else:
                x_pos_value = float(x_pos_name)
            if not is_numeric(y_pos_name):
                if self.name_space == "physics":
                    y_pos_value = self.value_at_time(y_pos_name, t)
                elif self.name_space == "render":
                    y_pos_value = self.value_at_time(y_pos_name, physics_t, t, physics_ctx)
            else:
                y_pos_value = float(y_pos_name)

            target_pos = np.array([x_pos_value, y_pos_value, 0.0])

            for i in range(n):
                mob = m.submobjects[i]
                ref = refs[i]
                mob.become(ref)  # 每帧重置，避免变换累积
                mob.move_to(target_pos)

                # 开始出现
                fi_start = times[i] - duration
                # 完全出现
                fi_end = times[i]
                # 开始消失
                fo_start = times[i + 1] - duration
                # 完全消失
                fo_end = times[i + 1]

                # 处理区间重合的几种情况
                # 如果完全出现的时刻等于完全消失的时刻，则设置从fi_start到fo_end的先显示再消失
                if fo_start <= fi_end and fi_end <= fo_end:
                    if t < fi_start:
                        mob.set_opacity(0)
                    elif t <= fo_end:
                        # 一半的duration用来出现，一半用来消失
                        span = (fo_end + fi_start) / 2
                        if t <= span:
                            apply_in(mob, ref, (t - fi_start) / (span - fi_start))
                        else:
                            apply_out(mob, ref, (t - span) / (fo_end - span))
                    else:
                        mob.set_opacity(0)
                elif fi_end < fo_start:
                    if t < fi_start:
                        mob.set_opacity(0)
                    elif t <= fi_end:
                        apply_in(mob, ref, (t - fi_start) / duration)
                    elif t < fo_start:
                        mob.set_opacity(1)
                    elif t <= fo_end:
                        apply_out(mob, ref, (t - fo_start) / duration)
                    else:
                        mob.set_opacity(0)
                        

        container.add_updater(updater)
        return container


    def get_entities(self, tracker: ValueTracker, physics_ctx: PhysicsContext) -> list[Mobject]:
        #================================transition开始================================
        # 执行annotation transition动画
        entities = []
        for trans_id, transition in self.transitions.items():
            groups = transition.content.groups
            group_container = self.make_sequential(transition, groups, self.timeline[trans_id], tracker, physics_ctx, transition.style, transition.duration, transition.smooth_func)
            entities.append(group_container)
        return entities


# ---------------------------------------------------------------------------
# Visual binding helpers
# ---------------------------------------------------------------------------

# 颜色名到 manim 颜色对象的映射表，供 visual_binding 的 color 属性使用。
_COLOR_MAP = {
    "red": RED, "RED": RED,
    "green": GREEN, "GREEN": GREEN,
    "blue": BLUE, "BLUE": BLUE,
    "white": WHITE, "WHITE": WHITE,
    "yellow": YELLOW, "YELLOW": YELLOW,
    "yellow_c": YELLOW_C, "YELLOW_C": YELLOW_C,
    "red_c": RED_C, "RED_C": RED_C,
    "blue_c": BLUE_C, "BLUE_C": BLUE_C,
    "green_c": GREEN_C, "GREEN_C": GREEN_C,
    "orange": ORANGE, "ORANGE": ORANGE,
    "purple": PURPLE, "PURPLE": PURPLE,
    "teal_c": TEAL_C, "TEAL_C": TEAL_C,
    "gold_c": GOLD_C, "GOLD_C": GOLD_C,
    "maroon_c": MAROON_C, "MAROON_C": MAROON_C,
    "pure_red": PURE_RED, "PURE_RED": PURE_RED,
    "pure_green": PURE_GREEN, "PURE_GREEN": PURE_GREEN,
    "pure_blue": PURE_BLUE, "PURE_BLUE": PURE_BLUE,
}


def _resolve_color(val) -> object | None:
    """把 visual binding 表达式返回值解析为 manim 颜色对象。

    支持两种形式：
    - 字符串颜色名（如 "RED"、"blue"）
    - 数值（取整后作为颜色索引，暂不支持）
    """
    if isinstance(val, str):
        return _COLOR_MAP.get(val)
    if isinstance(val, (int, float)):
        # 数值模式暂不支持，预留接口
        return None
    return None


# 每个 mobject 上记录上一次 scale/rotation 值的属性名
_SCALE_ATTR = "_phyanim_last_scale"
_ROT_ATTR = "_phyanim_last_rotation"


def _apply_scale(m, target_scale: float) -> None:
    """相对缩放：基于上一次的 scale 值计算缩放比。"""
    if target_scale <= 0:
        target_scale = 0.01
    last = getattr(m, _SCALE_ATTR, 1.0)
    ratio = target_scale / last
    m.scale(ratio)
    setattr(m, _SCALE_ATTR, target_scale)


def _apply_rotation(m, target_angle: float, center=None) -> None:
    """相对旋转：基于上一次的角度值计算旋转增量。"""
    last = getattr(m, _ROT_ATTR, 0.0)
    delta = target_angle - last
    if abs(delta) > 1e-10:
        if center is not None:
            m.rotate(delta, about_point=np.array([center[0], center[1], 0.0]))
        else:
            m.rotate(delta)
    setattr(m, _ROT_ATTR, target_angle)
