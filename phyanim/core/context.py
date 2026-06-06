from typing import Tuple, Callable

from phyanim.core.trajectory import InterpolatedStateFunction, Trajectory
from phyanim.core.segment import PhysicsSegment
from phyanim.core.expressions import CompiledExpression
from phyanim.core.objects import PhysicObject2D
from phyanim.core.enhance.annotation import Annotation
from phyanim.core.enhance.trigger import CrossingTrigger, Trigger
from phyanim.core.enhance.transition import Transition

import sympy as sp
from manim import ValueTracker, Mobject, VGroup, PI
import numpy as np
from phyanim.utils.util import is_numeric

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
        """评估 obj_id 在时间 t 处的位置，返回具体数值 x, y。"""

        tra_idx = self.find_trajectory_index(t)
        seg_id = self.tra_map[tra_idx]

        obj = self.objects[obj_id]

        obj_seg_state_funcs = self.state_funcs[seg_id][obj_id]
        derived_funcs = self.derived_funcs[seg_id]

        all_states_deriveds = {}
        all_states_deriveds.update(obj_seg_state_funcs)
        all_states_deriveds.update(derived_funcs)

        cartesian_positions = []

        other_obj_state_funcs = {}
        for other_obj_id in self.objects:
            if other_obj_id != obj_id:
                funcs = self.state_funcs[seg_id][other_obj_id]
                other_obj_state_funcs.update(funcs)

        cartesian_position_variables = obj.cartesian_position_variables()
        for cartesian_position_variable in cartesian_position_variables:
            x_name, y_name = cartesian_position_variable
            if x_name not in all_states_deriveds:
                if x_name in other_obj_state_funcs:
                    x_func = other_obj_state_funcs[x_name]
                else:
                    raise ValueError(
                        f"Object {obj_id} position x variables "
                        f"{x_name} not found in other objects."
                    )
            else:
                x_func = all_states_deriveds[x_name]

            if y_name not in all_states_deriveds:
                if y_name in other_obj_state_funcs:
                    y_func = other_obj_state_funcs[y_name]
                else:
                    raise ValueError(
                        f"Object {obj_id} position y variables "
                        f"{y_name} not found in other objects."
                    )
            else: 
                y_func = all_states_deriveds[y_name]

            cartesian_positions.append((float(x_func(t)), float(y_func(t))))

        return cartesian_positions

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
            # 给物理实体添加updater
            mob.add_updater(self.make_obj_position_updater(obj_id, callbacks, tracker))

        return mobs

    # 普通物理对象
    def make_obj_position_updater(
        self,
        current_obj_id: str,
        current_callbacks=None,
        tracker: ValueTracker = None,
    ):
        def updater(m):
            t = tracker.get_value()
            physics_t = self.render_to_physics_mapping_func(t)
            cartesian_positions = self.eval_position(current_obj_id, physics_t)

            if len(cartesian_positions) == 1:
                x, y = cartesian_positions[0]
                m.move_to(np.array([x, y, 0.0]))
                return

            for callback, position in zip(current_callbacks, cartesian_positions):
                x, y = position
                callback(x, y)
        return updater    


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
