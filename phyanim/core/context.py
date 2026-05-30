from typing import Tuple, Callable

from phyanim.core.trajectory import InterpolatedStateFunction, Trajectory
from phyanim.core.segment import PhysicsSegment
from phyanim.core.expressions import CompiledExpression
from phyanim.core.objects import PhysicObject2D
from phyanim.core.enhance.annotation import Annotation
from phyanim.core.enhance.trigger import CrossingTrigger, Trigger
from phyanim.core.enhance.transition import Transition

import sympy as sp

class Context:

    def __init__(self):
        self.total_time: float | None = None
        self.time_mapping_func: Callable[[float], float] | None = None

    def set_time_mapping_to_physics(self, time_mapping_func: Callable[[float], float]):
        self.time_mapping_func = time_mapping_func

    # context会在渲染层接受一个渲染时刻，不同的context需要将其映射到不同的物理时间
    def time_mapping_to_physics(self, t: float) -> float:
        """将动画时间映射到物理时间。"""
        if self.time_mapping_func is None:
            """将动画时间映射到物理时间。"""
            if t <= self.total_time:
                return t
            else:
                return self.total_time
        return self.time_mapping_func(t)


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
                for name in invalid_symbol_names
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


# 注释上下文，需要使用AnimationContext和annotationsolver来进行初始化
class AnnotationContext(Context):
    """注释上下文，管理整个动画的注释。"""
    def __init__(self, annotations: dict[str, Annotation], timeline: dict[str, list[Tuple[float, float]]], total_time: float):
        self.timeline = timeline
        self.annotations = annotations
        self.total_time = total_time

        # 已经触发的注释字典，键为注释规格，值为触发时间列表（在渲染时使用，记录注释触发的次数等上下文）
        self.triggered_annotations = {}

class TransitionContext(Context):
    """转换上下文，管理整个动画的转换。"""
    def __init__(self, transitions: dict[str, Transition], timeline: dict[str, list[float]], total_time: float):
        self.transitions = transitions
        self.timeline = timeline
        self.total_time = total_time
        
        self.triggered_transitions = {}

from phyanim.core.enhance.timewrapper import TimeWrapper
class TimeWrapperContext(Context):

    def __init__(self, time_wrappers: dict[str, TimeWrapper], total_time: float, time_mapping_func: Callable[[float], float]):
        self.time_wrappers = time_wrappers
        self.total_time = total_time
        self.time_mapping_func = time_mapping_func