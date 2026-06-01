from typing import Any

from phyanim.core.expressions import CompiledExpression

class Solver:
    def __init__(self):
        pass

    def _split_state_history_by_object(self,segment: Any,state_history: dict[str, list[float]]) -> dict[str, dict[str, list[float]]]:
        object_states = {object_id: {} for object_id in segment.object_ids or []}

        if segment.state_owners is None:
            raise ValueError(f"Segment '{segment.segment_id}' has no state_owners.")

        for name, values in state_history.items():
            object_id = segment.state_owners[name]
            object_states[object_id][name] = values

        return object_states


from phyanim.core.context import PhysicsContext
import sympy as sp
from phyanim.core.enhance.trigger import Trigger, CrossingTrigger

class ContentSolver:
    def __init__(self):
        pass
    
    # 这里加入local_t，如果当前render_t在某一个缩放段内(render_start, render_end)，则local_t = render_t - render_start
    # local_t 是局部动画的本地时间
    def render_value_at_time(self, name: str, physics_t: float, render_t: float, physics_ctx: PhysicsContext) -> float:
        if name == "t":
            return render_t
        """评估 name 在时间 render_t 处的数值。"""
        tra_idx = physics_ctx.find_trajectory_index(physics_t)
        seg_id = physics_ctx.tra_map[tra_idx]

        # 先查缓存
        seg_states_func = physics_ctx.cached_seg_funcs.setdefault(seg_id, {})
        if name in seg_states_func:
            return seg_states_func[name](physics_t)

        for obj_id in physics_ctx.state_funcs[seg_id].keys():
            for state_name in physics_ctx.state_funcs[seg_id][obj_id].keys():
                if state_name == name:
                    seg_states_func[name] = physics_ctx.state_funcs[seg_id][obj_id][name]
                    return seg_states_func[name](physics_t)

        for derived_name in physics_ctx.derived_funcs[seg_id].keys():
            if derived_name == name:
                seg_states_func[name] = physics_ctx.derived_funcs[seg_id][derived_name]
                return seg_states_func[name](physics_t)
        
        raise ValueError(
            f"Variable {name} not found in any state or derived function at segment {seg_id}."
        )

    def render_eval_expr(self, expr: str, return_type: str, physics_t: float, render_t: float, physics_ctx: PhysicsContext, time_wrapper_ranges: list[tuple[float, float, str]]) -> float:
        """评估 expr 在时间 render_t 处的数值。"""
        tra_idx = physics_ctx.find_trajectory_index(physics_t)
        seg_id = physics_ctx.tra_map[tra_idx]

        if expr not in physics_ctx.cached_expressions:
            parsed = sp.sympify(expr, locals=physics_ctx.cached_seg_symbols_map[seg_id])
            names = tuple(sorted(str(symbol) for symbol in parsed.free_symbols))
            symbols = [sp.Symbol(name) for name in names]
            function = sp.lambdify(symbols, parsed, modules="numpy")
            physics_ctx.cached_expressions[expr] = {
                "names": names,
                "compiled_expression": CompiledExpression(expression=expr, names=names, function=function),
            }
        
        compiled_expression = physics_ctx.cached_expressions[expr]["compiled_expression"]
        names = physics_ctx.cached_expressions[expr]["names"]
        values = []
        # 除了时间变量，其它都映射回去物理层的值
        for name in names:
            if name == "t":
                values.append(render_t)
            elif name == "local_t":
                # 查找区间
                range_idx = -1
                for idx, time_wrapper_range in enumerate(time_wrapper_ranges):
                    if time_wrapper_range[0] <= render_t <= time_wrapper_range[1]:
                        range_idx = idx
                        break
                if range_idx >= 0:
                    values.append(render_t - time_wrapper_ranges[range_idx][0])
                else:
                    # local_t在内部，大概率不会触发边界处的检查
                    # local_t只有大于0才有意义
                    values.append(-1)
            elif name == "physics_t":
                values.append(physics_t)
            else:
                # 其它变量表达式统一在physics_t层评估
                values.append(physics_ctx.value_at_time(name, physics_t))
        
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
    def render_build_eval_exper_func(self, trigger: Trigger, physics_ctx: PhysicsContext, time_wrapper_ranges: list[tuple[float, float, str]]) -> callable:
        """根据 trigger 构建评估函数。"""
        if isinstance(trigger, CrossingTrigger):
            def function(render_t: float) -> float:
                physics_t = physics_ctx.render_to_physics_mapping_func(render_t)
                return self.render_eval_expr(trigger.expression, "float", physics_t, render_t, physics_ctx, time_wrapper_ranges)
            return function
        else:
            return None