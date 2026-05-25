# phyanim/solver/heyoka_solver.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from phyanim.core.events import PhysicsEvent
from phyanim.core.keyframe import PhysicsKeyFrame
from phyanim.core.segment import PhysicsSegment
from phyanim.core.solution import SegmentResult

from typing import Any

import numpy as np

from phyanim.core.solution import SegmentSolution, StateFunction
from phyanim.core.trajectory import Trajectory
from phyanim.solver.solver import Solver



@dataclass
class HeyokaSegmentSolver(Solver):
    """
    基于 heyoka.py 的分段求解器。

    注意：
    1. 需要安装 heyoka.py。
    2. equations 和 event expression 必须能被 heyoka symbolic expression 表示。
    3. 返回的 StateFunction 默认基于采样点线性插值。
    """

    tol: float = 1e-18
    high_accuracy: bool = True
    compact_mode: bool = False
    sample_dt: float | None = None
    clamp_functions: bool = True
    event_time_tolerance: float = 1e-9
    initial_event_tolerance: float = 1e-12

    def solve(
        self,
        segment: PhysicsSegment,
        start_keyframe: PhysicsKeyFrame,
        parameters: dict[str, float],
        end_event: PhysicsEvent,
    ) -> SegmentResult:
        try:
            import numpy as np
            import heyoka as hy
        except ImportError as exc:
            raise RuntimeError(
                "HeyokaSegmentSolver requires heyoka.py and numpy."
            ) from exc

        state_dict = segment.flatten_keyframe_state(start_keyframe.object_states)

        missing = [
            name
            for name in segment.state_vector
            if name not in state_dict
        ]
        if missing:
            raise ValueError(
                f"Segment '{segment.segment_id}' missing state variables: {missing}"
            )

        state_vector = list(segment.state_vector)

        y0 = np.array(
            [state_dict[name] for name in state_vector],
            dtype=float,
        )

        t0 = float(start_keyframe.time)
        planned_t1 = float(t0 + segment.duration)

        # ------------------------------------------------------------
        # 1. 创建 heyoka symbolic variables
        # ------------------------------------------------------------
        hy_vars = hy.make_vars(*state_vector)
        var_map = {
            name: hy_vars[index]
            for index, name in enumerate(state_vector)
        }

        # heyoka 的时间符号通常用 hy.time。
        # 这里为了表达式里可以写 t，把 t 映射到 hy.time。
        namespace: dict[str, Any] = {}
        namespace.update(var_map)
        namespace["t"] = hy.time

        # 参数作为数值常量放进去。
        for name, value in parameters.items():
            namespace[name] = float(value)

        # 常见数学函数。
        namespace.update(
            {
                # 三角函数
                "sin": hy.sin,
                "cos": hy.cos,
                "tan": hy.tan,
                "asin": hy.asin,
                "acos": hy.acos,
                "atan": hy.atan,
                "atan2": hy.atan2,
                # 双曲函数
                "sinh": hy.sinh,
                "cosh": hy.cosh,
                "tanh": hy.tanh,
                "asinh": hy.asinh,
                "acosh": hy.acosh,
                "atanh": hy.atanh,
                # 指数与对数
                "exp": hy.exp,
                "log": hy.log,
                # 幂与根
                "sqrt": hy.sqrt,
            }
        )

        def to_hy_expr(value):
            """
            heyoka.py 新版本对类型比较严格：
            ODE system 必须是 tuple[hy.expression, hy.expression]。
            如果表达式求值结果是普通 float/int，需要显式转成 hy.expression。
            """
            if isinstance(value, hy.expression):
                return value

            if isinstance(value, bool):
                return hy.expression(float(value))

            if isinstance(value, int | float):
                return hy.expression(float(value))

            raise TypeError(
                f"Cannot convert value {value!r} of type {type(value)} to heyoka expression."
            )


        def parse_expr(expr: str):
            value = eval(expr, {"__builtins__": {}}, namespace)
            return to_hy_expr(value)

        # ------------------------------------------------------------
        # 2. 构造 ODE system
        # ------------------------------------------------------------
        sys = []

        for name, var in zip(state_vector, hy_vars):
            if name not in segment.equations:
                raise ValueError(
                    f"Segment '{segment.segment_id}' has no equation for state '{name}'."
                )

            rhs_expr = parse_expr(segment.equations[name])
            sys.append((var, rhs_expr))

        # ------------------------------------------------------------
        # 3. 构造 terminal event
        # ------------------------------------------------------------
        event_expr = parse_expr(end_event.condition.expression)

        direction = self._map_event_direction(hy, end_event.condition.direction)

        triggered = {
            "time": None,
        }

        def terminal_callback(ta, d_sgn):
            triggered["time"] = float(ta.time)
            # False 表示触发后停止 propagate
            return False

        t_event = hy.t_event(
            event_expr,
            callback=terminal_callback,
            direction=direction,
        )

        # ------------------------------------------------------------
        # 4. 构造 integrator
        # ------------------------------------------------------------
        ta = hy.taylor_adaptive(
            sys=sys,
            state=y0.tolist(),
            time=t0,
            tol=self.tol,
            high_accuracy=self.high_accuracy,
            compact_mode=self.compact_mode,
            t_events=[t_event],
        )

        # ------------------------------------------------------------
        # 5. 积分到 planned_t1，期待 terminal event 提前停止
        # ------------------------------------------------------------
        # heyoka 的 propagate_until 会在 terminal event 处停止。
        outcome = ta.propagate_until(planned_t1)

        actual_t1 = float(ta.time)

        if triggered["time"] is None:
            raise RuntimeError(
                f"End event '{end_event.name}' was not triggered in "
                f"segment '{segment.segment_id}'. Heyoka outcome: {outcome}"
            )

        event_time = float(triggered["time"])

        if abs(event_time - t0) <= self.initial_event_tolerance:
            raise RuntimeError(
                f"End event '{end_event.name}' triggered at segment start in "
                f"'{segment.segment_id}'. Split/merge segments so the end event "
                "is not already active at t0."
            )

        if abs(event_time - actual_t1) > self.event_time_tolerance:
            raise RuntimeError(
                f"End event '{end_event.name}' time does not match segment end "
                f"in '{segment.segment_id}'. event_time={event_time}, actual_t1={actual_t1}"
            )

        # ------------------------------------------------------------
        # 6. 重新从 t0 积分到采样网格，生成 trajectory
        # ------------------------------------------------------------
        times = self.sample_times(t0, actual_t1, self.sample_dt)

        ta_sample = hy.taylor_adaptive(
            sys=sys,
            state=y0.tolist(),
            time=t0,
            tol=self.tol,
            high_accuracy=self.high_accuracy,
            compact_mode=self.compact_mode,
        )

        # propagate_grid 返回 tuple，文档示例中状态历史在返回值 [5]
        grid_result = ta_sample.propagate_grid(times)
        y_hist = grid_result[5]

        # y_hist shape: (n_time, n_state)
        ys = np.asarray(y_hist, dtype=float).T

        # ------------------------------------------------------------
        # 7. derived quantities 仍然用你原本的 Python 函数计算
        # ------------------------------------------------------------
        derived_functions = segment.build_derived_quantities(parameters)

        return self.build_segment_result_from_samples(
            segment=segment,
            start_keyframe=start_keyframe,
            parameters=parameters,
            end_event=end_event,
            actual_t1=actual_t1,
            times=times,
            ys=ys,
            state_vector=state_vector,
            derived_functions=derived_functions,
            triggered_event_name=end_event.name,
            clamp_functions=self.clamp_functions,
        )

    def _map_event_direction(self, hy: Any, direction: int):
        if direction > 0:
            return hy.event_direction.positive
        if direction < 0:
            return hy.event_direction.negative
        return hy.event_direction.any



    def sample_times(
        self,
        t0: float,
        t1: float,
        sample_dt: float | None,
        include_t1: bool = True,
    ) -> np.ndarray:
        if sample_dt is None:
            return np.array([t0, t1], dtype=float)

        if t1 <= t0:
            return np.array([t0], dtype=float)

        times = np.arange(t0, t1, sample_dt, dtype=float)

        if include_t1:
            if len(times) == 0 or times[-1] < t1:
                times = np.append(times, t1)

        return times.astype(float)


    def make_interp_function(
        self,
        times: np.ndarray,
        values: np.ndarray,
        clamp: bool = False,
    ):
        times = np.asarray(times, dtype=float)
        values = np.asarray(values, dtype=float)

        t0 = float(times[0])
        t1 = float(times[-1])

        def evaluate(t: float) -> float:
            tt = float(t)

            if clamp:
                if tt <= t0:
                    return float(values[0])
                if tt >= t1:
                    return float(values[-1])
            else:
                if tt < t0 or tt > t1:
                    raise ValueError(
                        f"Time {tt} outside interpolation domain ({t0}, {t1})."
                    )

            return float(np.interp(tt, times, values))

        return evaluate


    def build_segment_result_from_samples(
        self,
        *,
        segment: PhysicsSegment,
        start_keyframe: PhysicsKeyFrame,
        parameters: dict[str, float],
        end_event: Any,
        actual_t1: float,
        times: np.ndarray,
        ys: np.ndarray,
        state_vector: list[str],
        derived_functions: dict[str, Any],
        triggered_event_name: str,
        clamp_functions: bool,
    ) -> SegmentResult:
        """
        ys shape: (n_state, n_time)
        """

        state_history = {
            name: ys[index].astype(float).tolist()
            for index, name in enumerate(state_vector)
        }

        derived_history: dict[str, list[float]] = {
            name: []
            for name in derived_functions
        }

        for column, time in enumerate(times):
            state = {
                state_name: float(ys[index, column])
                for index, state_name in enumerate(state_vector)
            }

            for name, function in derived_functions.items():
                derived_history[name].append(
                    float(function(float(time), dict(state), dict(parameters)))
                )

        object_state_history = self._split_state_history_by_object(segment, state_history)

        end_state = {
            name: float(ys[index, -1])
            for index, name in enumerate(state_vector)
        }

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
            time=float(actual_t1),
            object_states=end_object_states,
            parameters={
                name: value
                for name, value in parameters.items()
                if name not in ["t_start", "t_end"]
            },
            segment_id=segment.segment_id,
            event_name=triggered_event_name,
        )

        domain = (float(times[0]), float(times[-1]))

        state_functions: dict[str, StateFunction] = {}
        for index, name in enumerate(state_vector):
            values = ys[index].astype(float)

            state_functions[name] = StateFunction(
                name=name,
                domain=domain,
                function=self.make_interp_function(times, values, clamp=clamp_functions),
                clamp=clamp_functions,
            )

        derived_state_functions: dict[str, StateFunction] = {}
        for name, values_list in derived_history.items():
            values = np.array(values_list, dtype=float)

            derived_state_functions[name] = StateFunction(
                name=name,
                domain=domain,
                function=self.make_interp_function(times, values, clamp=clamp_functions),
                clamp=clamp_functions,
            )

        solution = SegmentSolution(
            segment_id=segment.segment_id,
            object_ids=tuple(segment.object_ids or []),
            state_vector=tuple(state_vector),
            state_owners=dict(segment.state_owners or {}),
            domain=domain,
            state_functions=state_functions,
            derived_functions=derived_state_functions,
        )

        return SegmentResult(
            trajectory=trajectory,
            solution=solution,
            start_keyframe=start_keyframe,
            end_keyframe=end_keyframe,
            triggered_event=triggered_event_name,
        )
