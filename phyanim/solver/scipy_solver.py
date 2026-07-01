from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from phyanim.core.events import PhysicsEvent
from phyanim.core.keyframe import PhysicsKeyFrame
from phyanim.core.segment import PhysicsSegment
from phyanim.core.solution import SegmentResult, SegmentSolution, StateFunction
from phyanim.core.trajectory import Trajectory

from phyanim.solver.solver import Solver


@dataclass
class ScipySegmentSolver(Solver):
    """基于 solve_ivp 的高精度连续段求解器。"""

    method: str = "DOP853"
    rtol: float = 1e-10
    atol: float = 1e-12
    max_step: float | None = 0.001
    sample_dt: float | None = None
    clamp_functions: bool = False
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
            from scipy.integrate import solve_ivp
        except ImportError as exc:
            raise RuntimeError(
                "ScipySegmentSolver requires scipy and numpy."
            ) from exc
        
        # 从关键帧构建初始状态变量
        state_dict = segment.flatten_keyframe_state(start_keyframe.object_states)
        missing = [name for name in segment.state_vector if name not in state_dict]
        if missing:
            raise ValueError(f"Segment '{segment.segment_id}' missing state variables: {missing}")

        # 初始状态向量
        y0 = np.array([state_dict[name] for name in segment.state_vector], dtype=float)
        t0 = float(start_keyframe.time)
        # 预计结束时间，也就是积分上限（积分可能被事件打断）
        planned_t1 = float(t0 + segment.duration)
        # 保存映射关系
        name_to_index = {name: index for index, name in enumerate(segment.state_vector)}
        # 从状态向量到状态表
        def array_to_state(y: Any) -> dict[str, float]:
            return {name: float(y[index]) for name, index in name_to_index.items()}

        derivative = segment.build_derivative(parameters)
        derived_functions = segment.build_derived_quantities(parameters)

        # 根据某一时刻状态值获取状态的导数值
        def rhs(time: float, y: Any) -> Any:
            state = array_to_state(y)
            derivative_values = derivative(float(time), state, dict(parameters))
            return np.array([derivative_values[name] for name in segment.state_vector], dtype=float)

        # 事件表达式可以引用状态变量、derived 变量、参数、t
        symbol_names = set(segment.state_vector) | set(parameters) | set(segment.derived_equations) | {"t"}
        event_function = end_event.condition.build_function(symbol_names=symbol_names)

        def make_event(current_event, current_function):
            def scipy_event(time: float, y: Any) -> float:
                state = array_to_state(y)
                # 实时计算 derived 变量，合并到 state 中供事件表达式使用
                for dname, dfunc in derived_functions.items():
                    state[dname] = float(dfunc(float(time), dict(state), dict(parameters)))
                return float(current_function(float(time), state, dict(parameters)))

            # 事件方向直接交给 solve_ivp：-1 表示正到负，1 表示负到正，0 表示任意方向。
            scipy_event.terminal = current_event.condition.terminal
            scipy_event.direction = current_event.condition.direction
            return scipy_event

        scipy_event = make_event(end_event, event_function)

        kwargs: dict[str, Any] = {
            "fun": rhs,
            "t_span": (t0, planned_t1),
            "y0": y0,
            "method": self.method,
            "rtol": self.rtol,
            "atol": self.atol,
            "dense_output": True,
            "events": [scipy_event],
        }
        if self.max_step is not None:
            kwargs["max_step"] = self.max_step

        result = solve_ivp(**kwargs)
        if not result.success:
            raise RuntimeError(
                f"SciPy solver failed in segment '{segment.segment_id}': {result.message}"
            )

        # 真实结束时间
        actual_t1 = float(result.t[-1])
        # 事件必然要触发，否则分段不合理
        if result.t_events is None or len(result.t_events[0]) == 0:
            raise RuntimeError(
                f"End event '{end_event.name}' was not triggered in segment '{segment.segment_id}'."
            )
        # t_events与传入的 events 列表一一对应。每个元素是一个数组，包含该事件被触发的所有时间点。如果某事件从未触发，对应数组为空。
        event_time = float(result.t_events[0][0])
        if abs(event_time - t0) <= self.initial_event_tolerance:
            raise RuntimeError(
                f"End event '{end_event.name}' triggered at segment start in "
                f"'{segment.segment_id}'. Split/merge segments so the end event "
                "is not already active at t0."
            )
        if abs(event_time - actual_t1) > self.event_time_tolerance:
            raise RuntimeError(
                f"End event '{end_event.name}' time does not match segment end in '{segment.segment_id}'."
            )
        triggered_event_name = end_event.name
        times = self._sample_times(np, result.t, t0, actual_t1, self.sample_dt)
        ys = result.sol(times)

        state_history = {
            name: ys[index].astype(float).tolist()
            for index, name in enumerate(segment.state_vector)
        }
        derived_functions = segment.build_derived_quantities(parameters)
        derived_history = self._sample_derived(segment, times, ys, parameters, derived_functions)

        # 获取末状态
        end_y = result.sol(actual_t1)
        end_state = array_to_state(end_y)
        object_state_history = self._split_state_history_by_object(segment, state_history)
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
        # 生成的结束关键帧
        # end_keyframe = trajectory.end_keyframe(parameters)
        end_keyframe = PhysicsKeyFrame(
            time=actual_t1,
            object_states=end_object_states,
            parameters={name: value for name, value in parameters.items() if name not in ["t_start", "t_end"]},
            segment_id=segment.segment_id,
            event_name=triggered_event_name,
        )
        solution = self._make_solution(
            segment,
            result,
            t0,
            actual_t1,
            parameters,
            derived_functions,
        )
        return SegmentResult(
            trajectory=trajectory,
            solution=solution,
            start_keyframe=start_keyframe,
            end_keyframe=end_keyframe,
            triggered_event=triggered_event_name,
        )

    def _sample_times(
        self,
        np: Any,
        solver_times: Any,
        t0: float,
        t1: float,
        sample_dt: float | None,
    ) -> Any:
        # 采样时间步长为 None 时，返回原始时间点
        if sample_dt is None:
            return solver_times.astype(float)
        # 结束时间小于等于开始时间，返回单个时间点
        if t1 <= t0:
            return np.array([t0], dtype=float)
        times = np.arange(t0, t1, sample_dt, dtype=float)
        if len(times) == 0 or times[-1] < t1:
            times = np.append(times, t1)
        return times.astype(float)

    def _sample_derived(
        self,
        segment: PhysicsSegment,
        times: Any,
        ys: Any,
        parameters: dict[str, float],
        derived_functions: dict[str, Any],
    ) -> dict[str, list[float]]:
        derived_history = {name: [] for name in derived_functions}
        for column, time in enumerate(times):
            # 获取这一时刻的状态值
            state = {
                state_name: float(ys[index, column])
                for index, state_name in enumerate(segment.state_vector)
            }
            for name, function in derived_functions.items():
                derived_history[name].append(float(function(float(time), dict(state), dict(parameters))))
        return derived_history

    def _make_solution(
        self,
        segment: PhysicsSegment,
        result: Any,
        t0: float,
        t1: float,
        parameters: dict[str, float],
        derived_functions_source: dict[str, Any],
    ) -> SegmentSolution:
        domain = (t0, t1)
        state_functions: dict[str, StateFunction] = {}
        for index, name in enumerate(segment.state_vector):
            def make_state_function(state_index: int):
                def evaluate(time: float) -> float:
                    return float(result.sol(time)[state_index])

                return evaluate

            state_functions[name] = StateFunction(
                name=name,
                domain=domain,
                function=make_state_function(index),
                clamp=self.clamp_functions,
            )

        derived_functions: dict[str, StateFunction] = {}
        for derived_name, derived_function in derived_functions_source.items():
            def make_derived_function(function):
                def evaluate(time: float) -> float:
                    y = result.sol(time)
                    state = {
                        state_name: float(y[index])
                        for index, state_name in enumerate(segment.state_vector)
                    }
                    return float(function(float(time), state, dict(parameters)))

                return evaluate

            derived_functions[derived_name] = StateFunction(
                name=derived_name,
                domain=domain,
                function=make_derived_function(derived_function),
                clamp=self.clamp_functions,
            )

        return SegmentSolution(
            segment_id=segment.segment_id,
            object_ids=tuple(segment.object_ids or []),
            state_vector=tuple(segment.state_vector),
            state_owners=dict(segment.state_owners or {}),
            domain=domain,
            state_functions=state_functions,
            derived_functions=derived_functions,
        )
