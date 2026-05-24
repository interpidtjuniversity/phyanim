from __future__ import annotations

import numpy as np
from typing import Tuple

from manim import Scene, ValueTracker, linear

from phyanim.core.animation import PhysicsAnimation

class PhyAnimationScene2D(Scene):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.animation: PhysicsAnimation | None = None

        self.state_funcs = None
        self.derived_funcs = None

        self.start_times = []
        self.end_times = []
        self.tra_map = {}
        self.total_time = 0.0

    def set_animation(self, animation: PhysicsAnimation) -> None:
        """注册一个已求解的 PhysicsAnimation。"""
        self.animation = animation
        self.state_funcs = animation.build_state_functions()
        self.derived_funcs = animation.build_derived_functions()

        self.start_times = []
        self.end_times = []
        self.tra_map = {}

        for tra_idx, tra in enumerate(animation.trajectories):
            self.start_times.append(tra.times[0])
            self.end_times.append(tra.times[-1])
            self.tra_map[tra_idx] = tra.segment_id

        self.total_time = max(self.end_times)

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

    def eval_position(self, obj_id: str, t: float) -> Tuple[float, float]:
        """评估 obj_id 在时间 t 处的位置，返回具体数值 x, y。"""

        tra_idx = self.find_trajectory_index(t)
        seg_id = self.tra_map[tra_idx]

        obj = self.animation.objects[obj_id]

        obj_seg_state_funcs = self.state_funcs[seg_id][obj_id]
        derived_funcs = self.derived_funcs[seg_id]

        all_states_deriveds = {}
        all_states_deriveds.update(obj_seg_state_funcs)
        all_states_deriveds.update(derived_funcs)

        x_name, y_name = obj.cartesian_position_variables()

        if x_name not in all_states_deriveds or y_name not in all_states_deriveds:
            raise ValueError(
                f"Object {obj_id} does not have position variables "
                f"{x_name} and {y_name}"
            )

        x_func = all_states_deriveds[x_name]
        y_func = all_states_deriveds[y_name]

        return float(x_func(t)), float(y_func(t))
    
    def set_frame_size(self, width: float, height: float) -> None:
        self.frame_width = width
        self.frame_height = height

    def construct(self) -> None:
        tracker = ValueTracker(0.0)

        for obj_id, obj in self.animation.objects.items():
            if obj.mobject is None:
                raise ValueError(f"Object '{obj_id}' has no mobject for rendering.")
            self.add(obj.mobject)

            def make_updater(current_obj_id: str):
                def updater(m):
                    t = tracker.get_value()
                    x, y = self.eval_position(current_obj_id, t)
                    m.move_to(np.array([x, y, 0.0]))
                return updater

            obj.mobject.add_updater(make_updater(obj_id))

        self.play(
            tracker.animate.set_value(self.total_time),
            run_time=self.total_time,
            rate_func=linear,
        )

        for obj in self.animation.objects.values():
            if obj.mobject is not None:
                obj.mobject.clear_updaters()
