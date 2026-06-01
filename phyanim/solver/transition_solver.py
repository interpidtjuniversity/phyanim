from dataclasses import dataclass, field

from phyanim.core.context import PhysicsContext
from phyanim.core.context import TransitionContext
from phyanim.solver.solver import ContentSolver


from phyanim.core.enhance.transition import Transition

import numpy as np

@dataclass
class DefaultTransitionSolver(ContentSolver):

    timeline: dict[str, list[float]] = field(default_factory=dict)
    transitions: dict[str, Transition] = field(default_factory=dict)
    sample_dt: float = 1/60

    def solve_physics(self, physics_ctx: PhysicsContext) -> TransitionContext:
        old_time = None

        transition_triggers = {}
        for transition_id, transition in self.transitions.items():
            transition_triggers[transition_id] = len(transition.triggers)

        for t in physics_ctx.times:
            # 评估某个事件是否已经开始
            for transition_id, transition in self.transitions.items():
                triggers = transition.triggers

                for trigger in triggers:
                    event_t = trigger(old_time, t, physics_ctx.build_eval_exper_func(trigger))
                    if event_t is not None:
                        self.timeline.setdefault(transition_id, []).append(event_t)

            old_time = t
            
        
        # 收尾时间，看有没有还没触发的trigger
        for transition_id, transition in self.transitions.items():
            left_count = transition_triggers[transition_id] - len(self.timeline.get(transition_id, []))
            if left_count > 0:
                # 没触发的都累计到最后一个时间
                self.timeline.setdefault(transition_id, []).extend([physics_ctx.total_time] * left_count)
            
            # 排序
            self.timeline[transition_id].sort()
        
        return TransitionContext(self.transitions, self.timeline, physics_ctx.value_at_time, physics_ctx.eval_expr, physics_ctx.build_eval_exper_func, "physics")

    # render_range_map: time_wrapper的触发区间
    # transition_belongs_map: transition所属的time_wrapper
    """
    transition的求解必须在对应的time_wrapper的区间内, 否则是无意义的
    """
    def solve_render(self, physics_ctx: PhysicsContext, time_wrapper_ranges: list[tuple[float, float, str]], render_range_map: dict[str, dict[str, tuple[float, float]]], transition_belongs_map: dict[str, str]) -> TransitionContext:
        physics_start_time = physics_ctx.times[0]
        physics_end_time = physics_ctx.times[-1]
        render_start_time = physics_ctx.physics_to_render_mapping_func(physics_start_time)
        render_end_time = physics_ctx.physics_to_render_mapping_func(physics_end_time)
        render_times = np.arange(render_start_time, render_end_time, self.sample_dt, dtype=float)
        # 渲染层的时间是物理层的时间映射
        old_time = None

        transition_triggers = {}
        for transition_id, transition in self.transitions.items():
            transition_triggers[transition_id] = len(transition.triggers)

        for render_t in render_times:
            # 评估某个事件是否已经开始
            for transition_id, transition in self.transitions.items():
                wrapper_id = transition_belongs_map[transition_id]
                render_range = render_range_map[wrapper_id]["render_range"]
                triggers = transition.triggers

                for trigger in triggers:
                    event_t = trigger(old_time, render_t, self.render_build_eval_exper_func(trigger, physics_ctx, time_wrapper_ranges))
                    # 确保在对应的wrapper区间以内
                    if event_t is not None and render_range[0] <= render_t <= render_range[1]:
                        self.timeline.setdefault(transition_id, []).append(event_t)

            old_time = render_t
            
        
        # 收尾时间，看有没有还没触发的trigger
        for transition_id, transition in self.transitions.items():
            left_count = transition_triggers[transition_id] - len(self.timeline.get(transition_id, []))
            if left_count > 0:
                # 没触发的都累计到对应wrapper的最后一个时间
                wrapper_id = transition_belongs_map[transition_id]
                render_range = render_range_map[wrapper_id]["render_range"]
                self.timeline.setdefault(transition_id, []).extend([render_range[1]] * left_count)
            
            # 排序
            self.timeline[transition_id].sort()
        
        return TransitionContext(self.transitions, self.timeline, self.render_value_at_time, self.render_eval_expr, self.render_build_eval_exper_func, "render")
