from dataclasses import dataclass, field

from phyanim.core.context import PhysicsContext
from phyanim.core.context import TransitionContext
from phyanim.solver.solver import ContentSolver


from phyanim.core.enhance.transition import Transition

@dataclass
class DefaultTransitionSolver(ContentSolver):

    timeline: dict[str, list[float]] = field(default_factory=dict)
    transitions: dict[str, Transition] = field(default_factory=dict)

    def solve(self, physics_ctx: PhysicsContext, name_space: str) -> TransitionContext:
        if name_space == "physics":
            return self.solve_physics(physics_ctx)
        elif name_space == "render":
            return self.solve_render(physics_ctx)

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

    
    def solve_render(self, physics_ctx: PhysicsContext) -> TransitionContext:
        # 渲染层的时间是物理层的时间映射
        old_time = None

        transition_triggers = {}
        for transition_id, transition in self.transitions.items():
            transition_triggers[transition_id] = len(transition.triggers)

        for physics_t in physics_ctx.times:
            render_t = physics_ctx.physics_to_render_mapping_func(physics_t)
            # 评估某个事件是否已经开始
            for transition_id, transition in self.transitions.items():
                triggers = transition.triggers

                for trigger in triggers:
                    event_t = trigger(old_time, render_t, self.render_build_eval_exper_func(trigger))
                    if event_t is not None:
                        self.timeline.setdefault(transition_id, []).append(event_t)

            old_time = render_t
            
        
        # 收尾时间，看有没有还没触发的trigger
        total_time = physics_ctx.physics_to_render_mapping_func(physics_ctx.total_time)
        for transition_id, transition in self.transitions.items():
            left_count = transition_triggers[transition_id] - len(self.timeline.get(transition_id, []))
            if left_count > 0:
                # 没触发的都累计到最后一个时间
                self.timeline.setdefault(transition_id, []).extend([total_time] * left_count)
            
            # 排序
            self.timeline[transition_id].sort()
        
        return TransitionContext(self.transitions, self.timeline, self.render_value_at_time, self.render_eval_expr, self.render_build_eval_exper_func, "render")
