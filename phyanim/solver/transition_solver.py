from dataclasses import dataclass, field

from phyanim.core.context import PhysicsContext
from phyanim.core.context import TransitionContext

from phyanim.core.enhance.transition import Transition

@dataclass
class DefaultTransitionSolver:

    timeline: dict[str, list[float]] = field(default_factory=dict)
    transitions: dict[str, Transition] = field(default_factory=dict)

    def solve(self, physics_ctx: PhysicsContext) -> TransitionContext:
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
        
        return TransitionContext(self.transitions, self.timeline, physics_ctx.total_time)
