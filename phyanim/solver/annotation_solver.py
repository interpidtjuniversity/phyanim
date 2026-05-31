from typing import Tuple
from dataclasses import dataclass, field

from phyanim.core.context import PhysicsContext
from phyanim.core.context import AnnotationContext
from phyanim.core.enhance.annotation import Annotation
from phyanim.solver.solver import ContentSolver

from phyanim.core.enhance.annotation import Annotation, ActiveWhile, ActivateEventTimeRange, ActiveBetween

@dataclass
class DefaultAnnotationSolver(ContentSolver):

    timeline: dict[str, list[Tuple[float, float]]] = field(default_factory=dict)
    annotations: dict[str, Annotation] = field(default_factory=dict)

    def solve(self, physics_ctx: PhysicsContext, name_space: str) -> AnnotationContext:
        if name_space == "physics":
            return self.solve_physics(physics_ctx)
        elif name_space == "render":
            return self.solve_render(physics_ctx)

    def solve_physics(self, physics_ctx: PhysicsContext) -> AnnotationContext:
        start_map = {}
        time_map = {}

        for t in physics_ctx.times:
            # 评估某个事件是否已经开始
            for annotation_id, annotation in self.annotations.items():

                if isinstance(annotation.rule, ActiveWhile):
                    eval_result = physics_ctx.eval_expr(annotation.rule.trigger.expression, "bool", t)
                    # 事件还没开始，就评估事件是否开始
                    if annotation_id not in start_map:
                        # 事件已经开始，标记开始时间
                        if eval_result:
                            start_map[annotation_id] = t
                    # 事件已经开始，就评估事件是否结束
                    else:
                        # 事件已经结束，标记结束时间
                        if not eval_result:
                            self.timeline.setdefault(annotation_id, []).append(
                                (start_map[annotation_id], t)
                            )
                            start_map.pop(annotation_id)
                
                elif isinstance(annotation.rule, ActivateEventTimeRange):
                    # 如果在t时刻事件发生，则append(t-advance, t+delay) 
                    # 获取旧值
                    old_time = time_map.get(annotation_id, None)
                    # 更新新值
                    time_map[annotation_id] = t
                    # 回调trigger
                    event_t = annotation.rule.trigger(old_time, t, physics_ctx.build_eval_exper_func(annotation.rule.trigger))
                    if event_t is not None:
                        self.timeline.setdefault(annotation_id, []).append(
                            (event_t - annotation.rule.advance, event_t + annotation.rule.delay)
                        )                       

                elif isinstance(annotation.rule, ActiveBetween):
                    # 如果在t时刻事件发生，则append(start, end) 
                    # 获取旧值
                    old_time = time_map.get(annotation_id, None)
                    # 更新新值
                    time_map[annotation_id] = t
                    if annotation_id not in start_map:
                        event_t = annotation.rule.start_trigger(old_time, t, physics_ctx.build_eval_exper_func(annotation.rule.start_trigger))
                        if event_t is not None:
                            start_map[annotation_id] = event_t
                    else:
                        event_t = annotation.rule.end_trigger(old_time, t, physics_ctx.build_eval_exper_func(annotation.rule.end_trigger))
                        if event_t is not None:
                            self.timeline.setdefault(annotation_id, []).append(
                                (start_map[annotation_id], event_t)
                            )
                            # 事件已经结束，清除开始时间和结束时间
                            start_map.pop(annotation_id)
        
        # 收尾时间
        for annotation_id in start_map.keys():
            self.timeline.setdefault(annotation_id, []).append(
                (start_map[annotation_id], physics_ctx.total_time)
            )
        
        return AnnotationContext(self.annotations, self.timeline, physics_ctx.value_at_time, physics_ctx.eval_expr, physics_ctx.build_eval_exper_func, "physics")
        

    def solve_render(self, physics_ctx: PhysicsContext) -> AnnotationContext:
        start_map = {}
        time_map = {}
        print(self.annotations)

        # 评估在render层上的事件
        for physics_t in physics_ctx.times:
            render_t = physics_ctx.physics_to_render_mapping_func(physics_t)
            # 评估某个事件是否已经开始
            for annotation_id, annotation in self.annotations.items():

                if isinstance(annotation.rule, ActiveWhile):
                    eval_result = self.render_eval_expr(annotation.rule.trigger.expression, "bool", physics_t, render_t, physics_ctx)
                    # 事件还没开始，就评估事件是否开始
                    if annotation_id not in start_map:
                        # 事件已经开始，标记开始时间
                        if eval_result:
                            start_map[annotation_id] = render_t
                    # 事件已经开始，就评估事件是否结束
                    else:
                        # 事件已经结束，标记结束时间
                        if not eval_result:
                            self.timeline.setdefault(annotation_id, []).append(
                                (start_map[annotation_id], render_t)
                            )
                            start_map.pop(annotation_id)
                
                elif isinstance(annotation.rule, ActivateEventTimeRange):
                    # 如果在t时刻事件发生，则append(t-advance, t+delay) 
                    # 获取旧值
                    old_time = time_map.get(annotation_id, None)
                    # 更新新值
                    time_map[annotation_id] = render_t
                    # 回调trigger
                    event_t = annotation.rule.trigger(old_time, render_t, self.render_build_eval_exper_func(annotation.rule.trigger, physics_ctx))
                    if event_t is not None:
                        self.timeline.setdefault(annotation_id, []).append(
                            (event_t - annotation.rule.advance, event_t + annotation.rule.delay)
                        )                       

                elif isinstance(annotation.rule, ActiveBetween):
                    # 如果在t时刻事件发生，则append(start, end) 
                    # 获取旧值
                    old_time = time_map.get(annotation_id, None)
                    # 更新新值
                    time_map[annotation_id] = render_t
                    if annotation_id not in start_map:
                        event_t = annotation.rule.start_trigger(old_time, render_t, self.render_build_eval_exper_func(annotation.rule.start_trigger, physics_ctx))
                        if event_t is not None:
                            start_map[annotation_id] = event_t
                    else:
                        event_t = annotation.rule.end_trigger(old_time, render_t, self.render_build_eval_exper_func(annotation.rule.end_trigger, physics_ctx))
                        if event_t is not None:
                            self.timeline.setdefault(annotation_id, []).append(
                                (start_map[annotation_id], event_t)
                            )
                            # 事件已经结束，清除开始时间和结束时间
                            start_map.pop(annotation_id)
        
        # 收尾时间
        for annotation_id in start_map.keys():
            self.timeline.setdefault(annotation_id, []).append(
                (start_map[annotation_id], physics_ctx.physics_to_render_mapping_func(physics_ctx.total_time))
            )
        
        print(self.timeline)
        return AnnotationContext(self.annotations, self.timeline, self.render_value_at_time, self.render_eval_expr, self.render_build_eval_exper_func, "render")
