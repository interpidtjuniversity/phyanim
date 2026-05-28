from typing import Tuple
from dataclasses import dataclass, field

from phyanim.core.context import AnimationContext
from phyanim.core.context import AnnotationContext

from phyanim.core.annotation.annotation import Annotation

@dataclass
class DefaultAnnotationSolver:

    timeline: dict[str, list[Tuple[float, float]]] = field(default_factory=dict)
    annotations: dict[str, Annotation] = field(default_factory=dict)

    def solve(self, animation_ctx: AnimationContext) -> AnnotationContext:
        start_map = {}
        time_map = {}

        for t in animation_ctx.times:
            # 评估某个事件是否已经开始
            for annotation_id, annotation in self.annotations.items():

                if isinstance(annotation.rule, ActiveWhile):
                    eval_result = animation_ctx.eval_expr(annotation.rule.condition, "bool", t)
                    # 事件还没开始，就评估事件是否开始
                    if annotation_id not in start_map:
                        # 事件已经开始，标记开始时间
                        if eval_result:
                            start_map[annotation_id] = t
                    # 事件已经开始，就评估事件是否结束
                    else:
                        # 事件已经结束，标记结束时间
                        if not eval_result:
                            end_map[annotation_id] = t
                            # 标记事件发生时间段
                            self.timeline.setdefault(annotation_id, []).append(
                                (start_map[annotation_id], t)
                            )
                            # 事件已经结束，清除开始时间
                            start_map.pop(annotation_id)
                
                elif isinstance(annotation.rule, ActivateEventTimeRange):
                    # 如果在t时刻事件发生，则append(t-advance, t+delay) 
                    # 获取旧值
                    old_time = time_map.get(annotation_id, None)
                    # 更新新值
                    time_map[annotation_id] = t
                    # 回调trigger
                    event_t = annotation.rule.trigger(old_time, t, animation_ctx.build_eval_exper_func(annotation.rule.trigger))
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
                        event_t = annotation.rule.start_trigger(old_time, t, animation_ctx.build_eval_exper_func(annotation.rule.start_trigger))
                        if event_t is not None:
                            start_map[annotation_id] = event_t
                    else:
                        event_t = annotation.rule.end_trigger(old_time, t, animation_ctx.build_eval_exper_func(annotation.rule.end_trigger))
                        if event_t is not None:
                            self.timeline.setdefault(annotation_id, []).append(
                                (start_map[annotation_id], event_t)
                            )
                            # 事件已经结束，清除开始时间和结束时间
                            start_map.pop(annotation_id)
        
        # 收尾时间
        for annotation_id in start_map.keys():
            self.timeline.setdefault(annotation_id, []).append(
                (start_map[annotation_id], animation_ctx.total_time)
            )
        
        return AnnotationContext(self.annotations, self.timeline)
