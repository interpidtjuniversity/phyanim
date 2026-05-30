from __future__ import annotations

from manim import Scene, ValueTracker, linear



from phyanim.core.animation import PhysicsAnimation
from phyanim.solver import HeyokaSegmentSolver
from phyanim.solver.annotation_solver import DefaultAnnotationSolver
from phyanim.solver.transition_solver import DefaultTransitionSolver
from phyanim.solver.timewrapper_solver import DefaultTimeWrapperSolver

class PhyAnimationScene2D(Scene):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_animation(self, animation: PhysicsAnimation) -> None:
        """注册一个PhysicsAnimation。"""
        self.animation = animation
        # 变量管理上下文
        # 注释管理上下文
        self.main_layer = animation.solve(
            HeyokaSegmentSolver(sample_dt=1/20), 
            DefaultAnnotationSolver(annotations=self.animation.annotations), 
            DefaultTransitionSolver(transitions=self.animation.transitions),
            DefaultTimeWrapperSolver(time_wrappers=self.animation.time_wrappers)
        )
    
    def set_frame_size(self, width: float, height: float) -> None:
        self.frame_width = width
        self.frame_height = height

    def construct(self) -> None:
        tracker = ValueTracker(0.0)


        for ctx in self.main_layer.contexts:
            ctx_entities = ctx.get_entities(tracker)
            for entity in ctx_entities:
                self.add(entity)

        self.play(
            tracker.animate.set_value(self.main_layer.total_time),
            run_time=self.main_layer.total_time,
            rate_func=linear,
        )

        for obj in self.animation.objects.values():
            if obj.mobject is not None:
                obj.mobject.clear_updaters()


from phyanim.core.context import Context
class PhyAnimationMultiLayerScene2D(PhyAnimationScene2D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contexts: list[Context] = []
    
    def set_animation(self, animation: PhyAnimation):
        # physics obj、annotation、transition都在同一个本征时间layer上
        super().set_animation(animation)
        # 构建渲染上下文
        self.contexts.append(self.physics_ctx)
        self.contexts.append(self.anno_ctx)
        self.contexts.append(self.transition_ctx)
        self.contexts.append(self.time_wrapper_ctx)

        for ctx in self.contexts:
            ctx.set_time_mapping_to_physics(self.time_wrapper_ctx.time_mapping_func)


    def construct(self) -> None:
        # 主渲染器
        render_tracker = ValueTracker(0.0)
        max_time = max([ctx.total_time for ctx in self.contexts])

        self.add_entities(render_tracker)

        self.play(
            render_tracker.animate.set_value(max_time),
            run_time=max_time,
            rate_func=linear,
        )