from __future__ import annotations

from manim import Scene, ValueTracker, linear

from phyanim.core.animation import PhysicsAnimation

from phyanim.core.layer import Layer
        # for obj in self.animation.objects.values():
        #     if obj.mobject is not None:
        #         obj.mobject.clear_updaters()

from phyanim.core.timeline import Timeline
from phyanim.core.enhance.timewrapper import TimeWrapper

class PhyAnimationMultiLayerScene2D(Scene):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers: list[Layer] = []
        self.timeline = Timeline()

    def set_frame_size(self, width: float, height: float) -> None:
        self.frame_width = width
        self.frame_height = height

    def set_animation(self, animation: PhysicsAnimation) -> None:
        """注册一个PhysicsAnimation。"""
        self.animation = animation
    
    def add_time_wrapper(self, time_wrapper: TimeWrapper) -> None:
        self.timeline.add_time_wrapper(time_wrapper)

    def construct(self) -> None:
        if not self.animation.solved:
            self.animation.solve()
        
        # 主渲染器
        render_tracker = ValueTracker(0.0)
        result = self.timeline.solve(self.animation.physics_ctx)
        total_time = result.total_time
        render_to_physics_mapping_func = result.render_to_physics_mapping_func
        physics_to_render_mapping_func = result.physics_to_render_mapping_func
        time_wrapper_ranges = result.time_wrapper_ranges

        # 动画主体对象
        for obj in self.animation.physics_ctx.get_entities(render_tracker):
            self.add(obj)

        # 添加展示层对象
        physics_layer = self.animation.get_physics_layer()
        physics_layer.solve(self.animation.physics_ctx, time_wrapper_ranges)
        physics_layer.set_time_mapping_func(render_to_physics_mapping_func, physics_to_render_mapping_func)
        for entity in physics_layer.get_entities(render_tracker):
            self.add(entity)

        # 添加渲染层对象
        render_layer = self.animation.get_render_layer()
        render_layer.solve(self.animation.physics_ctx, time_wrapper_ranges)
        render_layer.set_time_mapping_func(render_to_physics_mapping_func, physics_to_render_mapping_func)
        for entity in render_layer.get_entities(render_tracker):
            self.add(entity)


        self.play(
            render_tracker.animate.set_value(total_time),
            run_time=total_time,
            rate_func=linear,
        )