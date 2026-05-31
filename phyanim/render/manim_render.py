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
        total_time, time_mapping_func = self.timeline.solve(self.animation.physics_ctx)
        # 动画主体对象
        for obj in self.animation.physics_ctx.get_entities(render_tracker):
            self.add(obj)

        for layer in self.animation.layers:
            layer.solve(self.animation.physics_ctx)
            layer.set_time_mapping_func(time_mapping_func)
            # 这里会在context内部添加updater
            entities = layer.get_entities(render_tracker)
            for entity in entities:
                self.add(entity)

        self.play(
            render_tracker.animate.set_value(total_time),
            run_time=total_time,
            rate_func=linear,
        )