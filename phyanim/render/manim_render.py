from __future__ import annotations

from manim import Scene, ValueTracker, linear, config

from phyanim.core.animation import PhysicsAnimation
from phyanim.core.timeline import Timeline

from phyanim.core.layer import Layer

class PhyAnimationMultiLayerScene2D(Scene):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers: list[Layer] = []

    def set_frame_size(self, width: float, height: float) -> None:
        self.frame_width = width
        self.frame_height = height

    def set_animation(self, animation: PhysicsAnimation) -> None:
        """注册一个PhysicsAnimation。"""
        self.animation = animation

    def construct(self) -> None:
        config["disable_caching"] = True

        if not self.animation.solved:
            self.animation.solve()

        physics_layer = self.animation.get_physics_layer()
        render_layer = self.animation.get_render_layer()
        # 求解timeline
        timeline = Timeline()
        for item in render_layer.sub_animation.values():
            timeline.add_time_wrapper(item["time_wrapper"])
        result = timeline.solve(self.animation.physics_ctx)
        # 主渲染器
        render_tracker = ValueTracker(0.0)
        # 动画主体对象
        for obj in self.animation.physics_ctx.get_entities(render_tracker):
            self.add(obj)

        # 添加展示层对象
        physics_layer.solve(self.animation.physics_ctx, result)
        for entity in physics_layer.get_entities(render_tracker, self.animation.physics_ctx):
            self.add(entity)

        # 添加渲染层对象
        render_layer.solve(self.animation.physics_ctx, result)
        for entity in render_layer.get_entities(render_tracker, self.animation.physics_ctx):
            self.add(entity)


        total_time = self.animation.physics_ctx.total_time
        total_time = self.animation.physics_ctx.physics_to_render_mapping_func(total_time)

        self.play(
            render_tracker.animate.set_value(total_time),
            run_time=total_time,
            rate_func=linear,
        )
        # for obj in self.animation.objects.values():
        #     if obj.mobject is not None:
        #         obj.mobject.clear_updaters()