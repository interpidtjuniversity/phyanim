from __future__ import annotations

import numpy as np

from manim import Scene, ValueTracker, linear

from phyanim.core.animation import PhysicsAnimation
from phyanim.solver import HeyokaSegmentSolver

class PhyAnimationScene2D(Scene):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_animation(self, animation: PhysicsAnimation) -> None:
        """注册一个PhysicsAnimation。"""
        self.animation = animation
        # 变量管理上下文
        # 注释管理上下文
        self.ctx, self.anno_ctx = animation.solve(HeyokaSegmentSolver(sample_dt=1/20))
    
    def set_frame_size(self, width: float, height: float) -> None:
        self.frame_width = width
        self.frame_height = height

    def construct(self) -> None:
        tracker = ValueTracker(0.0)

        for obj_id, obj in self.animation.objects.items():
            if obj.mobject is None:
                raise ValueError(f"Object '{obj_id}' has no mobject for rendering.")

            mob = obj.mobject
            self.add(mob)

            cartesian_position_variables = obj.cartesian_position_variables()

            callbacks = None

            if len(cartesian_position_variables) > 1:
                if hasattr(mob, "point_change_callbacks"):
                    callbacks = mob.point_change_callbacks()

                    if len(callbacks) != len(cartesian_position_variables):
                        raise ValueError(
                            f"Object '{obj_id}' has {len(cartesian_position_variables)} "
                            f"cartesian positions, but its mobject provides {len(callbacks)} callbacks."
                        )
                else:
                    raise TypeError(
                        f"Object '{obj_id}' uses multiple cartesian positions, "
                        "but its mobject implements neither set_control_points() "
                        "nor point_change_callbacks()."
                    )

            def make_position_updater(
                current_obj_id: str,
                current_callbacks=None,
            ):
                def updater(m):
                    t = tracker.get_value()
                    cartesian_positions = self.ctx.eval_position(current_obj_id, t)

                    if len(cartesian_positions) == 1:
                        x, y = cartesian_positions[0]
                        m.move_to(np.array([x, y, 0.0]))
                        return

                    for callback, position in zip(current_callbacks, cartesian_positions):
                        x, y = position
                        callback(x, y)

                return updater

            mob.add_updater(make_position_updater(obj_id, callbacks))

        self.play(
            tracker.animate.set_value(self.ctx.total_time),
            run_time=self.ctx.total_time,
            rate_func=linear,
        )

        for obj in self.animation.objects.values():
            if obj.mobject is not None:
                obj.mobject.clear_updaters()

