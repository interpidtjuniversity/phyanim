from __future__ import annotations

import numpy as np

from manim import Scene, ValueTracker, linear

from phyanim.core.animation import PhysicsAnimation
from phyanim.solver import HeyokaSegmentSolver
from phyanim.solver.annotation_solver import DefaultAnnotationSolver
from phyanim.core.annotation.asserts import ArrowAnnotation, MathTexAnnotation, TextAnnotation

from phyanim.utils.util import is_numeric

class PhyAnimationScene2D(Scene):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_animation(self, animation: PhysicsAnimation) -> None:
        """注册一个PhysicsAnimation。"""
        self.animation = animation
        # 变量管理上下文
        # 注释管理上下文
        self.ctx, self.anno_ctx = animation.solve(HeyokaSegmentSolver(sample_dt=1/20), DefaultAnnotationSolver(annotations=self.animation.annotations))
    
    def set_frame_size(self, width: float, height: float) -> None:
        self.frame_width = width
        self.frame_height = height

    def construct(self) -> None:
        tracker = ValueTracker(0.0)
        #================================物理实体开始================================
        def make_obj_position_updater(
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

        # 给mobject添加更新器
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
            # 给物理实体添加updater
            mob.add_updater(make_obj_position_updater(obj_id, callbacks))


        #================================注释实体开始================================
        def make_anno_position_updater(
            anno_id: str,
            content,
            callback=None,
        ):
            def updater(m):
                t = tracker.get_value()
                
                timeline = self.anno_ctx.timeline.get(anno_id, [])
                visible = any(start <= t <= end for start, end in timeline)
                m.set_opacity(1.0 if visible else 0.0)

                if not visible:
                    return

                x_pos_name, y_pos_name = content.pos_variables

                x_pos_value, y_pos_value = None, None
                if not is_numeric(x_pos_name):
                    x_pos_value = self.ctx.value_at_time(x_pos_name, t)
                else:
                    x_pos_value = float(x_pos_name)
                if not is_numeric(y_pos_name):
                    y_pos_value = self.ctx.value_at_time(y_pos_name, t)
                else:
                    y_pos_value = float(y_pos_name)

                if isinstance(content, ArrowAnnotation):
                    x_shift_name, y_shift_name = content.shift_variables
                    x_shift_value, y_shift_value = None, None
                    if not is_numeric(x_shift_name):
                        x_shift_value = self.ctx.value_at_time(x_shift_name, t)
                    else:
                        x_shift_value = float(x_shift_name)
                    if not is_numeric(y_shift_name):
                        y_shift_value = self.ctx.value_at_time(y_shift_name, t)
                    else:
                        y_shift_value = float(y_shift_name)
                    callback(x_pos_value, y_pos_value, x_shift_value, y_shift_value)
                
                elif isinstance(content, (MathTexAnnotation, TextAnnotation)):
                    callback(x_pos_value, y_pos_value)

            return updater

        for anno_id, anno in self.anno_ctx.annotations.items():
            self.add(anno.content.obj)
            call_back = anno.content.change_callback()
            anno.content.obj.add_updater(make_anno_position_updater(anno_id, anno.content, call_back))
   

        # 执行annotation transition动画

        self.play(
            tracker.animate.set_value(self.ctx.total_time),
            run_time=self.ctx.total_time,
            rate_func=linear,
        )

        for obj in self.animation.objects.values():
            if obj.mobject is not None:
                obj.mobject.clear_updaters()



   