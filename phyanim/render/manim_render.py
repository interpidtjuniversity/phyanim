from __future__ import annotations

import numpy as np
from typing import Callable

from manim import Scene, ValueTracker, linear, VGroup, PI



from phyanim.core.animation import PhysicsAnimation
from phyanim.solver import HeyokaSegmentSolver
from phyanim.solver.annotation_solver import DefaultAnnotationSolver
from phyanim.solver.transition_solver import DefaultTransitionSolver
from phyanim.solver.timewrapper_solver import DefaultTimeWrapperSolver
from phyanim.core.enhance.annotation import ArrowContent, MathTexContent, TextContent
from phyanim.core.enhance.transition import TransitionContent

from phyanim.utils.util import is_numeric

class PhyAnimationScene2D(Scene):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_animation(self, animation: PhysicsAnimation) -> None:
        """注册一个PhysicsAnimation。"""
        self.animation = animation
        # 变量管理上下文
        # 注释管理上下文
        self.physics_ctx, self.anno_ctx, self.transition_ctx, self.time_wrapper_ctx = animation.solve(
            HeyokaSegmentSolver(sample_dt=1/20), 
            DefaultAnnotationSolver(annotations=self.animation.annotations), 
            DefaultTransitionSolver(transitions=self.animation.transitions),
            DefaultTimeWrapperSolver(time_wrappers=self.animation.time_wrappers)
        )
    
    def set_frame_size(self, width: float, height: float) -> None:
        self.frame_width = width
        self.frame_height = height

    # 普通物理对象
    def make_obj_position_updater(
        self,
        current_obj_id: str,
        current_callbacks=None,
        tracker: ValueTracker = None,
    ):
        def updater(m):
            t = tracker.get_value()
            physics_t = self.physics_ctx.time_mapping_to_physics(t)
            cartesian_positions = self.physics_ctx.eval_position(current_obj_id, physics_t)

            if len(cartesian_positions) == 1:
                x, y = cartesian_positions[0]
                m.move_to(np.array([x, y, 0.0]))
                return

            for callback, position in zip(current_callbacks, cartesian_positions):
                x, y = position
                callback(x, y)
        return updater    
    
    def make_anno_position_updater(
        self,
        anno_id: str,
        content,
        callback=None,
        tracker: ValueTracker = None,
    ):
        def updater(m):
            t = tracker.get_value()
            physics_t = self.physics_ctx.time_mapping_to_physics(t)
                
            timeline = self.anno_ctx.timeline.get(anno_id, [])
            visible = any(start <= physics_t <= end for start, end in timeline)
            m.set_opacity(1.0 if visible else 0.0)

            if not visible:
                return

            x_pos_name, y_pos_name = content.pos_variables

            x_pos_value, y_pos_value = None, None
            if not is_numeric(x_pos_name):
                x_pos_value = self.physics_ctx.value_at_time(x_pos_name, physics_t)
            else:
                x_pos_value = float(x_pos_name)
            if not is_numeric(y_pos_name):
                y_pos_value = self.physics_ctx.value_at_time(y_pos_name, physics_t)
            else:
                y_pos_value = float(y_pos_name)

            if isinstance(content, ArrowContent):
                x_shift_name, y_shift_name = content.shift_variables
                x_shift_value, y_shift_value = None, None
                if not is_numeric(x_shift_name):
                    x_shift_value = self.physics_ctx.value_at_time(x_shift_name, physics_t)
                else:
                    x_shift_value = float(x_shift_name)
                if not is_numeric(y_shift_name):
                    y_shift_value = self.physics_ctx.value_at_time(y_shift_name, physics_t)
                else:
                    y_shift_value = float(y_shift_name)
                callback(x_pos_value, y_pos_value, x_shift_value, y_shift_value)
                
            elif isinstance(content, (MathTexContent, TextContent)):
                callback(x_pos_value, y_pos_value)

        return updater

    def make_sequential(self, transition, groups, times, tracker, style="scale", duration=0.5, smooth_func : Callable[[float], float] = lambda x: x * x * (3 - 2 * x)):
        """
        根据 tracker 在多个公式之间依次切换，支持多种动画风格。
        formulas: [mob_a, mob_b, mob_c, mob_d]
        times:    [2, 4, 6, 7, 8]  — 各切换时间点
        style:    动画风格:
            "fade"        — 纯淡入淡出
            "scale"       — 缩成一点再扩散（推荐）
            "slide_left"  — 左滑出 / 右滑入
            "slide_down"  — 下滑出 / 上滑入
            "spin"        — 旋转缩放出 / 旋转放大入
        duration: 过渡持续时间（秒）
        """
        container = VGroup(*groups)
        n = len(groups)
        refs = [f.copy() for f in groups]

        def apply_out(mob, ref, alpha):
            """alpha: 0=完全可见, 1=完全消失"""
            a = smooth_func(alpha)
            if style == "fade":
                mob.set_opacity(1 - a)
            elif style == "scale":
                s = 1 - a
                mob.scale(s)
                mob.set_opacity(max(0, 1 - a * 1.5))
            elif style == "slide_left":
                mob.shift(LEFT * a * 2)
                mob.set_opacity(1 - a)
            elif style == "slide_down":
                mob.shift(DOWN * a * 2)
                mob.set_opacity(1 - a)
            elif style == "spin":
                s = 1 - a
                mob.scale(s)
                mob.rotate(a * PI / 2)
                mob.set_opacity(max(0, 1 - a * 1.5))

        def apply_in(mob, ref, alpha):
            """alpha: 0=完全不可见, 1=完全可见"""
            a = smooth_func(alpha)
            if style == "fade":
                mob.set_opacity(a)
            elif style == "scale":
                mob.scale(max(0.01, a))
                mob.set_opacity(min(1, a * 1.5))
            elif style == "slide_left":
                mob.shift(RIGHT * (1 - a) * 2)
                mob.set_opacity(a)
            elif style == "slide_down":
                mob.shift(UP * (1 - a) * 2)
                mob.set_opacity(a)
            elif style == "spin":
                mob.scale(max(0.01, a))
                mob.rotate(-(1 - a) * PI / 2)
                mob.set_opacity(min(1, a * 1.5))

        # 这里m是最大的那个group(group1(str1,str2), group2(str3,str4,str5,str6))
        def updater(m):
            t = tracker.get_value()
            physics_t = self.physics_ctx.time_mapping_to_physics(t)
            
            x_pos_name, y_pos_name = transition.content.pos_variables

            x_pos_value, y_pos_value = None, None
            if not is_numeric(x_pos_name):
                x_pos_value = self.physics_ctx.value_at_time(x_pos_name, physics_t)
            else:
                x_pos_value = float(x_pos_name)
            if not is_numeric(y_pos_name):
                y_pos_value = self.physics_ctx.value_at_time(y_pos_name, physics_t)
            else:
                y_pos_value = float(y_pos_name)

            target_pos = np.array([x_pos_value, y_pos_value, 0.0])

            for i in range(n):
                mob = m.submobjects[i]
                ref = refs[i]
                mob.become(ref)  # 每帧重置，避免变换累积
                mob.move_to(target_pos)

                # 开始出现
                fi_start = times[i] - duration
                # 完全出现
                fi_end = times[i]
                # 开始消失
                fo_start = times[i + 1] - duration
                # 完全消失
                fo_end = times[i + 1]

                # 处理区间重合的几种情况
                # 如果完全出现的时刻等于完全消失的时刻，则设置从fi_start到fo_end的先显示再消失
                if fo_start <= fi_end and fi_end <= fo_end:
                    if physics_t < fi_start:
                        mob.set_opacity(0)
                    elif physics_t <= fo_end:
                        # 一半的duration用来出现，一半用来消失
                        span = (fo_end + fi_start) / 2
                        if physics_t <= span:
                            apply_in(mob, ref, (physics_t - fi_start) / (span - fi_start))
                        else:
                            apply_out(mob, ref, (physics_t - span) / (fo_end - span))
                    else:
                        mob.set_opacity(0)
                elif fi_end < fo_start:
                    if physics_t < fi_start:
                        mob.set_opacity(0)
                    elif physics_t <= fi_end:
                        apply_in(mob, ref, (physics_t - fi_start) / duration)
                    elif physics_t < fo_start:
                        mob.set_opacity(1)
                    elif physics_t <= fo_end:
                        apply_out(mob, ref, (physics_t - fo_start) / duration)
                    else:
                        mob.set_opacity(0)
                        

        container.add_updater(updater)
        return container

    def add_entities(self, tracker: ValueTracker):
                #================================物理实体开始================================    
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
            mob.add_updater(self.make_obj_position_updater(obj_id, callbacks, tracker))


        #================================注释实体开始================================
        for anno_id, anno in self.anno_ctx.annotations.items():
            # 这里的TransitionContent 需要单独运镜处理
            if isinstance(anno.content, TransitionContent):
                continue
            self.add(anno.content.obj)
            call_back = anno.content.change_callback()
            anno.content.obj.add_updater(self.make_anno_position_updater(anno_id, anno.content, call_back, tracker))
   
        #================================transition开始================================
        # 执行annotation transition动画
        for trans_id, transition in self.transition_ctx.transitions.items():
            groups = transition.content.groups
            group_container = self.make_sequential(transition, groups, self.transition_ctx.timeline[trans_id], tracker, transition.style, transition.duration, transition.smooth_func)
            self.add(group_container)


    def construct(self) -> None:
        tracker = ValueTracker(0.0)
        self.add_entities(tracker)

        self.play(
            tracker.animate.set_value(self.physics_ctx.total_time),
            run_time=self.physics_ctx.total_time,
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