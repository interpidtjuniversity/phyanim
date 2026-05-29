# from manim import *

# class Example3DirectorLoop(Scene):
#     def construct(self):
#         # --- 1. 基础设施 ---
#         time_tracker = ValueTracker(0)
        
#         # 模拟你的物理对象：一个上下浮动的圆
#         physics_obj = Circle().set_color(BLUE)
#         physics_obj.add_updater(lambda m: m.shift(UP * 0.1 * np.sin(time_tracker.get_value() * PI)))
#         self.add(physics_obj)
        
#         # 模拟你的 Annotation Mobjects
#         anno_A = Text("A").shift(UP * 2)
#         anno_B = Text("B").shift(UP * 2)
#         anno_C = Text("C").shift(UP * 2)
        
#         # --- 2. 定义动作列表 (Action List) ---
#         # 这就像剧本：[时间, 动作类型, 参数, 持续时长]
#         actions = [
#             (1.0, "fade_in", anno_A, 0.5),      # 1秒时，A淡入
#             (3.0, "transform", (anno_A, anno_B), 1.0), # 3秒时，A变B
#             (5.0, "transform", (anno_B, anno_C), 0.5), # 5秒时，B变C
#             (7.0, "fade_out", anno_C, 0.5),     # 7秒时，C淡出
#         ]
        
#         # --- 3. 导演循环 ---
#         current_time = 0.0
#         total_duration = 8.0
        
#         for action in actions:
#             t_trigger, action_type, params, duration = action
            
#             # A. 推进物理时间到事件发生点
#             dt = t_trigger - current_time
#             if dt > 0:
#                 self.play(
#                     time_tracker.animate.set_value(t_trigger),
#                     run_time=dt,
#                     rate_func=linear
#                 )
#                 current_time = t_trigger
            
#             # B. 执行事件 (同时推进物理时间)
#             if action_type == "fade_in":
#                 self.play(
#                     time_tracker.animate.set_value(current_time + duration),
#                     FadeIn(params),
#                     run_time=duration,
#                     rate_func=linear
#                 )
                
#             elif action_type == "fade_out":
#                 self.play(
#                     time_tracker.animate.set_value(current_time + duration),
#                     FadeOut(params),
#                     run_time=duration,
#                     rate_func=linear
#                 )
                
#             elif action_type == "transform":
#                 src, dst = params
#                 # FadeTransform 会自动处理 FadeOut(src) 和 FadeIn(dst)
#                 self.play(
#                     time_tracker.animate.set_value(current_time + duration),
#                     FadeTransform(src, dst),
#                     run_time=duration,
#                     rate_func=linear
#                 )
            
#             # 更新当前时间指针
#             current_time += duration

#         # C. 播放剩余时间
#         if current_time < total_duration:
#             self.play(
#                 time_tracker.animate.set_value(total_duration),
#                 run_time=total_duration - current_time,
#                 rate_func=linear
#             )




# from manim import *
# import numpy as np

# class Example3DirectorLoop(Scene):
#     def construct(self):
#         tracker = ValueTracker(0)

#         formula1 = MathTex("F", "=", "kx")
#         formula2 = MathTex("k", "=", "\\frac{F}{x}")

#         base_pos = ORIGIN

#         def get_formula_pos():
#             t = tracker.get_value()
#             return base_pos + UP * 0.5 * np.sin(t * PI)

#         def follow_tracker(m):
#             m.move_to(get_formula_pos())

#         formula1.add_updater(follow_tracker)
#         formula1.update()

#         self.add(formula1)

#         current_time = 0.0

#         # 先移动
#         waiting_time = 2.0
#         self.play(
#             tracker.animate.set_value(current_time + waiting_time),
#             run_time=waiting_time,
#             rate_func=linear,
#         )
#         current_time += waiting_time

#         # 关键：让 formula2 初始放到 formula1 当前的位置
#         formula2.move_to(formula1.get_center())

#         trans_time = 2.0
#         self.play(
#             tracker.animate.set_value(current_time + trans_time),

#             # 关键参数：suspend_mobject_updating=False
#             ReplacementTransform(
#                 formula1,
#                 formula2,
#             ),

#             run_time=trans_time,
#             rate_func=linear,
#         )

#         current_time += trans_time

#         # transform 结束后，formula2 才是新的显示对象
#         formula2.add_updater(follow_tracker)
#         formula2.update()

#         remain_time = 2.0
#         self.play(
#             tracker.animate.set_value(current_time + remain_time),
#             run_time=remain_time,
#             rate_func=linear,
#         )



from manim import *
import numpy as np


def _smoothstep(x):
    """三次缓动：首尾平滑，中间线性。"""
    return x * x * (3 - 2 * x)


def make_sequential(formulas, times, tracker, style="scale", duration=0.4):
    """
    根据 tracker 在多个公式之间依次切换，支持多种动画风格。

    formulas: [mob_a, mob_b, mob_c, mob_d]
    times:    [2, 4, 6, 7]  — 各切换时间点
    style:    动画风格:
        "fade"        — 纯淡入淡出
        "scale"       — 缩成一点再扩散（推荐）
        "slide_left"  — 左滑出 / 右滑入
        "slide_down"  — 下滑出 / 上滑入
        "spin"        — 旋转缩放出 / 旋转放大入
    duration: 过渡持续时间（秒）
    """
    container = VGroup(*formulas)
    n = len(formulas)
    refs = [f.copy() for f in formulas]

    def apply_out(mob, ref, alpha):
        """alpha: 0=完全可见, 1=完全消失"""
        a = _smoothstep(alpha)
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
        a = _smoothstep(alpha)
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

    def updater(m):
        t = tracker.get_value()

        for i in range(n):
            mob = m.submobjects[i]
            ref = refs[i]
            mob.become(ref)  # 每帧重置，避免变换累积

            if i == 0:
                # 第一个：t=0 起可见，在 times[1] 处消失
                fo_start = times[1] - duration
                fo_end = times[1]
                if t < fo_start:
                    pass  # become 后就是完整可见状态
                elif t <= fo_end:
                    apply_out(mob, ref, (t - fo_start) / duration)
                else:
                    mob.set_opacity(0)
            elif i == n - 1:
                # 最后一个：在 times[i] 处出现，之后一直可见
                fi_start = times[i]
                fi_end = times[i] + duration
                if t < fi_start:
                    mob.set_opacity(0)
                elif t <= fi_end:
                    apply_in(mob, ref, (t - fi_start) / duration)
                else:
                    pass  # 完整可见
            else:
                # 中间：在 times[i] 出现，在 times[i+1] 消失
                fi_start = times[i]
                fi_end = times[i] + duration
                fo_start = times[i + 1] - duration
                fo_end = times[i + 1]
                if t < fi_start:
                    mob.set_opacity(0)
                elif t <= fi_end:
                    apply_in(mob, ref, (t - fi_start) / duration)
                elif t < fo_start:
                    pass  # 完全可见
                elif t <= fo_end:
                    apply_out(mob, ref, (t - fo_start) / duration)
                else:
                    mob.set_opacity(0)

    container.add_updater(updater)
    return container


class MultiGroupOverlap(Scene):
    def construct(self):
        tracker = ValueTracker(0)

        # ===== A: scale 风格 — 缩成一点再扩散 =====
        a_mobs = [MathTex(t).shift(UP * 2.5) for t in
                  ["F=kx", "k=\\frac{F}{x}", "x=\\frac{F}{k}", "\\frac{F}{x}=k"]]
        grp_a = make_sequential(a_mobs, [2, 4, 6, 7], tracker, style="scale")

        # ===== B: slide_left 风格 — 左滑出右滑入 =====
        b_mobs = [MathTex(t).shift(DOWN * 0.5) for t in
                  ["x=vt", "v=\\frac{x}{t}", "t=\\frac{x}{v}", "\\frac{x}{t}=v"]]
        grp_b = make_sequential(b_mobs, [3, 6, 8, 10], tracker, style="slide_left")

        # ===== C: spin 风格 — 旋转缩放 =====
        c_mobs = [MathTex(t).shift(DOWN * 2.5) for t in
                  ["E=mc^2", "m=\\frac{E}{c^2}", "c^2=\\frac{E}{m}"]]
        grp_c = make_sequential(c_mobs, [1, 3, 5], tracker, style="spin")

        self.add(grp_a, grp_b, grp_c)

        # 一个 play 推到底
        self.play(tracker.animate.set_value(11), run_time=11, rate_func=linear)

