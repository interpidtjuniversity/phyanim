"""简谐振动 — 新特性综合演示。

展示以下功能：
1. register_event + is_local_max / is_local_min / is_global_max / is_global_min
2. register_event + bool 条件 (x > threshold)
3. trajectory.all_events() / trajectory.event_trigger_times()
4. attach_mobject_with_event（事件触发后开始更新，用 def 定义 update_fn）
5. detach_mobject_with_event（事件触发后移除 updater）
6. attach_expr_updater（实时显示物理量）
7. attach_line（速度箭头跟随）
8. MathTex 中文公式分块展示
9. 分段播放 + 暂停讲解
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
TTS_CONFIG = {
    "provider": "minimax",
    "api_key": os.environ.get("PHYANIM_TTS_API_KEY", "sk-api-3Zu_GYQdFjIoXCXFHGLwcjXL7sUOQGSiAAmbTns5cLpa36Xk9-F1fdezpn9hAwWEDFiLFlsMnavkbh-_GL5oJHOd8zhmtbIgm7dcNhi2keybZ5OtJ0XAOF4"),
    "model": "speech-02-turbo",
    "voice_id": "male-qn-qingse",
    "speed": 1.0,
    "vol": 1.0,
    "pitch": 0.0,
}

import numpy as np
from manim import *
from phyanim.api import *
from phyanim.voiceover import PhyAnimScene
from math import pi

BG_COLOR = "#1C2333"
ACCENT = GOLD_C
BALL_COLOR = YELLOW
GLOW_COLOR = "#FFD966"
SPRING_COLOR = "#E8E8E8"
WALL_COLOR = GREY_D
WALL_X = -4.2


def build_animation() -> PhysicsAnimation:
    """简谐振动: x'' = -omega^2 * x, omega = 2."""
    omega = 2.0
    animation = PhysicsAnimation(
        global_parameters={"omega": omega, "m": 1.0},
        engine="scipy",
        sample_dt=1 / 120,
    )

    ball = PhysicObject2D(
        object_id="ball",
        state_variables={
            "x": StateVariable("x", "m", "位移"),
            "vx": StateVariable("vx", "m/s", "速度"),
        },
        cartesian_position=[("x", "0")],
    )
    animation.add_object(ball, {"x": 3.0, "vx": 0.0})

    animation.add_segment(
        PhysicsSegment(
            segment_id="shm",
            object_ids=["ball"],
            state_vector=["x", "vx"],
            equations={"x": "vx", "vx": "-omega**2 * x"},
            state_owners={"x": "ball", "vx": "ball"},
            derived_equations={
                "ke": "0.5 * m * vx**2",
                "pe": "0.5 * m * omega**2 * x**2",
                "e_total": "0.5 * m * omega**2 * 3.0**2",
            },
            duration=100,
        ),
        end_event=time_countdown_event(3 * pi),  # ~3 periods
    )

    # --- 注册事件 ---

    # 1. 零点穿越: x 从正变负 (向左过零)
    animation.register_event("cross_zero_left", "x", direction=-1)

    # 2. 零点穿越: x 从负变正 (向右过零)
    animation.register_event("cross_zero_right", "x", direction=1)

    # 3. 局部极大值: x 达到正方向极值
    animation.register_event("at_right_peak", "is_local_max(x)", event_type="bool")

    # 4. 局部极小值: x 达到负方向极值
    animation.register_event("at_left_peak", "is_local_min(x)", event_type="bool")

    # 5. 全局最大值
    animation.register_event("global_max", "is_global_max(x)", event_type="bool")

    # 6. 全局最小值
    animation.register_event("global_min", "is_global_min(x)", event_type="bool")

    # 7. 布尔条件: 速度大于某阈值
    animation.register_event("fast_speed", "abs(vx) > 4", event_type="bool")

    return animation


class SHMFeatureDemoScene(PhyAnimScene):
    def construct(self):
        self.setup_speech(TTS_CONFIG)
        self.camera.background_color = BG_COLOR

        animation = build_animation()
        trajectory = solve_animation(animation)
        t_total = trajectory.total_time

        # 打印所有事件
        print("=== Registered Events ===")
        for eid, times in trajectory.all_events().items():
            print(f"  {eid}: {[f'{t:.4f}' for t in times]}")

        # 相机设置
        self.camera.frame_width = 16
        self.camera.frame_center = np.array([0, 0, 0])

        # --- 场景元素 ---
        # 平衡位置参考线
        equilibrium = DashedLine(UP * 1.0, DOWN * 1.0, color=GRAY, stroke_width=2)
        eq_label = MathTex("x=0", font_size=20, color=GRAY).next_to(equilibrium, UP, buff=0.1)

        # 振幅边界
        amp_line_r = DashedLine(UP * 1.0, DOWN * 1.0, color=GRAY_B, stroke_width=1).shift(RIGHT * 3)
        amp_line_l = DashedLine(UP * 1.0, DOWN * 1.0, color=GRAY_B, stroke_width=1).shift(LEFT * 3)

        # 墙壁（带斜线纹理）
        wall = Rectangle(width=0.3, height=2.0, color=WALL_COLOR, fill_opacity=0.5)
        wall.move_to([WALL_X, 0, 0])
        wall_pattern = VGroup(*[
            Line(
                [WALL_X - 0.15, y, 0],
                [WALL_X + 0.15, y - 0.2, 0],
                color=WALL_COLOR, stroke_width=1.5
            )
            for y in np.linspace(-0.8, 0.8, 9)
        ])

        # 小球光晕（下层）
        glow = Circle(radius=0.35, color=GLOW_COLOR, fill_opacity=0.3, stroke_opacity=0)
        glow.move_to([3, 0, 0])

        # 小球
        ball = Circle(radius=0.2, color=BALL_COLOR, fill_opacity=1.0)
        ball.move_to([3, 0, 0])

        # 弹簧（从墙壁到球左边缘）
        spring = Spring(
            start=[WALL_X + 0.15, 0, 0],
            end=[2.8, 0, 0],
            coils=12,
            radius=0.25,
            color=SPRING_COLOR,
            stroke_width=3,
        )

        # 速度箭头
        speed_arrow = Arrow(color=RED, buff=0, max_tip_length_to_length_ratio=0.3, stroke_width=4)

        # 能量条
        ke_bar = Rectangle(width=0.3, height=0.1, color=GREEN, fill_opacity=0.6)
        pe_bar = Rectangle(width=0.3, height=0.1, color=BLUE, fill_opacity=0.6)
        ke_label = MathTex("E_k", font_size=20, color=GREEN).next_to(ke_bar, LEFT, buff=0.2)
        pe_label = MathTex("E_p", font_size=20, color=BLUE).next_to(pe_bar, LEFT, buff=0.2)

        # tracker
        tracker = create_tracker(0.0)

        # 绑定球位置
        attach_position_updater(ball, trajectory, "ball", tracker)

        # 光晕跟随球位置
        def update_glow(mob):
            t = tracker.get_value()
            x = trajectory.value_at("x", t)
            mob.move_to([x, 0, 0])
        glow.add_updater(update_glow)

        # 弹簧端点跟随球位置（球左边缘 = x - 0.2）
        def update_spring(mob):
            t = tracker.get_value()
            x = trajectory.value_at("x", t)
            mob.set_end_point([x - 0.2, 0, 0])
        spring.add_updater(update_spring)

        # 速度标签（顶部）
        v_label = Text("vx=0.00m/s", font_size=24, color=WHITE)
        v_label.move_to([3, 2.5, 0])
        def update_v_label(mob, vx):
            new_label = Text(f"vx={vx:.2f}m/s", font_size=24, color=WHITE)
            new_label.move_to([3, 2.5, 0])
            mob.become(new_label)
        attach_expr_updater(v_label, trajectory, tracker, "vx", update_v_label)
        self.add(v_label)

        # 时间标签（底部）
        t_label = Text("t=0.00", font_size=24, color=WHITE)
        t_label.move_to([3, -2.5, 0])
        def update_t_label(mob, t):
            new_label = Text(f"t={t:.2f}", font_size=24, color=WHITE)
            new_label.move_to([3, -2.5, 0])
            mob.become(new_label)
        attach_expr_updater(t_label, trajectory, tracker, "t", update_t_label)
        self.add(t_label)

        # 速度箭头：从动画开始就持续更新（修复原来只在 cross_zero_right 事件后才更新的 bug）
        def update_speed_arrow(mob):
            t = tracker.get_value()
            x = trajectory.value_at("x", t)
            vx = trajectory.value_at("vx", t)
            start = np.array([x, 0, 0])
            if abs(vx) > 1e-6:
                end = start + np.array([vx * 0.3, 0, 0])
                mob.put_start_and_end_on(start, end)
                mob.set_opacity(1.0)
            else:
                mob.set_opacity(0.0)
        speed_arrow.add_updater(update_speed_arrow)
        self.add(speed_arrow)

        # 绑定能量条
        max_e = trajectory.eval_expr("e_total", 0.0)

        def update_ke_bar(mob, v):
            h = max(0.01, v / max_e * 3.0)
            mob.stretch(h / mob.height, 1, about_edge=DOWN)

        def update_pe_bar(mob, v):
            h = max(0.01, v / max_e * 3.0)
            mob.stretch(h / mob.height, 1, about_edge=DOWN)

        attach_expr_updater(ke_bar, trajectory, tracker, "ke", update_ke_bar)
        attach_expr_updater(pe_bar, trajectory, tracker, "pe", update_pe_bar)

        # 事件触发时显示的标注
        peak_text = MathTex(r"\text{正方向振幅}", font_size=24, color=ACCENT)
        valley_text = MathTex(r"\text{负方向振幅}", font_size=24, color=ACCENT)

        def update_peak_text(mob, traj, t):
            mob.next_to(ball, UP, buff=0.3)

        def update_valley_text(mob, traj, t):
            mob.next_to(ball, UP, buff=0.3)

        # ====== 阶段1：介绍 ======
        title = Text("简谐振动 — 能量与极值分析", font_size=32, color=ACCENT)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        self.play(
            Create(equilibrium), Write(eq_label),
            Create(amp_line_r), Create(amp_line_l),
        )

        with self.voiceover(text="这是一个简谐振动系统。小球在弹性力作用下做周期运动。") as vo:
            self.play(
                FadeIn(wall), Create(wall_pattern),
                Create(spring),
                FadeIn(glow), FadeIn(ball),
                run_time=0.8,
            )
            self.wait(max(0.1, vo.duration - 0.8))

        self.play(FadeOut(title))

        # ====== 阶段2：物理仿真（分段播放） ======
        # 添加元素
        ke_bar.move_to([5, -2, 0])
        pe_bar.move_to([5.8, -2, 0])
        ke_label.next_to(ke_bar, LEFT, buff=0.2)
        pe_label.next_to(pe_bar, LEFT, buff=0.2)
        self.add(ke_bar, pe_bar, ke_label, pe_label)

        # 段1: 播放到第一个正方向峰值
        peak_times = trajectory.event_trigger_times("at_right_peak")
        t_peak1 = peak_times[0] if peak_times else t_total / 4

        with self.voiceover(text="小球从正方向振幅处释放，向平衡位置加速。") as vo:
            self.play(
                tracker.animate.set_value(t_peak1),
                run_time=max(vo.duration, t_peak1),
                rate_func=linear,
            )

        # 暂停: 在峰值处标注
        # tracker 已在峰值位置，直接显示标注
        peak_text.set_opacity(1.0)
        peak_text.next_to(ball, UP, buff=0.3)
        # 用普通 updater 让标注跟随小球
        def peak_follow(m):
            m.next_to(ball, UP, buff=0.3)
        peak_text.add_updater(peak_follow)
        self.add(peak_text)

        with self.voiceover(text="此处为正方向振幅，速度为零，势能最大。") as vo:
            self.wait(vo.duration)

        # 段2播放时标注跟随小球，到下次向左过零时淡出移除
        detach_mobject_with_event(peak_text, trajectory, tracker, "cross_zero_left", peak_follow, event_index=1, fade_out=0.3)

        # 段2: 播放到负方向峰值
        valley_times = trajectory.event_trigger_times("at_left_peak")
        t_valley1 = valley_times[1] if valley_times else t_total / 2

        with self.voiceover(text="小球过平衡位置时速度最大，然后减速到负方向振幅。") as vo:
            self.play(
                tracker.animate.set_value(t_valley1),
                run_time=max(vo.duration, t_valley1 - t_peak1),
                rate_func=linear,
            )

        # 暂停: 在谷值处标注
        valley_text.set_opacity(1.0)
        valley_text.next_to(ball, UP, buff=0.3)
        def valley_follow(m):
            m.next_to(ball, UP, buff=0.3)
        valley_text.add_updater(valley_follow)
        self.add(valley_text)

        with self.voiceover(text="此处为负方向振幅，速度再次为零，势能最大。") as vo:
            self.wait(vo.duration)

        detach_mobject_with_event(valley_text, trajectory, tracker, "cross_zero_right", valley_follow, event_index=2, fade_out=0.3)

        # 段3: 播放剩余
        with self.voiceover(text="小球在两个极值之间往复运动，机械能守恒。") as vo:
            self.play(
                tracker.animate.set_value(t_total),
                run_time=max(vo.duration, t_total - t_valley1),
                rate_func=linear,
            )

        self.wait(0.5)

        # ====== 阶段3：公式推导（分块展示） ======
        self.play(
            FadeOut(ball), FadeOut(glow), FadeOut(speed_arrow),
            FadeOut(spring), FadeOut(wall), FadeOut(wall_pattern),
            FadeOut(ke_bar), FadeOut(pe_bar), FadeOut(ke_label), FadeOut(pe_label),
            FadeOut(equilibrium), FadeOut(eq_label),
            FadeOut(amp_line_r), FadeOut(amp_line_l),
            FadeOut(v_label), FadeOut(t_label),
            run_time=0.5,
        )

        # 第一块
        block1 = VGroup(
            MathTex(r"F = -kx", font_size=32),
            MathTex(r"a = -\omega^2 x, \quad \omega = \sqrt{k/m}", font_size=32),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        block1.move_to(ORIGIN).shift(UP * 0.5)

        with self.voiceover(text="简谐振动的回复力与位移成正比，方向相反。") as vo:
            self.play(Write(block1[0]), run_time=1.0)
            self.wait(max(0.1, vo.duration - 1.0))
        with self.voiceover(text="加速度等于负 omega 平方乘以位移。") as vo:
            self.play(Write(block1[1]), run_time=1.0)
            self.wait(max(0.1, vo.duration - 1.0))

        self.play(FadeOut(block1), run_time=0.3)

        # 第二块
        block2 = VGroup(
            MathTex(r"E_k = \frac{1}{2}mv^2", font_size=32),
            MathTex(r"E_p = \frac{1}{2}kx^2 = \frac{1}{2}m\omega^2 x^2", font_size=32),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        block2.move_to(ORIGIN).shift(UP * 0.5)

        with self.voiceover(text="动能为二分之一mv方，势能为二分之一kx方。") as vo:
            self.play(Write(block2[0]), run_time=1.0)
            self.wait(max(0.1, vo.duration - 1.0))
        with self.voiceover(text="势能也可以写成二分之一m omega方 x方。") as vo:
            self.play(Write(block2[1]), run_time=1.0)
            self.wait(max(0.1, vo.duration - 1.0))

        self.play(FadeOut(block2), run_time=0.3)

        # 最终结论
        conclusion = MathTex(
            r"E = E_k + E_p = \frac{1}{2}m\omega^2 A^2 = \text{const}",
            font_size=36, color=ACCENT,
        )
        conclusion.move_to(ORIGIN)

        with self.voiceover(text="总机械能等于动能加势能，恒等于二分之一m omega方振幅方，保持不变。") as vo:
            self.play(Write(conclusion), run_time=1.5)
            box = SurroundingRectangle(conclusion, color=ACCENT)
            self.play(Create(box), run_time=0.5)
            self.wait(max(0.1, vo.duration - 2.0))

        self.wait(1.0)
        self.play(FadeOut(conclusion), FadeOut(box), run_time=0.5)


if __name__ == "__main__":
    SHMFeatureDemoScene().render()
