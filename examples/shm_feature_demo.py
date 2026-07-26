"""斜面弹簧振子 — 等效平衡位置与能量分析。

30° 光滑斜面上的弹簧振子，置于重力场（向下）和匀强电场（水平向右）中。
展示：等效平衡位置推导、简谐运动证明、极值受力分析、能量转化曲线。
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
from math import pi, cos, sin

# 使用 pdflatex 替代 xelatex，避免 MiKTeX 间歇性崩溃
_tex_template = TexTemplate()
_tex_template.tex_compiler = "latex"
_tex_template.output_format = ".dvi"
Tex.default_template = _tex_template
MathTex.default_template = _tex_template

# 配色
BG_COLOR = "#1C2333"
ACCENT = GOLD_C
BALL_COLOR = YELLOW
GLOW_COLOR = "#FFD966"
SPRING_COLOR = "#E8E8E8"
WALL_COLOR = GREY_D
GRAVITY_COLOR = RED_C
ELECTRIC_COLOR = BLUE_C
SPRING_FORCE_COLOR = GREEN_C

# 物理参数
THETA = pi / 6
COS_T = cos(THETA)
SIN_T = sin(THETA)
MASS = 1.0
K_SPRING = 4.0
G_GRAV = 10.0
Q_CHARGE = 1.0
L0 = 0.5
S_EQ = 2.0
E_FIELD = (K_SPRING * (S_EQ - L0) + MASS * G_GRAV * SIN_T) / (Q_CHARGE * COS_T)
OMEGA = (K_SPRING / MASS) ** 0.5

# 屏幕布局
INCLINE_X = -4.0
INCLINE_Y = -1.5
INCLINE_LENGTH = 6.0
TANGENT = np.array([COS_T, SIN_T, 0.0])
NORMAL = np.array([-SIN_T, COS_T, 0.0])


def build_animation() -> PhysicsAnimation:
    """斜面弹簧振子: m*s'' = -k*(s - s_eq)."""
    animation = PhysicsAnimation(
        global_parameters={
            "theta": THETA,
            "m": MASS,
            "k": K_SPRING,
            "g": G_GRAV,
            "q": Q_CHARGE,
            "E": E_FIELD,
            "L0": L0,
            "s_eq": S_EQ,
        },
        engine="scipy",
        sample_dt=1 / 120,
    )

    ball = PhysicObject2D(
        object_id="ball",
        state_variables={
            "s": StateVariable("s", "m", "沿斜面位移"),
            "vs": StateVariable("vs", "m/s", "沿斜面速度"),
        },
        cartesian_position=[("screen_x", "screen_y")],
    )
    animation.add_object(ball, {"s": 3.0, "vs": 0.0})

    animation.add_segment(
        PhysicsSegment(
            segment_id="incline_shm",
            object_ids=["ball"],
            state_vector=["s", "vs"],
            equations={"s": "vs", "vs": "-(k/m) * (s - s_eq)"},
            state_owners={"s": "ball", "vs": "ball"},
            derived_equations={
                "screen_x": f"{INCLINE_X} + s * cos(theta)",
                "screen_y": f"{INCLINE_Y} + s * sin(theta)",
                "ke": "0.5 * m * vs**2",
                "pe_g": "m * g * s * sin(theta)",
                "pe_s": "0.5 * k * (s - L0)**2",
                "pe_e": "-q * E * s * cos(theta)",
                "mech_e": "0.5 * m * vs**2 + m * g * s * sin(theta) + 0.5 * k * (s - L0)**2",
                "total_e": "0.5 * m * vs**2 + m * g * s * sin(theta) + 0.5 * k * (s - L0)**2 - q * E * s * cos(theta)",
                "f_gravity_par": "-m * g * sin(theta)",
                "f_electric_par": "q * E * cos(theta)",
                "f_spring": "-k * (s - L0)",
                "s_minus_eq": "s - s_eq",
            },
            duration=100,
        ),
        end_event=time_countdown_event(4 * pi),
    )

    animation.register_event("cross_eq_left", "s_minus_eq", direction=-1)
    animation.register_event("cross_eq_right", "s_minus_eq", direction=1)
    animation.register_event("at_max_pos", "is_local_max(s)", event_type="bool")
    animation.register_event("at_min_pos", "is_local_min(s)", event_type="bool")

    return animation


def screen_pos(s: float) -> np.ndarray:
    """物理坐标 s -> 屏幕坐标 (x, y, 0)."""
    return np.array([INCLINE_X + s * COS_T, INCLINE_Y + s * SIN_T, 0.0])


class SHMFeatureDemoScene(PhyAnimScene):
    def construct(self):
        self.setup_speech(TTS_CONFIG)
        self.camera.background_color = BG_COLOR
        self.camera.frame_width = 16

        # 覆盖 PhyAnimScene.__init__ 设置的 xelatex 模板，改用 pdflatex 避免 MiKTeX 崩溃
        # _pdflatex_template = TexTemplate()
        # _pdflatex_template.tex_compiler = "latex"
        # _pdflatex_template.output_format = ".dvi"
        # MathTex.set_default(tex_template=_pdflatex_template)
        # Tex.set_default(tex_template=_pdflatex_template)

        animation = build_animation()
        trajectory = solve_animation(animation)
        t_total = trajectory.total_time

        print("=== Registered Events ===")
        for eid, times in trajectory.all_events().items():
            print(f"  {eid}: {[f'{t:.4f}' for t in times]}")

        # ====== 场景元素 ======
        # 斜面
        incline = InclinedPlane(
            length=INCLINE_LENGTH,
            angle=THETA,
            start=np.array([INCLINE_X, INCLINE_Y, 0.0]),
            color=GREY,
            fill_opacity=0.1,
            stroke_width=2,
        )

        # 挡板（垂直于斜面的小矩形）
        wall_bottom = np.array([INCLINE_X, INCLINE_Y, 0.0])
        wall_top = wall_bottom + NORMAL * 0.5
        wall_left = wall_bottom - TANGENT * 0.05
        wall_right = wall_bottom + TANGENT * 0.05
        wall = Polygon(
            wall_left, wall_right,
            wall_right + NORMAL * 0.5, wall_left + NORMAL * 0.5,
            color=WALL_COLOR, fill_opacity=0.8, stroke_width=2,
        )

        # 弹簧（从挡板顶部到球边缘）
        spring_start = wall_bottom + TANGENT * 0.1 + NORMAL * 0.05
        ball_init_pos = screen_pos(3.0)
        spring_end = ball_init_pos - TANGENT * 0.2
        spring = Spring(
            start=spring_start,
            end=spring_end,
            coils=12,
            radius=0.2,
            color=SPRING_COLOR,
            stroke_width=3,
        )

        # 球光晕 + 球
        glow = Circle(radius=0.35, color=GLOW_COLOR, fill_opacity=0.3, stroke_opacity=0)
        glow.move_to(ball_init_pos)
        ball = Circle(radius=0.2, color=BALL_COLOR, fill_opacity=1.0)
        ball.move_to(ball_init_pos)

        # 平衡位置标记（虚线，垂直于斜面）
        eq_pos = screen_pos(S_EQ)
        eq_mark = DashedLine(
            eq_pos - NORMAL * 0.4, eq_pos + NORMAL * 0.4,
            color=ACCENT, stroke_width=2,
        )
        eq_label = MathTex("O'", font_size=20, color=ACCENT).next_to(eq_mark, NORMAL * 0.5, buff=0.1)

        # 重力场可视化（向下红色箭头阵列）
        gravity_arrows = VGroup(*[
            Arrow(
                np.array([x, 2.5, 0.0]),
                np.array([x, 1.8, 0.0]),
                color=GRAVITY_COLOR, buff=0, stroke_width=2,
                max_tip_length_to_length_ratio=0.3,
            )
            for x in [-3, -1.5, 0, 1.5, 3]
        ])
        gravity_label = MathTex(r"\vec{g}", font_size=24, color=GRAVITY_COLOR)
        gravity_label.move_to([3.8, 2.15, 0])

        # 电场可视化（向右蓝色箭头阵列）
        electric_arrows = VGroup(*[
            Arrow(
                np.array([-5.5, y, 0.0]),
                np.array([-4.8, y, 0.0]),
                color=ELECTRIC_COLOR, buff=0, stroke_width=2,
                max_tip_length_to_length_ratio=0.3,
            )
            for y in [0.5, 1.0, 1.5, 2.0]
        ])
        electric_label = MathTex(r"\vec{E}", font_size=24, color=ELECTRIC_COLOR)
        electric_label.move_to([-5.15, 2.5, 0])

        # 速度箭头
        speed_arrow = Arrow(color=RED, buff=0, max_tip_length_to_length_ratio=0.2, stroke_width=3)

        # 数值标签
        s_label = Text("s=3.00m", font_size=22, color=WHITE)
        s_label.move_to([4, 2.5, 0])
        vs_label = Text("vs=0.00m/s", font_size=22, color=WHITE)
        vs_label.move_to([4, 2.0, 0])

        # tracker
        tracker = create_tracker(0.0)

        # 绑定球位置
        attach_position_updater(ball, trajectory, "ball", tracker)

        # 光晕跟随
        def update_glow(mob):
            t = tracker.get_value()
            sx = trajectory.value_at("screen_x", t)
            sy = trajectory.value_at("screen_y", t)
            mob.move_to([sx, sy, 0])
        glow.add_updater(update_glow)

        # 弹簧端点跟随球
        def update_spring(mob):
            t = tracker.get_value()
            sx = trajectory.value_at("screen_x", t)
            sy = trajectory.value_at("screen_y", t)
            ball_pos = np.array([sx, sy, 0])
            mob.set_end_point(ball_pos - TANGENT * 0.2)
        spring.add_updater(update_spring)

        # 速度箭头沿斜面方向，起点在球心
        def update_speed_arrow(mob):
            t = tracker.get_value()
            sx = trajectory.value_at("screen_x", t)
            sy = trajectory.value_at("screen_y", t)
            vs = trajectory.value_at("vs", t)
            ball_pos = np.array([sx, sy, 0])
            start = ball_pos
            if abs(vs) > 1e-6:
                end = start + TANGENT * vs * 0.4
                mob.put_start_and_end_on(start, end)
                mob.set_opacity(1.0)
            else:
                mob.set_opacity(0.0)
        speed_arrow.add_updater(update_speed_arrow)

        # 数值标签 updater
        def update_s_label(mob, s_val):
            new = Text(f"s={s_val:.2f}m", font_size=22, color=WHITE)
            new.move_to([4, 2.5, 0])
            mob.become(new)
        attach_expr_updater(s_label, trajectory, tracker, "s", update_s_label)

        def update_vs_label(mob, vs_val):
            new = Text(f"vs={vs_val:.2f}m/s", font_size=22, color=WHITE)
            new.move_to([4, 2.0, 0])
            mob.become(new)
        attach_expr_updater(vs_label, trajectory, tracker, "vs", update_vs_label)

        # ====== 阶段1：场景介绍 ======
        title = Text("斜面弹簧振子 — 等效平衡位置与能量分析", font_size=28, color=ACCENT)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        self.play(Create(incline), Create(wall))

        with self.voiceover(text="30度光滑斜面上，弹簧一端固定在挡板，另一端连接小球。空间中有向下的重力场和向右的匀强电场。") as vo:
            self.play(
                Create(spring),
                FadeIn(glow), FadeIn(ball),
                Create(gravity_arrows), Write(gravity_label),
                Create(electric_arrows), Write(electric_label),
                run_time=max(0.1, vo.duration),
            )

        self.play(FadeOut(title))

        # ====== 阶段2：受力分析与等效平衡位置 ======
        # 显示平衡位置标记
        self.play(Create(eq_mark), Write(eq_label))

        # 受力分析箭头（在初始位置 s=3）
        ball_pos = screen_pos(3.0)
        f_grav = MASS * G_GRAV * SIN_T      # 5.0 N, 沿斜面向下
        f_elec = Q_CHARGE * E_FIELD * COS_T  # 11.0 N, 沿斜面向上
        f_spr = K_SPRING * (3.0 - L0)        # 10.0 N, 沿斜面向下（弹簧拉伸）

        force_scale = 0.15
        grav_arrow = Arrow(
            ball_pos, ball_pos - TANGENT * f_grav * force_scale,
            color=GRAVITY_COLOR, buff=0, stroke_width=4,
        )
        elec_arrow = Arrow(
            ball_pos, ball_pos + TANGENT * f_elec * force_scale,
            color=ELECTRIC_COLOR, buff=0, stroke_width=4,
        )
        spr_arrow = Arrow(
            ball_pos, ball_pos - TANGENT * f_spr * force_scale,
            color=SPRING_FORCE_COLOR, buff=0, stroke_width=4,
        )

        grav_label = MathTex(f"mg\\sin\\theta={f_grav:.0f}N", font_size=20, color=GRAVITY_COLOR)
        grav_label.next_to(grav_arrow, NORMAL * 0.5, buff=0.1)
        elec_label = MathTex(f"qE\\cos\\theta={f_elec:.0f}N", font_size=20, color=ELECTRIC_COLOR)
        elec_label.next_to(elec_arrow, NORMAL * 0.5, buff=0.1)
        spr_label = MathTex(f"k(s-L_0)={f_spr:.0f}N", font_size=20, color=SPRING_FORCE_COLOR)
        spr_label.next_to(spr_arrow, -NORMAL * 0.5, buff=0.1)

        with self.voiceover(text="小球在最大拉伸处受力分析：重力沿斜面向下分量、电场力沿斜面向上分量、弹簧力沿斜面向下。") as vo:
            self.play(
                GrowArrow(grav_arrow), Write(grav_label),
                run_time=1.0,
            )
            self.wait(0.5)
            self.play(
                GrowArrow(elec_arrow), Write(elec_label),
                run_time=1.0,
            )
            self.wait(0.5)
            self.play(
                GrowArrow(spr_arrow), Write(spr_label),
                run_time=1.0,
            )
            self.wait(max(0.1, vo.duration - 3.5))

        # 推导等效平衡位置公式
        eq_formula = VGroup(
            MathTex(r"s_{eq} = L_0 + \frac{qE\cos\theta - mg\sin\theta}{k}", font_size=28, color=ACCENT),
            MathTex(r"s_{eq} = 0.5 + \frac{11 - 5}{4} = 2.0\,m", font_size=28),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        eq_formula.to_edge(RIGHT, buff=0.5).shift(UP * 1.5)

        with self.voiceover(text="等效平衡位置是重力、电场力和弹簧力合力为零的位置。它是小球振动的中心，小球围绕该点做简谐运动。代入参数，平衡位置在 s 等于 2 米处。") as vo:
            self.play(Write(eq_formula[0]), run_time=1.5)
            self.wait(0.3)
            self.play(Write(eq_formula[1]), run_time=1.5)
            self.wait(max(0.1, vo.duration - 3.3))

        # 清理受力分析
        self.play(
            FadeOut(grav_arrow), FadeOut(grav_label),
            FadeOut(elec_arrow), FadeOut(elec_label),
            FadeOut(spr_arrow), FadeOut(spr_label),
            FadeOut(eq_formula),
            run_time=0.5,
        )

        # ====== 阶段3：证明简谐运动 ======
        proof = VGroup(
            MathTex(r"m\ddot{s} = -k(s - s_{eq})", font_size=32),
            MathTex(r"\text{令 } x = s - s_{eq}", font_size=28),
            MathTex(r"m\ddot{x} = -kx \Rightarrow \ddot{x} = -\omega^2 x", font_size=32),
            MathTex(r"\omega = \sqrt{k/m} = 2\,\text{rad/s}", font_size=28, color=ACCENT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        proof.to_edge(RIGHT, buff=0.5).shift(UP * 1.0)

        with self.voiceover(text="运动方程为 m 乘以 s 二阶导等于负 k 乘以 s 减 s_eq。令 x 等于 s 减 s_eq，得 m x 二阶导等于负 k x，即简谐运动方程。角频率 omega 等于根号 k 比 m 等于 2。") as vo:
            for i, line in enumerate(proof):
                self.play(Write(line), run_time=1.0)
                if i < len(proof) - 1:
                    self.wait(0.3)
            self.wait(max(0.1, vo.duration - 4.5))

        self.play(FadeOut(proof), run_time=0.5)

        # ====== 阶段4：仿真与极值受力分析 ======
        self.add(speed_arrow, s_label, vs_label)

        # 段1: 从 s=3 到平衡位置
        cross_times = trajectory.event_trigger_times("cross_eq_left")
        t_cross1 = cross_times[0] if cross_times else t_total / 4

        with self.voiceover(text="小球从最大拉伸处释放，向平衡位置加速运动。") as vo:
            self.play(
                tracker.animate.set_value(t_cross1),
                run_time=max(vo.duration, t_cross1),
                rate_func=linear,
            )

        # 段2: 到最小位置 s=1
        min_times = trajectory.event_trigger_times("at_min_pos")
        t_min1 = min_times[0] if min_times else t_total / 2

        with self.voiceover(text="小球过平衡位置后继续减速，到达负方向最大偏离处。") as vo:
            self.play(
                tracker.animate.set_value(t_min1),
                run_time=max(vo.duration, t_min1 - t_cross1),
                rate_func=linear,
            )

        # 暂停：在 s=1 做受力分析
        self._show_force_analysis(trajectory, tracker, 1.0)

        # 段3: 回到最大位置 s=3
        max_times = trajectory.event_trigger_times("at_max_pos")
        t_max2 = max_times[0] if max_times else 3 * t_total / 4

        with self.voiceover(text="小球反向加速，回到正方向最大偏离处。") as vo:
            self.play(
                tracker.animate.set_value(t_max2),
                run_time=max(vo.duration, t_max2 - t_min1),
                rate_func=linear,
            )

        # 暂停：在 s=3 做受力分析
        self._show_force_analysis(trajectory, tracker, 3.0)

        # 段4: 播放剩余
        with self.voiceover(text="小球在两个极值之间往复运动，总能量守恒。") as vo:
            self.play(
                tracker.animate.set_value(t_total),
                run_time=max(vo.duration, t_total - t_max2),
                rate_func=linear,
            )

        # ====== 阶段5：能量分析 ======
        self.play(
            FadeOut(speed_arrow), FadeOut(s_label), FadeOut(vs_label),
            FadeOut(gravity_arrows), FadeOut(gravity_label),
            FadeOut(electric_arrows), FadeOut(electric_label),
            FadeOut(ball), FadeOut(glow), FadeOut(spring),
            FadeOut(incline), FadeOut(wall),
            FadeOut(eq_mark), FadeOut(eq_label),
            run_time=0.5,
        )

        # 清除 updater，防止 tracker 重置时这些 mobject 跳回起点
        for mob in [ball, glow, spring]:
            mob.clear_updaters()

        # 预采样能量数据
        times_arr, ke_arr = trajectory.sample("ke", dt=0.05)
        _, pe_g_arr = trajectory.sample("pe_g", dt=0.05)
        _, pe_s_arr = trajectory.sample("pe_s", dt=0.05)
        _, pe_e_arr = trajectory.sample("pe_e", dt=0.05)
        _, mech_e_arr = trajectory.sample("mech_e", dt=0.05)

        # 创建 Axes
        energy_axes = Axes(
            x_range=[0, t_total, t_total / 4],
            y_range=[-35, 30, 10],
            x_length=5.0,
            y_length=3.5,
            tips=False,
            axis_config={"color": GREY, "font_size": 16},
        )
        energy_axes.to_edge(RIGHT, buff=0.5).shift(DOWN * 0.5)

        # 轴标签
        x_label = MathTex("t", font_size=20, color=GREY).next_to(energy_axes.x_axis, DOWN, buff=0.2)
        y_label = MathTex("E", font_size=20, color=GREY).next_to(energy_axes.y_axis, UP, buff=0.2)

        # 能量曲线（逐步绘制）
        def make_curve_updater(data_arr, axes_obj):
            def updater(mob):
                t = tracker.get_value()
                mask = [i for i, tt in enumerate(times_arr) if tt <= t]
                if len(mask) < 2:
                    mob.set_points_as_corners([axes_obj.c2p(0, 0), axes_obj.c2p(0, 0)])
                    return
                pts = [axes_obj.c2p(times_arr[i], data_arr[i]) for i in mask]
                mob.set_points_as_corners(pts)
            return updater

        ke_curve = VMobject().set_stroke(GREEN, width=2)
        pe_g_curve = VMobject().set_stroke(BLUE, width=2)
        pe_s_curve = VMobject().set_stroke(PURPLE, width=2)
        pe_e_curve = VMobject().set_stroke(ORANGE, width=2)
        mech_e_curve = VMobject().set_stroke(YELLOW, width=3)

        ke_curve.add_updater(make_curve_updater(ke_arr, energy_axes))
        pe_g_curve.add_updater(make_curve_updater(pe_g_arr, energy_axes))
        pe_s_curve.add_updater(make_curve_updater(pe_s_arr, energy_axes))
        pe_e_curve.add_updater(make_curve_updater(pe_e_arr, energy_axes))
        mech_e_curve.add_updater(make_curve_updater(mech_e_arr, energy_axes))

        # 图例
        legend = VGroup(
            Text("动能 Ek", font_size=14, color=GREEN),
            Text("重力势能 Ep_g", font_size=14, color=BLUE),
            Text("弹性势能 Ep_s", font_size=14, color=PURPLE),
            Text("电势能 Ep_e", font_size=14, color=ORANGE),
            Text("机械能 Em", font_size=14, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        legend.next_to(energy_axes, UP, buff=0.2)

        with self.voiceover(text="现在分析能量变化。绿色是动能，蓝色是重力势能，紫色是弹性势能，橙色是电势能，黄色是机械能。注意机械能不守恒，但总能量守恒。") as vo:
            self.play(Create(energy_axes), Write(legend), Write(x_label), Write(y_label), run_time=1.0)
            self.add(ke_curve, pe_g_curve, pe_s_curve, pe_e_curve, mech_e_curve)

            # 重置 tracker 重新播放
            tracker.set_value(0.0)
            self.play(
                tracker.animate.set_value(t_total),
                run_time=max(vo.duration, t_total),
                rate_func=linear,
            )

        # 能量公式总结
        energy_formula = VGroup(
            MathTex(r"E_k = \frac{1}{2}mv^2", font_size=24, color=GREEN),
            MathTex(r"E_{p,g} = mgs\sin\theta", font_size=24, color=BLUE),
            MathTex(r"E_{p,s} = \frac{1}{2}k(s-L_0)^2", font_size=24, color=PURPLE),
            MathTex(r"E_{p,e} = -qEs\cos\theta", font_size=24, color=ORANGE),
            MathTex(r"E_{total} = E_k + E_{p,g} + E_{p,s} + E_{p,e} = \text{const}", font_size=24, color=ACCENT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        energy_formula.to_edge(LEFT, buff=0.5).shift(UP * 1.5)

        with self.voiceover(text="总能量等于动能加重力势能加弹性势能加电势能，恒定不变。机械能等于前三项之和，不守恒，因为电场力做功。") as vo:
            for i, line in enumerate(energy_formula):
                self.play(Write(line), run_time=0.8)
                if i < len(energy_formula) - 1:
                    self.wait(0.2)
            self.wait(max(0.1, vo.duration - 4.5))

        self.wait(1.0)
        self.play(
            FadeOut(energy_formula), FadeOut(energy_axes), FadeOut(legend),
            FadeOut(ke_curve), FadeOut(pe_g_curve), FadeOut(pe_s_curve),
            FadeOut(pe_e_curve), FadeOut(mech_e_curve),
            FadeOut(x_label), FadeOut(y_label),
            run_time=0.8,
        )

    def _show_force_analysis(self, trajectory, tracker, s_val):
        """在当前球位置逐个显示三个力分量箭头 + 合力箭头。"""
        ball_pos = screen_pos(s_val)
        f_grav = MASS * G_GRAV * SIN_T
        f_elec = Q_CHARGE * E_FIELD * COS_T
        f_spr = K_SPRING * (s_val - L0)

        # 合力（沿 TANGENT 正方向 = 沿斜面向上）
        f_net = -f_grav + f_elec - f_spr

        force_scale = 0.15

        # 重力分量（沿斜面向下）
        grav_arrow = Arrow(
            ball_pos, ball_pos - TANGENT * f_grav * force_scale,
            color=GRAVITY_COLOR, buff=0, stroke_width=4,
        )
        # 电场力分量（沿斜面向上）
        elec_arrow = Arrow(
            ball_pos, ball_pos + TANGENT * f_elec * force_scale,
            color=ELECTRIC_COLOR, buff=0, stroke_width=4,
        )
        # 弹簧力（方向取决于 s vs L0）
        spr_dir = -TANGENT if f_spr > 0 else TANGENT
        spr_arrow = Arrow(
            ball_pos, ball_pos + spr_dir * abs(f_spr) * force_scale,
            color=SPRING_FORCE_COLOR, buff=0, stroke_width=4,
        )
        # 合力箭头（偏移至斜面上方避免重叠）
        net_start = ball_pos + NORMAL * 0.15
        net_dir = TANGENT if f_net > 0 else -TANGENT
        net_arrow = Arrow(
            net_start, net_start + net_dir * abs(f_net) * force_scale,
            color=WHITE, buff=0, stroke_width=5,
        )

        grav_lbl = MathTex(f"mg\\sin\\theta={f_grav:.0f}N", font_size=20, color=GRAVITY_COLOR)
        grav_lbl.next_to(grav_arrow, NORMAL * 0.5, buff=0.1)
        elec_lbl = MathTex(f"qE\\cos\\theta={f_elec:.0f}N", font_size=20, color=ELECTRIC_COLOR)
        elec_lbl.next_to(elec_arrow, NORMAL * 0.5, buff=0.1)
        spr_lbl = MathTex(f"F_{{spring}}={f_spr:.0f}N", font_size=20, color=SPRING_FORCE_COLOR)
        spr_lbl.next_to(spr_arrow, -NORMAL * 0.5, buff=0.1)
        net_lbl = MathTex(f"F_{{net}}={abs(f_net):.0f}N", font_size=22, color=WHITE)
        net_lbl.next_to(net_arrow, NORMAL * 0.3, buff=0.1)

        # 方向描述
        dir_text = "沿斜面向上" if f_net > 0 else "沿斜面向下"

        pos_text = Text(f"s={s_val:.0f}m", font_size=22, color=ACCENT)
        pos_text.next_to(ball_pos, NORMAL * 0.8, buff=0.3)

        with self.voiceover(
            text=f"在 s 等于 {s_val:.0f} 米处受力分析。"
            f"重力沿斜面向下 {f_grav:.0f} 牛，"
            f"电场力沿斜面向上 {f_elec:.0f} 牛，"
            f"弹簧力沿斜面向下 {abs(f_spr):.0f} 牛。"
            f"合力大小为 {abs(f_net):.0f} 牛，方向{dir_text}，指向平衡位置。"
        ) as vo:
            self.play(Write(pos_text), run_time=0.5)
            self.play(GrowArrow(grav_arrow), Write(grav_lbl), run_time=1.0)
            self.wait(0.5)
            self.play(GrowArrow(elec_arrow), Write(elec_lbl), run_time=1.0)
            self.wait(0.5)
            self.play(GrowArrow(spr_arrow), Write(spr_lbl), run_time=1.0)
            self.wait(0.5)
            self.play(GrowArrow(net_arrow), Write(net_lbl), run_time=1.0)
            self.wait(max(0.1, vo.duration - 5.5))

        self.play(
            FadeOut(grav_arrow), FadeOut(grav_lbl),
            FadeOut(elec_arrow), FadeOut(elec_lbl),
            FadeOut(spr_arrow), FadeOut(spr_lbl),
            FadeOut(net_arrow), FadeOut(net_lbl),
            FadeOut(pos_text),
            run_time=0.5,
        )


if __name__ == "__main__":
    SHMFeatureDemoScene().render()
