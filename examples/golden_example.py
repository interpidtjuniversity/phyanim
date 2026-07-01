from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# TTS_CONFIG 由运行环境注入；直接运行时使用测试配置
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

# --- 物理参数 ---
R = 1       # 半圆半径
W = 4       # 轨道宽度
H = 2       # 轨道高度
m_ball = 5.0  # 小球质量
M_track = 10.0  # 轨道质量
g = 9.8       # 重力加速度

BG_COLOR = BLUE_E
MAIN_COLOR = PURPLE_B
STROKE_COLOR = TEAL_C
CONCLUSION_COLOR = GOLD_C


def build_animation() -> PhysicsAnimation:
    """构建物理动画：小球在凹槽轨道内滑动，水平动量守恒。"""
    animation = PhysicsAnimation(
        global_parameters={"g": g, "m": m_ball, "M": M_track, "R": R},
        engine="scipy",
        sample_dt=1 / 60,
    )

    # 小球对象：广义坐标 phi（角度）和 dphi（角速度）
    # 轨道位置 X 由动量守恒约束决定，不是独立状态，作为 derived 计算
    ball = PhysicObject2D(
        object_id="ball",
        state_variables={
            "theta": StateVariable("theta", "rad", "角度"),
            "omega": StateVariable("omega", "rad/s", "角速度"),
        },
        cartesian_position=[("x_ball", "y_ball")],
    )

    # 轨道对象：无独立状态，位置由 derived 驱动
    track = PhysicObject2D(
        object_id="track",
        state_variables={
            "x_track": StateVariable("x_track", "m", "轨道中心水平位置"),
            "y_track": StateVariable("y_track", "m", "轨道中心垂直位置"),
        },
        cartesian_position=[("x_track", "y_track")],
    )

    animation.add_object(ball, {"theta": -pi / 2, "omega": 0.0})
    animation.add_object(track, {"x_track": 0.0, "y_track": 0.0})

    # 拉格朗日方程约化后的 ODE（无约束广义坐标）
    animation.add_segment(
        PhysicsSegment(
            segment_id="swing",
            object_ids=["ball", "track"],
            state_vector = ["theta", "omega", "x_track", "y_track"],
            equations={
                "theta": "omega",
                "omega": "-(m*omega**2*sin(theta)*cos(theta) + g*(M+m)*sin(theta)/R) / (M + m*sin(theta)**2)",
                "x_track": "-m*R*cos(theta)*omega/(M+m)",
                "y_track": "0",
            },
                state_owners = {
                "theta": "ball",
                "omega": "ball",
                "x_track": "track",
                "y_track": "track",
            },
            # derived: 笛卡尔渲染坐标
            derived_equations = {
                "x_ball": "x_track + R*sin(theta)",
                "y_ball": "R*(1 - cos(theta))",
            },
            duration=100,
        ),
        end_event=time_countdown_event(15),
    )

    return animation


def create_track_geom():
    """构建内凹轨道几何体。"""
    track = VMobject()
    left = W / 2
    top = H / 2
    track.start_new_path(np.array([-R, top, 0]))
    track.add_line_to(np.array([-left, top, 0]))
    track.add_line_to(np.array([-left, -top, 0]))
    track.add_line_to(np.array([left, -top, 0]))
    track.add_line_to(np.array([left, top, 0]))
    track.add_line_to(np.array([R, top, 0]))
    for theta in np.linspace(0, -np.pi, 60):
        track.add_line_to(np.array([R * np.cos(theta), top + R * np.sin(theta), 0]))
    track.close_path()
    track.set_fill(MAIN_COLOR, opacity=0.6)
    track.set_stroke(STROKE_COLOR, width=3)
    return track


class GoldenExampleScene(PhyAnimScene):
    def construct(self):
        self.setup_speech(TTS_CONFIG)
        MathTex.set_default(tex_template=TexTemplateLibrary.ctex)
        self.camera.background_color = BG_COLOR

        # 求解物理
        animation = build_animation()
        trajectory = solve_animation(animation)
        t_total = trajectory.total_time

        # ==================== 阶段1：题目介绍 ====================
        title = Text("凹槽轨道与小球的动量守恒", font="SimSun", font_size=32, color=CONCLUSION_COLOR)
        title.to_edge(UP, buff=0.5)

        ground = Line(LEFT * 6 + DOWN * H/2, RIGHT * 6 + DOWN * H/2, color=GRAY, stroke_width=4)
        track = create_track_geom()
        track.move_to(np.array([0, 0, 0]))
        ball = Circle(radius=0.1, color=YELLOW)
        ball.set_fill(YELLOW, opacity=1.0)
        ball.move_to(np.array([-R, H/2, 0]))

        track_label = Text("M = 10 kg", font="SimSun", font_size=20, color=WHITE)
        track_label.move_to(track.get_center() + DOWN * 0.4)
        ball_label = Text("m = 5 kg", font="SimSun", font_size=18, color=WHITE)
        ball_label.next_to(ball, UP, buff=0.15)

        self.play(FadeIn(title), Create(ground))

        with self.voiceover(text="我们有一个质量为10千克的内凹轨道，放置在光滑水平面上。") as tracker:
            self.play(DrawBorderThenFill(track), Write(track_label))
            self.wait(tracker.duration)

        with self.voiceover(text="在轨道左侧最高点，放置一个质量为5千克的小球，由静止释放。") as tracker:
            self.play(FadeIn(ball), Write(ball_label))
            self.wait(tracker.duration)

        self.play(FadeOut(ball_label), FadeOut(title))

        # ==================== 阶段2：物理仿真（分段播放） ====================
        # 质心红线
        center_line = DashedLine(UP * 1.5 + LEFT * (m_ball * R + M_track * 0) / (m_ball + M_track), DOWN * 3 + (m_ball * R + M_track * 0) / (m_ball + M_track) * LEFT, color=RED, stroke_width=2)
        center_label = Text(f"系统水平质心 (x = {-(m_ball * R + M_track * 0) / (m_ball + M_track):.2f})", font="SimSun", font_size=16, color=RED)
        center_label.next_to(center_line, UP, buff=0.1)

        # 时间驱动器
        t_tracker = ValueTracker(0.0)

        # 绑定物体位置到轨迹
        attach_position_updater(track, trajectory, "track", t_tracker)
        attach_position_updater(ball, trajectory, "ball", t_tracker)

        # 轨道标签跟随
        track_label.add_updater(lambda mob: mob.move_to(track.get_center() + DOWN * 0.4))

        with self.voiceover(text="由于水平面完全光滑，系统在水平方向不受外力，其水平质心始终保持静止。") as tracker:
            self.play(Create(center_line), Write(center_label))
            self.wait(tracker.duration)

        # --- 段1：播放前 6 秒物理动画 ---
        with self.voiceover(text="现在释放小球，可以看到轨道在小球下落过程中会向右反冲运动。") as tracker:
            t1 = min(6.0, t_total)
            self.play(
                t_tracker.animate.set_value(t1),
                run_time=max(tracker.duration, t1),
                rate_func=linear,
            )

        # --- 暂停：受力分析 ---
        with self.voiceover(text="此时小球受重力和轨道支持力，轨道受小球的反作用力水平向右。") as tracker:
            ball_center = ball.get_center()
            # 重力箭头
            g_arrow = Arrow(ball_center, ball_center + DOWN * 0.8, color=RED, buff=0.1, stroke_width=5)
            g_label = MathTex("mg", color=RED, font_size=24).next_to(g_arrow, RIGHT, buff=0.1)
            # 轨道受力箭头
            track_x = trajectory.value_at("x_track", t_tracker.get_value())
            track_y = trajectory.value_at("y_track", t_tracker.get_value())
            track_center = np.array([track_x, track_y + H/2, 0])
            f_arrow = Arrow(ball_center, (track_center - ball_center) * 0.8 + ball_center, color=BLUE, buff=0.1, stroke_width=5)
            f_label = MathTex("f", color=BLUE, font_size=24).next_to(f_arrow, UP, buff=0.1)

            self.play(FadeIn(g_arrow, g_label, f_arrow, f_label), run_time=0.5)
            self.wait(max(0.1, tracker.duration - 0.5))
            self.play(FadeOut(g_arrow, g_label, f_arrow, f_label), run_time=0.3)

        # --- 段2：继续播放剩余物理动画 ---
        with self.voiceover(text="小球在轨道内来回摆动，但系统质心始终不动。") as tracker:
            self.play(
                t_tracker.animate.set_value(t_total),
                run_time=max(tracker.duration, t_total - 6.0),
                rate_func=linear,
            )


        self.wait(0.5)

        # ==================== 阶段3：公式推导 ====================
        track.clear_updaters()
        ball.clear_updaters()
        track_label.clear_updaters()

        sim_group = VGroup(track, ball, track_label, center_line, center_label, ground)
        self.play(
            sim_group.animate.move_to(LEFT * 3.5),
            run_time=0.5,
        )

        # 公式组（注意排版：检查高度，留足间距）
        formulas_tex = [
            r"m v_x + M V_x = 0",
            r"m \Delta x_m + M \Delta x_M = 0",
            r"\Delta x_M = -\frac{m}{M} \Delta x_m",
            r"\Delta x_M = -0.5 \, \Delta x_m",
        ]
        formulas = VGroup(*[MathTex(tex, font_size=32) for tex in formulas_tex])
        formulas.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        # 检查是否超出画面
        if formulas.height > 6:
            formulas.scale_to_fit_height(6)
        formulas.to_edge(RIGHT, buff=1.0).shift(UP * 0.5)

        with self.voiceover(text="在水平方向上，小球与轨道组成的系统动量始终守恒。") as tracker:
            self.play(Write(formulas[0]), run_time=1.0)
            self.wait(max(0.1, tracker.duration - 1.0))

        with self.voiceover(text="将动量关系对时间积分，可得水平位移的守恒关系。") as tracker:
            self.play(Write(formulas[1]), run_time=1.0)
            self.wait(max(0.1, tracker.duration - 1.0))

        with self.voiceover(text="轨道位移与小球水平位移比值恰好与质量比相反。") as tracker:
            self.play(Write(formulas[2]), run_time=1.0)
            self.wait(max(0.1, tracker.duration - 1.0))

        with self.voiceover(text="轨道质量为10千克，小球为5千克，轨道位移为小球的一半，方向相反。") as tracker:
            formulas[3].set_color(CONCLUSION_COLOR)
            self.play(Write(formulas[3]), run_time=1.0)
            box = SurroundingRectangle(formulas[3], color=CONCLUSION_COLOR)
            self.play(Create(box), run_time=0.5)
            self.wait(max(0.1, tracker.duration - 1.5))

        self.wait(1.0)


if __name__ == "__main__":
    GoldenExampleScene().render()
