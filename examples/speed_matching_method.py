"""复合场运动中的配速法详解。

配速法核心思想：
1. 粒子初始静止，我们"配上"一个速度 v₀
2. 为了保持初始状态不变，同时"配上"反向速度 -v₀
3. v₀ 产生匀速直线运动（漂移）
4. -v₀ 与初始静止合成，粒子在漂移系中做圆周运动
5. 两个分运动可以独立分析，再叠加得到实际运动
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

# --- 物理参数 ---
m = 1.0       # 质量 (kg)
q = 1.0       # 电荷 (C)
g = 1.0       # 重力加速度 (m/s²)
E = 1.0       # 水平电场强度 (V/m)
B = 2.0       # 垂直磁场强度 (T)

# 力的计算
F_gravity = m * g          # 重力，竖直向下
F_electric = q * E         # 电场力，水平向右
F_total = np.sqrt(F_gravity**2 + F_electric**2)  # 合力大小

# 配速计算：qv0×B = -F_constant
# 在2D中，B垂直纸面向外(B_z = B)：v×B = (vy*B, -vx*B)
# 条件：q*v0y*B = -qE, -q*v0x*B = mg
# 解得：v0x = -mg/(qB), v0y = -E/B
v0x = -m * g / (q * B)
v0y = -E / B
v0_magnitude = np.sqrt(v0x**2 + v0y**2)
v0_angle = np.arctan2(v0y, v0x)

# 圆周运动参数
v_prime_init = v0_magnitude
R_circle = m * v_prime_init / (q * B)
omega_c = q * B / m
T_circle = 2 * pi / omega_c

print("=" * 60)
print("配速法参数计算")
print("=" * 60)
print(f"重力 F_g = {F_gravity:.2f} N (竖直向下)")
print(f"电场力 F_E = {F_electric:.2f} N (水平向右)")
print(f"合力 F = {F_total:.2f} N")
print(f"配速: v0x = {v0x:.2f}, v0y = {v0y:.2f}, |v0| = {v0_magnitude:.2f} m/s")
print(f"圆周半径 R = {R_circle:.2f} m")
print(f"回旋周期 T = {T_circle:.2f} s")

BG_COLOR = BLUE_E
PARTICLE_COLOR = YELLOW
GRAVITY_COLOR = RED
ELECTRIC_COLOR = GREEN
MAGNETIC_COLOR = PURPLE
RESULTANT_COLOR = ORANGE
VELOCITY_COLOR = BLUE_C
CIRCLE_COLOR = WHITE
TRAIL_COLOR = GOLD
SUM_COLOR = GREEN


def build_animation() -> PhysicsAnimation:
    """构建物理动画：带电粒子在复合场中运动。"""
    animation = PhysicsAnimation(
        global_parameters={"m": m, "q": q, "g": g, "E": E, "B": B, "v0x_val": v0x, "v0y_val": v0y},
        engine="scipy",
        sample_dt=1 / 60,
    )

    particle = PhysicObject2D(
        object_id="particle",
        state_variables={
            "x": StateVariable("x", "m", "水平位置"),
            "y": StateVariable("y", "m", "竖直位置"),
            "vx": StateVariable("vx", "m/s", "水平速度"),
            "vy": StateVariable("vy", "m/s", "竖直速度"),
        },
        cartesian_position=[("x", "y")],
    )

    # 初始条件：从原点静止释放
    animation.add_object(particle, {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0})

    animation.add_segment(
        PhysicsSegment(
            segment_id="motion",
            object_ids=["particle"],
            state_vector=["x", "y", "vx", "vy"],
            equations={
                "x": "vx",
                "y": "vy",
                "vx": "(q*E + q*vy*B)/m",
                "vy": "(-m*g - q*vx*B)/m",
            },
            state_owners={
                "x": "particle", "y": "particle",
                "vx": "particle", "vy": "particle",
            },
            derived_equations={
                "vx_prime": "vx - v0x_val",
                "vy_prime": "vy - v0y_val",
                "v_prime": "sqrt((vx - v0x_val)**2 + (vy - v0y_val)**2)",
                "x_drift": "v0x_val*t",
                "y_drift": "v0y_val*t",
                "x_circle": "x - v0x_val*t",
                "y_circle": "y - v0y_val*t",
            },
            duration=100,
        ),
        end_event=time_countdown_event(5.0),
    )

    return animation


def create_field_visualization(axes):
    """创建电场和磁场的可视化。"""
    # 磁场：垂直纸面向外的符号（点）
    B_symbols = VGroup()
    for x in np.arange(-3, 4, 1.5):
        for y in np.arange(-2.5, 3, 1.5):
            dot = Dot(axes.c2p(x, y), radius=0.08, color=MAGNETIC_COLOR, fill_opacity=0.5)
            B_symbols.add(dot)

    B_label = MathTex(r"\vec{B} \odot", color=MAGNETIC_COLOR, font_size=28)
    B_label.move_to(axes.c2p(3.5, 2.5))

    # 电场：水平向右的箭头
    E_arrow = Arrow(
        axes.c2p(-3, -2), axes.c2p(-1.5, -2),
        color=ELECTRIC_COLOR, buff=0, stroke_width=4
    )
    E_label = MathTex(r"\vec{E}", color=ELECTRIC_COLOR, font_size=32)
    E_label.next_to(E_arrow, DOWN, buff=0.15)

    # 重力场：竖直向下的箭头
    g_arrow = Arrow(
        axes.c2p(-3, 2), axes.c2p(-3, 0.5),
        color=GRAVITY_COLOR, buff=0, stroke_width=4
    )
    g_label = MathTex(r"m\vec{g}", color=GRAVITY_COLOR, font_size=32)
    g_label.next_to(g_arrow, LEFT, buff=0.15)

    return VGroup(B_symbols, B_label, E_arrow, E_label, g_arrow, g_label)


class SpeedMatchingMethodScene(PhyAnimScene):
    def construct(self):
        self.setup_speech(TTS_CONFIG)
        MathTex.set_default(tex_template=TexTemplateLibrary.ctex)
        self.camera.background_color = BG_COLOR

        # 求解物理
        animation = build_animation()
        trajectory = solve_animation(animation)
        t_total = trajectory.total_time

        # ==================== 阶段1：问题介绍 ====================
        title = Text("复合场中的配速法", font="SimSun", font_size=36, color=GOLD_C)
        title.to_edge(UP, buff=0.5)

        # 创建坐标系 - 扩大范围以适应粒子运动
        axes = Axes(
            x_range=[-6, 2, 1],
            y_range=[-4, 2, 1],
            x_length=10,
            y_length=8,
            axis_config={"include_numbers": False},
        )
        axes.move_to(ORIGIN)

        x_label = axes.get_x_axis_label("x")
        y_label = axes.get_y_axis_label("y")

        self.play(FadeIn(title))

        with self.voiceover(text="考虑一个带电粒子，同时处于重力场、水平电场和垂直磁场中。") as tracker:
            self.play(Create(axes), FadeIn(x_label, y_label))
            self.wait(tracker.duration)

        # 添加场可视化
        field_vis = create_field_visualization(axes)

        with self.voiceover(text="磁场垂直纸面向外，电场水平向右，重力竖直向下。") as tracker:
            self.play(FadeIn(field_vis))
            self.wait(tracker.duration)

        # ==================== 阶段2：粒子静止释放 ====================
        self.play(FadeOut(title), run_time=0.5)

        # 粒子（静止在原点）
        particle_mob = Dot(point=axes.c2p(0, 0), radius=0.15, color=PARTICLE_COLOR)
        particle_label = Text("静止释放", font="SimSun", font_size=18, color=WHITE)
        particle_label.next_to(particle_mob, UR, buff=0.1)

        with self.voiceover(text="粒子从原点静止释放，初速度为零。") as tracker:
            self.play(FadeIn(particle_mob, particle_label))
            self.wait(tracker.duration)

        # 受力分析
        gravity_arrow = Arrow(
            axes.c2p(0, 0), axes.c2p(0, -0.8),
            color=GRAVITY_COLOR, buff=0, stroke_width=4
        )
        gravity_label = MathTex("mg", color=GRAVITY_COLOR, font_size=24)
        gravity_label.next_to(gravity_arrow, RIGHT, buff=0.1)

        electric_arrow = Arrow(
            axes.c2p(0, 0), axes.c2p(0.8, 0),
            color=ELECTRIC_COLOR, buff=0, stroke_width=4
        )
        electric_label = MathTex("qE", color=ELECTRIC_COLOR, font_size=24)
        electric_label.next_to(electric_arrow, UP, buff=0.1)

        with self.voiceover(text="粒子受重力和电场力，合力恒定。洛伦兹力初始为零，因为速度为零。") as tracker:
            self.play(GrowArrow(gravity_arrow), Write(gravity_label))
            self.play(GrowArrow(electric_arrow), Write(electric_label))
            self.wait(tracker.duration)

        self.play(FadeOut(gravity_arrow, gravity_label, electric_arrow, electric_label), run_time=0.3)

        # ==================== 阶段3：配速法核心思想 ====================
        # 配速法思想图示
        idea_title = Text("配速法思想", font="SimSun", font_size=28, color=GOLD_C)
        idea_title.to_edge(UP, buff=0.5)

        # 先展示合力
        resultant_arrow = Arrow(
            axes.c2p(0, 0), axes.c2p(0.8, -0.8),
            color=RESULTANT_COLOR, buff=0, stroke_width=4
        )
        resultant_label = MathTex(r"F_{\text{合}}", color=RESULTANT_COLOR, font_size=28)
        resultant_label.next_to(resultant_arrow, RIGHT, buff=0.1)

        with self.voiceover(text="首先分析合力。重力和电场力的合力大小恒定，方向斜向右下。") as tracker:
            self.play(GrowArrow(resultant_arrow), Write(resultant_label))
            self.wait(tracker.duration)

        # 步骤1：给粒子配上速度 v₀（向左下方）
        step1_text = Text("步骤1：配上速度 v₀", font="SimSun", font_size=22, color=WHITE)
        step1_text.move_to(RIGHT * 4.5 + DOWN * 2.5)

        # v₀ 箭头（向左下方，与合力方向垂直）
        v0_arrow1 = Arrow(
            axes.c2p(0, 0), axes.c2p(v0x * 1.2, v0y * 1.2),
            color=VELOCITY_COLOR, buff=0, stroke_width=5
        )
        v0_label1 = MathTex(r"\vec{v}_0", color=VELOCITY_COLOR, font_size=28)
        v0_label1.next_to(v0_arrow1, DL, buff=0.1)

        step1_formula = MathTex(r"qv_0 B = F_{\text{合}}", font_size=26)
        step1_formula.next_to(step1_text, DOWN, buff=0.3)

        with self.voiceover(text="配速法的关键：给粒子配上特殊速度v0，使得qv0叉乘B恰好抵消合力。v0方向向左下方，大小等于合力除以qB。") as tracker:
            self.play(FadeIn(idea_title))
            self.play(Write(step1_text), run_time=1.0)
            self.play(GrowArrow(v0_arrow1), Write(v0_label1), run_time=0.8)
            self.play(Write(step1_formula), run_time=1.0)
            self.wait(max(0.1, tracker.duration - 2.8))

        # 步骤2：配上反向速度 -v₀
        self.play(FadeOut(step1_text, step1_formula), run_time=0.3)

        step2_text = Text("步骤2：配上反向速度 -v₀", font="SimSun", font_size=22, color=WHITE)
        step2_text.move_to(RIGHT * 4.5 + DOWN * 2.5)

        # -v₀ 箭头（向右上方）
        v0_arrow2 = Arrow(
            axes.c2p(0, 0), axes.c2p(-v0x * 1.2, -v0y * 1.2),
            color=RED_C, buff=0, stroke_width=5
        )
        v0_label2 = MathTex(r"-\vec{v}_0", color=RED_C, font_size=28)
        v0_label2.next_to(v0_arrow2, UR, buff=0.1)

        step2_formula = MathTex(r"\vec{v}_0 + (-\vec{v}_0) = 0", font_size=26)
        step2_formula.next_to(step2_text, DOWN, buff=0.3)

        with self.voiceover(text="但为了保持初始静止状态不变，同时配上反向速度负v0。这样两个速度相加等于零，初始状态不变。") as tracker:
            self.play(Write(step2_text), run_time=1.0)
            self.play(GrowArrow(v0_arrow2), Write(v0_label2), run_time=0.8)
            self.play(Write(step2_formula), run_time=1.0)
            self.wait(max(0.1, tracker.duration - 2.8))

        # 步骤3：两个速度独立作用
        self.play(FadeOut(step2_text, step2_formula), run_time=0.3)

        step3_text = Text("步骤3：独立分析", font="SimSun", font_size=22, color=WHITE)
        step3_text.move_to(RIGHT * 4.5 + DOWN * 2.5)

        step3_formula1 = MathTex(r"\vec{v}_0 \rightarrow \text{匀速直线}", font_size=24)
        step3_formula1.next_to(step3_text, DOWN, buff=0.3)

        step3_formula2 = MathTex(r"-\vec{v}_0 \rightarrow \text{匀速圆周}", font_size=24)
        step3_formula2.next_to(step3_formula1, DOWN, buff=0.2)

        with self.voiceover(text="由于运动方程是线性的，两个速度可以独立分析。v0产生匀速直线运动，负v0产生匀速圆周运动。") as tracker:
            self.play(Write(step3_text), run_time=1.0)
            self.play(Write(step3_formula1), Write(step3_formula2), run_time=1.5)
            self.wait(max(0.1, tracker.duration - 2.5))

        self.wait(0.5)
        self.play(
            FadeOut(idea_title, step3_text, step3_formula1, step3_formula2,
                    v0_arrow1, v0_label1, v0_arrow2, v0_label2,
                    resultant_arrow, resultant_label, field_vis),
            run_time=0.5
        )

        # ==================== 阶段4：物理仿真演示 ====================
        # 创建粒子并绑定轨迹
        t_tracker = create_tracker(0.0)

        # 实际运动粒子
        actual_particle = Dot(radius=0.12, color=PARTICLE_COLOR)
        actual_particle.move_to(axes.c2p(0, 0))

        def update_actual_particle(mob):
            t = t_tracker.get_value()
            x_actual = trajectory.value_at("x", t)
            y_actual = trajectory.value_at("y", t)
            mob.move_to(axes.c2p(x_actual, y_actual))

        actual_particle.add_updater(update_actual_particle)

        # 漂移运动粒子（只显示配速导致的位移）
        drift_particle = Dot(radius=0.1, color=VELOCITY_COLOR, fill_opacity=0.6)
        drift_particle.move_to(axes.c2p(0, 0))

        def update_drift_particle(mob):
            t = t_tracker.get_value()
            x_drift = trajectory.value_at("x_drift", t)
            y_drift = trajectory.value_at("y_drift", t)
            mob.move_to(axes.c2p(x_drift, y_drift))

        drift_particle.add_updater(update_drift_particle)

        # 圆周运动粒子（相对运动）
        circle_particle = Dot(radius=0.1, color=CIRCLE_COLOR, fill_opacity=0.6)
        circle_particle.move_to(axes.c2p(0, 0))

        def update_circle_particle(mob):
            t = t_tracker.get_value()
            x_circle = trajectory.value_at("x_circle", t)
            y_circle = trajectory.value_at("y_circle", t)
            mob.move_to(axes.c2p(x_circle, y_circle))

        circle_particle.add_updater(update_circle_particle)

        # 实际运动轨迹
        actual_trail = VMobject().set_stroke(color=TRAIL_COLOR, width=2.5, opacity=0.8)
        actual_trail.set_points_as_corners([axes.c2p(0, 0), axes.c2p(0, 0)])
        times_arr, xs_arr = trajectory.sample("x", dt=0.02)
        _, ys_arr = trajectory.sample("y", dt=0.02)

        def update_actual_trail(mob):
            t = t_tracker.get_value()
            mask = [i for i, tt in enumerate(times_arr) if tt <= t]
            if len(mask) < 2:
                mob.set_points_as_corners([axes.c2p(0, 0), axes.c2p(0, 0)])
                return
            pts = [axes.c2p(xs_arr[i], ys_arr[i]) for i in mask]
            mob.set_points_as_corners(pts)
            mob.set_fill(opacity=0)

        actual_trail.add_updater(update_actual_trail)

        # 漂移轨迹（直线）
        drift_trail = VMobject().set_stroke(color=VELOCITY_COLOR, width=2, opacity=0.5)
        drift_trail.set_points_as_corners([axes.c2p(0, 0), axes.c2p(0, 0)])

        def update_drift_trail(mob):
            t = t_tracker.get_value()
            times_sample = np.linspace(0, t, 100)
            if len(times_sample) < 2:
                mob.set_points_as_corners([axes.c2p(0, 0), axes.c2p(0, 0)])
                return
            pts = [axes.c2p(trajectory.value_at("x_drift", tt), trajectory.value_at("y_drift", tt)) for tt in times_sample]
            mob.set_points_as_corners(pts)
            mob.set_fill(opacity=0)

        drift_trail.add_updater(update_drift_trail)

        # 圆周轨迹
        circle_trail = VMobject().set_stroke(color=CIRCLE_COLOR, width=2, opacity=0.5)
        circle_trail.set_points_as_corners([axes.c2p(0, 0), axes.c2p(0, 0)])

        def update_circle_trail(mob):
            t = t_tracker.get_value()
            times_sample = np.linspace(0, t, 100)
            if len(times_sample) < 2:
                mob.set_points_as_corners([axes.c2p(0, 0), axes.c2p(0, 0)])
                return
            pts = [axes.c2p(trajectory.value_at("x_circle", tt), trajectory.value_at("y_circle", tt)) for tt in times_sample]
            mob.set_points_as_corners(pts)
            mob.set_fill(opacity=0)

        circle_trail.add_updater(update_circle_trail)

        # 平行四边形法则：展示位移关系
        # 用 always_redraw 创建箭头，避免 put_start_and_end_on 拉伸箭头尖端
        # 边1：漂移位移（从原点到漂移位置）
        drift_arrow = always_redraw(lambda: Arrow(
            axes.c2p(0, 0),
            axes.c2p(trajectory.value_at("x_drift", t_tracker.get_value()),
                     trajectory.value_at("y_drift", t_tracker.get_value())),
            color=VELOCITY_COLOR, stroke_width=4, buff=0,
        ))
        
        # 边2：圆周位移（从漂移位置到实际位置）
        circle_arrow = always_redraw(lambda: Arrow(
            axes.c2p(trajectory.value_at("x_drift", t_tracker.get_value()),
                     trajectory.value_at("y_drift", t_tracker.get_value())),
            axes.c2p(trajectory.value_at("x_drift", t_tracker.get_value()) + trajectory.value_at("x_circle", t_tracker.get_value()),
                     trajectory.value_at("y_drift", t_tracker.get_value()) + trajectory.value_at("y_circle", t_tracker.get_value())),
            color=CIRCLE_COLOR, stroke_width=4, buff=0,
        ))
        
        # 边3：合成位移（从原点到漂移+圆周位置）
        sum_arrow = always_redraw(lambda: Arrow(
            axes.c2p(0, 0),
            axes.c2p(trajectory.value_at("x_drift", t_tracker.get_value()) + trajectory.value_at("x_circle", t_tracker.get_value()),
                     trajectory.value_at("y_drift", t_tracker.get_value()) + trajectory.value_at("y_circle", t_tracker.get_value())),
            color=SUM_COLOR, stroke_width=4, buff=0,
        ))
        
        # 边4：实际位移（从原点到实际位置）
        actual_arrow = always_redraw(lambda: Arrow(
            axes.c2p(0, 0),
            axes.c2p(trajectory.value_at("x", t_tracker.get_value()),
                     trajectory.value_at("y", t_tracker.get_value())),
            color=TRAIL_COLOR, stroke_width=4, buff=0,
        ))
        
        # 边5：圆周位移（从原点到圆周位置）
        circle_disp_arrow = always_redraw(lambda: Arrow(
            axes.c2p(0, 0),
            axes.c2p(trajectory.value_at("x_circle", t_tracker.get_value()),
                     trajectory.value_at("y_circle", t_tracker.get_value())),
            color=CIRCLE_COLOR, stroke_width=4, buff=0,
        ))
        
        # 位移标签
        drift_disp_label = MathTex(r"\vec{s}_{\text{漂移}}", color=VELOCITY_COLOR, font_size=24)
        drift_disp_label.add_updater(lambda m: m.next_to(
            axes.c2p(trajectory.value_at("x_drift", t_tracker.get_value()) / 2,
                     trajectory.value_at("y_drift", t_tracker.get_value()) / 2),
            DL, buff=0.15
        ))
        
        circle_disp_label = MathTex(r"\vec{s}_{\text{圆周}}", color=CIRCLE_COLOR, font_size=24)
        circle_disp_label.add_updater(lambda m: m.next_to(
            axes.c2p(trajectory.value_at("x_drift", t_tracker.get_value()) + 
                     trajectory.value_at("x_circle", t_tracker.get_value()) / 2,
                     trajectory.value_at("y_drift", t_tracker.get_value()) + 
                     trajectory.value_at("y_circle", t_tracker.get_value()) / 2),
            UR, buff=0.15
        ))
        
        sum_disp_label = MathTex(r"\vec{s}_{\text{合成}}", color=SUM_COLOR, font_size=24)
        sum_disp_label.add_updater(lambda m: m.next_to(
            axes.c2p((trajectory.value_at("x_drift", t_tracker.get_value()) + 
                      trajectory.value_at("x_circle", t_tracker.get_value())) / 2,
                     (trajectory.value_at("y_drift", t_tracker.get_value()) + 
                      trajectory.value_at("y_circle", t_tracker.get_value())) / 2),
            UL, buff=0.15
        ))
        
        actual_disp_label = MathTex(r"\vec{s}_{\text{实际}}", color=TRAIL_COLOR, font_size=24)
        actual_disp_label.add_updater(lambda m: m.next_to(
            axes.c2p(trajectory.value_at("x", t_tracker.get_value()) / 2,
                     trajectory.value_at("y", t_tracker.get_value()) / 2),
            UR, buff=0.15
        ))

        # 标签
        actual_label = Text("实际运动", font="SimSun", font_size=18, color=TRAIL_COLOR)
        actual_label.to_edge(LEFT, buff=0.3).shift(DOWN * 0.5)
        drift_label = Text("漂移运动 (v₀)", font="SimSun", font_size=18, color=VELOCITY_COLOR)
        drift_label.to_edge(LEFT, buff=0.3).shift(DOWN * 1.0)
        circle_label = Text("圆周运动 (-v₀)", font="SimSun", font_size=18, color=CIRCLE_COLOR)
        circle_label.to_edge(LEFT, buff=0.3).shift(DOWN * 1.5)

        # 运动合成说明
        synthesize_text = Text(
            "平行四边形法则：合成位移 = 实际位移",
            font="SimSun", font_size=20, color=GOLD_C
        )
        synthesize_text.to_edge(UP, buff=0.5)

        # 添加所有元素
        self.add(actual_trail, drift_trail, circle_trail)
        self.add(drift_arrow, circle_arrow, sum_arrow, actual_arrow, circle_disp_arrow)
        self.add(drift_disp_label, circle_disp_label, sum_disp_label, actual_disp_label)
        self.add(actual_particle, drift_particle, circle_particle)
        self.add(actual_label, drift_label, circle_label, synthesize_text)

        with self.voiceover(text="现在观察三个运动的同步演示。黄色是实际运动轨迹，蓝色是v0产生的漂移运动，白色是负v0产生的圆周运动。") as tracker:
            self.play(
                t_tracker.animate.set_value(t_total),
                run_time=max(tracker.duration, t_total),
                rate_func=linear,
            )

        with self.voiceover(text="看这三个箭头构成的平行四边形。蓝色箭头是漂移位移，白色箭头是圆周位移。绿色箭头是它们的合成位移。") as tracker:
            self.wait(tracker.duration)

        with self.voiceover(text="现在比较绿色合成箭头和黄色实际箭头。你会发现它们完全重合！这证明了平行四边形法则：漂移位移加圆周位移等于实际位移。") as tracker:
            # 高亮显示合成箭头和实际箭头
            self.play(
                sum_arrow.animate.set_stroke(width=10),
                actual_arrow.animate.set_stroke(width=10),
                run_time=1.0
            )
            self.wait(1.0)
            self.play(
                sum_arrow.animate.set_stroke(width=6),
                actual_arrow.animate.set_stroke(width=6),
                run_time=0.5
            )
            self.wait(tracker.duration - 2.5)

        self.wait(1.0)


if __name__ == "__main__":
    SpeedMatchingMethodScene().render()
