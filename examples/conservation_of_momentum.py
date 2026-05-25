from __future__ import annotations

import sys
from math import pi
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phyanim.core.segment import PhysicsSegment
from phyanim.core.animation import PhysicsAnimation
from phyanim.core.events import time_countdown_event
from phyanim.render import PhyAnimationScene2D
from phyanim.solver import ScipySegmentSolver, HeyokaSegmentSolver
from phyanim.core.objects import PointParticle
from phyanim.core.state import StateVariable
from phyanim.core.objects import PhysicObject2D

from phyanim.core.entity import ConcaveTrack
from phyanim.core import Parameter

if __name__ == "__main__":
    # ============================================================
    # 光滑地面上可移动的半圆形轨道，小球从左侧最高点释放
    # 广义坐标 θ (小球相对轨道最低点的角度)
    # ============================================================

    animation = PhysicsAnimation(global_parameters={
        "g": 9.8,
    })

    # --- 轨道对象：只有水平位置 x_track 作为状态变量 ---
    track = PhysicObject2D(
        object_id="track",
        parameters={"M": Parameter("M", 5.0, "kg", "轨道质量"), "R": Parameter("R", 1.0, "m", "轨道半径")},
        state_variables={
            "x_track": StateVariable("x_track", "m", "轨道中心水平位置"),
            "y_track": StateVariable("y_track", "m", "轨道中心垂直位置"),
        },
        cartesian_position={
            "x": "x_track",
            "y": "y_track",
        },
        mobject=ConcaveTrack(width=4, height=2, radius=1, color="red")
    )

    # --- 小球对象：用广义坐标 θ 和 ω 描述 ---
    ball = PointParticle(
        object_id="ball",
        mass=1.0,
        color="blue",
        radius=0.1,
        parameter_names={"mass": "m"},
        state_names={"theta": "theta", "omega": "omega"},
        cartesian_position={"x": "x_ball", "y": "y_ball"}
    )

    # 初始条件：θ = -π/2（左侧最高点），静止释放
    animation.add_object(track, {"x_track": 0.0, "y_track": 0.0})
    animation.add_object(ball, {"theta": -pi / 2, "omega": 0.0})

    # --- 状态向量 ---
    state_vector = ["theta", "omega", "x_track", "y_track"]

    # --- 运动方程（由 Lagrangian 推导） ---
    # dθ/dt = ω
    # dω/dt = -(m·ω²·sinθ·cosθ + g·(M+m)·sinθ/R) / (M + m·sin²θ)
    # dx_track/dt = -m·R·cosθ·ω / (M+m)
    state_equations = {
        "theta": "omega",
        "omega": "-(m*omega**2*sin(theta)*cos(theta) + g*(M+m)*sin(theta)/R) / (M + m*sin(theta)**2)",
        "x_track": "-m*R*cos(theta)*omega/(M+m)",
        "y_track": "0",
    }

    # --- 派生量（每个方程只能引用状态变量和参数，不能引用其他派生量） ---
    # vx_track = -m*R*cosθ*ω/(M+m)       （轨道水平速度，由动量守恒导出）
    # vx_ball  = R*cosθ*ω*M/(M+m)        （小球水平速度 = vx_track + R*cosθ*ω）
    # vy_ball  = R*sinθ*ω                （小球竖直速度）
    # E_total  = ½*m*R²*ω²*(M+m*sin²θ)/(M+m) + m*g*R*(1-cosθ)   （总机械能）
    # Px_total = m*vx_ball + M*vx_track   （水平动量，恒为零）
    derived_equations = {
        "x_ball": "x_track + R*sin(theta)",
        "y_ball": "R*(1 - cos(theta))",
    }

    state_owners = {
        "theta": "ball",
        "omega": "ball",
        "x_track": "track",
        "y_track": "track",
    }

    # --- 单段仿真，运行足够长时间观察往复运动 ---
    animation.add_segment(
        PhysicsSegment(
            segment_id="sliding",
            object_ids=["ball", "track"],
            state_vector=state_vector,
            equations=state_equations,
            derived_equations=derived_equations,
            state_owners=state_owners,
            duration=200,
        ),
        end_event=time_countdown_event(20),
    )

    # 求解
    # animation.solve(ScipySegmentSolver(sample_dt=1 / 30))
    animation.solve(HeyokaSegmentSolver(sample_dt=1/20))

    scene = PhyAnimationScene2D()
    # scene.set_frame_size(width=1000, height=1000)
    scene.set_animation(animation)
    scene.render()