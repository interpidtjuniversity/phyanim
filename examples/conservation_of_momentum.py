from __future__ import annotations

import sys
from math import pi
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phyanim import (
    PhysicsAnimation,
    PhysicsSegment,
    PhysicObject,
    time_countdown_event,
)
from phyanim.core.state import StateVariable
from phyanim.render import TimelineExporter
from phyanim.solver import ScipySegmentSolver

if __name__ == "__main__":
    # ============================================================
    # 光滑地面上可移动的半圆形轨道，小球从左侧最高点释放
    # 广义坐标 θ (小球相对轨道最低点的角度)
    # ============================================================

    animation = PhysicsAnimation(global_parameters={
        "g": 9.8,
        "R": 1.0,
        "m": 1.0,
        "M": 5.0,
    })

    # --- 轨道对象：只有水平位置 x_track 作为状态变量 ---
    track = PhysicObject(
        object_id="track",
        state_variables={
            "x_track": StateVariable("x_track", "m", "轨道中心水平位置"),
        },
    )

    # --- 小球对象：用广义坐标 θ 和 ω 描述 ---
    ball = PhysicObject(
        object_id="ball",
        state_variables={
            "theta": StateVariable("theta", "rad", "小球相对轨道最低点的角度"),
            "omega": StateVariable("omega", "rad/s", "角速度"),
        },
    )

    # 初始条件：θ = -π/2（左侧最高点），静止释放
    animation.add_object(track, {"x_track": 0.0})
    animation.add_object(ball, {"theta": -pi / 2, "omega": 0.0})

    # --- 状态向量 ---
    state_vector = ["theta", "omega", "x_track"]

    # --- 运动方程（由 Lagrangian 推导） ---
    # dθ/dt = ω
    # dω/dt = -(m·ω²·sinθ·cosθ + g·(M+m)·sinθ/R) / (M + m·sin²θ)
    # dx_track/dt = -m·R·cosθ·ω / (M+m)
    state_equations = {
        "theta": "omega",
        "omega": "-(m*omega**2*sin(theta)*cos(theta) + g*(M+m)*sin(theta)/R) / (M + m*sin(theta)**2)",
        "x_track": "-m*R*cos(theta)*omega/(M+m)",
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
        "vx_track": "-m*R*cos(theta)*omega/(M+m)",
        "vx_ball": "R*cos(theta)*omega*M/(M+m)",
        "vy_ball": "R*sin(theta)*omega",
        "KE_ball": "0.5*m*R**2*omega**2*(M**2*cos(theta)**2/(M+m)**2 + sin(theta)**2)",
        "KE_track": "0.5*M*m**2*R**2*cos(theta)**2*omega**2/(M+m)**2",
        "PE": "m*g*R*(1 - cos(theta))",
        "E_total": "0.5*m*R**2*omega**2*(M + m*sin(theta)**2)/(M+m) + m*g*R*(1 - cos(theta))",
        "Px_total": "m*R*cos(theta)*omega*M/(M+m) - M*m*R*cos(theta)*omega/(M+m)",
    }

    state_owners = {
        "theta": "ball",
        "omega": "ball",
        "x_track": "track",
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
        end_event=time_countdown_event(100),
    )

    # 求解
    animation.solve(ScipySegmentSolver(sample_dt=1 / 120))

    # --- 输出结果 ---
    traj = animation.trajectories[0]

    print("=== 守恒量验证 ===")
    E0 = traj.derived["E_total"][0]
    E_final = traj.derived["E_total"][-1]
    print(f"初始总能量 E0 = {E0:.6f} J")
    print(f"最终总能量 E  = {E_final:.6f} J")
    print(f"能量相对漂移 = {(E_final - E0) / E0:.2e}")

    Px0 = traj.derived["Px_total"][0]
    Px_final = traj.derived["Px_total"][-1]
    print(f"初始水平动量 = {Px0:.6e} kg·m/s")
    print(f"最终水平动量 = {Px_final:.6e} kg·m/s")

    # 找出小球到达最低点 (theta ≈ 0) 的时刻
    theta_values = traj.states["theta"]
    times = traj.times
    print("\n=== 关键事件 ===")
    for i in range(1, len(theta_values)):
        if theta_values[i - 1] < 0 and theta_values[i] >= 0:
            print(f"小球到达最低点: t = {times[i]:.4f} s")
            print(f"  轨道位移 x_track = {traj.states['x_track'][i]:.4f} m")
            print(f"  小球坐标 x_ball = {traj.derived['x_ball'][i]:.4f} m")
            print(f"  小球坐标 y_ball = {traj.derived['y_ball'][i]:.4f} m")
            break

    # 找出小球到达右侧最高点 (theta ≈ π/2) 的时刻
    for i in range(1, len(theta_values)):
        if theta_values[i - 1] < pi / 2 - 0.01 and theta_values[i] >= pi / 2 - 0.01:
            print(f"\n小球到达右侧最高点: t = {times[i]:.4f} s")
            print(f"  theta = {theta_values[i]:.4f} rad")
            print(f"  轨道位移 x_track = {traj.states['x_track'][i]:.4f} m")
            print(f"  小球坐标 x_ball = {traj.derived['x_ball'][i]:.4f} m")
            break

    print(f"\n共 {len(times)} 个采样点，时间范围 [{times[0]:.2f}, {times[-1]:.2f}] s")

    # --- 绘制 ---
    exporter = TimelineExporter()
    exporter.plot_variables(
        animation,
        variables=["theta", "omega", "x_track", "x_ball", "y_ball", "E_total", "Px_total"],
        title="小球在可移动半圆轨道上的运动",
        show=True,
    )

    # 保存 JSON
    output_path = Path("phyanim/outputs/conservation_of_momentum.json")
    exporter.write_json(animation, output_path)
    print(f"\n已保存到: {output_path}")