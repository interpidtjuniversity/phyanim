from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phyanim.core.segment import PhysicsSegment
from phyanim.core.animation import PhysicsAnimation
from phyanim.core.events import time_countdown_event
from phyanim.render import PhyAnimationScene2D
from phyanim.solver import HeyokaSegmentSolver, ScipySegmentSolver
from phyanim.core.state import StateVariable
from phyanim.core.objects import PhysicObject2D
from phyanim.core.entity.TwoD import Spring
from phyanim.core.events import PhysicsEvent, EventCondition, StateTransition

from manim import Circle


if __name__ == "__main__":
    # 弹簧劲度系数为 1N/m
    animation = PhysicsAnimation(global_parameters={"k":1, "m1":1, "m2":1})
    # 添加弹簧对象
    spring = PhysicObject2D(
        object_id="spring",
        state_variables={},
        cartesian_position=[("ball1x", "ball1y"), ("ball2x", "ball2y")],
        mobject=Spring(start=[-6, 0], end=[-4, 0], radius=0.1, color="red")
    )
    animation.add_object(spring, {})

    # 添加小球1
    ball1 = PhysicObject2D(
        object_id="ball1",
        state_variables={
            "ball1x": StateVariable("ball1x", "m", "小球1水平位置"),
            "ball1y": StateVariable("ball1y", "m", "小球1垂直位置"),
            "ball1vx": StateVariable("ball1vx", "m/s", "小球1水平速度"),
        },
        cartesian_position=[("ball1x", "ball1y")],
        mobject=Circle(radius=0.1, color="blue")
    )
    animation.add_object(ball1, {"ball1x": "-6.0", "ball1y": "0.0", "ball1vx": "2.0"})

    # 添加小球2
    ball2 = PhysicObject2D(
        object_id="ball2",
        state_variables={
            "ball2x": StateVariable("ball2x", "m", "小球2水平位置"),
            "ball2y": StateVariable("ball2y", "m", "小球2垂直位置"),
            "ball2vx": StateVariable("ball2vx", "m/s", "小球2水平速度"),
        },
        cartesian_position=[("ball2x", "ball2y")],
        mobject=Circle(radius=0.1, color="green")
    )
    animation.add_object(ball2, {"ball2x": "-4.0", "ball2y": "0.0", "ball2vx": "0.0"})
    
    state_vector=["ball1x", "ball1y", "ball2x", "ball2y", "ball1vx", "ball2vx"]
    # equations可能不同段不同
    # 碰撞
    state_equations2={
        "ball1y": "0.0",
        "ball2y": "0.0",

        "ball1x": "ball1vx",
        "ball2x": "ball2vx",

        "ball1vx": "-k*(2.0-ball2x+ball1x)/m1",
        "ball2vx": "k*(2.0-ball2x+ball1x)/m2",
    }
    drived_equations={
    }
    # 这个必须公用同一套
    state_owners={
        "ball1x": "ball1",
        "ball2x": "ball2",
        "ball1vx": "ball1",
        "ball2vx": "ball2",
        "ball1y": "ball1",
        "ball2y": "ball2",
    }
    event = time_countdown_event(10)

    animation.add_segment(
        PhysicsSegment(
            segment_id="seg1",
            object_ids=["spring", "ball1", "ball2"],
            state_vector=state_vector,
            equations=state_equations2,
            derived_equations=drived_equations,
            state_owners=state_owners,
            duration=100,
        ),
        end_event=event,
    )

    animation.solve(HeyokaSegmentSolver(sample_dt=1/60))

    scene = PhyAnimationScene2D()
    # scene.set_frame_size(width=1000, height=1000)
    scene.set_animation(animation)
    scene.render()