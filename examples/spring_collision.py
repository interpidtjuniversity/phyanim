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
        state_variables={
            "start_x": StateVariable("start_x", "m", "弹簧开始点水平位置"),
            "start_y": StateVariable("start_y", "m", "弹簧开始点垂直位置"),
            "end_x": StateVariable("end_x", "m", "弹簧结束点水平位置"),
            "end_y": StateVariable("end_y", "m", "弹簧结束点垂直位置"),
        },
        cartesian_position=[("start_x", "start_y"), ("end_x", "end_y")],
        mobject=Spring(start=[-1, 0], end=[1, 0], radius=0.1, color="red")
    )
    animation.add_object(spring, {"start_x": "-1.0", "start_y": "0.0", "end_x": "1.0", "end_y": "0.0"})

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
    animation.add_object(ball1, {"ball1x": "-5.0", "ball1y": "0.0", "ball1vx": "1.0"})

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
    animation.add_object(ball2, {"ball2x": "1.0", "ball2y": "0.0", "ball2vx": "0.0"})
    
    state_vector=["start_x", "start_y", "end_x", "end_y", "ball1x", "ball1y", "ball2x", "ball2y", "ball1vx", "ball2vx"]
    # equations可能不同段不同
    # 靠近
    state_equations1={
        "start_x": "0.0",
        "start_y": "0.0",
        "end_x": "0.0",
        "end_y": "0.0",
        "ball1y": "0.0",
        "ball2y": "0.0",
        "ball1x": "ball1vx",
        "ball2x": "ball2vx",
        "ball1vx": "0.0",
        "ball2vx": "0.0",
    }
    # 碰撞
    state_equations2={
        "start_x": "ball1vx",
        "start_y": "0.0",
        "end_x": "ball2vx",
        "end_y": "0.0",
        "ball1y": "0.0",
        "ball2y": "0.0",

        "ball1x": "ball1vx",
        "ball2x": "ball2vx",

        "ball1vx": "-k*(2.0-end_x+start_x)/m1",
        "ball2vx": "k*(2.0-end_x+start_x)/m2",
    }
    # 分离
    state_equations3={
        "start_x": "ball1vx",
        "start_y": "0.0",
        "end_x": "ball1vx",
        "end_y": "0.0",
        "ball1y": "0.0",
        "ball2y": "0.0",

        "ball1x": "ball1vx",
        "ball2x": "ball2vx",

        "ball1vx": "0.0",
        "ball2vx": "0.0",
    }
    drived_equations={
    }
    # 这个必须公用同一套
    state_owners={
        "start_x": "spring",
        "start_y": "spring",
        "end_x": "spring",
        "end_y": "spring",
        "ball1x": "ball1",
        "ball2x": "ball2",
        "ball1vx": "ball1",
        "ball2vx": "ball2",
        "ball1y": "ball1",
        "ball2y": "ball2",
    }
    event1 = PhysicsEvent("event1", condition=EventCondition(expression="ball1x - start_x", terminal=True, direction=1), transition=StateTransition(name="collision_impulse", equations={}))
    event2 = PhysicsEvent("event2", condition=EventCondition(expression="end_x - start_x - 2", terminal=True, direction=1), transition=StateTransition(name="collision_impulse", equations={}))
    event3 = time_countdown_event(2)

    animation.add_segment(
        PhysicsSegment(
            segment_id="seg1",
            object_ids=["spring", "ball1", "ball2"],
            state_vector=state_vector,
            equations=state_equations1,
            derived_equations=drived_equations,
            state_owners=state_owners,
            duration=100,
        ),
        end_event=event1,
    )
    animation.add_segment(
        PhysicsSegment(
            segment_id="seg2",
            object_ids=["spring", "ball1", "ball2"],
            state_vector=state_vector,
            equations=state_equations2,
            derived_equations=drived_equations,
            state_owners=state_owners,
            duration=100,
        ),
        end_event=event2,
    )
    animation.add_segment(
        PhysicsSegment(
            segment_id="seg3",
            object_ids=["spring", "ball1", "ball2"],
            state_vector=state_vector,
            equations=state_equations3,
            derived_equations=drived_equations,
            state_owners=state_owners,
            duration=100,
        ),
        end_event=event3,
    )

    animation.solve(HeyokaSegmentSolver(sample_dt=1/10))

    scene = PhyAnimationScene2D()
    # scene.set_frame_size(width=1000, height=1000)
    scene.set_animation(animation)
    scene.render()