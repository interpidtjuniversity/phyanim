from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phyanim.core.segment import PhysicsSegment
from phyanim.core.animation import PhysicsAnimation
from phyanim.core.events import time_countdown_event
from phyanim.render import PhyAnimationScene2D
from phyanim.solver import HeyokaSegmentSolver
from phyanim.core.state import StateVariable
from phyanim.core.objects import PhysicObject2D
from phyanim.core.entity.TwoD import Spring


if __name__ == "__main__":
    # 弹簧劲度系数为 1N/m
    animation = PhysicsAnimation(global_parameters={"k":1})
    spring = PhysicObject2D(
        object_id="spring",
        state_variables={
            "start_x": StateVariable("start_x", "m", "弹簧开始点水平位置"),
            "start_y": StateVariable("start_y", "m", "弹簧开始点垂直位置"),
            "end_x": StateVariable("end_x", "m", "弹簧结束点水平位置"),
            "end_y": StateVariable("end_y", "m", "弹簧结束点垂直位置"),
        },
        cartesian_position=[("start_x", "start_y"), ("end_x", "end_y")],
        mobject=Spring(start=[-2, 0], end=[2, 0], radius=0.3, color="red")
    )
    animation.add_object(spring, {"start_x": "-2.0", "start_y": "0.0", "end_x": "2.0", "end_y": "0.0"})
    
    state_vector=["start_x", "start_y", "end_x", "end_y"]
    # equations可能不同段不同
    state_equations1={
        "start_x": "-1.0",
        "start_y": "0.0",
        "end_x": "0.0",
        "end_y": "0.0",
    }
    state_equations2={
        "start_x": "0.0",
        "start_y": "0.0",
        "end_x": "1.0",
        "end_y": "0.0",
    }
    drived_equations={
    }
    # 这个必须公用同一套
    state_owners={
        "start_x": "spring",
        "start_y": "spring",
        "end_x": "spring",
        "end_y": "spring",
    }
    animation.add_segment(
        PhysicsSegment(
            segment_id="compress1",
            object_ids=["spring"],
            state_vector=state_vector,
            equations=state_equations1,
            derived_equations=drived_equations,
            state_owners=state_owners,
            duration=100,
        ),
        end_event=time_countdown_event(5),
    )
    animation.add_segment(
        PhysicsSegment(
            segment_id="compress2",
            object_ids=["spring"],
            state_vector=state_vector,
            equations=state_equations2,
            derived_equations=drived_equations,
            state_owners=state_owners,
            duration=100,
        ),
        end_event=time_countdown_event(5),
    )

    #animation.solve(ScipySegmentSolver(sample_dt=1/20))
    #animation.solve(HeyokaSegmentSolver(sample_dt=1/10))

    scene = PhyAnimationScene2D()
    # scene.set_frame_size(width=1000, height=1000)
    scene.set_animation(animation)
    scene.render()