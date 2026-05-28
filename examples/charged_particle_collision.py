from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phyanim.core.segment import PhysicsSegment
from phyanim.core.animation import PhysicsAnimation
from phyanim.core.events import PhysicsEvent, time_countdown_event
from phyanim.render import PhyAnimationScene2D
from phyanim.core.objects import PointParticle
from phyanim.core.events import EventCondition, StateTransition


if __name__ == "__main__":
    # 向右的匀强场，场强为1N/C
    animation = PhysicsAnimation(global_parameters={"E":1})
    # 定义一个带电点粒子A
    particleA = PointParticle("particleA",  mass=1.0, charge=1.0, radius=0.01, color="red", parameter_names={"mass":"mA", "charge":"qA"}, state_names={"x":"xA", "y":"yA", "vx":"vxA", "vy":"vyA"}, cartesian_position=("xA", "yA"))
    # 定义一个带电点粒子B
    particleB = PointParticle("particleB",  mass=1.0, charge=-1.0, radius=0.01, color="blue", parameter_names={"mass":"mB", "charge":"qB"}, state_names={"x":"xB", "y":"yB", "vx":"vxB", "vy":"vyB"}, cartesian_position=("xB", "yB"))
    animation.add_object(particleA, {"xA":-2, "vxA":1, "yA":0.0, "vyA":0.0})
    animation.add_object(particleB, {"xB":2, "vxB":-1, "yB":0.0, "vyB":0.0})

    collision_event = PhysicsEvent("collision_event", condition=EventCondition(expression="xA - xB", terminal=True, direction=1), transition=StateTransition(name="collision_impulse", equations={"vxA": "vxB", "vxB": "vxA"}))
    
    state_vector=["xA", "xB", "vxA", "vxB", "yA", "yB", "vyA", "vyB"]
    # equations可能不同段不同
    state_equations={
        "xA": "vxA",
        "xB": "vxB",
        "vxA": "E*qA/mA",
        "vxB": "E*qB/mB",
        "yA": "0.0",
        "yB": "0.0",
        "vyA": "0.0",
        "vyB": "0.0",
    }
    drived_equations={
    }
    # 这个必须公用同一套
    state_owners={
        "xA": "particleA",
        "xB": "particleB",
        "vxA": "particleA",
        "vxB": "particleB",
        "yA": "particleA",
        "yB": "particleB",
        "vyA": "particleA",
        "vyB": "particleB",
    }
    animation.add_segment(
        PhysicsSegment(
            segment_id="before_collision1",
            object_ids=["particleA", "particleB"],
            state_vector=state_vector,
            equations=state_equations,
            derived_equations=drived_equations,
            state_owners=state_owners,
            duration=100,
        ),
        end_event=collision_event,
    )
    animation.add_segment(
        PhysicsSegment(
            segment_id="before_collision2",
            object_ids=["particleA", "particleB"],
            state_vector=state_vector,
            equations=state_equations,
            derived_equations=drived_equations,
            state_owners=state_owners,
            duration=100,
        ),
        end_event=collision_event,
    )
    # animation.add_segment(
    #     PhysicsSegment(
    #         segment_id="before_collision",
    #         object_ids=["particleA", "particleB"],
    #         state_vector=state_vector,
    #         equations=state_equations,
    #         state_owners=state_owners,
    #         duration=100,
    #     ),
    #     end_event=collision_event,
    # )
    animation.add_segment(
        PhysicsSegment(
            segment_id="after_collision",
            object_ids=["particleA", "particleB"],
            state_vector=state_vector,
            equations=state_equations,
            derived_equations=drived_equations,
            state_owners=state_owners,
            duration=100,
        ),
        end_event=time_countdown_event(5),
    )

    scene = PhyAnimationScene2D()
    # scene.set_frame_size(width=1000, height=1000)
    scene.set_animation(animation)
    scene.render()