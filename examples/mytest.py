from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phyanim import EventCondition, PhysicsAnimation, PhysicsEvent, PhysicsSegment, PointParticle, StateTransition, time_countdown_event
from phyanim.render import TimelineExporter
from phyanim.solver import ScipySegmentSolver

if __name__ == "__main__":
    # 向右的匀强场，场强为1N/C
    animation = PhysicsAnimation(global_parameters={"E":1})
    # 定义一个带电点粒子A
    particleA = PointParticle("particleA",  mass=1.0, charge=1.0, radius=0.0, parameter_names={"mass":"mA", "charge":"qA"}, state_names={"x":"xA", "y":"yA", "vx":"vxA", "vy":"vyA"})
    # 定义一个带电点粒子B
    particleB = PointParticle("particleB",  mass=1.0, charge=-1.0, radius=0.0, parameter_names={"mass":"mB", "charge":"qB"}, state_names={"x":"xB", "y":"yB", "vx":"vxB", "vy":"vyB"})
    animation.add_object(particleA, {"xA":10, "vxA":1})
    animation.add_object(particleB, {"xB":20, "vxB":-1})
    # 定义一个事件：粒子A和B弹性相撞
    collision_event = PhysicsEvent("collision_event", condition=EventCondition(expression="xA - xB", terminal=True, direction=1), transition=StateTransition(name="collision_impulse", equations={"vxA": "vxB", "vxB": "vxA"}))
    
    state_vector=["xA", "xB", "vxA", "vxB"]
    state_equations={
        "xA": "vxA",
        "xB": "vxB",
        "vxA": "E*qA/mA",
        "vxB": "E*qB/mB",
    }
    # 这个必须公用同一套
    state_owners={
        "xA": "particleA",
        "xB": "particleB",
        "vxA": "particleA",
        "vxB": "particleB",
    }
    animation.add_segment(
        PhysicsSegment(
            segment_id="before_collision",
            object_ids=["particleA", "particleB"],
            state_vector=state_vector,
            equations=state_equations,
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
            state_owners=state_owners,
            duration=100,
        ),
        end_event=time_countdown_event(3),
    )
    animation.solve(ScipySegmentSolver(sample_dt=1/20))
    exporter = TimelineExporter()
    exporter.plot_variables(animation, show=True)
    # output_path = Path("phyanim/outputs/mytest.json")
    # exporter = TimelineExporter()
    # TimelineExporter().write_json(animation, output_path)
    # print(f"Wrote {output_path}")
    # for keyframe in animation.keyframes:
    #     print(keyframe)