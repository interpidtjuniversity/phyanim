from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phyanim import EventCondition, PhysicsAnimation, PhysicsEvent, PhysicsSegment, PointParticle, time_end_event
from phyanim.render import TimelineExporter


def main() -> None:
    animation = PhysicsAnimation()
    particle = PointParticle("particle", mass=1.0, charge=1.0)
    animation.add_object(particle, {"x": -2.0, "y": 0.0, "vx": 3.0, "vy": 0.0})

    enter_field = PhysicsEvent(
        "enter_field",
        EventCondition(expression="x", terminal=True, direction=1),
    )
    exit_field = PhysicsEvent(
        "exit_field",
        EventCondition(expression="x - 2", terminal=True, direction=1),
    )

    state_vector = ["x", "y", "vx", "vy"]
    uniform_equations = {"x": "vx", "y": "vy", "vx": "0", "vy": "0"}
    kinetic_energy = {"kinetic_energy": "0.5*m*(vx**2 + vy**2)"}

    animation.add_segment(
        PhysicsSegment(
            segment_id="before_field",
            object_ids=["particle"],
            state_vector=state_vector,
            equations=uniform_equations,
            derived_equations=kinetic_energy,
            duration=2.0,
        ),
        end_event=enter_field,
    )
    animation.add_segment(
        PhysicsSegment(
            segment_id="inside_field",
            object_ids=["particle"],
            state_vector=state_vector,
            equations={
                "x": "vx",
                "y": "vy",
                "vx": "q*B*vy/m",
                "vy": "-q*B*vx/m",
            },
            parameters={"B": 0.4},
            derived_equations=kinetic_energy,
            duration=3.0,
        ),
        end_event=exit_field,
    )
    animation.add_segment(
        PhysicsSegment(
            segment_id="after_field",
            object_ids=["particle"],
            state_vector=state_vector,
            equations=uniform_equations,
            derived_equations=kinetic_energy,
            duration=1.0,
        ),
        end_event=time_end_event("after_field_end"),
    )

    from phyanim.solver import ScipySegmentSolver

    animation.solve(ScipySegmentSolver(sample_dt=0.01))
    output_path = Path("outputs/charged_particle_timeline.json")
    TimelineExporter().write_json(animation, output_path)
    print(f"Wrote {output_path}")
    for keyframe in animation.keyframes:
        print(keyframe)


if __name__ == "__main__":
    main()
