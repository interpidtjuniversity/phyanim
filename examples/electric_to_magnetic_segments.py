from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phyanim import EventCondition, PhysicsAnimation, PhysicsEvent, PhysicsSegment, PointParticle, time_end_event
from phyanim.render import TimelineExporter


STATE_VECTOR = ["x", "y", "vx", "vy"]
KINETIC_ENERGY = {"kinetic_energy": "0.5*m*(vx**2 + vy**2)"}


def build_animation() -> PhysicsAnimation:
    animation = PhysicsAnimation()
    animation.add_object(
        PointParticle("particle", mass=1.0, charge=1.0),
        {"x": 0.0, "y": 0.0, "vx": 0.6, "vy": 0.0},
    )

    exit_electric_region = PhysicsEvent(
        "exit_electric_region",
        EventCondition(expression="x - 1.0", terminal=True, direction=1),
    )
    exit_magnetic_region = PhysicsEvent(
        "exit_magnetic_region",
        EventCondition(expression="x - 2.0", terminal=True, direction=1),
    )

    animation.add_segment(
        PhysicsSegment(
            segment_id="electric_acceleration",
            object_ids=["particle"],
            state_vector=STATE_VECTOR,
            equations={"x": "vx", "y": "vy", "vx": "q*E/m", "vy": "0"},
            parameters={"E": 1.4},
            derived_equations=KINETIC_ENERGY,
            duration=3.0,
        ),
        end_event=exit_electric_region,
    )
    animation.add_segment(
        PhysicsSegment(
            segment_id="magnetic_deflection",
            object_ids=["particle"],
            state_vector=STATE_VECTOR,
            equations={
                "x": "vx",
                "y": "vy",
                "vx": "q*B*vy/m",
                "vy": "-q*B*vx/m",
            },
            parameters={"B": 0.55},
            derived_equations=KINETIC_ENERGY,
            duration=4.0,
        ),
        end_event=exit_magnetic_region,
    )
    animation.add_segment(
        PhysicsSegment(
            segment_id="free_motion",
            object_ids=["particle"],
            state_vector=STATE_VECTOR,
            equations={"x": "vx", "y": "vy", "vx": "0", "vy": "0"},
            derived_equations=KINETIC_ENERGY,
            duration=0.8,
        ),
        end_event=time_end_event("free_motion_end"),
    )
    return animation


def main() -> None:
    animation = build_animation()
    from phyanim.solver import ScipySegmentSolver

    animation.solve(ScipySegmentSolver(sample_dt=1 / 90))
    output_path = Path("outputs/electric_to_magnetic_timeline.json")
    TimelineExporter().write_json(animation, output_path)
    print(f"Wrote {output_path}")
    for keyframe in animation.keyframes:
        print(keyframe)


if __name__ == "__main__":
    main()
