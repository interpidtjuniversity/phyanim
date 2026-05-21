from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phyanim import (
    EventCondition,
    PhysicsAnimation,
    PhysicsEvent,
    PhysicsSegment,
    PointParticle,
    StateTransition,
)
from phyanim.render import TimelineExporter


STATE_VECTOR = ["x", "y", "vx", "vy"]
PROJECTILE_EQUATIONS = {"x": "vx", "y": "vy", "vx": "0", "vy": "-g"}
ENERGY_EQUATIONS = {
    "mechanical_energy": "0.5*m*(vx**2 + vy**2) + m*g*y",
}


def build_animation() -> PhysicsAnimation:
    animation = PhysicsAnimation(global_parameters={"g": 9.8, "e": 0.72})
    animation.add_object(
        PointParticle("ball", mass=1.0),
        {"x": 0.0, "y": 2.4, "vx": 2.0, "vy": 0.8},
    )

    ground_hit = PhysicsEvent(
        "hit_ground",
        EventCondition(expression="y", terminal=True, direction=-1),
        StateTransition(name="restitution", equations={"vy": "-e*vy"}),
    )
    second_ground_hit = PhysicsEvent(
        "second_hit_ground",
        EventCondition(expression="y", terminal=True, direction=-1),
    )

    animation.add_segment(
        PhysicsSegment(
            segment_id="first_flight",
            object_ids=["ball"],
            state_vector=STATE_VECTOR,
            equations=PROJECTILE_EQUATIONS,
            derived_equations=ENERGY_EQUATIONS,
            duration=3.0,
        ),
        end_event=ground_hit,
    )
    animation.add_segment(
        PhysicsSegment(
            segment_id="after_bounce",
            object_ids=["ball"],
            state_vector=STATE_VECTOR,
            equations=PROJECTILE_EQUATIONS,
            derived_equations=ENERGY_EQUATIONS,
            duration=3.0,
        ),
        end_event=second_ground_hit,
    )
    return animation


def main() -> None:
    animation = build_animation()
    from phyanim.solver import ScipySegmentSolver

    animation.solve(ScipySegmentSolver(sample_dt=1 / 60))
    output_path = Path("outputs/bouncing_projectile_timeline.json")
    TimelineExporter().write_json(animation, output_path)
    print(f"Wrote {output_path}")
    for keyframe in animation.keyframes:
        print(keyframe)


if __name__ == "__main__":
    main()
