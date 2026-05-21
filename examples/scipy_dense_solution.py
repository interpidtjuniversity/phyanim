from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phyanim import PhysicsAnimation, PhysicsSegment, PointParticle, time_end_event
from phyanim.solver import ScipySegmentSolver


def main() -> None:
    animation = PhysicsAnimation(global_parameters={"g": 9.8})
    animation.add_object(PointParticle("ball", mass=1.0), {"x": 0.0, "y": 0.0, "vx": 6.0, "vy": 8.0})
    animation.add_segment(
        PhysicsSegment(
            segment_id="projectile",
            object_ids=["ball"],
            state_vector=["x", "y", "vx", "vy"],
            equations={"x": "vx", "y": "vy", "vx": "0", "vy": "-g"},
            duration=1.5,
        ),
        end_event=time_end_event("projectile_end"),
    )

    results = animation.solve_segments(
        ScipySegmentSolver(method="DOP853", rtol=1e-10, atol=1e-12, sample_dt=1 / 30)
    )
    solution = results[0].solution
    x_of_t = solution.state_functions["x"]
    y_of_t = solution.state_functions["y"]

    query_time = 0.75
    print(solution.state_at(query_time))
    print({"x": x_of_t(query_time), "y": y_of_t(query_time)})


if __name__ == "__main__":
    main()
