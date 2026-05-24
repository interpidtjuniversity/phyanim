from __future__ import annotations

from math import pi, sqrt

from phyanim import (
    EventCondition,
    PhysicsAnimation,
    PhysicsEvent,
    PhysicsSegment,
    PhysicObject2D,
    PointParticle,
    StateTransition,
    StateVariable,
    time_countdown_event,
)
from phyanim.solver import ScipySegmentSolver


def build_sliding_track_animation(duration: float = 4.0) -> PhysicsAnimation:
    animation = PhysicsAnimation(
        global_parameters={
            "g": 9.8,
            "R": 1.0,
            "m": 1.0,
            "M": 5.0,
        }
    )
    track = PhysicObject2D(
        object_id="track",
        state_variables={
            "x_track": StateVariable("x_track", "m", "track center position"),
        },
    )
    ball = PhysicObject2D(
        object_id="ball",
        state_variables={
            "theta": StateVariable("theta", "rad", "ball angle"),
            "omega": StateVariable("omega", "rad/s", "angular velocity"),
        },
    )

    animation.add_object(track, {"x_track": 0.0})
    animation.add_object(ball, {"theta": -pi / 2, "omega": 0.0})
    animation.add_segment(
        PhysicsSegment(
            segment_id="sliding",
            object_ids=["ball", "track"],
            state_vector=["theta", "omega", "x_track"],
            equations={
                "theta": "omega",
                "omega": "-(m*omega**2*sin(theta)*cos(theta) + g*(M+m)*sin(theta)/R) / (M + m*sin(theta)**2)",
                "x_track": "-m*R*cos(theta)*omega/(M+m)",
            },
            derived_equations={
                "x_ball": "x_track + R*sin(theta)",
                "y_ball": "R*(1 - cos(theta))",
                "E_total": "0.5*m*R**2*omega**2*(M + m*sin(theta)**2)/(M+m) + m*g*R*(1 - cos(theta))",
                "Px_total": "m*R*cos(theta)*omega*M/(M+m) - M*m*R*cos(theta)*omega/(M+m)",
            },
            state_owners={
                "theta": "ball",
                "omega": "ball",
                "x_track": "track",
            },
            duration=duration + 1.0,
        ),
        end_event=time_countdown_event(duration, name="sliding_end"),
    )
    return animation


def build_charged_collision_animation() -> PhysicsAnimation:
    animation = PhysicsAnimation(global_parameters={"E": 1.0})
    particle_a = PointParticle(
        "particleA",
        mass=1.0,
        charge=1.0,
        parameter_names={"mass": "mA", "charge": "qA"},
        state_names={"x": "xA", "y": "yA", "vx": "vxA", "vy": "vyA"},
    )
    particle_b = PointParticle(
        "particleB",
        mass=1.0,
        charge=-1.0,
        parameter_names={"mass": "mB", "charge": "qB"},
        state_names={"x": "xB", "y": "yB", "vx": "vxB", "vy": "vyB"},
    )
    animation.add_object(particle_a, {"xA": -2.0, "vxA": 1.0, "yA": 0.0, "vyA": 0.0})
    animation.add_object(particle_b, {"xB": 2.0, "vxB": -1.0, "yB": 0.0, "vyB": 0.0})

    collision_event = PhysicsEvent(
        "collision_event",
        condition=EventCondition(expression="xA - xB", terminal=True, direction=1),
        transition=StateTransition(
            name="collision_impulse",
            equations={"vxA": "vxB", "vxB": "vxA"},
        ),
    )
    state_vector = ["xA", "xB", "vxA", "vxB", "yA", "yB", "vyA", "vyB"]
    state_equations = {
        "xA": "vxA",
        "xB": "vxB",
        "vxA": "E*qA/mA",
        "vxB": "E*qB/mB",
        "yA": "vyA",
        "yB": "vyB",
        "vyA": "0.0",
        "vyB": "0.0",
    }
    state_owners = {
        "xA": "particleA",
        "xB": "particleB",
        "vxA": "particleA",
        "vxB": "particleB",
        "yA": "particleA",
        "yB": "particleB",
        "vyA": "particleA",
        "vyB": "particleB",
    }

    for segment_id in ("before_collision1", "before_collision2"):
        animation.add_segment(
            PhysicsSegment(
                segment_id=segment_id,
                object_ids=["particleA", "particleB"],
                state_vector=state_vector,
                equations=state_equations,
                state_owners=state_owners,
                duration=100.0,
            ),
            end_event=collision_event,
        )
    animation.add_segment(
        PhysicsSegment(
            segment_id="after_collision",
            object_ids=["particleA", "particleB"],
            state_vector=state_vector,
            equations=state_equations,
            state_owners=state_owners,
            duration=100.0,
        ),
        end_event=time_countdown_event(5, name="after_collision_end"),
    )
    return animation


def test_conservation_of_momentum_example_keeps_invariants_stable() -> None:
    animation = build_sliding_track_animation(duration=4.0)
    results = animation.solve_segments(ScipySegmentSolver(sample_dt=1 / 60))

    trajectory = results[0].trajectory
    energies = trajectory.derived["E_total"]
    momenta = trajectory.derived["Px_total"]

    assert results[0].triggered_event == "sliding_end"
    assert abs(results[0].end_keyframe.time - 4.0) < 1e-10
    assert max(abs(value - energies[0]) for value in energies) < 1e-6
    assert max(abs(value) for value in momenta) < 1e-10
    assert trajectory.derived["y_ball"][0] > trajectory.derived["y_ball"][-1]
    assert len(trajectory.times) > 100


def test_charged_particle_collision_example_handles_repeated_transitions() -> None:
    animation = build_charged_collision_animation()
    results = animation.solve_segments(ScipySegmentSolver(sample_dt=0.05))

    assert [result.triggered_event for result in results] == [
        "collision_event",
        "collision_event",
        "after_collision_end",
    ]
    assert abs(results[0].end_keyframe.time - (sqrt(5) - 1)) < 1e-9
    assert abs(results[1].end_keyframe.time - (2 * sqrt(5) + sqrt(5) - 1)) < 1e-9

    after_first_collision = animation.keyframes[2].object_states
    before_second_collision = results[1].end_keyframe.object_states
    after_second_collision = animation.keyframes[4].object_states
    assert after_first_collision["particleA"]["vxA"] < 0.0
    assert after_first_collision["particleB"]["vxB"] > 0.0
    assert before_second_collision["particleA"]["vxA"] > 0.0
    assert before_second_collision["particleB"]["vxB"] < 0.0
    assert after_second_collision["particleA"]["vxA"] < 0.0
    assert after_second_collision["particleB"]["vxB"] > 0.0

    final_state = results[-1].end_keyframe.object_states
    assert final_state["particleA"]["xA"] > final_state["particleB"]["xB"]
    assert final_state["particleA"]["yA"] == 0.0
    assert final_state["particleB"]["yB"] == 0.0


def test_unchanged_object_states_are_carried_across_segment_functions() -> None:
    animation = PhysicsAnimation()
    animation.add_object(PointParticle("moving", mass=1.0), {"x": 0.0, "y": 0.0, "vx": 1.0, "vy": 0.0})
    animation.add_object(
        PointParticle(
            "marker",
            mass=1.0,
            parameter_names={"mass": "marker_m"},
            state_names={
                "x": "marker_x",
                "y": "marker_y",
                "vx": "marker_vx",
                "vy": "marker_vy",
            },
        ),
        {"marker_x": 10.0, "marker_y": -1.0, "marker_vx": 0.0, "marker_vy": 0.0},
    )
    animation.add_segment(
        PhysicsSegment(
            segment_id="moving_only",
            object_ids=["moving"],
            state_vector=["x", "y", "vx", "vy"],
            equations={"x": "vx", "y": "vy", "vx": "0", "vy": "0"},
            duration=1.0,
        ),
        end_event=time_countdown_event(1, name="moving_only_end"),
    )

    animation.solve_segments()
    state_functions = animation.build_state_functions(clamp=False)
    marker_x = state_functions["moving_only"]["marker"]["marker_x"]

    assert marker_x(0.5) == 10.0
    try:
        marker_x(2.0)
    except ValueError as exc:
        assert "after trajectory end" in str(exc)
    else:
        raise AssertionError("Expected carried state function to honor clamp=False.")


def test_derived_quantities_are_carried_across_later_segments() -> None:
    animation = PhysicsAnimation()
    animation.add_object(PointParticle("p", mass=2.0), {"x": 0.0, "y": 0.0, "vx": 3.0, "vy": 0.0})
    animation.add_segment(
        PhysicsSegment(
            segment_id="with_energy",
            object_ids=["p"],
            state_vector=["x", "y", "vx", "vy"],
            equations={"x": "vx", "y": "vy", "vx": "0", "vy": "0"},
            derived_equations={"kinetic": "0.5*m*(vx**2 + vy**2)"},
            duration=1.0,
        ),
        end_event=time_countdown_event(1, name="with_energy_end"),
    )
    animation.add_segment(
        PhysicsSegment(
            segment_id="without_energy",
            object_ids=["p"],
            state_vector=["x", "y", "vx", "vy"],
            equations={"x": "vx", "y": "vy", "vx": "0", "vy": "0"},
            duration=1.0,
        ),
        end_event=time_countdown_event(1, name="without_energy_end"),
    )

    animation.solve_segments()
    derived_functions = animation.build_derived_functions(clamp=False)
    carried_kinetic = derived_functions["without_energy"]["kinetic"]

    assert abs(carried_kinetic(1.5) - 9.0) < 1e-12
    try:
        carried_kinetic(3.0)
    except ValueError as exc:
        assert "after trajectory end" in str(exc)
    else:
        raise AssertionError("Expected carried derived function to honor clamp=False.")
