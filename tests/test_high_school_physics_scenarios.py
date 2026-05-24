from __future__ import annotations

from math import exp, pi, sqrt

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


def terminal_event(name: str, expression: str, *, direction: int = 0) -> PhysicsEvent:
    return PhysicsEvent(
        name=name,
        condition=EventCondition(expression=expression, terminal=True, direction=direction),
        transition=StateTransition(name=f"{name}_identity", equations={}),
    )


def test_projectile_returns_to_ground_with_expected_range_and_energy() -> None:
    g = 9.8
    vx0 = 6.0
    vy0 = 8.0
    expected_flight_time = 2 * vy0 / g

    animation = PhysicsAnimation(global_parameters={"g": g})
    animation.add_object(PointParticle("projectile", mass=2.0), {"x": 0.0, "y": 0.0, "vx": vx0, "vy": vy0})
    animation.add_segment(
        PhysicsSegment(
            segment_id="flight",
            object_ids=["projectile"],
            state_vector=["x", "y", "vx", "vy"],
            equations={"x": "vx", "y": "vy", "vx": "0", "vy": "-g"},
            derived_equations={"energy": "0.5*m*(vx**2 + vy**2) + m*g*y"},
            duration=expected_flight_time + 1.0,
        ),
        end_event=terminal_event("hit_ground", "y", direction=-1),
    )

    result = animation.solve_segments(ScipySegmentSolver(sample_dt=0.02))[0]
    end_state = result.end_keyframe.object_states["projectile"]
    energy = result.trajectory.derived["energy"]

    assert result.triggered_event == "hit_ground"
    assert abs(result.end_keyframe.time - expected_flight_time) < 1e-9
    assert abs(end_state["x"] - vx0 * expected_flight_time) < 1e-8
    assert abs(end_state["y"]) < 1e-8
    assert abs(end_state["vy"] + vy0) < 1e-8
    assert max(abs(value - energy[0]) for value in energy) < 1e-8


def test_mass_spring_oscillator_completes_one_period_and_conserves_energy() -> None:
    mass = 2.0
    spring_k = 8.0
    amplitude = 0.75
    omega = sqrt(spring_k / mass)
    period = 2 * pi / omega

    animation = PhysicsAnimation(global_parameters={"k": spring_k})
    animation.add_object(PointParticle("block", mass=mass), {"x": amplitude, "y": 0.0, "vx": 0.0, "vy": 0.0})
    animation.add_segment(
        PhysicsSegment(
            segment_id="oscillation",
            object_ids=["block"],
            state_vector=["x", "y", "vx", "vy"],
            equations={"x": "vx", "y": "vy", "vx": "-k*x/m", "vy": "0"},
            derived_equations={"energy": "0.5*m*vx**2 + 0.5*k*x**2"},
            duration=period + 0.1,
        ),
        end_event=time_countdown_event(period, name="one_period"),
    )

    result = animation.solve_segments(ScipySegmentSolver(sample_dt=period / 200))[0]
    end_state = result.end_keyframe.object_states["block"]
    energy = result.trajectory.derived["energy"]

    assert abs(result.end_keyframe.time - period) < 1e-9
    assert abs(end_state["x"] - amplitude) < 1e-8
    assert abs(end_state["vx"]) < 1e-8
    assert max(abs(value - energy[0]) for value in energy) < 1e-8


def test_uniform_magnetic_field_returns_particle_after_one_cyclotron_period() -> None:
    charge = 2.0
    mass = 1.0
    magnetic_b = 3.0
    speed = 5.0
    omega = charge * magnetic_b / mass
    radius = speed / omega
    period = 2 * pi / omega

    animation = PhysicsAnimation(global_parameters={"B": magnetic_b})
    animation.add_object(PointParticle("ion", mass=mass, charge=charge), {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": speed})
    animation.add_segment(
        PhysicsSegment(
            segment_id="cyclotron_orbit",
            object_ids=["ion"],
            state_vector=["x", "y", "vx", "vy"],
            equations={"x": "vx", "y": "vy", "vx": "q*B*vy/m", "vy": "-q*B*vx/m"},
            derived_equations={
                "speed_sq": "vx**2 + vy**2",
                "orbit_radius_sq": "(x - v0_over_omega)**2 + y**2",
            },
            parameters={"v0_over_omega": radius},
            duration=period + 0.1,
        ),
        end_event=time_countdown_event(period, name="one_cyclotron_period"),
    )

    result = animation.solve_segments(ScipySegmentSolver(sample_dt=period / 240))[0]
    end_state = result.end_keyframe.object_states["ion"]
    speed_sq = result.trajectory.derived["speed_sq"]
    orbit_radius_sq = result.trajectory.derived["orbit_radius_sq"]

    assert abs(result.end_keyframe.time - period) < 1e-9
    assert abs(end_state["x"]) < 1e-8
    assert abs(end_state["y"]) < 1e-8
    assert abs(end_state["vx"]) < 1e-8
    assert abs(end_state["vy"] - speed) < 1e-8
    assert max(abs(value - speed**2) for value in speed_sq) < 2e-8
    assert max(abs(value - radius**2) for value in orbit_radius_sq) < 2e-8


def test_lc_circuit_oscillation_conserves_total_electromagnetic_energy() -> None:
    capacitance = 0.5
    inductance = 2.0
    initial_charge = 3.0
    omega = 1 / sqrt(inductance * capacitance)
    period = 2 * pi / omega

    animation = PhysicsAnimation(global_parameters={"C": capacitance, "L": inductance})
    animation.add_object(
        PhysicObject2D(
            object_id="circuit",
            state_variables={
                "charge": StateVariable("charge", "C", "capacitor charge"),
                "current": StateVariable("current", "A", "loop current"),
            },
        ),
        {"charge": initial_charge, "current": 0.0},
    )
    animation.add_segment(
        PhysicsSegment(
            segment_id="lc_oscillation",
            object_ids=["circuit"],
            state_vector=["charge", "current"],
            equations={"charge": "current", "current": "-charge/(L*C)"},
            derived_equations={"energy": "0.5*charge**2/C + 0.5*L*current**2"},
            duration=period + 0.1,
        ),
        end_event=time_countdown_event(period, name="lc_one_period"),
    )

    result = animation.solve_segments(ScipySegmentSolver(sample_dt=period / 200))[0]
    end_state = result.end_keyframe.object_states["circuit"]
    energy = result.trajectory.derived["energy"]

    assert abs(result.end_keyframe.time - period) < 1e-9
    assert abs(end_state["charge"] - initial_charge) < 1e-8
    assert abs(end_state["current"]) < 1e-8
    assert max(abs(value - energy[0]) for value in energy) < 1e-8


def test_linear_drag_motion_matches_terminal_velocity_solution() -> None:
    mass = 2.0
    drag = 0.5
    gravity = 9.8
    duration = 6.0
    beta = drag / mass
    terminal_velocity = gravity / beta
    expected_v = terminal_velocity * (1 - exp(-beta * duration))
    expected_y = terminal_velocity * (duration - (1 - exp(-beta * duration)) / beta)

    animation = PhysicsAnimation(global_parameters={"g": gravity, "b": drag})
    animation.add_object(PointParticle("drop", mass=mass), {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0})
    animation.add_segment(
        PhysicsSegment(
            segment_id="linear_drag_fall",
            object_ids=["drop"],
            state_vector=["x", "y", "vx", "vy"],
            equations={"x": "vx", "y": "vy", "vx": "0", "vy": "g - b*vy/m"},
            derived_equations={"terminal_gap": "m*g/b - vy"},
            duration=duration + 1.0,
        ),
        end_event=time_countdown_event(duration, name="drag_duration"),
    )

    result = animation.solve_segments(ScipySegmentSolver(sample_dt=0.05))[0]
    end_state = result.end_keyframe.object_states["drop"]
    terminal_gaps = result.trajectory.derived["terminal_gap"]

    assert abs(result.end_keyframe.time - duration) < 1e-9
    assert abs(end_state["vy"] - expected_v) < 1e-8
    assert abs(end_state["y"] - expected_y) < 1e-8
    assert all(left > right for left, right in zip(terminal_gaps, terminal_gaps[1:]))


def test_inelastic_bounce_then_apex_uses_event_transition_between_segments() -> None:
    gravity = 9.8
    height = 5.0
    restitution = 0.6
    impact_speed = sqrt(2 * gravity * height)
    first_hit_time = sqrt(2 * height / gravity)
    expected_apex_time = first_hit_time + restitution * impact_speed / gravity
    expected_apex_height = restitution**2 * height

    animation = PhysicsAnimation(global_parameters={"g": gravity, "e": restitution})
    animation.add_object(PointParticle("ball", mass=1.0), {"x": 0.0, "y": height, "vx": 0.0, "vy": 0.0})

    animation.add_segment(
        PhysicsSegment(
            segment_id="drop_to_floor",
            object_ids=["ball"],
            state_vector=["x", "y", "vx", "vy"],
            equations={"x": "vx", "y": "vy", "vx": "0", "vy": "-g"},
            duration=first_hit_time + 1.0,
        ),
        end_event=PhysicsEvent(
            name="hit_floor",
            condition=EventCondition(expression="y", terminal=True, direction=-1),
            transition=StateTransition(name="inelastic_bounce", equations={"vy": "-e*vy"}),
        ),
    )
    animation.add_segment(
        PhysicsSegment(
            segment_id="rise_to_apex",
            object_ids=["ball"],
            state_vector=["x", "y", "vx", "vy"],
            equations={"x": "vx", "y": "vy", "vx": "0", "vy": "-g"},
            derived_equations={"energy": "0.5*m*vy**2 + m*g*y"},
            duration=2.0,
        ),
        end_event=terminal_event("apex", "vy", direction=-1),
    )

    results = animation.solve_segments(ScipySegmentSolver(sample_dt=0.01))
    impact_state = results[0].end_keyframe.object_states["ball"]
    post_bounce_state = animation.keyframes[2].object_states["ball"]
    apex_state = results[1].end_keyframe.object_states["ball"]
    post_bounce_energy = results[1].trajectory.derived["energy"][0]
    apex_energy = results[1].trajectory.derived["energy"][-1]

    assert [result.triggered_event for result in results] == ["hit_floor", "apex"]
    assert abs(results[0].end_keyframe.time - first_hit_time) < 1e-9
    assert impact_state["vy"] < 0.0
    assert abs(post_bounce_state["vy"] - restitution * impact_speed) < 1e-8
    assert abs(results[1].end_keyframe.time - expected_apex_time) < 1e-9
    assert abs(apex_state["vy"]) < 1e-8
    assert abs(apex_state["y"] - expected_apex_height) < 1e-8
    assert abs(apex_energy - post_bounce_energy) < 1e-8
