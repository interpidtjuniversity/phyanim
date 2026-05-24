from __future__ import annotations

from phyanim import (
    EventCondition,
    PhysicsAnimation,
    PhysicsEvent,
    PhysicsSegment,
    PointParticle,
    StateTransition,
)
from manim import Circle
from phyanim.solver import ScipySegmentSolver


def no_op_transition(name: str = "identity") -> StateTransition:
    return StateTransition(name=name, equations={})


def terminal_event(name: str, expression: str, *, direction: int = 0) -> PhysicsEvent:
    return PhysicsEvent(
        name,
        EventCondition(expression=expression, terminal=True, direction=direction),
        no_op_transition(),
    )


def terminal_time_event(name: str) -> PhysicsEvent:
    return terminal_event(name, "t - t_end", direction=1)


def test_sympy_equations_stop_on_event() -> None:
    animation = PhysicsAnimation()
    animation.add_object(PointParticle("p", mass=1.0), {"x": 0.0, "y": 0.0, "vx": 2.0, "vy": 0.0})

    event = terminal_event("reach_x_1", "x - 1", direction=1)
    animation.add_segment(
        PhysicsSegment(
            segment_id="move",
            object_ids=["p"],
            state_vector=["x", "y", "vx", "vy"],
            equations={"x": "vx", "y": "vy", "vx": "0", "vy": "0"},
            duration=3.0,
        ),
        end_event=event,
    )

    animation.solve()
    end = animation.keyframes[-1]
    assert end.event_name == "reach_x_1"
    assert abs(end.time - 0.5) < 1e-10
    assert abs(end.object_states["p"]["x"] - 1.0) < 1e-10


def test_dense_solution_returns_state_functions() -> None:
    animation = PhysicsAnimation()
    animation.add_object(PointParticle("p", mass=1.0), {"x": 0.0, "y": 0.0, "vx": 2.0, "vy": 3.0})
    animation.add_segment(
        PhysicsSegment(
            segment_id="projectile",
            object_ids=["p"],
            state_vector=["x", "y", "vx", "vy"],
            equations={"x": "vx", "y": "vy", "vx": "0", "vy": "-10"},
            derived_equations={"energy_like": "0.5*m*(vx**2 + vy**2) + m*10*y"},
            duration=1.0,
            ),
        end_event=terminal_time_event("projectile_end"),
    )

    results = animation.solve_segments(ScipySegmentSolver(sample_dt=0.2))
    solution = results[0].solution

    assert abs(solution.state_functions["x"](0.25) - 0.5) < 1e-9
    assert abs(solution.state_at(0.25)["y"] - (3.0 * 0.25 - 5.0 * 0.25**2)) < 1e-9
    assert abs(solution.state_at(0.25)["vy"] - 0.5) < 1e-9
    assert abs(solution.derived_at(0.25)["energy_like"] - 6.5) < 1e-8


def test_state_function_can_reject_out_of_range_time() -> None:
    animation = PhysicsAnimation()
    animation.add_object(PointParticle("p", mass=1.0, state_names={"x": "x", "y": "y", "vx": "vx", "vy": "vy"}, cartesian_position={"x": "x", "y": "y"}), {"x": 0.0, "y": 0.0, "vx": 1.0, "vy": 0.0})
    animation.add_segment(
        PhysicsSegment(
            segment_id="move",
            object_ids=["p"],
            state_vector=["x", "y", "vx", "vy"],
            equations={"x": "vx", "y": "vy", "vx": "0", "vy": "0"},
            duration=1.0,
            ),
        end_event=terminal_time_event("move_end"),
    )

    animation.solve()
    x_of_t = animation.build_state_functions(clamp=False)["move"]["p"]["x"]
    try:
        x_of_t(2.0)
    except ValueError as exc:
        assert "after trajectory end" in str(exc)
    else:
        raise AssertionError("Expected out-of-range lookup to raise ValueError.")


def test_event_transition_is_applied_by_framework_at_segment_end() -> None:
    animation = PhysicsAnimation()
    animation.add_object(PointParticle("p", mass=1.0), {"x": 0.0, "y": 0.0, "vx": 2.0, "vy": 0.0})

    event = PhysicsEvent(
        "bounce",
        EventCondition(expression="x - 1", terminal=True, direction=1),
        StateTransition(name="reverse_vx", equations={"vx": "-vx"}),
    )
    animation.add_segment(
        PhysicsSegment(
            segment_id="hit_wall",
            object_ids=["p"],
            state_vector=["x", "y", "vx", "vy"],
            equations={"x": "vx", "y": "vy", "vx": "0", "vy": "0"},
            duration=3.0,
        ),
        end_event=event,
    )

    result = animation.solve_segments()[0]
    assert result.triggered_event == "bounce"
    assert abs(result.end_keyframe.object_states["p"]["x"] - 1.0) < 1e-10
    assert abs(result.end_keyframe.object_states["p"]["vx"] - 2.0) < 1e-10
    assert abs(result.solution.state_at(result.solution.t1)["vx"] - 2.0) < 1e-10
    assert abs(animation.keyframes[-1].object_states["p"]["vx"] + 2.0) < 1e-10


def test_sympy_physics_symbols_do_not_collide_with_builtin_constants() -> None:
    animation = PhysicsAnimation(global_parameters={"E": 3.0})
    animation.add_object(PointParticle("p", mass=2.0, charge=4.0), {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0})
    animation.add_segment(
        PhysicsSegment(
            segment_id="electric_acceleration",
            object_ids=["p"],
            state_vector=["x", "y", "vx", "vy"],
            equations={"x": "vx", "y": "vy", "vx": "q*E/m", "vy": "0"},
            duration=1.0,
            ),
        end_event=terminal_time_event("electric_acceleration_end"),
    )

    result = animation.solve_segments()[0]
    assert abs(result.solution.state_at(1.0)["vx"] - 6.0) < 1e-9


def test_segment_supports_multiple_objects_and_transition_between_segments() -> None:
    animation = PhysicsAnimation()
    animation.add_object(
        PointParticle(
            "cart1",
            mass=1.0,
            parameter_names={"mass": "m1"},
            state_names={
                "x": "cart1_x",
                "y": "cart1_y",
                "vx": "cart1_vx",
                "vy": "cart1_vy",
            },
        ),
        {"cart1_x": 0.0, "cart1_vx": 2.0},
    )
    animation.add_object(
        PointParticle(
            "cart2",
            mass=2.0,
            parameter_names={"mass": "m2"},
            state_names={
                "x": "cart2_x",
                "y": "cart2_y",
                "vx": "cart2_vx",
                "vy": "cart2_vy",
            },
        ),
        {"cart2_x": 3.0, "cart2_vx": -1.0},
    )

    collision = PhysicsEvent(
        "collision",
        EventCondition(expression="cart2_x - cart1_x", terminal=True, direction=-1),
        StateTransition(
            name="elastic_collision",
            equations={
                "cart1_vx": "((m1-m2)/(m1+m2))*cart1_vx + (2*m2/(m1+m2))*cart2_vx",
                "cart2_vx": "(2*m1/(m1+m2))*cart1_vx + ((m2-m1)/(m1+m2))*cart2_vx",
            },
        ),
    )
    common = {
        "object_ids": ["cart1", "cart2"],
        "state_vector": ["cart1_x", "cart1_vx", "cart2_x", "cart2_vx"],
        "state_owners": {
            "cart1_x": "cart1",
            "cart1_vx": "cart1",
            "cart2_x": "cart2",
            "cart2_vx": "cart2",
        },
        "equations": {
            "cart1_x": "cart1_vx",
            "cart1_vx": "0",
            "cart2_x": "cart2_vx",
            "cart2_vx": "0",
        },
    }
    animation.add_segment(
        PhysicsSegment(
            segment_id="before_collision",
            duration=3.0,
            **common,
        ),
        end_event=collision,
    )
    animation.add_segment(
        PhysicsSegment(
            segment_id="after_collision",
            duration=0.5,
            **common,
            ),
        end_event=terminal_time_event("after_collision_end"),
    )

    results = animation.solve_segments()
    assert results[0].triggered_event == "collision"
    assert abs(results[0].end_keyframe.time - 1.0) < 1e-10
    assert abs(results[0].end_keyframe.object_states["cart1"]["cart1_vx"] - 2.0) < 1e-10
    assert abs(animation.keyframes[2].object_states["cart1"]["cart1_vx"] + 2.0) < 1e-10
    assert abs(animation.keyframes[2].object_states["cart2"]["cart2_vx"] - 1.0) < 1e-10
    assert abs(results[1].end_keyframe.object_states["cart1"]["cart1_x"] - 1.0) < 1e-10
    assert abs(results[1].end_keyframe.object_states["cart2"]["cart2_x"] - 2.5) < 1e-10
    assert results[1].solution.object_states_at(results[1].solution.t1)["cart1"]["cart1_x"] == results[1].solution.state_at(results[1].solution.t1)["cart1_x"]


def test_boundary_start_can_return_later_when_initial_motion_opposes_event_direction() -> None:
    animation = PhysicsAnimation()
    animation.add_object(PointParticle("p", mass=1.0), {"x": 0.0, "y": 0.0, "vx": 1.0, "vy": 0.0})

    boundary_return = terminal_event("boundary_return", "x", direction=-1)
    animation.add_segment(
        PhysicsSegment(
            segment_id="inside_region",
            object_ids=["p"],
            state_vector=["x", "y", "vx", "vy"],
            equations={"x": "vx", "y": "vy", "vx": "-1", "vy": "0"},
            duration=3.0,
        ),
        end_event=boundary_return,
    )

    result = animation.solve_segments()[0]
    assert result.triggered_event == "boundary_return"
    assert abs(result.end_keyframe.time - 2.0) < 1e-9


def test_boundary_start_can_return_later_when_crossing_direction_matches_later() -> None:
    animation = PhysicsAnimation()
    animation.add_object(PointParticle("p", mass=1.0), {"x": 0.0, "y": 0.0, "vx": -1.0, "vy": 0.0})

    boundary_return = terminal_event("boundary_return", "x", direction=1)
    animation.add_segment(
        PhysicsSegment(
            segment_id="inside_region",
            object_ids=["p"],
            state_vector=["x", "y", "vx", "vy"],
            equations={"x": "vx", "y": "vy", "vx": "1", "vy": "0"},
            duration=3.0,
        ),
        end_event=boundary_return,
    )

    result = animation.solve_segments()[0]
    assert result.triggered_event == "boundary_return"
    assert abs(result.end_keyframe.time - 2.0) < 1e-9


def test_end_event_triggered_at_segment_start_is_rejected() -> None:
    animation = PhysicsAnimation()
    animation.add_object(PointParticle("p", mass=1.0), {"x": 0.0, "y": 0.0, "vx": 1.0, "vy": 0.0})

    malformed_event = terminal_event("already_at_boundary", "x", direction=1)
    animation.add_segment(
        PhysicsSegment(
            segment_id="malformed",
            object_ids=["p"],
            state_vector=["x", "y", "vx", "vy"],
            equations={"x": "vx", "y": "vy", "vx": "0", "vy": "0"},
            duration=1.0,
        ),
        end_event=malformed_event,
    )

    try:
        animation.solve_segments()
    except RuntimeError as exc:
        assert "triggered at segment start" in str(exc)
    else:
        raise AssertionError("Expected an end event active at segment start to fail.")


def test_point_particle_defaults_to_renderable_circle_and_cartesian_coordinates() -> None:
    particle = PointParticle("ball", mass=1.0, state_names={"theta": "theta", "omega": "omega"}, cartesian_position={"x": "x_ball", "y": "y_ball"})

    assert isinstance(particle.mobject, Circle)
    assert particle.cartesian_position_variables() == ("x_ball", "y_ball")
    assert list(particle.state_variables) == ["theta", "omega"]


def test_point_particle_default_cartesian_coordinates_remain_simple() -> None:
    particle = PointParticle("p", mass=1.0, state_names={"x": "x", "y": "y", "vx": "vx", "vy": "vy"}, cartesian_position={"x": "x", "y": "y"})

    assert isinstance(particle.mobject, Circle)
    assert particle.cartesian_position_variables() == ("x", "y")
    assert list(particle.state_variables) == ["x", "y", "vx", "vy"]
