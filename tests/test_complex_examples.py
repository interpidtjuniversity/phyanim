from __future__ import annotations

from examples.bouncing_projectile_segments import build_animation as build_bouncing_projectile
from examples.electric_to_magnetic_segments import build_animation as build_electric_to_magnetic
from phyanim import PhysicsAnimation, StateTransition


def ensure_terminal_transitions(animation: PhysicsAnimation) -> PhysicsAnimation:
    for event in animation.segment_end_events:
        if event.transition is None:
            event.transition = StateTransition(name="identity", equations={})
    return animation


def test_bouncing_projectile_example() -> None:
    animation = ensure_terminal_transitions(build_bouncing_projectile())
    results = animation.solve_segments()

    first_event_after = animation.keyframes[2].object_states["ball"]
    second_end = results[1].end_keyframe.object_states["ball"]

    assert results[0].triggered_event == "hit_ground"
    assert results[1].triggered_event == "second_hit_ground"
    assert abs(first_event_after["y"]) < 1e-9
    assert first_event_after["vy"] > 0.0
    assert abs(second_end["y"]) < 1e-8
    assert second_end["vy"] < 0.0
    assert results[1].end_keyframe.time > results[0].end_keyframe.time


def test_electric_to_magnetic_example() -> None:
    animation = ensure_terminal_transitions(build_electric_to_magnetic())
    results = animation.solve_segments()

    electric_end = results[0].end_keyframe.object_states["particle"]
    magnetic_start_speed_sq = electric_end["vx"] ** 2 + electric_end["vy"] ** 2
    magnetic_end = results[1].end_keyframe.object_states["particle"]
    magnetic_end_speed_sq = magnetic_end["vx"] ** 2 + magnetic_end["vy"] ** 2
    free_end = results[2].end_keyframe.object_states["particle"]

    assert [result.triggered_event for result in results] == [
        "exit_electric_region",
        "exit_magnetic_region",
        "free_motion_end",
    ]
    assert abs(electric_end["x"] - 1.0) < 1e-9
    assert abs(magnetic_end["x"] - 2.0) < 1e-8
    assert abs(magnetic_end_speed_sq - magnetic_start_speed_sq) < 1e-8
    assert free_end["x"] > magnetic_end["x"]
