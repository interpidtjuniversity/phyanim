"""Multi-scene + kinematic + visual bindings demo.

Scene 1 (title): A ball sits at rest; visual binding changes its color
                 based on a static parameter (demonstrates the binding API).

Scene 2 (simulation): The ball undergoes free-fall using the *kinematic*
                       mode (closed-form expression x = x0 + v0*t,
                       y = y0 - 0.5*g*t**2) — no ODE integration.

Scene 3 (ode): The ball bounces on the floor using a spring ODE, with
                a visual binding that changes scale based on velocity.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manim import Circle, Text, VGroup, DOWN

from phyanim.core.animation import PhysicsAnimation
from phyanim.core.segment import PhysicsSegment
from phyanim.core.events import time_countdown_event
from phyanim.core.state import StateVariable
from phyanim.core.objects import PhysicObject2D
from phyanim.core.enhance.visual_binding import VisualBinding
from phyanim.render import PhyAnimationMultiLayerScene2D


def build_scene_01_title() -> PhysicsAnimation:
    """Static title scene — ball with a color visual binding."""
    animation = PhysicsAnimation(
        global_parameters={"g": 9.8},
        engine="kinematic",
        sample_dt=1 / 30,
    )

    ball = PhysicObject2D(
        object_id="ball",
        state_variables={
            "x": StateVariable("x", "m", "水平位置"),
            "y": StateVariable("y", "m", "垂直位置"),
        },
        cartesian_position=[("x", "y")],
        mobject=Circle(radius=0.2, color="YELLOW"),
    )
    # Visual binding: scale based on y (demonstrates the API).
    ball.visual_bindings.append(
        VisualBinding(
            attribute="scale",
            variables=["y"],
            expression="1.0 + 0.05*y",
        )
    )
    animation.add_object(ball, {"x": "0.0", "y": "0.0"})

    # Kinematic: ball stays still.
    animation.add_segment(
        PhysicsSegment.from_kinematic(
            segment_id="title_hold",
            objects=["ball"],
            state_vector=["x", "y"],
            expressions={"x": "0.0", "y": "0.0"},
            duration=2.0,
        ),
        end_event=time_countdown_event(2.0),
    )
    return animation


def build_scene_02_kinematic_fall() -> PhysicsAnimation:
    """Free-fall using closed-form kinematic expressions."""
    animation = PhysicsAnimation(
        global_parameters={"g": 9.8},
        engine="kinematic",
        sample_dt=1 / 30,
    )

    ball = PhysicObject2D(
        object_id="ball",
        state_variables={
            "x": StateVariable("x", "m", "水平位置"),
            "y": StateVariable("y", "m", "垂直位置"),
        },
        cartesian_position=[("x", "y")],
        mobject=Circle(radius=0.2, color="RED"),
    )
    animation.add_object(ball, {"x": "0.0", "y": "4.0"})

    # Kinematic mode: closed-form free-fall.
    animation.add_segment(
        PhysicsSegment.from_kinematic(
            segment_id="free_fall",
            objects=["ball"],
            state_vector=["x", "y"],
            expressions={
                "x": "0.0",
                "y": "4.0 - 0.5*9.8*t**2",
            },
            duration=1.0,
        ),
        end_event=time_countdown_event(1.0),
    )
    return animation


def build_scene_03_ode_bounce() -> PhysicsAnimation:
    """Spring bounce using ODE mode + visual binding on velocity."""
    animation = PhysicsAnimation(
        global_parameters={"k": 50.0, "m": 1.0, "g": 9.8},
        engine="scipy",
        sample_dt=1 / 30,
    )

    ball = PhysicObject2D(
        object_id="ball",
        state_variables={
            "x": StateVariable("x", "m", "水平位置"),
            "y": StateVariable("y", "m", "垂直位置"),
            "vy": StateVariable("vy", "m/s", "垂直速度"),
        },
        cartesian_position=[("x", "y")],
        mobject=Circle(radius=0.2, color="BLUE"),
    )
    # Visual binding: opacity changes based on velocity magnitude.
    ball.visual_bindings.append(
        VisualBinding(
            attribute="opacity",
            variables=["vy"],
            expression="0.5 + 0.5*Abs(vy)/(Abs(vy)+5)",
        )
    )
    animation.add_object(ball, {"x": "0.0", "y": "3.0", "vy": "0.0"})

    # ODE mode: spring + gravity.
    animation.add_segment(
        PhysicsSegment(
            segment_id="bounce",
            object_ids=["ball"],
            state_vector=["x", "y", "vy"],
            equations={
                "x": "0.0",
                "y": "vy",
                "vy": "-g - k*y/m",
            },
            state_owners={"x": "ball", "y": "ball", "vy": "ball"},
            duration=5.0,
        ),
        end_event=time_countdown_event(5.0),
    )
    return animation


if __name__ == "__main__":
    scene = PhyAnimationMultiLayerScene2D()
    scene.add_scene("title", build_scene_01_title(), transition="fade")
    scene.add_scene("kinematic_fall", build_scene_02_kinematic_fall(), transition="fade")
    scene.add_scene("ode_bounce", build_scene_03_ode_bounce(), transition="slide")
    scene.render()
