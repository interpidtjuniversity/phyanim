from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manim import BLUE, GREEN, WHITE, YELLOW

from phyanim import PhysicsAnimation, PhysicsEvent, PointParticle, StateTransition, object2d
from phyanim.core.entity.TwoD import RightSemicircleTrack, StraightTrack
from phyanim.core.enhance.annotation import Annotation, AnnotationActivation, ArrowContent, TextContent
from phyanim.core.enhance.transition import Transition
from phyanim.core.enhance.trigger import CrossingTrigger, Trigger
from phyanim.render import PhyAnimationMultiLayerScene2D
from phyanim.core.enhance.timewrapper import TimeWrapper

def build_animation() -> PhysicsAnimation:
    """A ball slides down an incline, crosses rough flat track, loops, and lands."""

    g = 9.8
    radius = 0.8
    theta = math.radians(30.0)
    incline_mu = 0.10
    horizontal_mu = 0.05
    horizontal_length = 2.0

    # Just-pass condition at the top of a smooth vertical circle:
    # v_top^2 / R = g, so v_bottom^2 = v_top^2 + 4gR = 5gR.
    loop_entry_speed_squared = 5.0 * g * radius
    horizontal_loss = 2.0 * horizontal_mu * g * horizontal_length
    incline_accel = g * (math.sin(theta) - incline_mu * math.cos(theta))
    incline_length = (loop_entry_speed_squared + horizontal_loss) / (2.0 * incline_accel)

    incline_start_x = -incline_length * math.cos(theta)
    incline_start_y = incline_length * math.sin(theta)
    horizontal_start_x = 0.0
    horizontal_end_x = horizontal_length
    track_y = 0.0
    loop_center_x = horizontal_end_x
    loop_center_y = radius

    ball = PointParticle(
        "ball",
        mass=1.0,
        radius=0.08,
        color=YELLOW,
        state_names={
            "x": "ball_x",
            "y": "ball_y",
            "vx": "ball_vx",
            "vy": "ball_vy",
            "phi": "ball_phi",
            "omega": "ball_omega",
        },
        cartesian_position=("ball_x", "ball_y"),
    )

    incline_track = object2d(
        "incline_track",
        states={
            "incline_start_x": "incline start x",
            "incline_start_y": "incline start y",
            "incline_end_x": "incline end x",
            "incline_end_y": "incline end y",
        },
        cartesian_position=[
            ("incline_start_x", "incline_start_y"),
            ("incline_end_x", "incline_end_y"),
        ],
        mobject=StraightTrack(
            start=[incline_start_x, incline_start_y, 0.0],
            end=[horizontal_start_x, track_y, 0.0],
            color=WHITE,
            stroke_width=5,
        ),
    )
    horizontal_track = object2d(
        "horizontal_track",
        states={
            "horizontal_start_x": "horizontal start x",
            "horizontal_start_y": "horizontal start y",
            "horizontal_end_x": "horizontal end x",
            "horizontal_end_y": "horizontal end y",
        },
        cartesian_position=[
            ("horizontal_start_x", "horizontal_start_y"),
            ("horizontal_end_x", "horizontal_end_y"),
        ],
        mobject=StraightTrack(
            start=[horizontal_start_x, track_y, 0.0],
            end=[horizontal_end_x, track_y, 0.0],
            color=GREEN,
            stroke_width=5,
        ),
    )
    loop_track = object2d(
        "loop_track",
        states={
            "loop_center_x": "loop center x",
            "loop_center_y": "loop center y",
        },
        cartesian_position=("loop_center_x", "loop_center_y"),
        mobject=RightSemicircleTrack(radius=radius, color=BLUE, stroke_width=5),
    )

    animation = PhysicsAnimation(
        sample_dt=1 / 60,
        global_parameters={
            "g": g,
            "R": radius,
            "theta": theta,
            "mu_incline": incline_mu,
            "mu_horizontal": horizontal_mu,
            "track_end_x": horizontal_end_x,
        },
    )
    animation.add_object(
        incline_track,
        incline_track.initial_state(
            incline_start_x=incline_start_x,
            incline_start_y=incline_start_y,
            incline_end_x=horizontal_start_x,
            incline_end_y=track_y,
        ),
    )
    animation.add_object(
        horizontal_track,
        horizontal_track.initial_state(
            horizontal_start_x=horizontal_start_x,
            horizontal_start_y=track_y,
            horizontal_end_x=horizontal_end_x,
            horizontal_end_y=track_y,
        ),
    )
    animation.add_object(
        loop_track,
        loop_track.initial_state(loop_center_x=loop_center_x, loop_center_y=loop_center_y),
    )
    animation.add_object(
        ball,
        ball.initial_state(
            ball_x=incline_start_x,
            ball_y=incline_start_y,
            ball_vx=0.0,
            ball_vy=0.0,
            ball_phi=0.0,
            ball_omega=0.0,
        ),
    )

    ball_states = ["ball_x", "ball_y", "ball_vx", "ball_vy", "ball_phi", "ball_omega"]
    base_owners = {name: "ball" for name in ball_states}

    derived = {
        "incline_formula_x": str(incline_start_x / 2),
        "incline_formula_y": str(incline_start_y / 2 + 1.2),
        "horizontal_formula_x": str(horizontal_length / 2),
        "horizontal_formula_y": "1.0",
        "loop_top_text_x": str(loop_center_x),
        "loop_top_text_y": str(2 * radius + 0.5),
        "track_end_x_derived": str(horizontal_end_x),
    }

    animation.add_equation_segment(
        "slide_down_incline",
        objects=["ball"],
        equations={
            "ball_x": "ball_vx",
            "ball_y": "ball_vy",
            "ball_vx": "g*(sin(theta)-mu_incline*cos(theta))*cos(theta)",
            "ball_vy": "-g*(sin(theta)-mu_incline*cos(theta))*sin(theta)",
            "ball_phi": "0.0",
            "ball_omega": "0.0",
        },
        owners=base_owners,
        derived=derived,
        duration=20,
        end_event=PhysicsEvent.terminal(
            "reaches_horizontal_track",
            "ball_x",
            direction=1,
            transition=StateTransition.from_equations(
                "align_velocity_to_horizontal",
                {
                    "ball_x": "0.0",
                    "ball_y": "0.0",
                    "ball_vx": "sqrt(ball_vx**2 + ball_vy**2)",
                    "ball_vy": "0.0",
                },
            ),
        ),
    )

    animation.add_equation_segment(
        "rough_horizontal_track",
        objects=["ball"],
        equations={
            "ball_x": "ball_vx",
            "ball_y": "0.0",
            "ball_vx": "-mu_horizontal*g",
            "ball_vy": "0.0",
            "ball_phi": "0.0",
            "ball_omega": "0.0",
        },
        owners=base_owners,
        derived=derived,
        duration=20,
        end_event=PhysicsEvent.terminal(
            "enters_smooth_loop",
            "ball_x - track_end_x",
            direction=1,
            transition=StateTransition.from_equations(
                "convert_to_loop_coordinates",
                {
                    "ball_x": "track_end_x",
                    "ball_y": "0.0",
                    "ball_vy": "0.0",
                    "ball_phi": "0.0",
                    "ball_omega": "ball_vx/R",
                },
            ),
        ),
    )

    animation.add_equation_segment(
        "smooth_vertical_loop",
        objects=["ball"],
        equations={
            "ball_x": "R*cos(ball_phi)*ball_omega",
            "ball_y": "R*sin(ball_phi)*ball_omega",
            "ball_vx": "-R*sin(ball_phi)*ball_omega**2 - g*cos(ball_phi)*sin(ball_phi)",
            "ball_vy": "R*cos(ball_phi)*ball_omega**2 - g*sin(ball_phi)**2",
            "ball_phi": "ball_omega",
            "ball_omega": "-g*sin(ball_phi)/R",
        },
        owners=base_owners,
        derived=derived,
        duration=20,
        end_event=PhysicsEvent.terminal(
            "leaves_loop_top",
            f"ball_phi - {math.pi}",
            direction=1,
            transition=StateTransition.from_equations(
                "launch_horizontally_from_top",
                {
                    "ball_x": "track_end_x",
                    "ball_y": "2*R",
                    "ball_vx": "-R*ball_omega",
                    "ball_vy": "0.0",
                },
            ),
        ),
    )

    animation.add_equation_segment(
        "projectile_lands_on_horizontal_track",
        objects=["ball"],
        equations={
            "ball_x": "ball_vx",
            "ball_y": "ball_vy",
            "ball_vx": "0.0",
            "ball_vy": "-g",
            "ball_phi": "0.0",
            "ball_omega": "0.0",
        },
        owners=base_owners,
        derived=derived,
        duration=20,
        end_event=PhysicsEvent.terminal(
            "lands_on_horizontal_track",
            "ball_y",
            direction=-1,
            transition=StateTransition.from_equations(
                "settle_on_horizontal_track",
                {
                    "ball_y": "0.0",
                    "ball_vx": "0.0",
                    "ball_vy": "0.0",
                },
            ),
        ),
    )

    animation.add_equation_segment(
        "hold_landing_frame",
        objects=["ball"],
        equations={
            "ball_x": "0.0",
            "ball_y": "0.0",
            "ball_vx": "0.0",
            "ball_vy": "0.0",
            "ball_phi": "0.0",
            "ball_omega": "0.0",
        },
        owners=base_owners,
        derived=derived,
        duration=1.0,
        end_event=PhysicsEvent.terminal(
            "finish_after_landing_hold",
            "t - t_start - 0.6",
            direction=1,
        ),
    )

    # === Annotations and Transitions ===
    physics_layer = animation.get_physics_layer()

    # Velocity arrow (always visible while ball is moving)
    physics_layer.add_annotation(
        Annotation(
            id="velocity_arrow",
            ann_type="while",
            content=ArrowContent(
                pos_variables=("ball_x", "ball_y"),
                shift_variables=("ball_vx", "ball_vy"),
                scale=0.15,
                color="yellow",
                tip_length=0.12,
            ),
            activation=AnnotationActivation(
                trigger=Trigger(expression="ball_vx**2 + ball_vy**2 > 0.01"),
            ),
        )
    )

    # Incline formula transition: mgsinθ - μmgcosθ = ma
    physics_layer.add_transition(
        Transition(
            id="incline_formula",
            group_strings=[
                ["\\text{牛顿第二定律:} mg\\sin\\theta - \\mu mg\\cos\\theta = ma"],
            ],
            triggers=[
                CrossingTrigger(expression="ball_vx**2 + ball_vy**2 - 0.01", direction=1),
                CrossingTrigger(expression="ball_x", direction=1),
            ],
            pos_variables=("incline_formula_x", "incline_formula_y"),
            group_dir={},
            font_size=30,
            style="scale",
            duration=0.1,
        )
    )

    # Horizontal track formula transition: -μmg = ma
    physics_layer.add_transition(
        Transition(
            id="horizontal_formula",
            group_strings=[
                ["-\\mu mg = ma"],
            ],
            triggers=[
                CrossingTrigger(expression="ball_x - 0.01", direction=1),
                CrossingTrigger(expression="ball_x - track_end_x_derived", direction=1),
            ],
            pos_variables=("horizontal_formula_x", "horizontal_formula_y"),
            group_dir={},
            font_size=30,
            style="scale",
            duration=0.1,
        )
    )

    # Loop top text: v = sqrt(gr)
    physics_layer.add_annotation(
        Annotation(
            id="loop_top_text",
            ann_type="while",
            content=TextContent(
                txt="v=sqrt(gr)",
                pos_variables=("loop_top_text_x", "loop_top_text_y"),
                font_size=28,
            ),
            activation=AnnotationActivation(
                trigger=Trigger(expression="ball_phi > 2.5"),
            ),
        )
    )

    return animation


if __name__ == "__main__":
    scene = PhyAnimationMultiLayerScene2D()
    scene.set_animation(build_animation())
    scene.render()
