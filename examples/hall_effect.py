from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manim import Circle, Text, VGroup, WHITE, BLUE, RED

from phyanim.core.animation import PhysicsAnimation
from phyanim.core.entity.TwoD import StraightTrack
from phyanim.core.enhance.annotation import Annotation, AnnotationActivation, ArrowContent, TextContent
from phyanim.core.enhance.timewrapper import TimeWrapper
from phyanim.core.enhance.transition import Transition
from phyanim.core.enhance.trigger import CrossingTrigger, Trigger
from phyanim.core.events import EventCondition, PhysicsEvent, StateTransition, time_countdown_event
from phyanim.core.objects import PhysicObject2D, PointParticle
from phyanim.core.segment import PhysicsSegment
from phyanim.core.state import StateVariable
from phyanim.render import PhyAnimationMultiLayerScene2D


if __name__ == "__main__":
    animation = PhysicsAnimation(global_parameters={"B": 0.6, "E0": 0.03})

    conductor_top = PhysicObject2D(
        object_id="conductor_top",
        state_variables={
            "top_start_x": StateVariable("top_start_x", "m", "导体上表面起点x"),
            "top_start_y": StateVariable("top_start_y", "m", "导体上表面起点y"),
            "top_end_x": StateVariable("top_end_x", "m", "导体上表面终点x"),
            "top_end_y": StateVariable("top_end_y", "m", "导体上表面终点y"),
        },
        cartesian_position=[("top_start_x", "top_start_y"), ("top_end_x", "top_end_y")],
        mobject=StraightTrack(start=[-2.25, 2.0, 0.0], end=[2.25, 2.0, 0.0], color=WHITE, stroke_width=5),
    )
    conductor_bottom = PhysicObject2D(
        object_id="conductor_bottom",
        state_variables={
            "bottom_start_x": StateVariable("bottom_start_x", "m", "导体下表面起点x"),
            "bottom_start_y": StateVariable("bottom_start_y", "m", "导体下表面起点y"),
            "bottom_end_x": StateVariable("bottom_end_x", "m", "导体下表面终点x"),
            "bottom_end_y": StateVariable("bottom_end_y", "m", "导体下表面终点y"),
        },
        cartesian_position=[("bottom_start_x", "bottom_start_y"), ("bottom_end_x", "bottom_end_y")],
        mobject=StraightTrack(start=[-2.25, -2.0, 0.0], end=[2.25, -2.0, 0.0], color=WHITE, stroke_width=5),
    )
    animation.add_object(conductor_top, {"top_start_x": -2.25, "top_start_y": 2.0, "top_end_x": 2.25, "top_end_y": 2.0})
    animation.add_object(conductor_bottom, {"bottom_start_x": -2.25, "bottom_start_y": -2.0, "bottom_end_x": 2.25, "bottom_end_y": -2.0})

    object_ids = ["conductor_top", "conductor_bottom"]
    electrons = []
    for i in range(5):
        electron = PointParticle(
            f"electron{i + 1}",
            mass=1.0,
            charge=-1.0,
            radius=0.09,
            color=BLUE,
            parameter_names={"mass": f"me{i + 1}", "charge": f"qe{i + 1}"},
            state_names={"x": f"ex{i + 1}", "y": f"ey{i + 1}", "vx": f"evx{i + 1}", "vy": f"evy{i + 1}"},
            cartesian_position=(f"ex{i + 1}", f"ey{i + 1}"),
        )
        electrons.append(electron)
        object_ids.append(electron.object_id)
        animation.add_object(electron, {f"ex{i + 1}": -2.25, f"ey{i + 1}": 0.0, f"evx{i + 1}": 1.1 if i == 0 else 0.0, f"evy{i + 1}": 0.0})

    positives = []
    for i in range(4):
        positive = PhysicObject2D(
            object_id=f"positive{i + 1}",
            state_variables={
                f"px{i + 1}": StateVariable(f"px{i + 1}", "m", "正电荷x"),
                f"py{i + 1}": StateVariable(f"py{i + 1}", "m", "正电荷y"),
            },
            cartesian_position=[(f"px{i + 1}", f"py{i + 1}")],
            mobject=VGroup(Text("+", color=RED, font_size=36), Circle(radius=0.12, color=RED)),
        )
        positives.append(positive)
        object_ids.append(positive.object_id)
        animation.add_object(positive, {f"px{i + 1}": -2.25, f"py{i + 1}": 4.0})

    state_vector = [
        "top_start_x", "top_start_y", "top_end_x", "top_end_y",
        "bottom_start_x", "bottom_start_y", "bottom_end_x", "bottom_end_y",
    ]
    state_owners = {
        "top_start_x": "conductor_top",
        "top_start_y": "conductor_top",
        "top_end_x": "conductor_top",
        "top_end_y": "conductor_top",
        "bottom_start_x": "conductor_bottom",
        "bottom_start_y": "conductor_bottom",
        "bottom_end_x": "conductor_bottom",
        "bottom_end_y": "conductor_bottom",
    }
    equations_base = {
        "top_start_x": "0.0",
        "top_start_y": "0.0",
        "top_end_x": "0.0",
        "top_end_y": "0.0",
        "bottom_start_x": "0.0",
        "bottom_start_y": "0.0",
        "bottom_end_x": "0.0",
        "bottom_end_y": "0.0",
    }

    for i in range(5):
        n = i + 1
        state_vector.extend([f"ex{n}", f"ey{n}", f"evx{n}", f"evy{n}"])
        state_owners.update({
            f"ex{n}": f"electron{n}",
            f"ey{n}": f"electron{n}",
            f"evx{n}": f"electron{n}",
            f"evy{n}": f"electron{n}",
        })
        equations_base.update({
            f"ex{n}": f"evx{n}",
            f"ey{n}": f"evy{n}",
            f"evx{n}": "0.0",
            f"evy{n}": "0.0",
        })
    for i in range(4):
        n = i + 1
        state_vector.extend([f"px{n}", f"py{n}"])
        state_owners.update({
            f"px{n}": f"positive{n}",
            f"py{n}": f"positive{n}",
        })
        equations_base.update({
            f"px{n}": "0.0",
            f"py{n}": "0.0",
        })

    derived_equations = {
        "title_x": "0.0", "title_y": "3.2",
        "analysis_x": "-1.9", "analysis_y": "-0.05",
        "v_dx": "1.0", "v_dy": "0.0",
        "i_dx": "-1.0", "i_dy": "0.0",
        "f_dx": "0.0", "f_dy": "-1.0",
        "v_label_x": "-1.5", "v_label_y": "0.2",
        "i_label_x": "-2.5", "i_label_y": "0.2",
        "f_label_x": "-2.1", "f_label_y": "-0.5",
        "hall_formula_x": "3.7", "hall_formula_y": "1.2",
        "fe_dx": "0.0", "fe_dy": "1.0",
        "fl_dx": "0.0", "fl_dy": "-1.0",
        "fe_label_x": "ex5 - 0.4", "fe_label_y": "ey5 + 0.9",
        "fl_label_x": "ex5 + 0.45", "fl_label_y": "ey5 - 0.9",
    }
    for row, y in enumerate([1.2, 0.4, -0.4, -1.2], start=1):
        for col, x in enumerate([-1.8, -0.9, 0.0, 0.9, 1.8], start=1):
            derived_equations[f"b_label_x{row}{col}"] = str(x)
            derived_equations[f"b_label_y{row}{col}"] = str(y)
    for i, x in enumerate([-1.0, 1.0, 0.0, -0.5, 0.5, -0.75, -0.25, 0.25, 0.75], start=1):
        derived_equations[f"hall_e_x{i}"] = str(x)
        derived_equations[f"hall_e_y{i}"] = "1.35"
        derived_equations[f"hall_e_dx{i}"] = "0.0"
        derived_equations[f"hall_e_dy{i}"] = "-1.0"

    for i in range(4):
        n = i + 1
        equations = dict(equations_base)
        equations[f"evx{n}"] = f"-B*evy{n}*qe{n}/me{n}"
        equations[f"evy{n}"] = f"B*evx{n}*qe{n}/me{n} - E*qe{n}/me{n}"
        transition_equations = {f"evx{n}": "0.0", f"evy{n}": "0.0", f"px{n}": f"ex{n}", f"py{n}": "2.0"}
        if n < 4:
            transition_equations[f"evx{n + 1}"] = "1.1"
            transition_equations[f"evy{n + 1}"] = "0.0"
        else:
            transition_equations["evx5"] = "4*E0/B"
            transition_equations["evy5"] = "0.0"
        event = PhysicsEvent(
            f"electron{n}_hits_bottom",
            condition=EventCondition(expression=f"ey{n} + 1.85", terminal=True, direction=-1),
            transition=StateTransition(name=f"electron{n}_stops_and_updates_hall_field", equations=transition_equations),
        )
        animation.add_segment(
            PhysicsSegment(
                segment_id=f"electron{n}_deflecting",
                object_ids=object_ids,
                state_vector=state_vector,
                equations=equations,
                parameters={"E": i * 0.03},
                derived_equations=derived_equations,
                state_owners=state_owners,
                duration=40,
            ),
            end_event=event,
        )

    balance_equations = dict(equations_base)
    balance_equations["evx5"] = "0.0"
    balance_equations["evy5"] = "B*evx5*qe5/me5 - E*qe5/me5"
    animation.add_segment(
        PhysicsSegment(
            segment_id="electron5_force_balance",
            object_ids=object_ids,
            state_vector=state_vector,
            equations=balance_equations,
            parameters={"E": 4 * 0.03},
            derived_equations=derived_equations,
            state_owners=state_owners,
            duration=100,
        ),
        end_event=time_countdown_event(15, name="force_balance_display"),
    )

    physics_layer = animation.get_physics_layer()
    physics_layer.add_annotation(
        Annotation(
            id="title",
            ann_type="while",
            content=TextContent(txt="Hall effect: accumulated charges create Hall field", pos_variables=("title_x", "title_y"), font_size=28),
            activation=AnnotationActivation(trigger=Trigger(expression="1")),
        )
    )
    for row in range(1, 5):
        for col in range(1, 6):
            physics_layer.add_annotation(
                Annotation(
                    id=f"magnetic_field_x_{row}_{col}",
                    ann_type="while",
                    content=TextContent(txt="×", pos_variables=(f"b_label_x{row}{col}", f"b_label_y{row}{col}"), font_size=38),
                    activation=AnnotationActivation(trigger=Trigger(expression="1")),
                )
            )
    for i in range(4):
        n = i + 1
        physics_layer.add_annotation(
            Annotation(
                id=f"positive_charge_label_{n}",
                ann_type="while",
                content=TextContent(txt="+", pos_variables=(f"px{n}", f"py{n}"), font_size=36),
                activation=AnnotationActivation(trigger=Trigger(expression=f"py{n} < 2.5")),
            )
        )
    hall_arrow_thresholds = [1, 1, 2, 3, 3, 4, 4, 4, 4]
    for i, threshold in enumerate(hall_arrow_thresholds, start=1):
        physics_layer.add_annotation(
            Annotation(
                id=f"hall_e_arrow_{i}",
                ann_type="while",
                content=ArrowContent(pos_variables=(f"hall_e_x{i}", f"hall_e_y{i}"), shift_variables=(f"hall_e_dx{i}", f"hall_e_dy{i}"), scale=2.6, color="yellow", tip_length=0.12),
                activation=AnnotationActivation(trigger=Trigger(expression=f"py{threshold} < 2.5")),
            )
        )
    physics_layer.add_annotation(
        Annotation(
            id="electric_force_arrow",
            ann_type="while",
            content=ArrowContent(pos_variables=("ex5", "ey5"), shift_variables=("fe_dx", "fe_dy"), scale=0.75, color="red", tip_length=0.12),
            activation=AnnotationActivation(trigger=Trigger(expression="evx5 > 0")),
        )
    )
    physics_layer.add_annotation(
        Annotation(
            id="electric_force_label",
            ann_type="while",
            content=TextContent(txt="Fe", pos_variables=("fe_label_x", "fe_label_y"), font_size=24),
            activation=AnnotationActivation(trigger=Trigger(expression="evx5 > 0")),
        )
    )
    physics_layer.add_annotation(
        Annotation(
            id="lorentz_force_arrow",
            ann_type="while",
            content=ArrowContent(pos_variables=("ex5", "ey5"), shift_variables=("fl_dx", "fl_dy"), scale=0.75, color="blue", tip_length=0.12),
            activation=AnnotationActivation(trigger=Trigger(expression="evx5 > 0")),
        )
    )
    physics_layer.add_annotation(
        Annotation(
            id="lorentz_force_label",
            ann_type="while",
            content=TextContent(txt="FL", pos_variables=("fl_label_x", "fl_label_y"), font_size=24),
            activation=AnnotationActivation(trigger=Trigger(expression="evx5 > 0")),
        )
    )
    physics_layer.add_transition(
        Transition(
            id="hall_field_formula",
            group_strings=[
                ["Bvq=Eq"],
                ["E=Bv"],
                ["U_H=Ed"],
            ],
            triggers=[
                CrossingTrigger(expression="evx5 - 0.1", direction=1),
                CrossingTrigger(expression="ex5 + 1.7", direction=1),
                CrossingTrigger(expression="ex5 - 0.1", direction=1),
                CrossingTrigger(expression="ex5 - 2.0", direction=1),
            ],
            pos_variables=("hall_formula_x", "hall_formula_y"),
            group_dir={},
            font_size=34,
            style="scale",
            duration=0.4,
        )
    )

    render_layer = animation.get_render_layer()
    render_layer.add_sub_animation(
        time_wrapper=TimeWrapper(
            id="first_electron_force_analysis",
            type="freeze",
            trigger=CrossingTrigger(expression="ex1 + 1.9", direction=1),
            extend_to=5,
        ),
        annotations=[
            Annotation(
                id="analysis_v_arrow",
                ann_type="while",
                content=ArrowContent(pos_variables=("analysis_x", "analysis_y"), shift_variables=("v_dx", "v_dy"), scale=1.0, color="blue", tip_length=0.12),
                activation=AnnotationActivation(trigger=Trigger(expression="local_t > 1")),
            ),
            Annotation(
                id="analysis_v_label",
                ann_type="while",
                content=TextContent(txt="v", pos_variables=("v_label_x", "v_label_y"), font_size=26),
                activation=AnnotationActivation(trigger=Trigger(expression="local_t > 1")),
            ),
            Annotation(
                id="analysis_i_arrow",
                ann_type="while",
                content=ArrowContent(pos_variables=("analysis_x", "analysis_y"), shift_variables=("i_dx", "i_dy"), scale=1.0, color="yellow", tip_length=0.12),
                activation=AnnotationActivation(trigger=Trigger(expression="local_t > 2")),
            ),
            Annotation(
                id="analysis_i_label",
                ann_type="while",
                content=TextContent(txt="I", pos_variables=("i_label_x", "i_label_y"), font_size=26),
                activation=AnnotationActivation(trigger=Trigger(expression="local_t > 2")),
            ),
            Annotation(
                id="analysis_f_arrow",
                ann_type="while",
                content=ArrowContent(pos_variables=("analysis_x", "analysis_y"), shift_variables=("f_dx", "f_dy"), scale=1.0, color="red", tip_length=0.12),
                activation=AnnotationActivation(trigger=Trigger(expression="local_t > 3")),
            ),
            Annotation(
                id="analysis_f_label",
                ann_type="while",
                content=TextContent(txt="F", pos_variables=("f_label_x", "f_label_y"), font_size=26),
                activation=AnnotationActivation(trigger=Trigger(expression="local_t > 3")),
            ),
        ],
    )

    scene = PhyAnimationMultiLayerScene2D()
    scene.set_animation(animation)
    scene.render()
