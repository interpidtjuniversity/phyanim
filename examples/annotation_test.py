from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phyanim.core.segment import PhysicsSegment
from phyanim.core.animation import PhysicsAnimation
from phyanim.core.events import time_countdown_event
from phyanim.render import PhyAnimationMultiLayerScene2D
from phyanim.core.state import StateVariable
from phyanim.core.objects import PhysicObject2D
from phyanim.core.entity.TwoD import Spring

from phyanim.core.enhance.annotation import Annotation, AnnotationActivation
from phyanim.core.enhance.trigger import CrossingTrigger, Trigger
from phyanim.core.enhance.transition import Transition
from phyanim.core.enhance.annotation import ArrowContent, TextContent

if __name__ == "__main__":
    # 弹簧劲度系数为 1N/m
    animation = PhysicsAnimation(global_parameters={"k":1})
    spring = PhysicObject2D(
        object_id="spring",
        state_variables={
            "start_x": StateVariable("start_x", "m", "弹簧开始点水平位置"),
            "start_y": StateVariable("start_y", "m", "弹簧开始点垂直位置"),
            "end_x": StateVariable("end_x", "m", "弹簧结束点水平位置"),
            "end_y": StateVariable("end_y", "m", "弹簧结束点垂直位置"),
        },
        cartesian_position=[("start_x", "start_y"), ("end_x", "end_y")],
        mobject=Spring(start=[-2, 0], end=[2, 0], radius=0.3, color="red", coils=20)
    )
    animation.add_object(spring, {"start_x": "-2.0", "start_y": "0.0", "end_x": "2.0", "end_y": "0.0"})
    
    state_vector=["start_x", "start_y", "end_x", "end_y"]
    # equations可能不同段不同
    state_equations1={
        "start_x": "-0.5",
        "start_y": "0.0",
        "end_x": "0.0",
        "end_y": "0.0",
    }
    state_equations2={
        "start_x": "0.0",
        "start_y": "0.0",
        "end_x": "0.5",
        "end_y": "0.0",
    }
    drived_equations={
    }
    # 这个必须公用同一套
    state_owners={
        "start_x": "spring",
        "start_y": "spring",
        "end_x": "spring",
        "end_y": "spring",
    }
    animation.add_segment(
        PhysicsSegment(
            segment_id="compress1",
            object_ids=["spring"],
            state_vector=state_vector,
            equations=state_equations1,
            derived_equations=drived_equations,
            state_owners=state_owners,
            duration=100,
        ),
        end_event=time_countdown_event(5),
    )
    animation.add_segment(
        PhysicsSegment(
            segment_id="compress2",
            object_ids=["spring"],
            state_vector=state_vector,
            equations=state_equations2,
            derived_equations=drived_equations,
            state_owners=state_owners,
            duration=100,
        ),
        end_event=time_countdown_event(5),
    )
    main_layer = animation.create_layer(layer_id="id")

    main_layer.add_annotation(
        Annotation(
            id="anno1",
            ann_type="while",
            content=TextContent(txt="while文本", pos_variables=("start_x", "start_y")),
            activation=AnnotationActivation(trigger=Trigger(expression="t > 2")),
        )
    )
    main_layer.add_annotation(
        Annotation(
            id="anno2",
            ann_type="between",
            content=TextContent(txt="between文本", pos_variables=("end_x", "end_y")),
            activation=AnnotationActivation(start_trigger=CrossingTrigger(expression="t-3", direction=1), end_trigger=CrossingTrigger(expression="t-6", direction=1)),
        )
    )
    main_layer.add_annotation(
        Annotation(
            id="anno3",
            ann_type="time_range",
            content=TextContent(txt="time_range文本", pos_variables=("0", "2")),
            activation=AnnotationActivation(trigger=CrossingTrigger(expression="t - 5", direction=1), advance=1, delay=1),
        )
    )
    main_layer.add_annotation(
        Annotation(
            id="vector_arrow",
            ann_type="while",
            content=ArrowContent(pos_variables=("end_x", "end_y"), shift_variables=("1", "1"), scale=0.5, tip_length=0.1),
            activation=AnnotationActivation(trigger=Trigger(expression="t > 0")),
        )
    )
    """
        "fade"        — 纯淡入淡出
        "scale"       — 缩成一点再扩散（推荐）
        "slide_left"  — 左滑出 / 右滑入
        "slide_down"  — 下滑出 / 上滑入
        "spin"        — 旋转缩放出 / 旋转放大入
    """
    main_layer.add_transition(
        Transition(
            id="formula_derivation",
            group_strings=[["a=1","b=2","c=3","d=4"],["a+b=3", "c+d=7", "a+b+c+d=10"]], 
            triggers=[CrossingTrigger(expression="t-3", direction=1), CrossingTrigger(expression="t-6", direction=1), CrossingTrigger(expression="t-9", direction=1)],
            pos_variables=("end_x", "end_y"),
            group_dir={},
            style="spin",
            duration=0.5,
        )
    )
    
    scene = PhyAnimationMultiLayerScene2D()
    # scene.set_frame_size(width=1000, height=1000)
    scene.set_animation(animation)
    scene.render()