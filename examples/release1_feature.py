from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phyanim.core.segment import PhysicsSegment
from phyanim.core.animation import PhysicsAnimation
from phyanim.core.events import time_countdown_event
from phyanim.render import PhyAnimationScene2D
from phyanim.core.state import StateVariable
from phyanim.core.objects import PhysicObject2D
from phyanim.core.entity.TwoD import Spring
from phyanim.core.events import PhysicsEvent, EventCondition, StateTransition
from phyanim.core.annotation.asserts import ArrowContent, TextContent
from phyanim.core.annotation.annotation import Annotation, AnnotationActivation, Trigger, CrossingTrigger, Transition


from manim import Circle


if __name__ == "__main__":
    # 弹簧劲度系数为 1N/m
    animation = PhysicsAnimation(global_parameters={"k":1, "m1":2, "m2":1})
    # 添加弹簧对象
    spring = PhysicObject2D(
        object_id="spring",
        state_variables={
            "start_x": StateVariable("start_x", "m", "弹簧开始点水平位置"),
            "start_y": StateVariable("start_y", "m", "弹簧开始点垂直位置"),
            "end_x": StateVariable("end_x", "m", "弹簧结束点水平位置"),
            "end_y": StateVariable("end_y", "m", "弹簧结束点垂直位置"),
        },
        cartesian_position=[("start_x", "start_y"), ("end_x", "end_y")],
        mobject=Spring(start=[-1, 0], end=[1, 0], radius=0.1, color="red")
    )
    animation.add_object(spring, {"start_x": "-1.0", "start_y": "0.0", "end_x": "1.0", "end_y": "0.0"})

    # 添加小球1
    ball1 = PhysicObject2D(
        object_id="ball1",
        state_variables={
            "ball1x": StateVariable("ball1x", "m", "小球1水平位置"),
            "ball1y": StateVariable("ball1y", "m", "小球1垂直位置"),
            "ball1vx": StateVariable("ball1vx", "m/s", "小球1水平速度"),
        },
        cartesian_position=[("ball1x", "ball1y")],
        mobject=Circle(radius=0.1, color="blue")
    )
    animation.add_object(ball1, {"ball1x": "-5.0", "ball1y": "0.0", "ball1vx": "1.0"})

    # 添加小球2
    ball2 = PhysicObject2D(
        object_id="ball2",
        state_variables={
            "ball2x": StateVariable("ball2x", "m", "小球2水平位置"),
            "ball2y": StateVariable("ball2y", "m", "小球2垂直位置"),
            "ball2vx": StateVariable("ball2vx", "m/s", "小球2水平速度"),
        },
        cartesian_position=[("ball2x", "ball2y")],
        mobject=Circle(radius=0.1, color="green")
    )
    animation.add_object(ball2, {"ball2x": "1.0", "ball2y": "0.0", "ball2vx": "0.0"})
    
    state_vector=["start_x", "start_y", "end_x", "end_y", "ball1x", "ball1y", "ball2x", "ball2y", "ball1vx", "ball2vx"]
    # equations可能不同段不同
    # 靠近
    state_equations1={
        "start_x": "0.0",
        "start_y": "0.0",
        "end_x": "0.0",
        "end_y": "0.0",
        "ball1y": "0.0",
        "ball2y": "0.0",
        "ball1x": "ball1vx",
        "ball2x": "ball2vx",
        "ball1vx": "0.0",
        "ball2vx": "0.0",
    }
    # 碰撞
    state_equations2={
        "start_x": "ball1vx",
        "start_y": "0.0",
        "end_x": "ball2vx",
        "end_y": "0.0",
        "ball1y": "0.0",
        "ball2y": "0.0",

        "ball1x": "ball1vx",
        "ball2x": "ball2vx",

        "ball1vx": "-k*(2.0-end_x+start_x)/m1",
        "ball2vx": "k*(2.0-end_x+start_x)/m2",
    }
    # 分离
    state_equations3={
        "start_x": "ball1vx",
        "start_y": "0.0",
        "end_x": "ball1vx",
        "end_y": "0.0",
        "ball1y": "0.0",
        "ball2y": "0.0",

        "ball1x": "ball1vx",
        "ball2x": "ball2vx",

        "ball1vx": "0.0",
        "ball2vx": "0.0",
    }
    drived_equations={
        "spring_x": "(start_x + end_x)/2",    
        "spring_y": "-0.5",
        "ball1vy": "0.0",
        "ball2vy": "0.0",
    }
    # 这个必须公用同一套
    state_owners={
        "start_x": "spring",
        "start_y": "spring",
        "end_x": "spring",
        "end_y": "spring",
        "ball1x": "ball1",
        "ball2x": "ball2",
        "ball1vx": "ball1",
        "ball2vx": "ball2",
        "ball1y": "ball1",
        "ball2y": "ball2",
    }
    event1 = PhysicsEvent("event1", condition=EventCondition(expression="ball1x - start_x", terminal=True, direction=1), transition=StateTransition(name="collision_impulse", equations={}))
    event2 = PhysicsEvent("event2", condition=EventCondition(expression="end_x - start_x - 2", terminal=True, direction=1), transition=StateTransition(name="collision_impulse", equations={}))
    event3 = time_countdown_event(2)

    animation.add_segment(
        PhysicsSegment(
            segment_id="seg1",
            object_ids=["spring", "ball1", "ball2"],
            state_vector=state_vector,
            equations=state_equations1,
            derived_equations=drived_equations,
            state_owners=state_owners,
            duration=100,
        ),
        end_event=event1,
    )
    animation.add_segment(
        PhysicsSegment(
            segment_id="seg2",
            object_ids=["spring", "ball1", "ball2"],
            state_vector=state_vector,
            equations=state_equations2,
            derived_equations=drived_equations,
            state_owners=state_owners,
            duration=100,
        ),
        end_event=event2,
    )
    animation.add_segment(
        PhysicsSegment(
            segment_id="seg3",
            object_ids=["spring", "ball1", "ball2"],
            state_vector=state_vector,
            equations=state_equations3,
            derived_equations=drived_equations,
            state_owners=state_owners,
            duration=100,
        ),
        end_event=event3,
    )

    animation.add_annotation(
        Annotation(
            id="ball1_v_arrow",
            ann_type="while",
            content=ArrowContent(pos_variables=("ball1x", "ball1y"), shift_variables=("ball1vx", "ball1vy"), scale=0.5, tip_length=0.1),
            activation=AnnotationActivation(trigger=Trigger(expression="t > 0")),
        )
    )
    animation.add_annotation(
        Annotation(
            id="ball1_v_label",
            ann_type="while",
            content=TextContent(txt="ball1_v", pos_variables=("ball1x", "0.5")),
            activation=AnnotationActivation(trigger=Trigger(expression="t > 0")),
        )
    )
    animation.add_annotation(
        Annotation(
            id="ball2_v_arrow",
            ann_type="while",
            content=ArrowContent(pos_variables=("ball2x", "ball2y"), shift_variables=("ball2vx", "ball2vy"), scale=0.5, tip_length=0.1),
            activation=AnnotationActivation(trigger=Trigger(expression="t > 0")),
        )
    )
    animation.add_annotation(
        Annotation(
            id="ball2_v_label",
            ann_type="while",
            content=TextContent(txt="ball2_v", pos_variables=("ball2x", "0.5")),
            activation=AnnotationActivation(trigger=Trigger(expression="t > 0")),
        )
    )

    animation.add_annotation(
        Annotation(
            id="spring_compress",
            ann_type="between",
            content=TextContent(txt="弹簧被压缩", pos_variables=("spring_x", "spring_y")),
            activation=AnnotationActivation(start_trigger=CrossingTrigger(expression="ball1x - start_x", direction=1), end_trigger=CrossingTrigger(expression="ball2x - ball1x - 2", direction=1)),
        )
    )
    animation.add_annotation(
        Annotation(
            id="time_range1",
            ann_type="time_range",
            content=TextContent(txt="马上碰撞", pos_variables=("start_x", "1")),
            activation=AnnotationActivation(trigger=CrossingTrigger(expression="ball1x - start_x", direction=1), advance=2, delay=0),
        )
    )
    animation.add_annotation(
        Annotation(
            id="time_range2",
            ann_type="time_range",
            content=TextContent(txt="马上分离", pos_variables=("ball2x", "2")),
            activation=AnnotationActivation(trigger=CrossingTrigger(expression="ball2x - ball1x - 2", direction=1), advance=1, delay=0),
        )
    )
    animation.add_annotation(
        Annotation(
            id="time_range3",
            ann_type="time_range",
            content=TextContent(txt="碰撞完成", pos_variables=("ball2x", "2")),
            activation=AnnotationActivation(trigger=CrossingTrigger(expression="ball2x - ball1x - 2", direction=1), advance=0, delay=2),
        )
    )
    """
        "fade"        — 纯淡入淡出
        "scale"       — 缩成一点再扩散（推荐）
        "slide_left"  — 左滑出 / 右滑入
        "slide_down"  — 下滑出 / 上滑入
        "spin"        — 旋转缩放出 / 旋转放大入
    """
    animation.add_transition(
        Transition(
            id="formula_derivation",
            group_strings=[["m_{1}v_{1}+m_{2}v_{2}=m_{1}v_{1}'+m_{2}v_{2}'", "\\frac{1}{2}m_{1}v_{1}^2+\\frac{1}{2}m_{2}v_{2}^2=\\frac{1}{2}m_{1}{v_{1}'}^{2}+\\frac{1}{2}m_{2}{v_{2}'}^{2}", "m_{1}=2kg, m_{2}=1kg", "v_{1}=1m/s, v_{2}=0"],["v_{1}'=\\frac{1}{3}m/s", "v_{2}'=\\frac{4}{3}m/s"]], 
            triggers=[CrossingTrigger(expression="t-2", direction=1), CrossingTrigger(expression="t-7", direction=1), CrossingTrigger(expression="t-10", direction=1)],
            pos_variables=("0", "-2"),
            group_dir={},
            style="spin",
            duration=0.5,
        )
    )

    scene = PhyAnimationScene2D()
    # scene.set_frame_size(width=1000, height=1000)
    scene.set_animation(animation)
    scene.render()