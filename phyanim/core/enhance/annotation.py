from typing import Any
from phyanim.core.enhance.trigger import Trigger, ActiveRule, ActiveWhile, ActivateEventTimeRange, ActiveBetween
from phyanim.core.enhance.asserts import Content

from manim import Arrow, Text, MathTex
import numpy as np

class AnnotationActivation:
    """注释激活规则，定义了注释在什么时间激活。"""
    def __init__(self, trigger: Trigger | None = None, start_trigger: Trigger | None = None, end_trigger: Trigger | None = None, advance: float | None = None, delay: float | None = None):
        self.trigger = trigger
        self.start_trigger = start_trigger
        self.end_trigger = end_trigger
        self.advance = advance
        self.delay = delay
    
    # build的实质是赋值ctx的timeline
    def build_rule(self, ann_type: str) -> ActiveRule:

        match ann_type:
            case "while":
                if self.trigger is None:
                    raise ValueError("while annotation must have a trigger")
                return ActiveWhile(self.trigger)
            case "time_range":
                if self.trigger is None or self.advance is None or self.delay is None:
                    raise ValueError("time_range annotation must have a trigger, advance, delay")
                return ActivateEventTimeRange(self.trigger, self.advance, self.delay)
            case "between":
                if self.start_trigger is None or self.end_trigger is None:
                    raise ValueError("between annotation must have a start_trigger and end_trigger")
                return ActiveBetween(self.start_trigger, self.end_trigger)
            case _:
                raise ValueError(f"Unknown annotation type: {ann_type}")

class Annotation:
    def __init__(self, id: str, ann_type: str, content: Any, activation: AnnotationActivation):
        self.id = id
        self.ann_type = ann_type
        # 内容对象
        self.content = content
        self.rule : ActiveRule = activation.build_rule(self.ann_type)

class ArrowContent(Content):
    def __init__(self, pos_variables: tuple[str, str], shift_variables: tuple[str, str], scale: float = 1.0, color: str = "red", max_stroke_width_to_length_ratio: float = 2, tip_length: float = 0.15):
        self.pos_variables = pos_variables
        self.shift_variables = shift_variables
        self.scale = scale
        self.color = color
        
        self.obj = Arrow(
            color=self.color,
            max_stroke_width_to_length_ratio=max_stroke_width_to_length_ratio,
            tip_length=tip_length,
        )
        super().__init__(pos_variables, self.obj)
    
    def change_callback(self):
        return lambda x_pos, y_pos, x_shift, y_shift: self.update(x_pos, y_pos, x_shift, y_shift)
    
    # 更新箭头的位置
    def update(self, x_pos, y_pos, x_shift, y_shift):
        end_x = x_pos + x_shift * self.scale
        end_y = y_pos + y_shift * self.scale

        if abs(end_x - x_pos) < 1e-9 and abs(end_y - y_pos) < 1e-9:
            self.obj.set_opacity(0)
            return

        self.obj.set_opacity(1)
        self.obj.put_start_and_end_on(
            np.array([x_pos, y_pos, 0.0]),
            np.array([end_x, end_y, 0.0]),
        )

class MathTexContent(Content):
    def __init__(self, tex_strings: list[str], pos_variables: tuple[str, str]):
        self.tex_strings = tex_strings
        self.pos_variables = pos_variables
        self.obj = MathTex(*tex_strings)
        super().__init__(pos_variables, self.obj)


class TextContent(Content):
    def __init__(self, txt: str, pos_variables: tuple[str, str], font_size: int = 24):
        self.txt = txt
        self.pos_variables = pos_variables
        self.obj = Text(txt, font_size=font_size)
        super().__init__(pos_variables, self.obj)


