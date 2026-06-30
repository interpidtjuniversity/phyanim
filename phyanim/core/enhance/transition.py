from typing import Callable

from phyanim.core.enhance.asserts import Content
from phyanim.core.enhance.trigger import Trigger
from manim import VGroup, DOWN, RIGHT, LEFT, UP

from manim import MathTex, TexTemplate

import numpy as np


def _make_tex_template() -> TexTemplate:
    """Build a default TexTemplate configured for xelatex."""
    tpl = TexTemplate()
    tpl.tex_compiler = "xelatex"
    tpl.output_format = ".xdv"
    return tpl


_TEX_TEMPLATE = _make_tex_template()

class TransitionContent(Content):
    def __init__(self, tex_strings: list[list[str]], pos_variables: tuple[str, str], direction: dict[int, str] = {}, font_size: int = 24):
        self.tex_strings = tex_strings
        self.pos_variables = pos_variables

        self.groups : list[VGroup] = []

        # 这里一会加tex的类型判断，是普通文本还是math tex
        for group_idx, group_tex_strings in enumerate(tex_strings):
            if direction.get(group_idx, "down") == "down":
                self.groups.append(VGroup(*[MathTex(tex, font_size=font_size, tex_template=_TEX_TEMPLATE) for tex in group_tex_strings]).arrange(DOWN))
            elif direction.get(group_idx, "down") == "right":
                self.groups.append(VGroup(*[MathTex(tex, font_size=font_size, tex_template=_TEX_TEMPLATE) for tex in group_tex_strings]).arrange(RIGHT))
            elif direction.get(group_idx, "down") == "left":
                self.groups.append(VGroup(*[MathTex(tex, font_size=font_size, tex_template=_TEX_TEMPLATE) for tex in group_tex_strings]).arrange(LEFT))
            elif direction.get(group_idx, "down") == "up":
                self.groups.append(VGroup(*[MathTex(tex, font_size=font_size, tex_template=_TEX_TEMPLATE) for tex in group_tex_strings]).arrange(UP))
            else:
                raise ValueError(f"Invalid direction: {direction.get(group_idx, 'down')}")

        super().__init__(pos_variables, None)
    
    def update(self, x_pos, y_pos):
        for group in self.groups:
            group.move_to(np.array([x_pos, y_pos, 0.0]))

class Transition:
    def __init__(self, 
        id: str, 
        group_strings: list[list[str]], 
        triggers: list[Trigger], 
        pos_variables: tuple[str, str], 
        group_dir: dict[int, str] = {}, 
        font_size: int = 24, 
        style: str = "fade", 
        duration: float = 0.1,
        smooth_func : Callable[[float], float] = lambda x: x * x * (3 - 2 * x)
    ):
        self.id = id
        self.group_strings = group_strings
        self.triggers = triggers
        # 这里是否要添加断言triggers的size要和group_strings的size一致？
        self.pos_variables = pos_variables
        self.group_dir = group_dir
        self.font_size = font_size

        self.style = style
        self.duration = duration
        self.smooth_func = smooth_func
        self.build_content()

    def build_content(self):
        self.content = TransitionContent(self.group_strings, self.pos_variables, self.group_dir, self.font_size)