from manim import *

class AnnotationContent:
    def __init__(self, pos_variables: tuple[str, str], obj):
        self.pos_variables = pos_variables
        self.obj = obj
    
    def change_callback(self):
        return lambda x_pos, y_pos: self.update(x_pos, y_pos)
    
    def update(self, x_pos, y_pos):
        self.obj.move_to(np.array([x_pos, y_pos, 0.0]))

class ArrowContent(AnnotationContent):
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
        self.obj.put_start_and_end_on(
            np.array([x_pos, y_pos, 0.0]),
            np.array([x_pos + x_shift * self.scale, y_pos + y_shift * self.scale, 0.0]),
        )

class MathTexContent(AnnotationContent):
    def __init__(self, tex_strings: list[str], pos_variables: tuple[str, str]):
        self.tex_strings = tex_strings
        self.pos_variables = pos_variables
        self.obj = MathTex(*tex_strings)
        super().__init__(pos_variables, self.obj)


class TextContent(AnnotationContent):
    def __init__(self, txt: str, pos_variables: tuple[str, str], font_size: int = 24):
        self.txt = txt
        self.pos_variables = pos_variables
        self.obj = Text(txt, font_size=font_size)
        super().__init__(pos_variables, self.obj)


class TransitionContent(AnnotationContent):
    def __init__(self, tex_strings: list[list[str]], pos_variables: tuple[str, str], direction: dict[int, str] = {}, font_size: int = 24):
        self.tex_strings = tex_strings
        self.pos_variables = pos_variables

        self.groups : list[VGroup] = []

        # 这里一会加tex的类型判断，是普通文本还是math tex
        for group_idx, group_tex_strings in enumerate(tex_strings):
            if direction.get(group_idx, "down") == "down":
                self.groups.append(VGroup([MathTex(tex, font_size=font_size) for tex in group_tex_strings]).arrange(DOWN))
            elif direction.get(group_idx, "down") == "right":
                self.groups.append(VGroup([MathTex(tex, font_size=font_size) for tex in group_tex_strings]).arrange(RIGHT))
            elif direction.get(group_idx, "down") == "left":
                self.groups.append(VGroup([MathTex(tex, font_size=font_size) for tex in group_tex_strings]).arrange(LEFT))
            elif direction.get(group_idx, "down") == "up":
                self.groups.append(VGroup([MathTex(tex, font_size=font_size) for tex in group_tex_strings]).arrange(UP))
            else:
                raise ValueError(f"Invalid direction: {direction.get(group_idx, 'down')}")

        super().__init__(pos_variables, None)
    
    def update(self, x_pos, y_pos):
        for group in self.groups:
            group.move_to(np.array([x_pos, y_pos, 0.0]))
