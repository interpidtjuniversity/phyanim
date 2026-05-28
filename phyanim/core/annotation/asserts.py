from manim import *

class ArrowAnnotation:
    def __init__(self, pos_variables: tuple[str, str], shift_variables: tuple[str, str], scale: float = 1.0, color: str = "red", max_stroke_width_to_length_ratio: float = 2):
        self.pos_variables = pos_variables
        self.shift_variables = shift_variables
        # 缩放因子
        self.scale = scale
        self.color = color
        
        self.obj = Arrow(color=self.color, max_stroke_width_to_length_ratio=max_stroke_width_to_length_ratio)
    
    def change_callback(self):
        return lambda x_pos, y_pos, x_shift, y_shift: self.update(x_pos, y_pos, x_shift, y_shift)
    
    # 更新箭头的位置
    def update(self, x_pos, y_pos, x_shift, y_shift):
        self.obj.put_start_and_end_on(
            np.array([x_pos, y_pos, 0.0]),
            np.array([x_pos + x_shift * self.scale, y_pos + y_shift * self.scale, 0.0]),
        )

class MathTexAnnotation:
    def __init__(self, tex_strings: list[str], pos_variables: tuple[str, str]):
        self.tex_strings = tex_strings
        self.pos_variables = pos_variables
        self.obj = MathTex(*tex_strings)
    
    def change_callback(self):
        return lambda x_pos, y_pos: self.update(x_pos, y_pos)
    
    def update(self, x_pos, y_pos):
        self.obj.move_to(np.array([x_pos, y_pos, 0.0]))


class TextAnnotation:
    def __init__(self, txt: str, pos_variables: tuple[str, str], font_size: int = 24):
        self.txt = txt
        self.pos_variables = pos_variables
        self.obj = Text(txt, font_size=font_size)
    
    def change_callback(self):
        return lambda x_pos, y_pos: self.update(x_pos, y_pos)
    
    def update(self, x_pos, y_pos):
        self.obj.move_to(np.array([x_pos, y_pos, 0.0]))
