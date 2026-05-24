from manim import VMobject, WHITE, PI
import numpy as np


class ConcaveTrack(VMobject):
    def __init__(self, width: float, height: float, radius: float, color: str = WHITE, fill_opacity: float = 0.3, stroke_width: float = 2,**kwargs):
        if 2 * radius > width or radius > height:
            raise ValueError("Radius must be less than the height and half width of the track")

        super().__init__(**kwargs)
        self.width = width
        self.height = height
        self.radius = radius
        
        # 矩形左、右下角坐标
        left = width / 2
        bottom = height / 2
        
        # 关键点：左下角 → 右下角 → 右上角 → 一系列圆弧上的点 → 左上角 → 回到左下角
        points = []
        # 左下角
        points.append(np.array([-left, -bottom, 0]))
        # 右下角
        points.append(np.array([left, -bottom, 0]))
        # 右上角（半圆弧起点）
        points.append(np.array([left, bottom, 0]))
        
        # 生成半圆弧上的点（用很多个点近似）
        arc_center = np.array([0, bottom, 0])  # 圆心在矩形上边中点
        num_points = 60
        for i in range(num_points + 1):
            angle = i * PI / num_points  # 从 PI 到 0（从左到右画上半圆）
            x = arc_center[0] + self.radius * np.cos(angle)
            y = arc_center[1] - self.radius * np.sin(angle)
            points.append(np.array([x, y, 0]))
        
        # 左上角
        points.append(np.array([-left, bottom, 0]))
        # 回到左下角，形成闭合
        points.append(np.array([-left, -bottom, 0]))
        
        # 设置路径
        self.set_points_as_corners(points)
        self.set_stroke(color=color, width=stroke_width)
        self.set_fill(color=color, opacity=fill_opacity)
