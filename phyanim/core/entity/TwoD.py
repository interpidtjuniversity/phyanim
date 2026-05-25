from manim import VMobject, WHITE, PI
import numpy as np

# 内凹球形轨道
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

        # 把轨道放在地面上，height方向整体提高bottom
        for point in points:
            point[1] += bottom
        points = np.array(points)
        
        # 设置路径
        self.set_points_as_corners(points)
        self.set_stroke(color=color, width=stroke_width)
        self.set_fill(color=color, opacity=fill_opacity)

# 外凸半球轨道
class ConvexTrack(VMobject):
    def __init__(self, radius: float, color: str = WHITE, fill_opacity: float = 0.3, stroke_width: float = 2,**kwargs):

        super().__init__(**kwargs)
        self.radius = radius
        
        # 关键点：左下角 → 右下角 → 右上角 → 一系列圆弧上的点 → 左上角 → 回到左下角
        points = []
        num_points = 60
        for i in range(num_points + 1):
            angle = i * PI / num_points  # 从 PI 到 0（从左到右画上半圆）
            x = self.radius * np.cos(angle)
            y = self.radius * np.sin(angle)
            points.append(np.array([x, y, 0]))
        # 加入起始点
        x_start, y_start = points[0]
        points.append(np.array([x_start, y_start, 0]))

        self.set_points_as_corners(points)
        self.set_stroke(color=color, width=stroke_width)
        self.set_fill(color=color, opacity=fill_opacity)

# 标准直线轨道
class StraightTrack(VMobject):
    def __init__(self, start: np.ndarray, end: np.ndarray, color: str = WHITE, fill_opacity: float = 0.3, stroke_width: float = 2,**kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.end = end
        self.set_points_as_corners([self.start, self.end])
        self.set_stroke(color=color, width=stroke_width)
        self.set_fill(color=color, opacity=fill_opacity)

# 直线轨道组
class StraightTrackGroup(VMobject):
    def __init__(self, tracks: list[np.ndarray], color: str = WHITE, fill_opacity: float = 0.3, stroke_width: float = 2,**kwargs):
        super().__init__(**kwargs)
        self.points = []
        for track in tracks:
            self.points.append(track)
        self.set_points_as_corners(self.points)
        self.set_stroke(color=color, width=stroke_width)
        self.set_fill(color=color, opacity=fill_opacity)

