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
        x_start, y_start = points[0][0], points[0][1]
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
    def __init__(self, tracks: list[np.ndarray], color: str = WHITE, stroke_width: float = 2,**kwargs):
        super().__init__(**kwargs)
        self.points = []
        for track in tracks:
            self.points.append(track)
        self.set_points_as_corners(self.points)
        self.set_stroke(color=color, width=stroke_width)

# 动态弹簧：由两个端点决定长度、方向和位置。
# 把弹簧声明成一个形状，严格执行计算与渲染分离
from manim import LEFT, RIGHT
class Spring(VMobject):
    """动态弹簧：由两个端点决定长度、方向和位置。
        start: 起点坐标 (np.array)
        end:   终点坐标 (np.array)
        coils: 圈数（默认 10）
        radius: 弹簧半径（默认 0.2）
        color: 颜色
        stroke_width: 线宽
        num_points: 路径采样点数（默认 200，越大越光滑）
    """
    def __init__(
        self,
        start: np.ndarray = LEFT,
        end: np.ndarray = RIGHT,
        coils: int = 10,
        radius: float = 0.2,
        color: str = WHITE,
        stroke_width: float = 2,
        num_points: int = 200,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.start = np.array(start)
        self.end = np.array(end)
        self.coils = coils
        self.radius = radius
        self.num_points = num_points

        # 初始绘制
        self._update_path()
        self.set_stroke(color=color, width=stroke_width)

        # 添加自动更新器：每帧自动根据端点重绘
        self.add_updater(lambda mob, dt: mob._update_path())

    # 返回两个callback
    def point_change_callbacks(self):
        self.start_point_change_callback = lambda x, y: self.set_start_point(np.array([x, y]))
        self.end_point_change_callback = lambda x, y: self.set_end_point(np.array([x, y]))
        return [self.start_point_change_callback, self.end_point_change_callback]

    def set_start_end(self, start: np.ndarray, end: np.ndarray):
        self.start = np.array(start)
        self.end = np.array(end)
        self._update_path()

    def set_start_point(self, start: np.ndarray):
        self.start = np.array(start)
        self._update_path()
    
    def set_end_point(self, end: np.ndarray):
        self.end = np.array(end)
        self._update_path()

    # 重新生成模型
    def _update_path(self):
        """根据当前端点重新生成螺旋线路径"""
        D = self.end - self.start
        L = np.linalg.norm(D)
        if L < 1e-12:
            # 长度为零时退化为一个点
            self.set_points_as_corners([self.start, self.start])
            return

        # 标准弹簧参数 t (0..1)
        t = np.linspace(0, 1, self.num_points)
        # x = L*t, y = radius * sin(2pi * coils * t)
        y_std = self.radius * np.sin(2 * np.pi * self.coils * t)

        # 方向向量
        u = D / L  # 单位向量，平行弹簧方向
        v = np.array([-u[1], u[0], 0.0]) # 单位向量，垂直弹簧方向

        # 每个点的坐标 = start + x_std_i * u + y_std_i * v
        points = []
        for i in range(self.num_points):
            x = self.start[0] + t[i]*D[0] + y_std[i]*v[0]
            y = self.start[1] + t[i]*D[1] + y_std[i]*v[1]
            points.append(np.array([x, y, 0]))

        self.set_points_smoothly(points)  # 使用 set_points_smoothly 绘制平滑曲线

