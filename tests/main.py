from manim import *
import numpy as np
import math


class SimpleTimelineRuntime:
    """
    一个简化版 timeline。
    这里直接使用 render_t -> physics_t。
    你可以替换成自己的 TimelineRuntime。
    """

    def __init__(self):
        pass

    def physics_time_at(self, render_t: float) -> float:
        # 示例：正常播放
        return render_t


class FakePhysicsContext:
    """
    一个极简 physics context。
    用来模拟物理系统在 physics_t 下的状态。
    """

    def ball_pos(self, physics_t: float) -> np.ndarray:
        x = -4.0 + physics_t
        y = 1.0 + 0.4 * math.sin(2.0 * physics_t)
        return np.array([x, y, 0.0])

    def anchor_pos(self, physics_t: float) -> np.ndarray:
        return np.array([-4.5, 1.0, 0.0])


class LensRenderLayer:
    """
    RenderLayer 风格的放大镜。

    外层：
        用 render_t 控制显示、遮罩、边框等。

    内部物理副本：
        用 physics_t 采样物理状态。
    """

    def __init__(
        self,
        scene: Scene,
        tracker: ValueTracker,
        timeline_runtime: SimpleTimelineRuntime,
        physics_ctx: FakePhysicsContext,
        main_group: VGroup,
        center: np.ndarray = np.array([2.5, 0.8, 0.0]),
        radius: float = 1.3,
        zoom: float = 2.5,
        focus_func=None,
    ):
        self.scene = scene
        self.tracker = tracker
        self.timeline_runtime = timeline_runtime
        self.physics_ctx = physics_ctx
        self.main_group = main_group

        self.center = center
        self.radius = radius
        self.zoom = zoom

        if focus_func is None:
            self.focus_func = lambda physics_t: physics_ctx.ball_pos(physics_t)
        else:
            self.focus_func = focus_func

        self.layer_group = VGroup()

        self.blur_group = VGroup()
        self.dim_overlay = None

        self.lens_bg = None
        self.lens_frame = None

        self.lens_ball = None
        self.lens_anchor = None
        self.lens_spring = None

    def world_to_lens(self, world_point: np.ndarray, physics_t: float) -> np.ndarray:
        """
        世界坐标 -> 放大镜坐标。
        """
        focus = self.focus_func(physics_t)
        return self.center + self.zoom * (world_point - focus)

    def create_fake_blur_group(self) -> VGroup:
        """
        伪模糊：
        复制主画面若干份，做轻微偏移和低透明度。

        注意：
            这不是真正的高斯模糊，只是视觉近似。
        """
        blur_group = VGroup()

        offsets = [
            0.035 * RIGHT,
            0.035 * LEFT,
            0.035 * UP,
            0.035 * DOWN,
            0.025 * (RIGHT + UP),
            0.025 * (RIGHT + DOWN),
            0.025 * (LEFT + UP),
            0.025 * (LEFT + DOWN),
        ]

        for off in offsets:
            copied = self.main_group.copy()
            copied.shift(off)
            copied.set_opacity(0.08)
            blur_group.add(copied)

        return blur_group

    def create(self) -> VGroup:
        """
        创建放大镜渲染层。
        """

        # 1. 镜头外伪模糊
        self.blur_group = self.create_fake_blur_group()

        # 2. 全屏暗化遮罩
        self.dim_overlay = Rectangle(
            width=20,
            height=12,
            stroke_width=0,
            fill_color=BLACK,
            fill_opacity=0.35,
        )
        self.dim_overlay.move_to(ORIGIN)

        # 3. 放大镜背景
        # 用半透明填充盖住内部原画面，避免内外重叠太乱。
        self.lens_bg = Circle(
            radius=self.radius,
            stroke_width=0,
            fill_color=BLACK,
            fill_opacity=0.18,
        )
        self.lens_bg.move_to(self.center)

        # 4. 放大镜边框
        self.lens_frame = Circle(
            radius=self.radius,
            stroke_color=YELLOW,
            stroke_width=5,
            fill_opacity=0,
        )
        self.lens_frame.move_to(self.center)

        # 5. 放大镜内的物理副本
        self.lens_anchor = Dot(color=GRAY, radius=0.06)
        self.lens_ball = Dot(color=RED, radius=0.13)
        self.lens_spring = Line(color=BLUE, stroke_width=5)

        # 6. updater：用 render_t 得到 physics_t，然后采样物理状态
        def update_lens_objects(group):
            render_t = self.tracker.get_value()
            physics_t = self.timeline_runtime.physics_time_at(render_t)

            world_anchor = self.physics_ctx.anchor_pos(physics_t)
            world_ball = self.physics_ctx.ball_pos(physics_t)

            lens_anchor_pos = self.world_to_lens(world_anchor, physics_t)
            lens_ball_pos = self.world_to_lens(world_ball, physics_t)

            self.lens_anchor.move_to(lens_anchor_pos)
            self.lens_ball.move_to(lens_ball_pos)

            self.lens_spring.put_start_and_end_on(
                lens_anchor_pos,
                lens_ball_pos,
            )

            # 简单圆形可见性判断。
            # 如果对象中心离开镜头区域，则隐藏。
            # 对复杂线段的真实裁剪需要额外 clip/mask。
            if np.linalg.norm(lens_ball_pos - self.center) <= self.radius:
                self.lens_ball.set_opacity(1.0)
            else:
                self.lens_ball.set_opacity(0.0)

            if np.linalg.norm(lens_anchor_pos - self.center) <= self.radius:
                self.lens_anchor.set_opacity(1.0)
            else:
                self.lens_anchor.set_opacity(0.0)

            # 这里为了演示，线条不做精确裁剪。
            # 如果端点都远离镜头，可以隐藏。
            if (
                np.linalg.norm(lens_anchor_pos - self.center) <= self.radius * 1.4
                or np.linalg.norm(lens_ball_pos - self.center) <= self.radius * 1.4
            ):
                self.lens_spring.set_opacity(1.0)
            else:
                self.lens_spring.set_opacity(0.0)

        lens_physics_group = VGroup(
            self.lens_spring,
            self.lens_anchor,
            self.lens_ball,
        )
        lens_physics_group.add_updater(update_lens_objects)

        # 7. 镜头说明文字，RenderLayer 内容，用 render_t 播放
        title = Text("Lens View", font_size=28, color=YELLOW)
        title.next_to(self.lens_frame, UP, buff=0.15)

        def update_title(m):
            render_t = self.tracker.get_value()

            # 示例：前 1 秒淡入
            alpha = min(max(render_t / 1.0, 0.0), 1.0)
            m.set_opacity(alpha)

        title.add_updater(update_title)

        self.layer_group = VGroup(
            self.blur_group,
            self.dim_overlay,
            self.lens_bg,
            lens_physics_group,
            self.lens_frame,
            title,
        )

        return self.layer_group


class LensBlurDemo(Scene):
    def construct(self):
        tracker = ValueTracker(0.0)

        timeline_runtime = SimpleTimelineRuntime()
        physics_ctx = FakePhysicsContext()

        # =========================
        # 主物理画面
        # =========================

        anchor = Dot(color=GRAY, radius=0.06)
        ball = Dot(color=RED, radius=0.11)
        spring = Line(color=BLUE, stroke_width=4)

        def update_main_objects(group):
            render_t = tracker.get_value()
            physics_t = timeline_runtime.physics_time_at(render_t)

            anchor_pos = physics_ctx.anchor_pos(physics_t)
            ball_pos = physics_ctx.ball_pos(physics_t)

            anchor.move_to(anchor_pos)
            ball.move_to(ball_pos)
            spring.put_start_and_end_on(anchor_pos, ball_pos)

        main_group = VGroup(spring, anchor, ball)
        main_group.add_updater(update_main_objects)

        # 背景参考线
        axis = NumberLine(
            x_range=[-5, 5, 1],
            length=10,
            include_numbers=True,
            font_size=20,
        )
        axis.shift(2.2 * DOWN)

        bg_grid = VGroup(axis)

        self.add(bg_grid)
        self.add(main_group)

        # =========================
        # Lens RenderLayer
        # =========================

        lens_layer = LensRenderLayer(
            scene=self,
            tracker=tracker,
            timeline_runtime=timeline_runtime,
            physics_ctx=physics_ctx,
            main_group=main_group,
            center=np.array([2.6, 1.2, 0.0]),
            radius=1.25,
            zoom=2.8,
            focus_func=lambda physics_t: physics_ctx.ball_pos(physics_t),
        )

        self.add(lens_layer.create())

        # =========================
        # 播放
        # =========================

        self.play(
            tracker.animate.set_value(7.0),
            run_time=7.0,
            rate_func=linear,
        )