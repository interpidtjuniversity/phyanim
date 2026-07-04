# SHM 演示动画 — 添加弹簧与美感优化

## Context

`examples/shm_feature_demo.py` 是简谐振动的综合演示动画，使用 hybrid 模式（`solve_animation` + `attach_position_updater`）。当前画面只有黄色小球在做 SHM 运动，缺少弹簧这一 SHM 最经典的物理元素，视觉上不够直观。同时整体配色和布局还有优化空间。

此外，速度箭头存在一个 bug：原代码使用 `attach_mobject_with_event` 绑定到 `cross_zero_right` 事件，意味着球必须第一次向右过零后箭头才开始更新。但球初始在右侧振幅处、先向左运动，这段时间箭头不更新，导致第一次向左运动期间箭头方向错误。本次修改顺便修复此 bug。

## 修改目标

1. **添加弹簧 + 墙壁**：左侧画一面带斜线纹理的墙，弹簧从墙面水平延伸到球的左边缘，端点随球位置实时更新。
2. **美感优化**：配色协调化、布局合理化、视觉层次清晰。
3. **修复速度箭头 bug**：让箭头从动画开始就持续更新。

## 实施方案

### 文件
- `d:\clcy\pythonplayground\phyanim\examples\shm_feature_demo.py` （唯一修改文件）

### 关键复用
- `phyanim.api.Spring`（已在 `phyanim.api.__init__` 导出，来自 `phyanim/core/entity/TwoD.py#L345`）
  - 提供 `set_start_point` / `set_end_point` 方法
  - 自带 `_update_path` updater，端点变化时自动重绘螺旋路径
- `phyanim.api.attach_expr_updater`、`create_tracker`、`attach_position_updater` 等已有 API 保持不变

### 具体改动

#### 1. 顶部常量扩展
新增配色常量：
```python
WALL_COLOR = GREY_D
SPRING_COLOR = "#E8E8E8"
BALL_COLOR = YELLOW
GLOW_COLOR = "#FFD966"
```

#### 2. 添加墙壁与弹簧（在场景元素区段，球之前）
- 墙壁位置 `WALL_X = -4.2`，矩形 `width=0.3, height=2.0`，填充浅灰半透明
- 墙壁纹理：8 条斜线 `Line([WALL_X-0.15, y, 0], [WALL_X+0.15, y-0.2, 0])` 用 `VGroup` 组合
- 弹簧：`Spring(start=[WALL_X+0.15, 0.5, 0], end=[2.8, 0.5, 0], coils=12, radius=0.25, color=SPRING_COLOR, stroke_width=3)`
  - 球的 y 坐标固定为 0.5（来自 `cartesian_position=[("x", "0.5")]`），弹簧也在 y=0.5
  - 球半径 0.2，所以球左边缘 = `x - 0.2`

#### 3. 弹簧 updater
```python
def update_spring(mob):
    t = tracker.get_value()
    x = trajectory.value_at("x", t)
    mob.set_end_point([x - 0.2, 0.5, 0])
spring.add_updater(update_spring)
```

#### 4. 修复速度箭头 bug（L180-184）
删除原 `attach_mobject_with_event` 包装，改为直接 `add_updater`：
```python
def update_speed_arrow(mob):
    t = tracker.get_value()
    x = trajectory.value_at("x", t)
    vx = trajectory.value_at("vx", t)
    start = np.array([x, 0.5, 0])
    if abs(vx) > 1e-6:
        end = start + np.array([vx * 0.3, 0, 0])
        mob.put_start_and_end_on(start, end)
        mob.set_opacity(1.0)
    else:
        mob.set_opacity(0.0)
speed_arrow.add_updater(update_speed_arrow)
```

#### 5. 美感优化

**配色**：
- 小球：黄色填充 + 微弱外发光（用 `SurroundingRectangle` 或 `GlowDot` 模拟，保持简洁就用稍大半径的低透明度圆作为光晕）
- 弹簧：浅银色 `#E8E8E8`
- 墙壁：深灰 `GREY_D` 填充 + 斜线纹理
- 速度箭头：红色保持，箭头粗细稍增
- 能量条：动能绿、势能蓝保持

**布局**（用户已确认）：
- 球的 y 坐标改为 0（修改 `cartesian_position=[("x", "0")]`），让一切在 x 轴上更直观
- 同步更新弹簧 y=0、箭头 start y=0
- 能量条移到右下角 `[5, -2, 0]` 区域，避免与球运动轨迹重叠
- 速度/时间标签移到顶部 `[3, 2.5, 0]` 与 `[3, -2.5, 0]`，远离球
- 振幅边界线 `amp_line_r/l` 改为 `UP * 1.0` 到 `DOWN * 1.0`，更紧凑

**球的光晕**（用户已确认添加）：
- 在球下层添加一个稍大半径（0.35）的低透明度（0.3）黄色圆作为光晕
- 光晕用 updater 跟随球位置

**视觉层次**：
- 标题用渐入 + 缩放效果
- 公式推导阶段，公式块居中显示（`to_edge(LEFT, buff=2.0).shift(UP * 1.5)` 改为屏幕居中偏左）
- 结论公式用 `SurroundingRectangle` 高亮（已有）

#### 6. 同步调整
- `equilibrium` 线保持（垂直线 x=0）
- `FadeOut` 列表添加 `spring, wall, wall_pattern`
- 公式推导阶段前的 `FadeOut` 列表也要包含新增元素

## 验证

运行命令：
```
python examples/shm_feature_demo.py
```

检查点：
1. 弹簧正确显示，从左墙延伸到球
2. 弹簧随球运动实时拉伸/压缩
3. 球向左运动时速度箭头指向左（红色箭头向左），速度标签显示负值
4. 球向右运动时箭头指向右
5. 整体配色协调，布局无重叠
6. 公式推导阶段正常过渡
