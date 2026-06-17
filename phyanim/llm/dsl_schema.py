"""Authoritative JSON DSL schema definition for PhyAnim.

This module documents the complete DSL structure that LLMs are asked to produce
and that :mod:`phyanim.llm.validation` validates and :mod:`phyanim.llm.parser`
translates into executable Python code.

The DSL is a strict superset of what the existing ``examples/*.py`` files can
express: every example maps to exactly one DSL document and back.
"""

from __future__ import annotations

# Supported geometry kinds for objects.
GEOMETRY_KINDS = (
    "circle",
    "spring",
    "straight_track",
    "straight_track_group",
    "right_semicircle_track",
    "left_semicircle_track",
    "circular_arc_track",
    "concave_track",
    "convex_track",
    "inclined_plane",
    "pulley",
    "block",
    "vector_arrow",
    "text_plus",
    "none",
)

# Supported annotation activation types.
ANNOTATION_TYPES = ("while", "time_range", "between")

# Supported annotation content kinds.
CONTENT_KINDS = ("arrow", "text", "mathtex")

# Supported time wrapper types.
TIME_WRAPPER_TYPES = ("freeze", "slow")

# Supported transition (formula derivation) styles.
TRANSITION_STYLES = ("fade", "scale", "slide_left", "slide_down", "spin")

# Supported group arrangement directions inside a transition.
TRANSITION_DIRS = ("down", "right", "left", "up")

# Supported solver engines.
ENGINES = ("scipy", "heyoka")

# DSL top-level template, rendered into the LLM system prompt.
DSL_TEMPLATE = """{
  "scene": {
    "dimension": 2,
    "engine": "scipy",
    "sample_dt": 0.01667
  },
  "global_parameters": {"k": 1.0, "g": 9.8},
  "objects": [
    {
      "id": "ball1",
      "type": "point_particle",
      "mass": 2.0,
      "charge": null,
      "radius": 0.1,
      "color": "red",
      "parameters": {"R": {"value": 2.0, "unit": "m", "description": "track radius"}},
      "state_names": {"x": "x1", "y": "y1", "vx": "v1", "vy": "v1y"},
      "cartesian_position": [["x1", "y1"]],
      "geometry": "circle",
      "geometry_params": {}
    },
    {
      "id": "spring",
      "type": "object2d",
      "states": ["start_x", "start_y", "end_x", "end_y"],
      "cartesian_position": [["start_x", "start_y"], ["end_x", "end_y"]],
      "geometry": "spring",
      "geometry_params": {"start": [-1, 0], "end": [1, 0], "coils": 10, "radius": 0.2, "color": "red"}
    }
  ],
  "initial_states": {
    "ball1": {"x1": -5.0, "y1": 0.0, "v1": 1.0, "v1y": 0.0},
    "spring": {"start_x": -1.0, "start_y": 0.0, "end_x": 1.0, "end_y": 0.0}
  },
  "segments": [
    {
      "id": "approach",
      "object_ids": ["ball1", "spring"],
      "equations": {"x1": "v1", "v1": "0", "start_x": "0", "start_y": "0", "end_x": "0", "end_y": "0"},
      "owners": {"x1": "ball1", "v1": "ball1", "start_x": "spring", "start_y": "spring", "end_x": "spring", "end_y": "spring"},
      "parameters": {"E": 0.03},
      "derived": {"spring_x": "(start_x + end_x)/2", "spring_y": "-0.5"},
      "duration": 100,
      "end_event": {
        "expression": "x1 - start_x",
        "direction": 1,
        "transition": {"v1": "v2"}
      }
    }
  ],
  "physics_layer": {
    "annotations": [
      {
        "id": "v_arrow",
        "type": "while",
        "content": {
          "kind": "arrow",
          "pos_variables": ["x1", "y1"],
          "shift_variables": ["v1", "v1y"],
          "scale": 0.5,
          "color": "red",
          "tip_length": 0.1
        },
        "trigger": {"expression": "t > 0", "direction": 0}
      },
      {
        "id": "compress_label",
        "type": "between",
        "content": {"kind": "text", "text": "spring compressed", "pos_variables": ["spring_x", "spring_y"], "font_size": 24},
        "start_trigger": {"expression": "x1 - start_x", "direction": 1},
        "end_trigger": {"expression": "end_x - start_x - 2", "direction": 1}
      },
      {
        "id": "about_to_collide",
        "type": "time_range",
        "content": {"kind": "text", "text": "about to collide", "pos_variables": ["0", "1"], "font_size": 24},
        "trigger": {"expression": "x1 - start_x", "direction": 1},
        "advance": 2.0,
        "delay": 0.0
      }
    ],
    "transitions": [
      {
        "id": "formula_derivation",
        "groups": [["m1*v1 + m2*v2 = m1*v1' + m2*v2'", "\\frac{1}{2}m1*v1^2 = ..."], ["v1' = 1/3"]],
        "triggers": [{"expression": "t - 2", "direction": 1}, {"expression": "t - 7", "direction": 1}],
        "pos_variables": ["0", "-2"],
        "group_dir": {},
        "style": "spin",
        "duration": 0.5,
        "font_size": 24
      }
    ]
  },
  "render_layer": {
    "sub_animations": [
      {
        "time_wrapper": {
          "id": "freeze_collision",
          "type": "freeze",
          "trigger": {"expression": "x1 - start_x", "direction": 1},
          "extend_to": 5
        },
        "annotations": [
          {
            "id": "freeze_label",
            "type": "while",
            "content": {"kind": "text", "text": "frozen for analysis", "pos_variables": ["0", "2.5"], "font_size": 24},
            "trigger": {"expression": "local_t > 1", "direction": 0}
          }
        ],
        "transitions": []
      }
    ]
  }
}"""


# Human-readable field reference, embedded into the system prompt.
DSL_FIELD_REFERENCE = """\
顶层字段：
  scene: {dimension(必须=2), engine("scipy"|"heyoka", 缺省scipy), sample_dt(缺省1/60)}
  global_parameters: 全局常量字典 {name: number}
  objects: 物理对象列表，见下
  initial_states: {object_id: {state_name: number|表达式字符串}}，必须覆盖每个对象全部状态
  segments: 连续物理段列表，按时间顺序执行，见下
  physics_layer: {annotations: [...], transitions: [...]} 物理层标注（与物理时间同步）
  render_layer: {sub_animations: [...]} 渲染层子动画（时间缩放，可暂停/慢放物理）

对象 ObjectSpec：
  id: 唯一标识符（合法Python标识符）
  type: "point_particle" | "object2d"
  point_particle 专用: mass(number), charge(number|null), radius(number), color(str)
  object2d 专用: states([str]) 自定义状态名列表
  通用:
    parameters: {name: {value, unit, description}} 对象级常量
    state_names: {标准名x/y/vx/vy: 自定义名} 仅 point_particle 可选重命名
    cartesian_position: [[x_var, y_var], ...] 笛卡尔坐标变量对；可引用自身状态/其它对象状态/derived变量
    geometry: 见下方几何体参考
    geometry_params: 见下方几何体参考（每种几何体有自己的构造参数）
  特殊形态：
    无状态视觉对象: states=[] 且 cartesian_position 引用其它对象状态（如两端跟随两球的弹簧）
    位置由derived计算: cartesian_position 指向 derived_equations 计算的变量（如摆球的 x_ball/y_ball）

几何体参考（geometry 与 geometry_params）：
  几何体决定物体的 mobject 形态。每个几何体的 cartesian_position 对数必须等于其端点数。
  端点数=1：cartesian_position 给 1 对 [x,y]，整体平移跟随该点。
  端点数=2：cartesian_position 给 2 对 [x,y]，两端分别跟随两个点（如弹簧两端跟随两球）。
  端点数=0：cartesian_position 可省略或为空（静态装饰，不随状态移动）。

  geometry="circle"
    端点数=1。一个圆。用于质点/小球。
    geometry_params: {color(str, 缺省"red"), radius(number, 缺省0.1)}

  geometry="text_plus"
    端点数=1。一个"+"号叠圆，表示正电荷。
    geometry_params: {color(str, 缺省"red"), radius(number, 缺省0.12), font_size(int, 缺省36)}

  geometry="spring"
    端点数=2。动态弹簧，两端可分别绑定两个状态点，每帧按两端坐标重绘螺旋线。
    start/end 仅作初始绘制用，运行时由 cartesian_position 的两对坐标驱动。
    geometry_params: {start([x,y], 缺省[-1,0]), end([x,y], 缺省[1,0]),
                      coils(int, 圈数, 缺省10), radius(number, 螺旋半径, 缺省0.2),
                      color(str, 缺省"white"), stroke_width(number, 缺省2)}
    典型用法: object2d 弹簧，states=[start_x,start_y,end_x,end_y]，
              cartesian_position=[["start_x","start_y"],["end_x","end_y"]]，
              方程让 start_x/end_x 跟随两端物体的位置。

  geometry="straight_track"
    端点数=2。直线轨道/导轨，两端可移动。
    start/end 为 3D 坐标 [x,y,z]（z 通常为 0），仅作初始绘制。
    geometry_params: {start([x,y,z], 缺省[0,0,0]), end([x,y,z], 缺省[1,0,0]),
                      color(str, 缺省"white"), stroke_width(number, 缺省5),
                      fill_opacity(number, 缺省0.3)}

  geometry="straight_track_group"
    端点数=0。折线轨道组，由多个点连成，静态。
    geometry_params: {tracks([[x,y,z],...], 点序列), color(str, 缺省"white"),
                      stroke_width(number, 缺省2)}

  geometry="right_semicircle_track"
    端点数=1。右半圆竖直环形轨道，圆心跟随 cartesian_position 移动。
    geometry_params: {radius(number, 缺省1.0), color(str, 缺省"white"),
                      stroke_width(number, 缺省4), upward(bool, 缺省true, true=从下到上)}

  geometry="left_semicircle_track"
    端点数=1。左半圆竖直环形轨道，圆心跟随 cartesian_position 移动。
    geometry_params: 同 right_semicircle_track

  geometry="circular_arc_track"
    端点数=1。开放圆弧轨道，圆心跟随 cartesian_position 移动。
    角度约定：0=右，PI/2=上（数学约定，弧度制）。
    geometry_params: {radius(number, 缺省1.0), start_angle(number, 缺省0),
                      end_angle(number, 缺省PI), color(str, 缺省"white"),
                      stroke_width(number, 缺省4)}

  geometry="concave_track"
    端点数=0。内凹半球轨道（矩形+上凹半圆），整体静态。
    几何约束：2*radius <= width 且 radius <= height，否则报错。
    geometry_params: {width(number, 缺省4.0), height(number, 缺省2.0),
                      radius(number, 凹陷半径, 缺省1.0), color(str, 缺省"white"),
                      stroke_width(number, 缺省2), fill_opacity(number, 缺省0.3)}

  geometry="convex_track"
    端点数=0。外凸半球轨道，整体静态。
    geometry_params: {radius(number, 缺省1.0), color(str, 缺省"white"),
                      stroke_width(number, 缺省2), fill_opacity(number, 缺省0.3)}

  geometry="inclined_plane"
    端点数=0。三角斜面，整体静态。
    geometry_params: {length(number, 斜面长, 缺省3.0), angle(number, 倾角弧度, 缺省PI/6),
                      color(str, 缺省"white"), stroke_width(number, 缺省3),
                      fill_opacity(number, 缺省0.12)}

  geometry="pulley"
    端点数=0。定滑轮（圆环+轴心），整体静态。
    geometry_params: {radius(number, 缺省0.25), color(str, 缺省"white"),
                      stroke_width(number, 缺省3)}

  geometry="block"
    端点数=1。矩形物块，中心跟随 cartesian_position 移动。
    geometry_params: {width(number, 缺省0.6), height(number, 缺省0.35),
                      color(str, 缺省"white"), stroke_width(number, 缺省3),
                      fill_opacity(number, 缺省0.25)}

  geometry="vector_arrow"
    端点数=0。静态箭头（用于图示固定方向力/速度）。
    注意：动态力/速度箭头不要用这个，改用 annotation 的 arrow 内容，由 shift_variables 驱动。
    geometry_params: {start([x,y], 缺省[0,0]), end([x,y], 缺省[1,0]),
                      color(str, 缺省"white"), stroke_width(number, 缺省4)}

  geometry="none"
    端点数=0。不创建 mobject（纯计算对象，不渲染）。

段 SegmentSpec：
  id: 唯一标识符
  object_ids: 参与本段的对象id列表
  equations: {state_name: 导数表达式字符串}  key必须是裸状态名（不要用x_dot）
  owners: {state_name: object_id}  多对象时必填，单对象可省略
  parameters: {name: number} 段内参数（覆盖global_parameters同名项）
  derived: {name: 表达式字符串}  派生量；只能引用状态/参数/t，不能引用其它derived量
  duration: number  积分时间上限（通常被end_event提前终止）
  end_event:
    expression: 零点检测表达式（事件触发条件=该表达式穿越零点）
    direction: -1|0|1  穿越方向
    transition: {state_name: 表达式}  事件后状态跃迁；{}或省略=恒等跃迁
  或 end_event: {"type": "countdown", "value": 5}  倒计时结束

标注 AnnotationSpec：
  id, type("while"|"time_range"|"between")
  content:
    kind: "arrow"|"text"|"mathtex"
    pos_variables: [x, y]  位置变量（状态/derived/字面量数字字符串）
    shift_variables: [dx, dy]  仅arrow
    text: str  仅text/mathtex
    scale, color, tip_length, font_size  可选样式
  while: trigger
  time_range: trigger + advance + delay  显示区间(trigger时刻-advance, trigger时刻+delay)
  between: start_trigger + end_trigger
  trigger: {expression, direction}

公式推导 TransitionSpec：
  id, groups([[str], ...]), triggers([trigger, ...]), pos_variables([x,y])
  group_dir: {idx: "down"|"right"|"left"|"up"}
  style: "fade"|"scale"|"slide_left"|"slide_down"|"spin"
  duration, font_size

子动画 SubAnimationSpec（render层）：
  time_wrapper:
    id, type("freeze"|"slow"), trigger
    freeze: extend_to(number) 冻结秒数
    slow: speed(number<1) 慢放倍率；可选 advance, delay
  annotations: [...]  使用 local_t 触发（子动画局部时间）
  transitions: [...]

表达式语法（SymPy）：
  可用符号: 所有状态变量、参数、t(绝对时间)、t_start/t_end(段起止)、local_t(子动画局部时间)
  可用函数: sin, cos, tan, sqrt, exp, log, asin, acos, atan, abs, Piecewise
  运算: + - * / **，比较 > < >= <= ==，逻辑 & | ~
  字面量数字字符串（如"0.5"）在 pos_variables/shift_variables 中视为常量
"""


__all__ = [
    "GEOMETRY_KINDS",
    "ANNOTATION_TYPES",
    "CONTENT_KINDS",
    "TIME_WRAPPER_TYPES",
    "TRANSITION_STYLES",
    "TRANSITION_DIRS",
    "ENGINES",
    "DSL_TEMPLATE",
    "DSL_FIELD_REFERENCE",
]
