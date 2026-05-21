from __future__ import annotations

import json
from typing import Any

from phyanim.llm.client import DeepSeekClient
from phyanim.llm.validation import IRValidationError, validate_physics_ir


SYSTEM_PROMPT = """你是物理动画框架的建模规划器。
只输出 JSON 对象，不要输出 Markdown。
你的任务是把自然语言物理题目转换为可验证的 Physics Animation IR。
IR 必须包含 scene、objects、initial_keyframe、segments、render。
方程必须使用一阶 ODE 形式，阶段之间通过事件或 duration 衔接。
状态变量必须是合法 SymPy 标识符，例如 "x"、"vx"、"cart1_x"、"cart2_vx"。
多物体或多参数场景必须由你保证 m、q、x、y、vx、vy 等符号不重复、不冲突。
ODE 段的 equations 使用状态名作为 key，例如 "x": "vx"，不要使用 "x_dot"。
如果是碰撞瞬间等状态突变阶段，设置 "type": "transition"，并提供 object_ids 和 transition；transition 阶段不需要 equations。
不要直接生成逐帧动画代码。"""


class PhysicsLLMPlanner:
    """Converts problem text into structured physics IR with validation."""

    def __init__(self, client: DeepSeekClient) -> None:
        self.client = client

    def plan(self, problem_text: str) -> dict[str, Any]:
        user_prompt = self._build_user_prompt(problem_text)
        result = self.client.complete_json(SYSTEM_PROMPT, user_prompt)
        try:
            validate_physics_ir(result)
        except IRValidationError as exc:
            repair_prompt = f"""{user_prompt}

你上一次输出没有通过校验，错误是：{exc}
请重新输出完整 JSON。必须修复该错误，不要解释。"""
            result = self.client.complete_json(SYSTEM_PROMPT, repair_prompt)
            validate_physics_ir(result)
        return result

    def plan_as_text(self, problem_text: str) -> str:
        return json.dumps(self.plan(problem_text), ensure_ascii=False, indent=2)

    def _build_user_prompt(self, problem_text: str) -> str:
        return f"""请为下面题目生成 Physics Animation IR。

最低 JSON 结构：
{{
  "scene": {{"dimension": 2, "time_symbol": "t"}},
  "objects": [
    {{
      "id": "object_id",
      "type": "point_particle",
      "parameters": {{"m": {{"value": 1.0, "unit": "kg"}}}},
      "state_variables": ["x", "y", "vx", "vy"]
    }}
  ],
  "initial_keyframe": {{
    "time": 0.0,
    "object_states": {{"object_id": {{"x": 0.0, "y": 0.0, "vx": 1.0, "vy": 0.0}}}}
  }},
  "segments": [
    {{
      "id": "segment_id",
      "description": "阶段说明",
      "type": "ode",
      "object_ids": ["object_id"],
      "state_vector": ["x", "y", "vx", "vy"],
      "equations": {{"x": "vx", "y": "vy", "vx": "0", "vy": "0"}},
      "parameters": {{}},
      "end_condition": {{"type": "duration", "value": 1.0}}
    }}
  ],
  "render": {{
    "objects": [
      {{"object_id": "object_id", "geometry": "circle", "position_mapping": ["x", "y"], "show_trace": true}}
    ]
  }}
}}

题目：
{problem_text}
"""
