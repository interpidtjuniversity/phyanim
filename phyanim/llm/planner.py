"""Physics DSL planner: turns a problem description into a validated DSL.

Usage::

    from phyanim.llm import LLMConfig, make_client, PhysicsLLMPlanner

    config = LLMConfig.from_readme_tail("readme.txt")
    planner = PhysicsLLMPlanner(make_client(config))
    dsl = planner.plan("两个小球通过弹簧碰撞...", images=["problem.png"])
"""

from __future__ import annotations

import json
from typing import Any

from phyanim.llm.client import LLMClient
from phyanim.llm.dsl_schema import DSL_FIELD_REFERENCE, DSL_TEMPLATE
from phyanim.llm.validation import IRValidationError, validate_dsl


SYSTEM_PROMPT = """你是 PhyAnim 物理动画框架的建模规划器。
你的任务：把自然语言物理题目（可能附带图片、选项、答案解析）转换为可执行的 JSON DSL。
只输出一个 JSON 对象，不要输出 Markdown 代码块标记，不要解释。

==================== 核心建模原则 ====================

1. 广义坐标约束约化（最重要）：
   - 你必须自行用牛顿力学或拉格朗日力学，把题目中的几何约束（如"沿圆弧运动""摆长固定""弹簧两端绑定物体""轻杆刚性连接"）约化为独立广义坐标的无约束常微分方程组。
   - DSL 的 equations 中只能出现无约束 ODE，不得出现约束力、拉格朗日乘子、代数约束方程。
   - 例如：单摆应约化为 theta'' = -(g/L)*sin(theta)，用广义坐标 theta/omega，而不是 x/y 加约束。
   - 例如：沿圆弧滑动的小球，用角度 phi 和角速度 omega 作为状态，再用 derived 计算笛卡尔坐标 x=R*sin(phi)、y=R*(1-cos(phi)) 供渲染。
   - 笛卡尔渲染坐标必须通过段的 derived 字段计算，并由对象的 cartesian_position 引用这些 derived 变量名。

2. 连续物理段与事件：
   - 把整个运动过程拆分为若干个连续物理段（segment），每个段内方程不变。
   - 段与段之间通过 end_event 衔接：end_event 的 expression 是一个零点检测表达式，当它穿越零点时该段结束。
   - 若段边界发生状态突变（如碰撞、反弹、卡住），用 end_event.transition 给出跃迁方程组（基于事件发生前的同一状态快照计算，不要原地累加）。
   - 每段的 duration 设为一个足够大的上限（如 100），实际结束由 end_event 决定；若确实只想定时结束，用 {"type":"countdown","value":N}。

3. 方程写法：
   - equations 的 key 必须是裸状态名（如 "x"、"vx"、"theta"），值是该状态量的导数表达式字符串。
   - 绝对不要用 "x_dot" 之类的后缀作为 key。
   - 表达式用 SymPy 语法，可用 sin/cos/tan/sqrt/exp/log/abs/Piecewise 和 ** 运算。
   - 表达式中的符号可以是：状态变量名、参数名、t（绝对时间）、t_start/t_end（段起止）。

4. 对象与状态：
   - type="point_particle" 默认有 x/y/vx/vy 四个状态，可通过 state_names 重命名（多物体避免符号冲突）或扩展广义坐标。
   - type="object2d" 用于自定义状态的对象（如轨道端点、弹簧端点），用 states 显式声明状态名。
   - 无自身状态的纯视觉对象（如两端跟随两球的弹簧）：states=[]，cartesian_position 引用其它对象的状态变量。
   - 多物体段必须用 owners 字段声明每个状态变量归属哪个对象。

5. 渲染与标注：
   - physics_layer 的标注与物理时间同步；render_layer 的子动画可暂停/慢放物理（freeze/slow），其内部标注用 local_t 触发。
   - 标注的 pos_variables / shift_variables 可以是状态变量名、derived 变量名、或字面量数字字符串（如 "0.5"）。

==================== DSL 字段规范 ====================

__FIELD_REFERENCE__

==================== DSL 完整示例（仅作格式参考，不要照抄） ====================

__DSL_TEMPLATE__

==================== 输出要求 ====================
- 只输出一个 JSON 对象，顶层键：scene, global_parameters, objects, initial_states, segments, physics_layer, render_layer。
- physics_layer 和 render_layer 可省略或为空对象。
- 所有标识符必须是合法 Python 标识符（字母/数字/下划线，不以数字开头）。
- 数值用 number，不要带单位字符串。
- initial_states 必须覆盖每个对象的全部状态变量。
"""


class PhysicsLLMPlanner:
    """Converts a problem description into a validated PhyAnim DSL document."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def plan(
        self,
        problem_text: str,
        *,
        options: list[str] | None = None,
        images: list[str] | None = None,
        analysis: str | None = None,
        extra: str | None = None,
    ) -> dict[str, Any]:
        """Generate and validate a DSL document.

        Parameters
        ----------
        problem_text:
            The problem statement.
        options:
            Optional multiple-choice options.
        images:
            Optional list of image file paths or data URLs (for vision models).
        analysis:
            Optional reference answer / explanation.
        extra:
            Optional additional context (known constants, hints, etc.).

        Returns
        -------
        dict
            A validated DSL document.
        """
        # Use plain str.replace instead of str.format: the field reference and
        # DSL template contain literal { } braces (JSON, param specs) that would
        # break str.format's placeholder parsing.
        system_prompt = (
            SYSTEM_PROMPT
            .replace("__FIELD_REFERENCE__", DSL_FIELD_REFERENCE)
            .replace("__DSL_TEMPLATE__", DSL_TEMPLATE)
        )
        user_prompt = self._build_user_prompt(
            problem_text, options=options, analysis=analysis, extra=extra
        )
        result = self.client.complete_json(system_prompt, user_prompt, images=images)
        try:
            validate_dsl(result)
        except IRValidationError as exc:
            # One self-repair round-trip feeding the error back to the model.
            repair_prompt = (
                f"{user_prompt}\n\n"
                f"你上一次输出的 DSL 没有通过校验，错误是：{exc}\n"
                "请重新输出完整 JSON DSL。必须修复该错误，不要解释。"
            )
            result = self.client.complete_json(
                system_prompt, repair_prompt, images=images
            )
            validate_dsl(result)
        return result

    def plan_as_text(self, problem_text: str, **kwargs: Any) -> str:
        """Convenience wrapper returning the DSL as a pretty-printed JSON string."""
        return json.dumps(self.plan(problem_text, **kwargs), ensure_ascii=False, indent=2)

    def _build_user_prompt(
        self,
        problem_text: str,
        *,
        options: list[str] | None,
        analysis: str | None,
        extra: str | None,
    ) -> str:
        sections: list[str] = []
        sections.append("题目：\n" + problem_text.strip())
        if options:
            lines = "\n".join(
                f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
            )
            sections.append("选项：\n" + lines)
        if analysis:
            sections.append("答案解析：\n" + analysis.strip())
        if extra:
            sections.append("额外信息：\n" + extra.strip())
        sections.append(
            "请根据以上信息生成 PhyAnim JSON DSL。"
            "先用广义坐标把约束约化为无约束 ODE，再拆分为连续物理段。"
        )
        return "\n\n".join(sections)


__all__ = ["PhysicsLLMPlanner", "SYSTEM_PROMPT"]
