"""Physics LLM planner: turns a problem description into executable Python code.

Usage::

    from phyanim.llm import LLMConfig, make_client, PhysicsLLMPlanner

    config = LLMConfig(api_key="...", base_url="...", model="...")
    planner = PhysicsLLMPlanner(make_client(config))
    code = planner.plan("两个小球通过弹簧碰撞...")
    # code is a string of executable Python that renders a manim video
"""

from __future__ import annotations

import re
from typing import Any

from phyanim.llm.client import LLMClient
from phyanim.llm.prompt_builder import build_system_prompt


SYSTEM_PROMPT = build_system_prompt()


class PhysicsLLMPlanner:
    """Converts a problem description into executable PhyAnim Python code.

    The LLM outputs raw Python source code (not JSON DSL).  The planner
    extracts the code from the model's response, validates that it parses
    as Python, and does one self-repair round-trip if it doesn't.
    """

    def __init__(self, client: LLMClient, tts_config: dict | None = None) -> None:
        self.client = client
        self.tts_config = tts_config or {}

    def plan(
        self,
        problem_text: str,
        *,
        options: list[str] | None = None,
        images: list[str] | None = None,
        analysis: str | None = None,
        extra: str | None = None,
        scene_name: str | None = None,
    ) -> str:
        """Generate executable Python code from a problem description.

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
        scene_name:
            Optional name for the generated Scene class.  When given, the
            LLM-generated class name is replaced so manim's output video
            file carries this name (e.g. ``scene_name="Scene_20260701_161030"``
            produces ``Scene_20260701_161030.mp4``).

        Returns
        -------
        str
            A string of executable Python source code with TTS_CONFIG injected.
        """
        system_prompt = SYSTEM_PROMPT
        user_prompt = self._build_user_prompt(
            problem_text, options=options, analysis=analysis, extra=extra
        )
        raw_response = self.client.complete_text(
            system_prompt, user_prompt, images=images
        )
        code = _extract_code(raw_response)
        try:
            _validate_python(code)
        except CodeValidationError as exc:
            # One self-repair round-trip feeding the error back to the model.
            repair_prompt = (
                f"{user_prompt}\n\n"
                f"你上一次输出的代码没有通过校验，错误是：{exc}\n"
                "请重新输出完整的 Python 代码。必须修复该错误，不要解释。"
            )
            raw_response = self.client.complete_text(
                system_prompt, repair_prompt, images=images
            )
            code = _extract_code(raw_response)
            _validate_python(code)
        # Inject TTS_CONFIG into the generated code.
        code = _inject_tts_config(code, self.tts_config)
        # Rename Scene class if requested (affects manim output filename).
        if scene_name:
            code = _rename_scene_class(code, scene_name)
        return code

    def plan_to_file(
        self,
        problem_text: str,
        path: str,
        **kwargs: Any,
    ) -> str:
        """Generate code and write it to *path*. Returns the code string."""
        code = self.plan(problem_text, **kwargs)
        from pathlib import Path
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(code, encoding="utf-8")
        return code

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
            "请根据以上信息生成 PhyAnim Python 动画代码。"
            "先用广义坐标把约束约化为无约束 ODE，再选择合适的渲染模式。"
        )
        return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Code extraction and validation
# ---------------------------------------------------------------------------

class CodeValidationError(ValueError):
    """Raised when LLM-generated code fails to parse as Python."""


def _extract_code(raw: str) -> str:
    """Extract Python source code from an LLM response.

    Handles three cases:
    1. Raw code with no markdown fences.
    2. Code wrapped in ```python ... ``` fences.
    3. Code wrapped in ``` ... ``` fences.
    """
    # Try to find a ```python ... ``` block.
    match = re.search(r"```python\s*\n(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try a generic ``` ... ``` block.
    match = re.search(r"```\s*\n(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()

    # No fences — assume the entire response is code.
    return raw.strip()


def _validate_python(code: str) -> None:
    """Validate that *code* parses as valid Python."""
    import ast
    try:
        ast.parse(code)
    except SyntaxError as exc:
        raise CodeValidationError(f"Python syntax error: {exc}") from exc

    # Check for essential structure.
    if "def construct" not in code and "scene.render()" not in code:
        raise CodeValidationError(
            "Code must contain a 'def construct' method or a 'scene.render()' call."
        )


def _inject_tts_config(code: str, tts_config: dict) -> str:
    """Inject TTS_CONFIG definition into generated code.

    The LLM is told to reference ``TTS_CONFIG`` without defining it.
    This function inserts the actual definition after the sys.path.insert
    line so the code is self-contained when executed.
    """
    if not tts_config:
        # No TTS config — inject a disabled default so code doesn't NameError.
        tts_config = {"provider": "none"}

    config_line = f"TTS_CONFIG = {repr(tts_config)}\n"

    # Try to insert after the last sys.path.insert line.
    lines = code.split("\n")
    insert_idx = None
    for i, line in enumerate(lines):
        if "sys.path.insert" in line:
            insert_idx = i + 1

    if insert_idx is not None:
        lines.insert(insert_idx, "")
        lines.insert(insert_idx + 1, config_line.rstrip())
        return "\n".join(lines)

    # Fallback: prepend.
    return config_line + code


def _rename_scene_class(code: str, new_name: str) -> str:
    """Rename the generated Scene class to *new_name*.

    manim names its output video file after the Scene class name
    (e.g. ``class Foo(Scene)`` → ``Foo.mp4``).  By renaming the class
    we control the output filename.

    This replaces the first ``class <Name>(`` that inherits from a Scene
    subclass (PhyAnimScene, PhyAnimCodeScene, PhyAnimHybridScene,
    PhyAnimationMultiLayerScene2D, or Scene).
    """
    # Match: class <Identifier>(<Base>):
    pattern = re.compile(
        r"class\s+(\w+)\s*\(\s*"
        r"(?:PhyAnim\w*|PhyAnimationMultiLayerScene2D|Scene)"
        r"\s*\)\s*:"
    )
    match = pattern.search(code)
    if match:
        old_name = match.group(1)
        # Replace the class definition.
        code = code[: match.start()] + code[match.start():].replace(
            f"class {old_name}(", f"class {new_name}(", 1
        )
        # Replace references to the old name (e.g. in __main__ block).
        code = code.replace(f"{old_name}().render()", f"{new_name}().render()")
    return code


__all__ = ["PhysicsLLMPlanner", "SYSTEM_PROMPT", "CodeValidationError"]
