"""Tests for the LLM code generation pipeline.

These tests do not call any LLM. They exercise:
1. ``_extract_code`` — extracting Python from markdown-fenced responses.
2. ``_validate_python`` — validating that extracted code parses as Python.
3. ``build_system_prompt`` — the system prompt contains key API sections.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

# Ensure the project root is importable when running tests directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phyanim.llm.planner import _extract_code, _validate_python, CodeValidationError
from phyanim.llm.prompt_builder import build_system_prompt


# ---------------------------------------------------------------------------
# Code extraction tests
# ---------------------------------------------------------------------------

class TestExtractCode:
    def test_extract_python_fenced(self) -> None:
        raw = "Here is the code:\n```python\nprint('hello')\n```\nDone."
        code = _extract_code(raw)
        assert code == "print('hello')"

    def test_extract_generic_fenced(self) -> None:
        raw = "```\nprint('hello')\n```"
        code = _extract_code(raw)
        assert code == "print('hello')"

    def test_extract_no_fences(self) -> None:
        raw = "print('hello')"
        code = _extract_code(raw)
        assert code == "print('hello')"

    def test_extract_multiline_code(self) -> None:
        raw = "```python\nimport manim\n\nclass Scene(manim.Scene):\n    pass\n```"
        code = _extract_code(raw)
        assert "class Scene" in code

    def test_extract_prefers_python_fence(self) -> None:
        raw = "text\n```python\nx = 1\n```\nmore text\n```\ny = 2\n```"
        code = _extract_code(raw)
        assert code == "x = 1"


# ---------------------------------------------------------------------------
# Code validation tests
# ---------------------------------------------------------------------------

class TestValidatePython:
    def test_valid_code_passes(self) -> None:
        code = "class MyScene:\n    def construct(self):\n        pass\n"
        _validate_python(code)  # should not raise

    def test_syntax_error_rejected(self) -> None:
        code = "def construct(:\n    pass\n"
        with pytest.raises(CodeValidationError, match="syntax error"):
            _validate_python(code)

    def test_missing_construct_rejected(self) -> None:
        code = "x = 1\ny = 2\n"
        with pytest.raises(CodeValidationError, match="def construct"):
            _validate_python(code)

    def test_scene_render_accepted(self) -> None:
        code = (
            "from phyanim.render import PhyAnimationMultiLayerScene2D\n"
            "scene = PhyAnimationMultiLayerScene2D()\n"
            "scene.render()\n"
        )
        _validate_python(code)  # should not raise


# ---------------------------------------------------------------------------
# System prompt tests
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_prompt_contains_all_modes(self) -> None:
        prompt = build_system_prompt()
        assert "Engine 模式" in prompt
        assert "Code 模式" in prompt
        assert "Hybrid 模式" in prompt

    def test_prompt_contains_api_sections(self) -> None:
        prompt = build_system_prompt()
        assert "Engine 模式 API" in prompt
        assert "Code 模式 API" in prompt
        assert "Hybrid 模式 API" in prompt

    def test_prompt_contains_examples(self) -> None:
        prompt = build_system_prompt()
        assert "Engine 模式完整示例" in prompt
        assert "Code 模式完整示例" in prompt
        assert "Hybrid 模式完整示例" in prompt

    def test_prompt_contains_key_apis(self) -> None:
        prompt = build_system_prompt()
        for snippet in (
            "PhysicsAnimation",
            "PhysicObject2D",
            "PhysicsSegment",
            "solve_animation",
            "TrajectoryData",
            "PhyAnimScene",
            "voiceover",
            "enable_trace",
            "attach_position_updater",
        ):
            assert snippet in prompt, f"System prompt missing: {snippet}"

    def test_prompt_contains_tts_guide(self) -> None:
        prompt = build_system_prompt()
        assert "TTS" in prompt
        assert "TTS_CONFIG" in prompt
        assert "narration" in prompt

    def test_prompt_contains_output_requirements(self) -> None:
        prompt = build_system_prompt()
        assert "```python" in prompt
        assert "输出要求" in prompt
