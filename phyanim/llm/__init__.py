"""LLM planning layer for natural-language physics problems.

Pipeline: problem text (+ optional images) → multimodal LLM → executable
Python code → optional manim render.

The LLM outputs raw Python code directly (no JSON DSL).  The planner
extracts and validates the code, then the runner can execute it.
"""

from phyanim.llm.client import (
    DeepSeekClient,
    DoubaoClient,
    LLMClient,
    LLMConfig,
    OpenAICompatibleClient,
    make_client,
)
from phyanim.llm.planner import CodeValidationError
from phyanim.llm.prompt_builder import build_code_generate_system_prompt
from phyanim.llm.runner import render_code

__all__ = [
    "CodeValidationError",
    "DeepSeekClient",
    "DoubaoClient",
    "LLMClient",
    "LLMConfig",
    "OpenAICompatibleClient",
    "build_code_generate_system_prompt",
    "make_client",
    "render_code",
]
