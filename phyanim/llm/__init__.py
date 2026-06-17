"""LLM planning layer for natural-language physics problems.

Pipeline: problem text (+ optional images) → multimodal LLM → JSON DSL →
executable Python code → optional manim render.
"""

from phyanim.llm.client import (
    DeepSeekClient,
    DoubaoClient,
    LLMClient,
    LLMConfig,
    OpenAICompatibleClient,
    make_client,
)
from phyanim.llm.parser import DSLParser
from phyanim.llm.planner import PhysicsLLMPlanner
from phyanim.llm.runner import generate_and_render, render_code
from phyanim.llm.validation import IRValidationError, validate_dsl, validate_physics_ir

__all__ = [
    "DSLParser",
    "DeepSeekClient",
    "DoubaoClient",
    "IRValidationError",
    "LLMClient",
    "LLMConfig",
    "OpenAICompatibleClient",
    "PhysicsLLMPlanner",
    "generate_and_render",
    "make_client",
    "render_code",
    "validate_dsl",
    "validate_physics_ir",
]
