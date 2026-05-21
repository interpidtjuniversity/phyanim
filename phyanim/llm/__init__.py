"""LLM planning layer for natural-language physics problems."""

from phyanim.llm.client import DeepSeekClient, LLMConfig
from phyanim.llm.planner import PhysicsLLMPlanner
from phyanim.llm.validation import IRValidationError, validate_physics_ir

__all__ = [
    "DeepSeekClient",
    "IRValidationError",
    "LLMConfig",
    "PhysicsLLMPlanner",
    "validate_physics_ir",
]
