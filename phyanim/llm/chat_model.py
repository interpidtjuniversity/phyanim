"""Factory to create a LangChain ChatModel from phyanim's LLMConfig.

Since both DeepSeek and Doubao expose OpenAI-compatible APIs, we use
``ChatOpenAI`` from ``langchain_openai`` for all providers.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from phyanim.llm.client import LLMConfig


def make_chat_model(config: LLMConfig) -> ChatOpenAI:
    """Create a LangChain ``ChatOpenAI`` from an :class:`LLMConfig`.

    Works with DeepSeek, Doubao, and any OpenAI-compatible backend
    because they all implement the same ``/chat/completions`` schema.
    """
    return ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=0.2,
        timeout=config.timeout_seconds,
    )
