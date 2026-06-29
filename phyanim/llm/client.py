"""Unified LLM client interface with pluggable backends.

This module provides:

* :class:`LLMConfig` — connection configuration, loadable from a ``readme.txt``
  tail (backward compatible with the original DeepSeek-only setup).
* :class:`LLMClient` — the protocol every backend implements.
* :class:`DeepSeekClient` — text-only, pure standard-library HTTP (original).
* :class:`OpenAICompatibleClient` — generic OpenAI-compatible backend that
  supports vision (image input via base64 data URLs). Use this for Doubao,
  GPT-4o, Qwen-VL, etc.
* :func:`make_client` — factory selecting a backend from :class:`LLMConfig`.

All backends speak the same ``complete_json`` API so the planner does not care
which provider is configured.
"""

from __future__ import annotations

import ast
import base64
import json
import mimetypes
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class LLMConfig:
    """Connection configuration for an LLM backend."""

    api_key: str
    base_url: str
    model: str
    provider: str = "deepseek"
    timeout_seconds: int = 120

@runtime_checkable
class LLMClient(Protocol):
    """Every backend implements this single method."""

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        """Call the model and return a parsed JSON object.

        ``images`` is an optional list of file paths or ``data:`` URLs. Backends
        that do not support vision ignore it; backends that do embed each image
        as a base64 data URL in the user message content.
        """
        ...


# ---------------------------------------------------------------------------
# DeepSeek (text-only, pure stdlib)
# ---------------------------------------------------------------------------

class DeepSeekClient:
    """Pure standard-library DeepSeek / OpenAI-compatible text client."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "reasoning_effort": "max",
        }
        response = self._post("/chat/completions", payload)
        content = response["choices"][0]["message"]["content"]
        return json.loads(content)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LLM request failed with HTTP {exc.code}: {body}"
            ) from exc


# ---------------------------------------------------------------------------
# Generic OpenAI-compatible client with vision support
# ---------------------------------------------------------------------------

class OpenAICompatibleClient:
    """OpenAI-compatible client supporting text and image (vision) inputs.

    Works with Doubao (Volcano Ark), GPT-4o, Qwen-VL, and any provider that
    implements the OpenAI Chat Completions schema with multimodal content.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        user_content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image in images or []:
            data_url = _resolve_image_to_data_url(image)
            user_content.append(
                {"type": "image_url", "image_url": {"url": data_url}}
            )

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        response = self._post("/chat/completions", payload)
        content = response["choices"][0]["message"]["content"]
        return json.loads(content)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LLM request failed with HTTP {exc.code}: {body}"
            ) from exc


# Alias for clarity when configuring Doubao specifically.
DoubaoClient = OpenAICompatibleClient


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_client(config: LLMConfig) -> LLMClient:
    """Select a backend implementation based on ``config.provider``.

    ``deepseek`` → text-only :class:`DeepSeekClient`.
    ``doubao`` / ``openai`` / anything else → vision-capable
    :class:`OpenAICompatibleClient`.
    """
    if config.provider == "deepseek":
        return DeepSeekClient(config)
    return OpenAICompatibleClient(config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_image_to_data_url(image: str) -> str:
    """Convert a file path, raw ``data:`` URL, or base64 string to a data URL."""
    if image.startswith("data:"):
        return image
    # Treat as a file path.
    path = Path(image)
    if path.exists():
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "image/png"
        data = path.read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    # Fall back to assuming the string is already raw base64 (png).
    return f"data:image/png;base64,{image}"


def _extract_assignment(content: str, name: str) -> str:
    """Extract a required ``name = value`` assignment from a config file."""
    pattern = rf"^\s*{re.escape(name)}\s*=\s*(.+?)\s*$"
    for line in content.splitlines():
        match = re.match(pattern, line)
        if match:
            return _parse_value(match.group(1), name)
    raise ValueError(f"Could not find '{name}' assignment in config file.")


def _extract_optional(content: str, name: str) -> str | None:
    """Extract an optional ``name = value`` assignment; return None if absent."""
    pattern = rf"^\s*{re.escape(name)}\s*=\s*(.+?)\s*$"
    for line in content.splitlines():
        match = re.match(pattern, line)
        if match:
            return _parse_value(match.group(1), name)
    return None


def _parse_value(raw_value: str, name: str) -> str:
    raw_value = raw_value.strip()
    try:
        value = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        value = raw_value.strip("\"'")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid value for '{name}' in config file.")
    return value


__all__ = [
    "LLMConfig",
    "LLMClient",
    "DeepSeekClient",
    "DoubaoClient",
    "OpenAICompatibleClient",
    "make_client",
]
