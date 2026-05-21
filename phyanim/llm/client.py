from __future__ import annotations

import ast
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: int = 60

    @classmethod
    def from_readme_tail(cls, path: str | Path = "readme.txt") -> "LLMConfig":
        readme_path = Path(path)
        content = readme_path.read_text(encoding="utf-8")
        api_key = _extract_assignment(content, "api_key")
        base_url = _extract_assignment(content, "DEEPSEEK_BASE_URL")
        model = _extract_assignment(content, "DEEPSEEK_MODEL")
        return cls(api_key=api_key, base_url=base_url.rstrip("/"), model=model)


class DeepSeekClient:
    """Small OpenAI-compatible client using only the Python standard library."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
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
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {body}") from exc


def _extract_assignment(content: str, name: str) -> str:
    pattern = rf"^\s*{re.escape(name)}\s*=\s*(.+?)\s*$"
    for line in content.splitlines():
        match = re.match(pattern, line)
        if match:
            raw_value = match.group(1).strip()
            try:
                value = ast.literal_eval(raw_value)
            except (ValueError, SyntaxError):
                value = raw_value.strip("\"'")
            if not isinstance(value, str) or not value:
                raise ValueError(f"Invalid value for '{name}' in README.")
            return value
    raise ValueError(f"Could not find '{name}' assignment in README.")
