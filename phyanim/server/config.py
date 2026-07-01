"""Server configuration: LLM backend + TTS settings.

All configuration is centralized here so the server can be started with
a single choice of LLM provider and TTS settings, without the LLM
needing to output any TTS configuration in its generated code.

Usage::

    from phyanim.server.config import ServerConfig, LLMProvider

    config = ServerConfig(
        llm_provider=LLMProvider.DEEPSEEK,
        llm_api_key="sk-...",
        tts=TTSConfig(provider="minimax", voice_id="male-qn-qingse"),
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from phyanim.llm.client import LLMConfig, make_client
from phyanim.voiceover.tts_config import TTSConfig


class LLMProvider(str, Enum):
    """Supported LLM backends."""

    DEEPSEEK = "deepseek"
    DOUBAO = "doubao"


# Preset model names for each provider.
_PROVIDER_MODELS = {
    LLMProvider.DEEPSEEK: "deepseek-v4-pro",
    LLMProvider.DOUBAO: "Doubao-Seed-2.0-pro",
}

# Preset base URLs for each provider.
_PROVIDER_BASE_URLS = {
    LLMProvider.DEEPSEEK: "https://api.deepseek.com",
    LLMProvider.DOUBAO: "https://ark.cn-beijing.volces.com/api/coding/v3",
}


@dataclass
class ServerConfig:
    """Unified server configuration.

    Attributes
    ----------
    llm_provider:
        Which LLM backend to use (deepseek or doubao).
    llm_api_key:
        API key for the LLM provider.  If empty, reads from environment
        variable ``PHYANIM_LLM_API_KEY``.
    llm_model:
        Model name.  If empty, uses the provider's default preset.
    llm_base_url:
        Base URL for the LLM API.  If empty, uses the provider's default.
    tts:
        TTS configuration.  If None, TTS is disabled.
    code_output_dir:
        Directory where generated .py scripts are written.
        Defaults to project_root/outputs/code.
    video_output_dir:
        Directory where manim writes media/ (videos, audio, tex cache).
        This is set as the cwd when running the generated script.
        Defaults to project_root/outputs/video.
    """

    llm_provider: LLMProvider = LLMProvider.DEEPSEEK
    llm_api_key: str = ""
    llm_model: str = ""
    llm_base_url: str = ""
    tts: TTSConfig = field(default_factory=lambda: TTSConfig(provider="none"))
    code_output_dir: str = ""
    video_output_dir: str = ""
    timeout_seconds: int = 300

    def resolved_api_key(self) -> str:
        """Return the effective LLM API key (field or env var)."""
        return self.llm_api_key or os.environ.get("PHYANIM_LLM_API_KEY", "")

    def resolved_model(self) -> str:
        """Return the effective model name."""
        return self.llm_model or _PROVIDER_MODELS.get(self.llm_provider, "")

    def resolved_base_url(self) -> str:
        """Return the effective base URL."""
        return self.llm_base_url or _PROVIDER_BASE_URLS.get(self.llm_provider, "")

    def resolved_code_output_dir(self) -> Path:
        """Return the absolute code output directory, creating it if needed."""
        from pathlib import Path
        d = Path(self.code_output_dir).resolve() if self.code_output_dir else Path.cwd() / "outputs" / "code"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def resolved_video_output_dir(self) -> Path:
        """Return the absolute video output directory, creating it if needed."""
        from pathlib import Path
        d = Path(self.video_output_dir).resolve() if self.video_output_dir else Path.cwd() / "outputs" / "video"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def to_llm_config(self) -> LLMConfig:
        """Build an LLMConfig for the phyanim.llm client."""
        provider = "deepseek" if self.llm_provider == LLMProvider.DEEPSEEK else "doubao"
        return LLMConfig(
            api_key=self.resolved_api_key(),
            base_url=self.resolved_base_url(),
            model=self.resolved_model(),
            provider=provider,
            timeout_seconds=self.timeout_seconds,
        )

    def to_tts_dict(self) -> dict[str, Any]:
        """Return TTS config as a dict for injection into generated code."""
        from dataclasses import asdict
        d = asdict(self.tts)
        # Inject the resolved API key from env var.
        d["api_key"] = self.tts.resolved_api_key()
        return d

    def make_llm_client(self):
        """Create and return an LLM client instance."""
        return make_client(self.to_llm_config())

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Build config from environment variables.

        Required env vars:
            PHYANIM_LLM_PROVIDER: "deepseek" or "doubao"
            PHYANIM_LLM_API_KEY: API key

        Optional env vars:
            PHYANIM_LLM_MODEL: Override model name
            PHYANIM_LLM_BASE_URL: Override base URL
            PHYANIM_TTS_PROVIDER: "minimax" or "none"
            PHYANIM_TTS_API_KEY: MiniMax API key
            PHYANIM_TTS_VOICE: Voice ID
            PHYANIM_CODE_OUTPUT_DIR: Directory for generated .py scripts
            PHYANIM_VIDEO_OUTPUT_DIR: Directory for rendered videos (media/)
        """
        from pathlib import Path

        provider_str = os.environ.get("PHYANIM_LLM_PROVIDER", "deepseek").lower()
        provider = LLMProvider.DEEPSEEK if provider_str == "deepseek" else LLMProvider.DOUBAO

        tts_provider = os.environ.get("PHYANIM_TTS_PROVIDER", "none")
        tts = TTSConfig(
            provider=tts_provider,
            api_key=os.environ.get("PHYANIM_TTS_API_KEY", ""),
            voice_id=os.environ.get("PHYANIM_TTS_VOICE", "male-qn-qingse"),
        )

        return cls(
            llm_provider=provider,
            llm_api_key=os.environ.get("PHYANIM_LLM_API_KEY", ""),
            llm_model=os.environ.get("PHYANIM_LLM_MODEL", ""),
            llm_base_url=os.environ.get("PHYANIM_LLM_BASE_URL", ""),
            tts=tts,
            code_output_dir=os.environ.get("PHYANIM_CODE_OUTPUT_DIR", ""),
            video_output_dir=os.environ.get("PHYANIM_VIDEO_OUTPUT_DIR", ""),
        )
