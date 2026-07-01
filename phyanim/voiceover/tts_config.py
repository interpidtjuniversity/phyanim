"""TTS configuration for voiceover integration.

Supports MiniMax TTS (via manim-voiceover + minimax-tts) with graceful
degradation when the optional dependencies are not installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


TTS_PROVIDERS = ("minimax", "none")


@dataclass(frozen=True)
class TTSConfig:
    """Configuration for a TTS speech service.

    The API key is read from (in priority order):
    1. The ``api_key`` field if non-empty.
    2. The ``PHYANIM_TTS_API_KEY`` environment variable.
    """

    provider: str = "minimax"
    api_key: str = ""
    model: str = "speech-02-turbo"
    voice_id: str = "male-qn-qingse"
    speed: float = 1.0
    vol: float = 1.0
    pitch: float = 0.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "TTSConfig":
        """Build from a DSL dict. Returns a disabled config if *d* is None."""
        if d is None:
            return cls(provider="none")
        return cls(
            provider=d.get("provider", "minimax"),
            api_key=d.get("api_key", ""),
            model=d.get("model", "speech-02-turbo"),
            voice_id=d.get("voice_id", "male-qn-qingse"),
            speed=float(d.get("speed", 1.0)),
            vol=float(d.get("vol", 1.0)),
            pitch=float(d.get("pitch", 0.0)),
        )

    def resolved_api_key(self) -> str:
        """Return the effective API key (field or env var)."""
        return self.api_key or os.environ.get("PHYANIM_TTS_API_KEY", "")

    @property
    def enabled(self) -> bool:
        """Whether TTS is available (provider != none and deps installed)."""
        return self.provider != "none" and bool(self.resolved_api_key())

    def create_service(self) -> Any | None:
        """Create the speech service instance.

        Returns ``None`` if TTS is disabled or dependencies are missing.

        Uses the custom MinimaxSpeechService bundled in phyanim.voiceover.
        """
        if not self.enabled:
            return None
        try:
            from phyanim.voiceover.minimax_tts import MinimaxSpeechService
        except ImportError:
            return None
        return MinimaxSpeechService(
            api_key=self.resolved_api_key(),
            model=self.model,
            voice_id=self.voice_id,
            speed=self.speed,
            vol=self.vol,
            pitch=self.pitch,
        )
