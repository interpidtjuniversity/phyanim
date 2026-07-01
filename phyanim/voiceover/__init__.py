"""Voiceover / TTS integration for PhyAnim.

Provides:
- :class:`TTSConfig` — TTS service configuration (MiniMax, with graceful degradation).
- :class:`PhyAnimScene` — base Scene class for code/hybrid render modes.
- :class:`NarrationSegment` / :class:`NarrationPlayer` — engine-mode narration.
"""

from phyanim.voiceover.tts_config import TTSConfig, TTS_PROVIDERS
from phyanim.voiceover.scene import PhyAnimScene
from phyanim.voiceover.narration import NarrationSegment, NarrationPlayer

__all__ = [
    "TTSConfig",
    "TTS_PROVIDERS",
    "PhyAnimScene",
    "NarrationSegment",
    "NarrationPlayer",
]
