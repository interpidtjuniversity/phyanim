"""Base scene class for code/hybrid render modes.

Provides TTS voiceover integration (via manim-voiceover) with graceful
degradation when the optional dependencies are not installed.

Usage in generated code::

    class GeneratedScene(PhyAnimScene):
        def construct(self):
            self.setup_speech(TTS_CONFIG)
            ball = Circle(radius=0.2, color=YELLOW)
            self.add(ball)
            with self.voiceover(text="小球从静止开始下落") as tracker:
                self.play(ball.animate.shift(DOWN * 3), run_time=2.0)
                self.wait(tracker.duration)
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator

from manim import Scene, Mobject

from phyanim.voiceover.tts_config import TTSConfig

# Try to import VoiceoverScene at module level so we can subclass it.
try:
    from manim_voiceover import VoiceoverScene as _BaseVoiceoverScene

    _HAS_VOICEOVER = True
except ImportError:
    _HAS_VOICEOVER = False

    class _BaseVoiceoverScene(Scene):  # type: ignore[no-redef]
        """Fallback when manim-voiceover is not installed."""


class PhyAnimScene(_BaseVoiceoverScene):
    """Base scene for code/hybrid render modes.

    - Deep blue background by default.
    - ``setup_speech`` initializes TTS (no-op if deps missing).
    - ``voiceover`` context manager syncs narration with animation.
    - ``safe_layout`` auto-scales large mobjects to fit the screen.
    """

    BG_COLOR = "#1C2333"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.camera.background_color = self.BG_COLOR
        self._tts_config: TTSConfig = TTSConfig(provider="none")
        self._speech_ready: bool = False

    # ------------------------------------------------------------------
    # TTS setup
    # ------------------------------------------------------------------

    def setup_speech(self, tts_config: dict | TTSConfig | None = None) -> bool:
        """Initialize the TTS speech service.

        Returns ``True`` if TTS is active, ``False`` if degraded.
        """
        if isinstance(tts_config, dict):
            self._tts_config = TTSConfig.from_dict(tts_config)
        elif isinstance(tts_config, TTSConfig):
            self._tts_config = tts_config
        else:
            self._tts_config = TTSConfig(provider="none")

        if not _HAS_VOICEOVER:
            self._speech_ready = False
            return False

        service = self._tts_config.create_service()
        if service is None:
            self._speech_ready = False
            return False

        self.set_speech_service(service)
        self._speech_ready = True
        return True

    # ------------------------------------------------------------------
    # Voiceover context manager
    # ------------------------------------------------------------------

    @contextmanager
    def voiceover(
        self, text: str, **kwargs: Any
    ) -> Generator["_DummyTracker", None, None]:
        """Context manager that syncs narration with animation.

        When TTS is active, this delegates to manim-voiceover's
        ``self.voiceover(text=...)`` which generates audio and returns a
        tracker with ``.duration``.

        When TTS is degraded (no deps or disabled), this yields a dummy
        tracker with a positive ``duration`` estimated from text length
        so that ``self.wait(tracker.duration)`` does not crash.
        """
        if self._speech_ready and _HAS_VOICEOVER:
            # Delegate to the real VoiceoverScene.voiceover.
            with super().voiceover(text=text, **kwargs) as tracker:
                yield tracker
        else:
            yield _DummyTracker(estimated_duration(text))

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def safe_layout(self, obj: Mobject, center: bool = True) -> Mobject:
        """Auto-scale *obj* to fit the screen, optionally centering it."""
        if obj.width > 12:
            obj.scale_to_fit_width(12)
        if obj.height > 6.5:
            obj.scale_to_fit_height(6.5)
        if center:
            obj.move_to(obj.get_center())
        return obj


@dataclass
class _DummyTracker:
    """Fallback tracker when TTS is not available."""

    duration: float = 1.0


def estimated_duration(text: str) -> float:
    """Estimate speech duration from text length when TTS is unavailable.

    Uses ~4 characters per second (rough Chinese/English average).
    Returns at least 1.0 second to avoid manim's wait(0) crash.
    """
    chars = len(text.strip())
    return max(1.0, chars / 4.0)
