"""Code-mode scene: LLM provides the full construct method body.

The generated code subclasses :class:`PhyAnimCodeScene` and overrides
``construct``.  The framework handles TTS setup and provides the
``voiceover`` context manager.
"""

from __future__ import annotations

from phyanim.voiceover.scene import PhyAnimScene


class PhyAnimCodeScene(PhyAnimScene):
    """Base scene for code render mode.

    LLM-generated code subclasses this and writes a full ``construct``
    method body using manim APIs directly.

    The generated code typically looks like::

        class GeneratedScene(PhyAnimCodeScene):
            def construct(self):
                self.setup_speech(TTS_CONFIG)
                # ===== LLM construct code =====
                ball = Circle(radius=0.2, color=YELLOW)
                self.add(ball)
                with self.voiceover(text="小球开始下落") as tracker:
                    self.play(ball.animate.shift(DOWN * 3), run_time=2.0)
                    self.wait(tracker.duration)
                # ===== end LLM code =====
    """

    def construct(self) -> None:
        """Default construct — overridden by generated code."""
        self.setup_speech()
