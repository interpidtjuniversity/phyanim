"""Hybrid-mode scene: solver produces trajectory data, LLM code consumes it.

The generated code subclasses :class:`PhyAnimHybridScene` and overrides
``construct``.  Inside ``construct``, the code first builds and solves a
:class:`PhysicsAnimation` via :func:`solve_animation`, then uses the
returned :class:`TrajectoryData` to drive custom manim rendering.
"""

from __future__ import annotations

from phyanim.voiceover.scene import PhyAnimScene


class PhyAnimHybridScene(PhyAnimScene):
    """Base scene for hybrid render mode.

    LLM-generated code subclasses this. The generated ``construct``
    method typically looks like::

        class GeneratedScene(PhyAnimHybridScene):
            def construct(self):
                self.setup_speech(TTS_CONFIG)
                # 1. Build and solve physics animation
                animation = build_animation()
                trajectory = solve_animation(animation)

                # 2. Custom rendering using trajectory data
                # ===== LLM render code =====
                ball = Circle(radius=0.2, color=YELLOW)
                tracker = create_tracker(0.0)
                attach_position_updater(ball, trajectory, "ball", tracker)
                self.add(ball)
                with self.voiceover(text="...") as vo:
                    self.play(tracker.animate.set_value(trajectory.total_time),
                              run_time=trajectory.total_time, rate_func=linear)
                    self.wait(vo.duration)
                # ===== end LLM code =====
    """

    def construct(self) -> None:
        """Default construct — overridden by generated code."""
        self.setup_speech()
