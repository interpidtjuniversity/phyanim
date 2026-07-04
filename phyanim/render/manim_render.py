from __future__ import annotations

from typing import Any

import numpy as np
from manim import Scene, ValueTracker, linear, config, FadeOut, RIGHT

from phyanim.core.animation import PhysicsAnimation
from phyanim.core.timeline import Timeline

from phyanim.core.layer import Layer
from phyanim.voiceover.tts_config import TTSConfig
from phyanim.voiceover.narration import NarrationPlayer

try:
    from manim_voiceover import VoiceoverScene as _VoiceoverBase

    _HAS_VOICEOVER = True
except ImportError:
    _HAS_VOICEOVER = False

    class _VoiceoverBase(Scene):  # type: ignore[no-redef]
        """Fallback when manim-voiceover is not installed."""

        def set_speech_service(self, service) -> None:
            pass

# Supported transition types between scenes.
_TRANSITION_TYPES = ("fade", "cut", "slide")
_TRANSITION_DURATION = 0.3


class PhyAnimationMultiLayerScene2D(_VoiceoverBase):
    """Renders one or more :class:`PhysicsAnimation` instances sequentially.

    Supports optional TTS narration via :meth:`set_narration` and
    :meth:`set_tts_config`.

    Single-scene usage (backward compatible)::

        scene = PhyAnimationMultiLayerScene2D()
        scene.set_animation(animation)
        scene.render()

    Multi-scene usage::

        scene = PhyAnimationMultiLayerScene2D()
        scene.add_scene("title",       anim1, transition="fade")
        scene.add_scene("simulation",  anim2, transition="slide")
        scene.render()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.animations: list[tuple[str, str, PhysicsAnimation]] = []
        self._tts_config: dict | TTSConfig | None = None
        self._narration: list[dict] | None = None
        # Auto-configure xelatex + CJK font for MathTex.
        self._setup_tex_template()

    def _setup_tex_template(self) -> None:
        """Set up xelatex template with CJK font support.

        Called automatically in __init__ — LLM code does NOT need
        to call MathTex.set_default() or configure any template.
        """
        from manim import MathTex, TexTemplate
        import subprocess

        cjk_fonts = [
            "WenQuanYi Micro Hei",
            "Noto Sans CJK SC",
            "SimSun",
            "SimHei",
            "Microsoft YaHei",
            "AR PL UMing CN",
        ]

        chosen_font = None
        try:
            result = subprocess.run(
                ["fc-list", ":family"],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5,
            )
            available = set(result.stdout.strip().split("\n"))
            for font in cjk_fonts:
                if font in available:
                    chosen_font = font
                    break
        except Exception:
            pass

        tpl = TexTemplate()
        tpl.tex_compiler = "xelatex"
        tpl.output_format = ".xdv"
        if chosen_font:
            tpl.add_to_preamble("\\usepackage{fontspec}")
            tpl.add_to_preamble(f"\\setmainfont{{{chosen_font}}}")
        MathTex.set_default(tex_template=tpl)

    def set_animation(self, animation: PhysicsAnimation) -> None:
        """Register a single animation (backward compatible)."""
        self.animations = [("scene", "fade", animation)]

    def add_scene(self, name: str, animation: PhysicsAnimation, transition: str = "fade") -> None:
        """Register a named scene with an optional transition type.

        Args:
            name: Scene identifier used for ``next_section``.
            animation: The :class:`PhysicsAnimation` to render.
            transition: One of ``"fade"``, ``"cut"``, ``"slide"``.
        """
        if transition not in _TRANSITION_TYPES:
            raise ValueError(
                f"Unknown transition '{transition}'. Supported: {_TRANSITION_TYPES}"
            )
        self.animations.append((name, transition, animation))

    def set_tts_config(self, config: dict | TTSConfig) -> None:
        """Configure TTS for engine-mode narration."""
        self._tts_config = config

    def set_narration(self, narration: list[dict]) -> None:
        """Set narration segments for engine-mode voiceover.

        Each dict should have ``text`` (str) and ``at`` (float, render seconds).
        """
        self._narration = narration

    def set_camera_frame(
        self,
        width: float | None = None,
        height: float | None = None,
        center: tuple[float, float] | None = None,
    ) -> None:
        """Configure the camera frame to fit the animation content.

        Call this when objects may move outside the default frame bounds.

        Parameters
        ----------
        width:
            Frame width in manim units.  None = keep default.
        height:
            Frame height in manim units.  None = keep default.
        center:
            (x, y) center of the frame.  None = keep default.

        Example::

            scene.set_camera_frame(width=20, height=12, center=(0, 5))
        """
        if height is not None:
            self.camera.frame_height = height
        if width is not None:
            self.camera.frame_width = width
        if center is not None:
            self.camera.frame_center = np.array([center[0], center[1], 0.0])

    def set_camera_config(
        self,
        frame_height: float | None = None,
        frame_width: float | None = None,
        frame_center: tuple[float, float] | None = None,
    ) -> None:
        """Configure camera frame to fit the animation content.

        Call this when objects may move outside the default frame.

        Parameters
        ----------
        frame_height:
            Height of the visible area in manim units.
        frame_width:
            Width of the visible area (defaults to height * 16/9).
        frame_center:
            (x, y) center of the frame.
        """
        self._camera_config = {
            "frame_height": frame_height,
            "frame_width": frame_width,
            "frame_center": frame_center,
        }

    def set_camera_config(
        self,
        frame_width: float | None = None,
        frame_height: float | None = None,
        frame_center: tuple[float, float] | None = None,
    ) -> None:
        """Configure camera frame to fit the animation content.

        Call this when objects may move outside the default frame.

        Parameters
        ----------
        frame_width:
            Width of the visible area in manim units.
        frame_height:
            Height of the visible area in manim units.
        frame_center:
            (x, y) center of the frame.
        """
        self._camera_config = {
            "frame_width": frame_width,
            "frame_height": frame_height,
            "frame_center": frame_center,
        }

    def construct(self) -> None:
        config["disable_caching"] = True

        for name, transition, anim in self.animations:
            self.next_section(name)
            mobs = self._render_one(anim)

            # Inter-scene transition.
            if transition == "fade" and mobs:
                self.play(*[FadeOut(m) for m in mobs], run_time=_TRANSITION_DURATION)
            elif transition == "slide" and mobs:
                self.play(*[m.animate.shift(RIGHT * 8) for m in mobs], run_time=_TRANSITION_DURATION)
            elif transition == "cut":
                self.wait(0.2)

            # Mandatory cleanup: remove mobjects and clear updaters so that
            # the next scene's updaters don't fire on stale trackers.
            for m in mobs:
                m.clear_updaters()
            if mobs:
                self.remove(*mobs)

    def _render_one(self, animation: PhysicsAnimation) -> list:
        """Render a single PhysicsAnimation; return added mobjects for cleanup."""
        if not animation.solved:
            animation.solve()

        physics_layer = animation.get_physics_layer()
        render_layer = animation.get_render_layer()

        # Solve timeline (time wrappers).
        timeline = Timeline()
        for item in render_layer.sub_animation.values():
            timeline.add_time_wrapper(item["time_wrapper"])
        result = timeline.solve(animation.physics_ctx)

        # Independent tracker per scene.
        render_tracker = ValueTracker(0.0)
        mobs: list = []

        # Main physics entities.
        for obj in animation.physics_ctx.get_entities(render_tracker):
            self.add(obj)
            mobs.append(obj)

        # Physics layer (annotations / transitions on physics time).
        physics_layer.solve(animation.physics_ctx, result)
        for entity in physics_layer.get_entities(render_tracker, animation.physics_ctx):
            self.add(entity)
            mobs.append(entity)

        # Render layer (annotations / transitions on render time).
        render_layer.solve(animation.physics_ctx, result)
        for entity in render_layer.get_entities(render_tracker, animation.physics_ctx):
            self.add(entity)
            mobs.append(entity)

        # Drive the single linear play for this scene.
        total_time = animation.physics_ctx.total_time
        total_time = animation.physics_ctx.physics_to_render_mapping_func(total_time)

        # Setup TTS if configured.
        if self._tts_config is not None:
            tts = TTSConfig.from_dict(self._tts_config) if isinstance(self._tts_config, dict) else self._tts_config
            service = tts.create_service()
            if service is not None:
                self.set_speech_service(service)

        if self._narration:
            # Narration-driven playback: interleave voiceover with tracker.
            player = NarrationPlayer(self._narration, total_time)
            player.play(self, render_tracker)
        else:
            self.play(
                render_tracker.animate.set_value(total_time),
                run_time=total_time,
                rate_func=linear,
            )
        return mobs
