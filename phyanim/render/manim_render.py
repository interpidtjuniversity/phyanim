from __future__ import annotations

from manim import Scene, ValueTracker, linear, config, FadeOut

from phyanim.core.animation import PhysicsAnimation
from phyanim.core.timeline import Timeline

from phyanim.core.layer import Layer

# Supported transition types between scenes.
_TRANSITION_TYPES = ("fade", "cut", "slide")
_TRANSITION_DURATION = 0.3


class PhyAnimationMultiLayerScene2D(Scene):
    """Renders one or more :class:`PhysicsAnimation` instances sequentially.

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

    def construct(self) -> None:
        config["disable_caching"] = True

        for name, transition, anim in self.animations:
            self.next_section(name)
            mobs = self._render_one(anim)

            # Inter-scene transition.
            if transition == "fade" and mobs:
                self.play(*[FadeOut(m) for m in mobs], run_time=_TRANSITION_DURATION)
            elif transition == "slide" and mobs:
                self.play(*[m.animate.shift(8 * 8) for m in mobs], run_time=_TRANSITION_DURATION)
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

        self.play(
            render_tracker.animate.set_value(total_time),
            run_time=total_time,
            rate_func=linear,
        )
        return mobs
