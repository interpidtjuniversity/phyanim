
from typing import Callable

from phyanim.core.context import PhysicsContext, Context
from phyanim.core.enhance.annotation import Annotation
from phyanim.core.enhance.transition import Transition

from phyanim.solver.annotation_solver import DefaultAnnotationSolver
from phyanim.solver.transition_solver import DefaultTransitionSolver

from manim import Mobject, ValueTracker

class Layer:

    def __init__(
        self,
        id: str = None,
        name_space: str = None,
        sample_dt: float = 1 / 60,
    ):
        if id is None or name_space is None:
            raise ValueError("id and name_space must be provided")
        self.id = id
        self.name_space = name_space
        self.sample_dt = sample_dt

        self.annotations : dict[str, Annotation] = {}
        self.transitions : dict[str, Transition] = {}

        # 可以后续在layer层继续加入其它context
        self.contexts: list[Context] = []

    def add_annotation(self, annotation: Annotation) -> None:
        if annotation.id is None:
            raise ValueError(
                f"Annotation '{annotation}' id should not be None or empty"
            )
        self.annotations[annotation.id] = annotation
    
    def add_transition(self, transition: Transition) -> None:
        if transition.id is None:
            raise ValueError(
                f"Transition '{transition}' id should not be None or empty"
            )
        self.transitions[transition.id] = transition

    def solve(self, physics_ctx: PhysicsContext, time_wrapper_ranges: list[tuple[float, float, str]]) -> None:
        # 求解展示层的东西
        self.contexts.clear()
        self.solve_annotation(physics_ctx, time_wrapper_ranges)
        self.solve_transition(physics_ctx, time_wrapper_ranges)
    
    def solve_annotation(self, physics_ctx: PhysicsContext, time_wrapper_ranges: list[tuple[float, float, str]]) -> None:
        self.contexts.append(DefaultAnnotationSolver(annotations=self.annotations, sample_dt=self.sample_dt).solve(physics_ctx, self.name_space, time_wrapper_ranges))

    def solve_transition(self, physics_ctx: PhysicsContext, time_wrapper_ranges: list[tuple[float, float, str]]) -> None:
        self.contexts.append(DefaultTransitionSolver(transitions=self.transitions, sample_dt=self.sample_dt).solve(physics_ctx, self.name_space, time_wrapper_ranges))

    def get_entities(self, tracker: ValueTracker) -> list[Mobject]:
        """获取展示层中的实体，例如 annotations、transitions、analysis items。"""
        # physics_ctx 的实体不在这里添加
        entities = []
        for ctx in self.contexts:
            entities.extend(ctx.get_entities(tracker))
        return entities

    def set_time_mapping_func(self, render_to_physics_mapping_func: Callable[[float], float], physics_to_render_mapping_func: Callable[[float], float]) -> None:
        for ctx in self.contexts:
            ctx.set_render_to_physics_mapping_func(render_to_physics_mapping_func)
            ctx.set_physics_to_render_mapping_func(physics_to_render_mapping_func)