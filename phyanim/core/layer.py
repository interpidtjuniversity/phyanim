
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
        id: str = None
    ):
        if id is None:
            raise ValueError("id must be provided")
        self.id = id
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

    def solve(self, physics_ctx: PhysicsContext) -> None:
        # 求解展示层的东西
        self.contexts.clear()
        self.solve_annotation(physics_ctx)
        self.solve_transition(physics_ctx)
    
    def solve_annotation(self, physics_ctx: PhysicsContext):
        self.contexts.append(DefaultAnnotationSolver(annotations=self.annotations).solve(physics_ctx))

    def solve_transition(self, physics_ctx: PhysicsContext):
        self.contexts.append(DefaultTransitionSolver(transitions=self.transitions).solve(physics_ctx))

    def get_entities(self, tracker: ValueTracker) -> list[Mobject]:
        """获取展示层中的实体，例如 annotations、transitions、analysis items。"""
        # physics_ctx 的实体不在这里添加
        entities = []
        for ctx in self.contexts:
            entities.extend(ctx.get_entities(tracker))
        return entities

    def set_time_mapping_func(self, time_mapping_func: Callable[[float], float]) -> None:
        for ctx in self.contexts:
            ctx.set_time_mapping_to_physics(time_mapping_func)
