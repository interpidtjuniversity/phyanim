
from typing import Callable

from phyanim.core.context import PhysicsContext, Context
from phyanim.core.enhance.annotation import Annotation
from phyanim.core.enhance.transition import Transition

from phyanim.solver.annotation_solver import DefaultAnnotationSolver
from phyanim.solver.transition_solver import DefaultTransitionSolver
from phyanim.core.timeline import TimelineResult
from phyanim.core.enhance.timewrapper import TimeWrapper

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

        # 这里是render层的子动画
        self.sub_animation : dict[str, dict] = {}

        # 可以后续在layer层继续加入其它context
        self.contexts: list[Context] = []
    
    def add_sub_animation(self, time_wrapper: TimeWrapper, annotations: list[Annotation] = None, transitions: list[Transition] = None) -> None:
        if self.name_space != "render":
            raise ValueError("Only render layer can add sub animation")
        self.sub_animation[time_wrapper.id] = {
            "time_wrapper": time_wrapper,
            "annotations": annotations if annotations is not None else [],
            "transitions": transitions if transitions is not None else [],
        }

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

    def solve(self, physics_ctx: PhysicsContext, result: TimelineResult) -> None:
        
        if self.name_space == "physics":
            self.contexts.clear()
            self.solve_physics_annotation(physics_ctx)
            self.solve_physics_transition(physics_ctx)
        elif self.name_space == "render":
            self.contexts.clear()
            self.solve_render_annotation(physics_ctx, result.time_wrapper_ranges, result.render_range_map)
            self.solve_render_transition(physics_ctx, result.time_wrapper_ranges, result.render_range_map)
            
        self.set_time_mapping_func(result.render_to_physics_mapping_func, result.physics_to_render_mapping_func)
    
    def solve_physics_annotation(self, physics_ctx: PhysicsContext) -> None:
        self.contexts.append(DefaultAnnotationSolver(annotations=self.annotations, sample_dt=self.sample_dt).solve_physics(physics_ctx))

    def solve_physics_transition(self, physics_ctx: PhysicsContext) -> None:
        self.contexts.append(DefaultTransitionSolver(transitions=self.transitions, sample_dt=self.sample_dt).solve_physics(physics_ctx))
    
    def solve_render_annotation(self, physics_ctx: PhysicsContext, time_wrapper_ranges: list[tuple[float, float, str]], render_range_map: dict[str, dict[str, tuple[float, float]]]) -> None:
        annotation_belongs_map = {}
        annotations = {}
        for item in self.sub_animation.values():
            for annotation in item["annotations"]:
                annotations[annotation.id] = annotation
                annotation_belongs_map[annotation.id] = item["time_wrapper"].id
        self.contexts.append(DefaultAnnotationSolver(annotations=annotations, sample_dt=self.sample_dt).solve_render(physics_ctx, time_wrapper_ranges, render_range_map, annotation_belongs_map))

    def solve_render_transition(self, physics_ctx: PhysicsContext, time_wrapper_ranges: list[tuple[float, float, str]], render_range_map: dict[str, dict[str, tuple[float, float]]]) -> None:
        transition_belongs_map = {}
        transitions = {}
        for item in self.sub_animation.values():
            for transition in item["transitions"]:
                transitions[transition.id] = transition
                transition_belongs_map[transition.id] = item["time_wrapper"].id
        self.contexts.append(DefaultTransitionSolver(transitions=transitions, sample_dt=self.sample_dt).solve_render(physics_ctx, time_wrapper_ranges, render_range_map, transition_belongs_map))

    def get_entities(self, tracker: ValueTracker, physics_ctx: PhysicsContext) -> list[Mobject]:
        """获取展示层中的实体，例如 annotations、transitions、analysis items。"""
        entities = []
        for ctx in self.contexts:
            entities.extend(ctx.get_entities(tracker, physics_ctx))
        return entities

    def set_time_mapping_func(self, render_to_physics_mapping_func: Callable[[float], float], physics_to_render_mapping_func: Callable[[float], float]) -> None:
        for ctx in self.contexts:
            ctx.set_render_to_physics_mapping_func(render_to_physics_mapping_func)
            ctx.set_physics_to_render_mapping_func(physics_to_render_mapping_func)