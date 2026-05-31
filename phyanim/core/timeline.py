from typing import Tuple, Callable

from phyanim.core.enhance.timewrapper import TimeWrapper
from phyanim.core.context import PhysicsContext
from phyanim.solver.timewrapper_solver import DefaultTimeWrapperSolver

class Timeline:
    def __init__(self):
    # 时间缩放相关
        self.time_wrappers: dict[str, TimeWrapper] = {}

    def add_time_wrapper(self, time_wrapper: TimeWrapper):
        self.time_wrappers[time_wrapper.id] = time_wrapper

    def solve(self, phy_context: PhysicsContext) -> Tuple[float, Callable[[float], float], Callable[[float], float]]:
        total_time, render_to_physics_mapping_func, physics_to_render_mapping_func = DefaultTimeWrapperSolver(time_wrappers=self.time_wrappers).solve(phy_context)
        phy_context.set_render_to_physics_mapping_func(render_to_physics_mapping_func)
        phy_context.set_physics_to_render_mapping_func(physics_to_render_mapping_func)
        return total_time, render_to_physics_mapping_func, physics_to_render_mapping_func
