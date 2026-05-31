from typing import Tuple, Callable

from dataclasses import field
from phyanim.core.enhance.timewrapper import TimeWrapper
from phyanim.core.context import PhysicsContext
from phyanim.solver.timewrapper_solver import DefaultTimeWrapperSolver

class Timeline:
    def __init__(self):
    # 时间缩放相关
        self.time_wrappers: dict[str, TimeWrapper] = {}

    def add_time_wrapper(self, time_wrapper: TimeWrapper):
        self.time_wrappers[time_wrapper.id] = time_wrapper

    def solve(self, phy_context: PhysicsContext) -> Tuple[float, Callable[[float], float]]:
        total_time, time_mapping_func = DefaultTimeWrapperSolver(time_wrappers=self.time_wrappers).solve(phy_context)
        phy_context.set_time_mapping_to_physics(time_mapping_func)
        return total_time, time_mapping_func
