from phyanim.core.enhance.timewrapper import TimeWrapper
from phyanim.core.context import PhysicsContext
from phyanim.solver.timewrapper_solver import DefaultTimeWrapperSolver, TimelineResult

class Timeline:
    def __init__(self):
    # 时间缩放相关
        self.time_wrappers: dict[str, TimeWrapper] = {}

    def add_time_wrapper(self, time_wrapper: TimeWrapper):
        self.time_wrappers[time_wrapper.id] = time_wrapper

    def solve(self, phy_context: PhysicsContext) -> TimelineResult:
        result = DefaultTimeWrapperSolver(time_wrappers=self.time_wrappers).solve(phy_context)
        phy_context.set_render_to_physics_mapping_func(result.render_to_physics_mapping_func)
        phy_context.set_physics_to_render_mapping_func(result.physics_to_render_mapping_func)
        return result