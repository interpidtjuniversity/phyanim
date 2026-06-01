from dataclasses import dataclass, field

from phyanim.core.context import PhysicsContext

from phyanim.core.enhance.timewrapper import TimeWrapper
from typing import Any, Tuple, Callable

@dataclass
class DefaultTimeWrapperSolver:

    total_time: float = None
    time_wrappers: dict[str, TimeWrapper] = field(default_factory=dict)

    def solve(self, physics_ctx: PhysicsContext) -> Tuple[float, Callable[[float], float], Callable[[float], float]] :
        # 先评估触发器的触发时间
        old_time = None
        trigger_map = {}
        wrapper_map = {}

        # 先检查触发，wrapper上的触发器都在物理层上评估
        for t in physics_ctx.times:
            for wrapper_id, wrapper in self.time_wrappers.items():
                event_t = wrapper.trigger(old_time, t, physics_ctx.build_eval_exper_func(wrapper.trigger))
                if event_t is not None:
                    # 如果是slow，则该区间慢放
                    if wrapper.type == "slow":
                        trigger_map[wrapper_id] = (event_t - wrapper.advance, event_t + wrapper.delay, wrapper_id)
                    # 如果是freeze，则该区间冻结
                    elif wrapper.type == "freeze":
                        trigger_map[wrapper_id] = (event_t, event_t, wrapper_id)
                old_time = t


        # 先检查，原始物理区间不能重合
        time_ranges = []
        for wrapper_id, time_range in trigger_map.items():
            time_ranges.append(time_range)
        # 按照起始时间排序
        time_ranges.sort(key=lambda x: x[0])
        for i in range(len(time_ranges) - 1):
            # 如果某一段的结束时间大于下一段的起始时间，说明重合
            if time_ranges[i][1] > time_ranges[i + 1][0]:
                raise ValueError(f"Time wrappers cannot overlap: {time_ranges[i][2]} and {time_ranges[i + 1][2]})")
        self.time_ranges = time_ranges
        
        # 先计算总时间
        total_time = 0
        last_physics_stop = 0
        # wrapper 时间
        wrapper_time_ranges = []
        for time_range in time_ranges:
            total_time += time_range[0] - last_physics_stop
            wrapper_start = total_time
            if self.time_wrappers[time_range[2]].type == "slow":
                # 加上时间缩放
                total_time += (time_range[1] - time_range[0]) / self.time_wrappers[time_range[2]].speed
            elif self.time_wrappers[time_range[2]].type == "freeze":
                # 加上冻结时间
                total_time += self.time_wrappers[time_range[2]].extend_to
            wrapper_end = total_time
            last_physics_stop = time_range[1]

            wrapper_time_ranges.append((wrapper_start, wrapper_end, time_range[2]))
            wrapper_map[time_range[2]] = (wrapper_start, wrapper_end, time_range[2])
        self.wrapper_time_ranges = wrapper_time_ranges
        
        # 加上最后一个物理区间的时间，如果有的话
        total_time += (physics_ctx.total_time - last_physics_stop)
        self.total_time = total_time

        def render_to_physics_mapping_func(t: float) -> float:
            # 查找t属于哪个wrapper_time_ranges的区间
            for idx, (wrapper_start, wrapper_end, _) in enumerate(self.wrapper_time_ranges):
                if wrapper_start <= t <= wrapper_end:
                    # 在某个完整区间里，区间左端物理时间点 + 占wrap区间的比例 * 物理区间长度
                    return self.time_ranges[idx][0] + (t - wrapper_start) * (self.time_ranges[idx][1] - self.time_ranges[idx][0]) / (wrapper_end - wrapper_start)
            
            # 寻找两侧的区间，返回物理left区间的右端点 + t - wrap left的右端点
            left_wrapper_idx = -1
            for left_idx, (_, wrapper_end, _) in enumerate(self.wrapper_time_ranges):
                if t > wrapper_end:
                    left_wrapper_idx = left_idx
                else:
                    break
        
            # 没有左侧区间，直接返回
            if left_wrapper_idx == -1:
                return t
            else:
                return self.time_ranges[left_wrapper_idx][1] + t - self.wrapper_time_ranges[left_wrapper_idx][1]
        
        def physics_to_render_mapping_func(t: float) -> Any:
            for idx, (start, end, _) in enumerate(self.time_ranges):
                if start <= t <= end:
                    # 如果是slow，则该区间慢放
                    if self.time_wrappers[wrapper_id].type == "slow":
                        return self.wrapper_time_ranges[idx][0] + (t - start) * (self.wrapper_time_ranges[idx][1] - self.wrapper_time_ranges[idx][0]) / (end - start)
                    # 如果是freeze，返回整个render_t区间，需要业务层自己做遍历
                    elif self.time_wrappers[wrapper_id].type == "freeze":
                        return (self.wrapper_time_ranges[idx][0], self.wrapper_time_ranges[idx][1])
            
            left_idx = -1
            for left_wrapper_idx, (_, end, _) in enumerate(self.time_ranges):
                if t > end:
                    left_idx = left_wrapper_idx
                else:
                    break

            # 没有左侧区间，直接返回
            if left_idx == -1:
                return t
            else:
                return self.wrapper_time_ranges[left_idx][1] + t - self.time_ranges[left_idx][1]

        self.render_to_physics_mapping_func = render_to_physics_mapping_func
        self.physics_to_render_mapping_func = physics_to_render_mapping_func

        render_range_map = {}
        for wrapper_id, (start, end, _) in trigger_map.items():
            render_range_map[wrapper_id] = {
                "physics_range": (start, end),
                "render_range": (wrapper_map[wrapper_id][0], wrapper_map[wrapper_id][1])
            }
        return TimelineResult(self.total_time, self.render_to_physics_mapping_func, self.physics_to_render_mapping_func, self.time_ranges, self.wrapper_time_ranges, render_range_map)


class TimelineResult:
    def __init__(self, 
        total_time: float, 
        render_to_physics_mapping_func: Callable[[float], float], 
        physics_to_render_mapping_func: Callable[[float], float],
        time_ranges: list[tuple[float, float, str]],
        time_wrapper_ranges: list[tuple[float, float, str]],
        render_range_map: dict[str, dict[str, tuple[float, float]]]
    ):  
        # 总渲染时间
        self.total_time = total_time
        self.render_to_physics_mapping_func = render_to_physics_mapping_func
        self.physics_to_render_mapping_func = physics_to_render_mapping_func
        # 物理时间区块
        self.time_ranges = time_ranges
        # 渲染时间区块
        self.time_wrapper_ranges = time_wrapper_ranges
        # time_wrapper的物理时间区块和渲染时间区块
        self.render_range_map = render_range_map