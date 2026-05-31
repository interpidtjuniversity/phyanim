from dataclasses import dataclass, field

from phyanim.core.context import PhysicsContext

from phyanim.core.enhance.timewrapper import TimeWrapper
from typing import Tuple, Callable

@dataclass
class DefaultTimeWrapperSolver:

    total_time: float = None
    time_wrappers: dict[str, TimeWrapper] = field(default_factory=dict)

    def solve(self, physics_ctx: PhysicsContext) -> Tuple[float, Callable[[float], float]] :
        # 先评估触发器的触发时间
        old_time = None
        trigger_map = {}

        # 先检查触发
        for t in physics_ctx.times:
            for wrapper_id, wrapper in self.time_wrappers.items():
                event_t = wrapper.trigger(old_time, t, physics_ctx.build_eval_exper_func(wrapper.trigger))
                if event_t is not None:
                    trigger_map[wrapper_id] = (event_t - wrapper.advance, event_t + wrapper.delay, wrapper_id)
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
            # 加上时间缩放
            total_time += (time_range[1] - time_range[0]) / self.time_wrappers[time_range[2]].speed
            wrapper_end = total_time
            last_physics_stop = time_range[1]

            wrapper_time_ranges.append((wrapper_start, wrapper_end, time_range[2]))
        self.wrapper_time_ranges = wrapper_time_ranges
        
        # 加上最后一个物理区间的时间，如果有的话
        total_time += (physics_ctx.total_time - last_physics_stop)
        self.total_time = total_time

        def time_mapping_func(t: float) -> float:
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
        
        self.time_mapping_func = time_mapping_func
        return self.total_time, self.time_mapping_func
