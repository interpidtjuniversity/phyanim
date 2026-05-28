from typing import Any, Callable
from scipy.optimize import brentq
import math

from manim import Mobject

class ActiveRule:
    """注释激活规则，定义了注释在什么时间激活。"""
    def __init__(self):
        pass

class Trigger:
    """触发事件，定义了在什么时间注释激活。"""
    def __init__(self, expression: str, direction : int = 0):
        if direction not in (-1, 0, 1):
            raise ValueError("direction must be -1, 0, or 1")
        self.expression = expression
        self.direction = direction
    
    def __call__(self, old_time: float, new_time: float, func: Callable[[float], Any]) -> Any:
        """
            评估触发事件是否在指定时刻t小范围内范围内发生。
            :param old_time: 上次渲染的时刻
            :param new_time: 这次渲染的时刻
            :param func: 评估函数
            :return: 是否发生
        """
        raise NotImplementedError("Trigger is not implemented.")

class CrossingTrigger(Trigger):
    """触发事件，定义了在什么时间注释激活。"""
    def __init__(self, expression: str, direction : int = 0, eps: float = 1e-12, xtol: float = 1e-10, rtol: float = 1e-10, min_interval: float = 1e-6):
        super().__init__(expression, direction)
        self.eps = eps
        self.xtol = xtol
        self.rtol = rtol
        self.min_interval = min_interval
    
    def __call__(self, old_time: float | None, new_time: float, func: Callable[[float], float]) -> float:
        """
            评估触发事件是否在指定时刻t小范围内范围内发生。
            :param old_time: 上次渲染的时刻
            :param new_time: 这次渲染的时刻
            :param func: 评估函数
            :return: 是否发生
        """
        if old_time is None or new_time <= old_time:
            return None

        f0 = float(func(old_time))
        f1 = float(func(new_time))

        if not math.isfinite(f0) or not math.isfinite(f1):
            return None

        if not self._matched(f0, f1):
            return None

        return self._refine(func, old_time, new_time, f0, f1)

    # 方向判断 
    def _sign(self, x: float) -> int:
        if x > self.eps:
            return 1
        if x < -self.eps:
            return -1
        return 0

    def _matched(self, f0: float, f1: float) -> bool:
        s0 = self._sign(f0)
        s1 = self._sign(f1)

        # 存在一个零点或者穿过零点
        if self.direction == 0:
            return s0 == 0 or s1 == 0 or s0 != s1

        if self.direction == 1:
            return s0 < 0 and s1 >= 0

        if self.direction == -1:
            return s0 > 0 and s1 <= 0

        return False

    def _refine(
        self,
        func: Callable[[float], float],
        t0: float,
        t1: float,
        f0: float,
        f1: float,
    ) -> float:
        # 小于阈值全部直接视为真实零点
        if abs(f0) <= self.eps:
            return t0

        if abs(f1) <= self.eps:
            return t1

        # brentq 要求严格异号
        if f0 * f1 > 0:
            # eps 逻辑可能导致 matched，但不是严格 bracket
            return self._linear_refine(t0, t1, f0, f1)

        return float(
            brentq(
                func,
                t0,
                t1,
                xtol=self.xtol,
                rtol=self.rtol,
            )
        )

    # 二分查找
    def _linear_refine(
        self,
        t0: float,
        t1: float,
        f0: float,
        f1: float,
    ) -> float:
        denom = abs(f0) + abs(f1)
        if denom <= self.eps:
            return 0.5 * (t0 + t1)

        alpha = abs(f0) / denom
        return t0 + alpha * (t1 - t0)


class AnnotationActivation:
    """注释激活规则，定义了注释在什么时间激活。"""
    def __init__(self, trigger: Trigger, start_trigger: Trigger = None, end_trigger: Trigger = None, advance: float = None, delay: float = None):
        self.trigger = trigger
        self.start_trigger = start_trigger
        self.end_trigger = end_trigger
        self.advance = advance
        self.delay = delay
    
    # build的实质是赋值ctx的timeline
    def build_rule(self, ann_type: str) -> ActiveRule:

        match ann_type:
            case "while":
                if self.trigger is None:
                    raise ValueError("while annotation must have a trigger")
                return ActiveWhile(self.trigger)
            case "time_range":
                if self.trigger is None or self.advance is None or self.delay is None:
                    raise ValueError("time_range annotation must have a trigger, advance, delay")
                return ActivateEventTimeRange(self.trigger, self.advance, self.delay)
            case "between":
                if self.start_trigger is None or self.end_trigger is None:
                    raise ValueError("between annotation must have a start_trigger and end_trigger")
                return ActiveBetween(self.start_trigger, self.end_trigger)
            case _:
                raise ValueError(f"Unknown annotation type: {ann_type}")

class Annotation:
    def __init__(self, id: str, ann_type: str, position: tuple[str, str], content: Any, activation: AnnotationActivation):
        self.id = id
        self.ann_type = ann_type
        self.position = position
        # 内容对象
        self.content = content
        # 展示对象
        self.obj : Mobject = None
        self.rule : ActiveRule = activation.build_rule(self.ann_type)


class ActiveWhile(ActiveRule):
    """条件为真时激活注释。"""
    def __init__(
        self,
        trigger: Trigger,
    ):
        self.trigger = trigger

class ActivateEventTimeRange(ActiveRule):
    """在指定事件触发范围内激活注释。(t-advance, t+delay)"""
    def __init__(
        self,
        trigger: Trigger,
        advance: float,
        delay: float,
    ):
        self.trigger = trigger
        self.advance = advance
        self.delay = delay

class ActiveBetween(ActiveRule):
    """在区间内激活注释。"""
    def __init__(
        self,
        start_trigger: Trigger,
        end_trigger: Trigger
    ):
        self.start_trigger = start_trigger
        self.end_trigger = end_trigger


